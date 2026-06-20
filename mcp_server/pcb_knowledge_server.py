import os
import sys
import json
import contextlib
import traceback
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# ============================================================
# MCP RAG 检索服务：自动选择空闲 GPU
# ============================================================
# 这个文件只服务 MCP/RAG embedding 检索，不影响主进程 YOLO/Qwen。
#
# 关键点：
# 1. SentenceTransformer 默认会使用可见的第一张 CUDA 卡。
# 2. 因此必须在导入 rag.knowledge_search / sentence_transformers / torch 之前，
#    先设置 CUDA_VISIBLE_DEVICES。
# 3. 默认 PCB_MCP_DEVICE=auto：用 nvidia-smi 找空闲显存最多的物理 GPU。
# 4. 选中物理 GPU N 后，子进程内部会看到它为 cuda:0。
# 5. 如果没有满足阈值的 GPU，才降级 CPU。
#
# 可选环境变量：
#   PCB_MCP_DEVICE=auto        自动选空闲 GPU，默认
#   PCB_MCP_DEVICE=cuda:3      强制使用物理 GPU 3
#   PCB_MCP_DEVICE=3           强制使用物理 GPU 3
#   PCB_MCP_DEVICE=cpu         强制 CPU
#   PCB_MCP_MIN_FREE_MIB=2048  自动选择时要求至少 2048MiB 空闲显存
#   PCB_MCP_EXCLUDE_GPUS=0,1   自动选择时排除 GPU 0/1
# ============================================================


def _query_gpu_free_memory() -> List[Tuple[int, int]]:
    """Return [(physical_gpu_index, free_mib), ...] from nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception as e:
        print(f"[MCP SERVER] nvidia-smi query failed: {repr(e)}", file=sys.stderr, flush=True)
        return []

    gpus: List[Tuple[int, int]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        try:
            idx_s, free_s = [x.strip() for x in line.split(",")[:2]]
            gpus.append((int(idx_s), int(free_s)))
        except Exception:
            print(f"[MCP SERVER] failed to parse nvidia-smi line: {line!r}", file=sys.stderr, flush=True)
    return gpus


def _parse_physical_gpu_index(value: str) -> Optional[int]:
    value = (value or "").strip().lower()
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    if value.isdigit():
        return int(value)
    return None


def _configure_mcp_device() -> str:
    raw_device = (
        os.environ.get("PCB_MCP_DEVICE")
        or os.environ.get("PCB_RAG_DEVICE")
        or "auto"
    )
    raw_device = str(raw_device).strip().lower()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if raw_device in {"cpu", "none", "off", "disable"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[MCP SERVER] using CPU because PCB_MCP_DEVICE=cpu", file=sys.stderr, flush=True)
        return "cpu"

    forced_idx = _parse_physical_gpu_index(raw_device)
    if forced_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(forced_idx)
        print(
            f"[MCP SERVER] forced physical GPU {forced_idx}; internal device will be cuda:0",
            file=sys.stderr,
            flush=True,
        )
        return f"physical cuda:{forced_idx} -> internal cuda:0"

    # auto mode
    try:
        min_free_mib = int(os.environ.get("PCB_MCP_MIN_FREE_MIB", "2048"))
    except Exception:
        min_free_mib = 2048

    exclude_raw = os.environ.get("PCB_MCP_EXCLUDE_GPUS", "")
    exclude = set()
    for x in exclude_raw.split(","):
        x = x.strip()
        if x.isdigit():
            exclude.add(int(x))

    gpus = _query_gpu_free_memory()
    if not gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[MCP SERVER] no GPU info available; fallback to CPU", file=sys.stderr, flush=True)
        return "cpu"

    for idx, free in gpus:
        print(f"[MCP SERVER] physical cuda:{idx} free={free}MiB", file=sys.stderr, flush=True)

    candidates = [(idx, free) for idx, free in gpus if idx not in exclude and free >= min_free_mib]
    if not candidates:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(
            f"[MCP SERVER] no GPU has >= {min_free_mib}MiB free; fallback to CPU",
            file=sys.stderr,
            flush=True,
        )
        return "cpu"

    selected_idx, selected_free = max(candidates, key=lambda x: x[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_idx)
    print(
        f"[MCP SERVER] auto selected physical cuda:{selected_idx} "
        f"free={selected_free}MiB; internal device will be cuda:0",
        file=sys.stderr,
        flush=True,
    )
    return f"physical cuda:{selected_idx} -> internal cuda:0"


_MCP_DEVICE_LABEL = _configure_mcp_device()

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("pcb-knowledge-server")

_searcher: Optional[Any] = None
_searcher_cls: Optional[Any] = None


def _to_jsonable(value: Any) -> Any:
    """把 numpy / tensor / Path / set 等对象转换成 JSON 可序列化对象。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return str(value)


def _load_searcher_class():
    """
    MCP stdio 模式下 stdout 只能用于 JSON-RPC 协议。
    rag.knowledge_search 在 import 阶段如果有 print，会污染协议。
    所以 import 也重定向 stdout 到 stderr。
    """
    global _searcher_cls
    if _searcher_cls is None:
        with contextlib.redirect_stdout(sys.stderr):
            from rag.knowledge_search import PCBKnowledgeSearcher
        _searcher_cls = PCBKnowledgeSearcher
    return _searcher_cls


def get_searcher():
    global _searcher
    if _searcher is None:
        cls = _load_searcher_class()
        with contextlib.redirect_stdout(sys.stderr):
            _searcher = cls()

        # 打印实际 embedding 模型设备，便于确认是否跑到了自动选择的 GPU。
        try:
            model = getattr(_searcher, "model", None)
            device = getattr(model, "device", None)
            if device is None and model is not None and hasattr(model, "_target_device"):
                device = getattr(model, "_target_device")
            print(f"[MCP SERVER] searcher initialized on device={device}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MCP SERVER] warning: failed to inspect model device: {repr(e)}", file=sys.stderr, flush=True)

    return _searcher


def normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    return _to_jsonable(
        {
            "score": item.get("score"),
            "doc_id": item.get("doc_id"),
            "defect_type": item.get("defect_type"),
            "title": item.get("title"),
            "severity": item.get("severity"),
            "visual_features": item.get("visual_features"),
            "possible_causes": item.get("possible_causes"),
            "electrical_symptoms": item.get("electrical_symptoms"),
            "detection_methods": item.get("detection_methods"),
            "repair_suggestions": item.get("repair_suggestions"),
            "risk": item.get("risk"),
            "prevention": item.get("prevention"),
            "tags": item.get("tags"),
        }
    )


@mcp.tool()
def pcb_knowledge_search(
    query: str,
    defect_type: str = "",
    top_k: int = 3,
) -> str:
    """
    Search PCB fault-maintenance knowledge.

    Args:
        query: Defect description or maintenance question.
        defect_type: Optional defect type filter, such as 短路、开路、鼠咬、毛刺、多余铜、漏孔.
        top_k: Number of knowledge entries to return.

    Return:
        JSON string with fields: tool/query/defect_type/top_k/num_results/results/error/device.
    """
    try:
        query = (query or "").strip()
        defect_type = (defect_type or "").strip()

        if not query and not defect_type:
            query = "PCB 缺陷检测方法、风险分析和维修建议"

        try:
            top_k = int(top_k)
        except Exception:
            top_k = 3

        if top_k <= 0:
            top_k = 3
        if top_k > 10:
            top_k = 10

        defect_filter = defect_type or None

        searcher = get_searcher()

        # encode/search 过程中如果有 print，也不能进入 stdout。
        with contextlib.redirect_stdout(sys.stderr):
            results = searcher.search(
                query=query,
                defect_type=defect_filter,
                top_k=top_k,
            )

        normalized = [normalize_result(x) for x in results]

        payload = {
            "tool": "pcb_knowledge_search",
            "query": query,
            "defect_type": defect_type,
            "top_k": top_k,
            "num_results": len(normalized),
            "results": normalized,
            "error": "",
            "device": _MCP_DEVICE_LABEL,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        payload = {
            "tool": "pcb_knowledge_search",
            "query": query if "query" in locals() else "",
            "defect_type": defect_type if "defect_type" in locals() else "",
            "top_k": top_k if "top_k" in locals() else 3,
            "num_results": 0,
            "results": [],
            "error": repr(e),
            "device": _MCP_DEVICE_LABEL,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    return json.dumps(_to_jsonable(payload), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
