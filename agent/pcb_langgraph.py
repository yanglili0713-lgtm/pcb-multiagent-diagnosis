import sys
import os
import re
import json
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import torch
from PIL import Image
from ultralytics import YOLO
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from langgraph.graph import StateGraph, START, END

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from agent.pcb_graph_state import PCBGraphState
from agent.diagnosis_pipeline import run_diagnosis_pipeline

YOLO_MODEL = PROJECT_ROOT / "output/yolo_pcb_detect/yolo11n_pcb_1024_pretrained_noamp/weights/best.pt"
BASE_MODEL = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "output/qwen25vl_7b_pcb_crop_cls_full"
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server" / "pcb_knowledge_server.py"

DEFECT_TYPES = ["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"]
FINAL_TYPES = DEFECT_TYPES + ["未知"]
HIGH_RISK_DEFAULT = ["短路", "开路", "漏孔"]


def _normalize_cuda_device(value: str | None, default: str = "cuda:0") -> str:
    """Return a torch device string.

    Environment examples:
      PCB_DEVICE=cuda:2        # use physical/logical cuda device 2
      PCB_VLM_DEVICE=cuda:3    # override only Qwen
      CUDA_VISIBLE_DEVICES=2 with PCB_DEVICE=cuda:0 is also valid.
    """
    if not torch.cuda.is_available():
        return "cpu"
    value = (value or default).strip()
    if value.isdigit():
        return f"cuda:{value}"
    if value == "cuda":
        return "cuda:0"
    return value


def get_vlm_device() -> str:
    return _normalize_cuda_device(
        os.getenv("PCB_VLM_DEVICE") or os.getenv("PCB_DEVICE") or os.getenv("QWEN_DEVICE"),
        default="cuda:0",
    )


def get_yolo_device() -> str | int:
    """Return an Ultralytics-compatible device argument.

    If PCB_YOLO_DEVICE/PCB_DEVICE is set to cuda:2, return integer 2.
    If CUDA_VISIBLE_DEVICES=2 and PCB_DEVICE is unset, default 0 means the visible GPU.
    """
    if not torch.cuda.is_available():
        return "cpu"
    value = (os.getenv("PCB_YOLO_DEVICE") or os.getenv("PCB_DEVICE") or "0").strip()
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    if value == "cuda":
        return 0
    if value.isdigit():
        return int(value)
    return value

CLASS_EN2ZH = {
    "short": "短路",
    "open_circuit": "开路",
    "mouse_bite": "鼠咬",
    "spur": "毛刺",
    "spurious_copper": "多余铜",
    "missing_hole": "漏孔",
}

VISUAL_DIAGNOSIS_PROMPT_TEMPLATE = """
你是 PCB 维修诊断专家。

YOLO 已检测到该局部区域疑似为：{yolo_type}
YOLO 置信度：{yolo_conf:.3f}
检测框位置 xyxy：{bbox}

请不要只重复缺陷类别。请基于局部图像做视觉诊断。
只输出一个合法 JSON，不要输出 Markdown 代码块，不要输出额外解释。

JSON 字段如下：
1. defect_confirmed: yes / no / uncertain
2. final_type: 从 短路、开路、鼠咬、毛刺、多余铜、漏孔、未知 中选择
3. subtype: 更细的缺陷形态，例如焊料桥接、残铜短路、导线断裂、边缘缺口、铜刺、孔位缺失等
4. visual_evidence: 说明你从图像中看到的关键视觉证据
5. repairability: 可返修 / 需人工确认 / 不建议返修 / 未知
6. direct_repair_suggestion: 给出一条最直接的维修建议
7. risk_level: 低 / 中 / 高 / 未知
8. need_human_review: true / false
9. review_reason: 如果需要人工复核，说明原因；如果不需要，填空字符串

如果图像证据不足，不要猜测，请输出 defect_confirmed=uncertain，并说明“模糊，建议人工复核”。
""".strip()

_yolo_model = None
_vlm_model = None
_vlm_processor = None


def _append_msg(state: PCBGraphState, msg: str) -> List[str]:
    return list(state.get("messages", [])) + [msg]


def _append_error(state: PCBGraphState, err: str) -> List[str]:
    return list(state.get("errors", [])) + [err]


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_MODEL))
    return _yolo_model


def get_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is not None and _vlm_processor is not None:
        return _vlm_model, _vlm_processor

    vlm_device = get_vlm_device()

    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL),
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to(vlm_device)
    model = PeftModel.from_pretrained(
        model,
        str(ADAPTER_PATH),
        device_map=None,
    )
    model = model.to(vlm_device)
    model.eval()

    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    except Exception:
        pass

    _vlm_model = model
    _vlm_processor = processor
    return _vlm_model, _vlm_processor


def extract_defect_type(text: str) -> str:
    """Compatibility fallback for old LoRA outputs such as '缺陷类型：短路'."""
    m = re.search(r"缺陷类型[:：]\s*([^\n。；;，,]+)", text)
    if m:
        candidate = m.group(1).strip()
        for d in DEFECT_TYPES:
            if d in candidate:
                return d
    for d in DEFECT_TYPES:
        if d in text:
            return d
    return "未知"


def _normalize_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "是", "1"}:
            return True
        if v in {"false", "no", "否", "0"}:
            return False
    return default


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"```$", "", t).strip()
    return t


def parse_visual_diagnosis(raw_text: str, yolo_type: str) -> Dict[str, Any]:
    """Parse Qwen output into a stable schema. Falls back gracefully if the model emits old classification text."""
    text = _strip_json_fence(raw_text)
    data: Dict[str, Any] = {}

    try:
        data = json.loads(text)
    except Exception:
        # Try to extract the first JSON object from mixed text.
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start : end + 1])
        except Exception:
            data = {}

    if not isinstance(data, dict):
        data = {}

    fallback_type = extract_defect_type(raw_text)
    if fallback_type == "未知":
        fallback_type = yolo_type if yolo_type in FINAL_TYPES else "未知"

    defect_confirmed = str(data.get("defect_confirmed", "uncertain")).strip().lower()
    if defect_confirmed not in {"yes", "no", "uncertain"}:
        defect_confirmed = "uncertain"

    final_type = str(data.get("final_type", fallback_type)).strip()
    if final_type not in FINAL_TYPES:
        final_type = fallback_type if fallback_type in FINAL_TYPES else "未知"

    need_review_default = defect_confirmed != "yes"

    diagnosis = {
        "defect_confirmed": defect_confirmed,
        "final_type": final_type,
        "subtype": str(data.get("subtype", "未知细分形态")).strip() or "未知细分形态",
        "visual_evidence": str(
            data.get(
                "visual_evidence",
                "模型未能输出稳定的视觉证据描述，建议人工查看局部 crop。",
            )
        ).strip(),
        "repairability": str(data.get("repairability", "需人工确认")).strip() or "需人工确认",
        "direct_repair_suggestion": str(
            data.get(
                "direct_repair_suggestion",
                "建议先进行显微镜/AOI 局部复核，再结合通断或绝缘测试决定维修动作。",
            )
        ).strip(),
        "risk_level": str(data.get("risk_level", "未知")).strip() or "未知",
        "need_human_review": _normalize_bool(data.get("need_human_review"), default=need_review_default),
        "review_reason": str(
            data.get(
                "review_reason",
                "图像证据不足或模型输出不稳定，建议人工复核。" if need_review_default else "",
            )
        ).strip(),
    }

    if diagnosis["defect_confirmed"] == "uncertain" and not diagnosis["review_reason"]:
        diagnosis["need_human_review"] = True
        diagnosis["review_reason"] = "模糊，建议人工复核。"

    return diagnosis


def crop_with_margin(img: Image.Image, xyxy, margin_ratio=0.20, min_size=96):
    w, h = img.size
    x1, y1, x2, y2 = map(float, xyxy)
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(bw, bh)
    side = max(side * (1 + 2 * margin_ratio), min_size)
    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(w, int(cx + side / 2))
    ny2 = min(h, int(cy + side / 2))
    return img.crop((nx1, ny1, nx2, ny2)), [nx1, ny1, nx2, ny2]


def vlm_visual_diagnose_crop(
    crop_path: str,
    yolo_type: str,
    yolo_conf: float,
    bbox: List[float],
) -> Tuple[str, Dict[str, Any]]:
    model, processor = get_vlm()
    bbox_short = [round(float(x), 1) for x in bbox]
    prompt = VISUAL_DIAGNOSIS_PROMPT_TEMPLATE.format(
        yolo_type=yolo_type,
        yolo_conf=float(yolo_conf),
        bbox=bbox_short,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(get_vlm_device())

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    diagnosis = parse_visual_diagnosis(output_text, yolo_type=yolo_type)
    return output_text, diagnosis


def select_qwen_targets(
    detections: List[Dict[str, Any]],
    qwen_mode: str = "auto",
    low_conf_threshold: float = 0.65,
    representative_per_type: int = 1,
    high_risk_types: List[str] = None,
) -> Set[int]:
    """Decide which detection boxes deserve Qwen visual diagnosis.

    auto/report-like policy:
    - low-confidence boxes: call Qwen
    - high-risk classes: call representative regions
    - multiple boxes with same class: call top-N representative regions
    never: call none
    always: call all
    """
    qwen_mode = (qwen_mode or "auto").lower().strip()
    high_risk_types = high_risk_types or HIGH_RISK_DEFAULT
    representative_per_type = max(0, int(representative_per_type))

    if qwen_mode == "always":
        return {int(d["idx"]) for d in detections}
    if qwen_mode == "never":
        return set()

    targets: Set[int] = set()

    # 1) Low confidence or suspicious detections must be diagnosed.
    for det in detections:
        if float(det.get("yolo_conf", 0.0)) < low_conf_threshold:
            targets.add(int(det["idx"]))

    # 2) Representative regions per class for report-grade explanation.
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        by_type.setdefault(det.get("yolo_type", "未知"), []).append(det)

    for defect_type, items in by_type.items():
        items = sorted(items, key=lambda x: float(x.get("yolo_conf", 0.0)), reverse=True)
        n = representative_per_type
        if defect_type in high_risk_types:
            n = max(n, representative_per_type)
        for det in items[:n]:
            targets.add(int(det["idx"]))

    return targets


def build_yolo_only_visual_diagnosis(
    det: Dict[str, Any],
    prefer_yolo_conf: float,
    skip_reason: str,
) -> Dict[str, Any]:
    yolo_type = det.get("yolo_type", "未知")
    yolo_conf = float(det.get("yolo_conf", 0.0))
    confirmed = "yes" if yolo_conf >= prefer_yolo_conf else "uncertain"
    need_review = yolo_conf < prefer_yolo_conf
    risk_level = "高" if yolo_type in HIGH_RISK_DEFAULT else "中"

    return {
        "defect_confirmed": confirmed,
        "final_type": yolo_type if yolo_type in FINAL_TYPES else "未知",
        "subtype": "未调用 Qwen，采用 YOLO 初判",
        "visual_evidence": "本区域未调用 Qwen 视觉诊断；当前仅保留 YOLO 检测框、类别和置信度作为视觉依据。",
        "repairability": "需人工确认" if need_review else "未知",
        "direct_repair_suggestion": "建议结合 RAG 标准维修知识、局部放大图和电气测试结果决定维修动作。",
        "risk_level": risk_level,
        "need_human_review": need_review,
        "review_reason": "YOLO 置信度未达到高置信阈值，建议人工复核。" if need_review else "",
        "qwen_called": False,
        "qwen_skip_reason": skip_reason,
    }


def decide_with_visual_diagnosis(
    det: Dict[str, Any],
    diagnosis: Dict[str, Any],
    prefer_yolo_conf: float,
) -> Tuple[str, bool, str, str]:
    """Fuse YOLO confidence and structured Qwen diagnosis.

    Returns: final_type, defect_confirmed_bool, decision_note, decision_confidence
    """
    yolo_type = det.get("yolo_type", "未知")
    yolo_conf = float(det.get("yolo_conf", 0.0))
    qwen_type = diagnosis.get("final_type", "未知")
    qwen_confirmed = diagnosis.get("defect_confirmed", "uncertain")
    qwen_called = bool(diagnosis.get("qwen_called", False))

    if qwen_confirmed == "no":
        return "未知", False, "Qwen 视觉诊断认为缺陷不成立，建议人工复核确认是否为误检。", "低"

    if qwen_confirmed == "uncertain":
        final_type = yolo_type if yolo_type in FINAL_TYPES else "未知"
        return final_type, False, "Qwen 视觉证据不足，暂采用 YOLO 初判类别，但必须人工复核。", "中低"

    # qwen_confirmed == yes
    if qwen_type == yolo_type or qwen_type == "未知":
        final_type = yolo_type if qwen_type == "未知" else qwen_type
        note = "YOLO 检测类别与 Qwen 视觉诊断一致，缺陷形态证据较充分。" if qwen_called else "未调用 Qwen，采用 YOLO 高置信初判。"
        confidence = "高" if yolo_conf >= prefer_yolo_conf else "中"
        return final_type, True, note, confidence

    # Qwen and YOLO disagree. Still not the old classification-accuracy comparison;
    # this is a risk-control decision for report and human review.
    if yolo_conf >= prefer_yolo_conf:
        return yolo_type, True, "Qwen 诊断类型与 YOLO 初判不一致，但 YOLO 置信度较高；暂采用 YOLO 类别并触发人工复核。", "中"

    return qwen_type, True, "YOLO 置信度较低且 Qwen 给出不同视觉诊断；暂采用 Qwen 诊断类别并触发人工复核。", "中低"


def _safe_parse_mcp_payload(payload: Any) -> Dict[str, Any]:
    """
    将 MCP 返回的 structuredContent / text / dict / list 统一转成：
        {"results": [...]}。
    如果 MCP server 返回 error，也保留为 _mcp_error。
    """
    if payload is None:
        return {
            "results": [],
            "_mcp_error": "MCP 返回为空。",
            "_mcp_raw": "",
        }

    if isinstance(payload, dict):
        data = dict(payload)

        # FastMCP 某些版本会把工具返回值包成：
        #   {"result": "{...json string...}"}
        # 如果直接读顶层，就会误以为 results=[]。
        # 所以这里优先展开 result/text/content 这些嵌套字段。
        for nested_key in ("result", "text", "content"):
            nested = data.get(nested_key)
            if nested not in (None, ""):
                nested_parsed = _safe_parse_mcp_payload(nested)
                # 如果嵌套里有有效结果或错误信息，就以嵌套解析结果为准。
                if nested_parsed.get("results") or nested_parsed.get("_mcp_error") or nested_parsed.get("error"):
                    return nested_parsed

        data.setdefault("results", [])
        if data.get("error") and not data.get("_mcp_error"):
            data["_mcp_error"] = str(data.get("error"))
        return data

    if isinstance(payload, list):
        return {"results": payload}

    raw = str(payload)
    cleaned = raw.strip()

    if not cleaned:
        return {
            "results": [],
            "_mcp_error": "MCP 返回空文本，无法解析 JSON。",
            "_mcp_raw": raw,
        }

    try:
        data = json.loads(cleaned)
        return _safe_parse_mcp_payload(data)
    except Exception as first_error:
        # 尝试从混杂日志中截取第一个 JSON 对象。
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(cleaned[start : end + 1])
                parsed = _safe_parse_mcp_payload(data)
                parsed["_mcp_raw"] = raw[:2000]
                return parsed
        except Exception:
            pass

        return {
            "results": [],
            "_mcp_error": f"MCP 返回内容不是合法 JSON：{repr(first_error)}",
            "_mcp_raw": raw[:2000],
        }


async def _call_mcp_search_async(query: str, defect_type: str, top_k: int) -> Dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # 必须使用当前 Python 解释器，避免 command="python" 调到系统 Python 或错误 conda 环境。
    # MCP/RAG 使用 SentenceTransformer 做 embedding 检索。
    # 这里不要继承 Streamlit 主进程的 CUDA_VISIBLE_DEVICES，否则 MCP 可能只看见
    # 主进程给 Qwen 用的那张卡。让 MCP server 自己通过 nvidia-smi 自动选择
    # 空闲显存最多的物理 GPU；选中后在 MCP 子进程内部会映射为 cuda:0。
    server_env = dict(os.environ)
    server_env.pop("CUDA_VISIBLE_DEVICES", None)
    server_env.pop("PCB_DEVICE", None)
    server_env.pop("PCB_VLM_DEVICE", None)
    server_env.pop("PCB_YOLO_DEVICE", None)
    server_env.setdefault("PCB_MCP_DEVICE", os.getenv("PCB_MCP_DEVICE", "auto"))
    server_env.setdefault("PCB_MCP_MIN_FREE_MIB", os.getenv("PCB_MCP_MIN_FREE_MIB", "2048"))
    server_env.setdefault("PCB_MCP_EXCLUDE_GPUS", os.getenv("PCB_MCP_EXCLUDE_GPUS", ""))
    server_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    server_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    server_env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
        env=server_env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "pcb_knowledge_search",
                arguments={
                    "query": query,
                    "defect_type": defect_type,
                    "top_k": top_k,
                },
            )

            # 1) 优先读取 structuredContent / structured_content。
            structured = None
            for attr in ("structuredContent", "structured_content"):
                if hasattr(result, attr):
                    value = getattr(result, attr)
                    if value:
                        structured = value
                        break

            if structured:
                print("[MCP] structured return:", repr(str(structured)[:1000]), flush=True)
                return _safe_parse_mcp_payload(structured)

            # 2) 再从 pydantic model_dump 里找 structuredContent。
            try:
                if hasattr(result, "model_dump"):
                    dumped = result.model_dump(by_alias=True)
                    for key in ("structuredContent", "structured_content"):
                        if dumped.get(key):
                            print("[MCP] model_dump structured return:", repr(str(dumped[key])[:1000]), flush=True)
                            return _safe_parse_mcp_payload(dumped[key])
            except Exception as dump_error:
                print("[MCP] model_dump failed:", repr(dump_error), flush=True)

            # 3) 最后读取 content[].text。
            text_parts: List[str] = []
            for content in getattr(result, "content", []) or []:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                else:
                    text_parts.append(str(content))

            text = "\n".join(text_parts).strip()
            print("[MCP] text return:", repr(text[:1000]), flush=True)

            return _safe_parse_mcp_payload(text)


def call_mcp_search(query: str, defect_type: str, top_k: int) -> Dict[str, Any]:
    """
    同步调用 MCP。
    这里捕获 BaseException，是为了兜住 anyio 抛出的 BaseExceptionGroup/ExceptionGroup，
    否则 LangGraph 会直接中断。
    """
    try:
        return asyncio.run(
            _call_mcp_search_async(
                query=query,
                defect_type=defect_type,
                top_k=top_k,
            )
        )
    except BaseException as e:
        return {
            "results": [],
            "_mcp_error": f"MCP 调用失败：{type(e).__name__}: {repr(e)}",
            "_mcp_raw": "",
        }


def _run_local_rag_fallback(
    decision: Dict[str, Any],
    defect_type: str,
    top_k: int,
    reason: str = "",
) -> str:
    """
    MCP 检索失败时自动降级到本地 RAG，保证前端和报告继续生成。
    注意：这是兜底，不代表 MCP 问题被忽略；页面会显示 MCP 失败原因。
    """
    try:
        visual_diagnosis = dict(decision.get("visual_diagnosis", {}))
        visual_diagnosis["final_type"] = defect_type

        rag_result = run_diagnosis_pipeline(
            visual_diagnosis,
            top_k=top_k,
        )

        if isinstance(rag_result, dict) and "report" in rag_result:
            local_report = rag_result["report"]
        else:
            local_report = str(rag_result)

        prefix = ["⚠️ MCP RAG 检索失败，系统已自动降级为本地 RAG。"]
        if reason:
            prefix.append(f"失败原因：`{reason}`")
        prefix.append("")
        prefix.append(local_report)
        return "\n".join(prefix)

    except Exception as e:
        return (
            "RAG 检索失败：MCP 调用失败，且本地 RAG 降级也失败。\n\n"
            f"- MCP 失败原因：{reason}\n"
            f"- Local RAG 失败原因：{repr(e)}"
        )


def build_rag_query_from_decision(decision: Dict[str, Any]) -> str:
    vd = decision.get("visual_diagnosis", {})
    final_type = decision.get("final_type", vd.get("final_type", "未知"))
    subtype = vd.get("subtype", "")
    evidence = vd.get("visual_evidence", "")
    repairability = vd.get("repairability", "未知")
    direct_suggestion = vd.get("direct_repair_suggestion", "")
    return (
        f"PCB {final_type} 缺陷维修诊断。"
        f"细分形态：{subtype}。"
        f"图像证据：{evidence}。"
        f"可返修性：{repairability}。"
        f"初步维修动作：{direct_suggestion}。"
        "请返回风险等级、检测确认方法、标准维修建议、复测方法和预防措施。"
    )


def format_mcp_rag_report(decision: Dict[str, Any], mcp_result: Dict[str, Any]) -> str:
    results = mcp_result.get("results", [])
    vd = decision.get("visual_diagnosis", {})
    defect_type = decision.get("final_type", vd.get("final_type", "未知"))

    lines: List[str] = []
    lines.append("# PCB 视觉诊断 + MCP 知识库维修报告\n")
    lines.append("## 1. 视觉诊断摘要")
    lines.append(f"- 最终缺陷类型：{defect_type}")
    lines.append(f"- 细分形态：{vd.get('subtype', '未知')}")
    lines.append(f"- 视觉证据：{vd.get('visual_evidence', '无')}")
    lines.append(f"- 可返修性：{vd.get('repairability', '未知')}")
    lines.append(f"- 初步维修动作：{vd.get('direct_repair_suggestion', '无')}")
    lines.append(f"- 是否需要人工复核：{'是' if decision.get('need_human_review') else '否'}")
    if decision.get("need_human_review"):
        lines.append(f"- 复核原因：{decision.get('review_reason', vd.get('review_reason', '建议人工复核。'))}")
    lines.append("")

    lines.append("## 2. MCP 知识库检索结果")
    if not results:
        lines.append("未检索到相关知识条目。")
        return "\n".join(lines)

    for i, item in enumerate(results, 1):
        lines.append(f"### 知识条目 {i}")
        lines.append(f"- 文档编号：{item.get('doc_id')}")
        lines.append(f"- 缺陷类型：{item.get('defect_type')}")
        lines.append(f"- 标题：{item.get('title')}")
        lines.append(f"- 风险等级：{item.get('severity')}")
        if item.get("visual_features"):
            lines.append(f"- 标准视觉特征：{item.get('visual_features')}")
        if item.get("risk"):
            lines.append(f"- 风险说明：{item.get('risk')}")
        if item.get("detection_methods"):
            lines.append(f"- 检测方法：{item.get('detection_methods')}")
        lines.append(f"- 维修建议：{item.get('repair_suggestions')}")
        if item.get("prevention"):
            lines.append(f"- 预防措施：{item.get('prevention')}")
        lines.append("")

    top = results[0]
    lines.append("## 3. 综合建议")
    lines.append(f"- 优先参考知识条目：{top.get('title')}")
    lines.append(f"- 建议措施：{top.get('repair_suggestions')}")
    lines.append("- 维修后执行外观复检、通断测试/绝缘测试，确认缺陷已消除且未引入二次损伤。")
    return "\n".join(lines)


# -------------------------
# LangGraph Nodes
# -------------------------

def detection_agent(state: PCBGraphState) -> PCBGraphState:
    image_path = Path(state["image_path"])
    conf = float(state.get("conf", 0.50))
    imgsz = int(state.get("imgsz", 1024))
    max_det = int(state.get("max_det", 20))

    output_dir = Path(
        state.get("output_dir")
        or PROJECT_ROOT / "output" / "langgraph_runs" / f"{image_path.stem}_{uuid.uuid4().hex[:8]}"
    )
    crop_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        err = f"输入图像不存在：{image_path}"
        return {
            "output_dir": str(output_dir),
            "crop_dir": str(crop_dir),
            "detections": [],
            "messages": _append_msg(state, "DetectionAgent failed."),
            "errors": _append_error(state, err),
        }

    yolo = get_yolo()
    results = yolo.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        max_det=max_det,
        save=False,
        verbose=False,
        device=get_yolo_device(),
    )
    r = results[0]
    names = r.names

    annotated = r.plot()
    annotated_rgb = annotated[..., ::-1]
    annotated_path = output_dir / "yolo_detect_annotated.jpg"
    Image.fromarray(annotated_rgb).save(annotated_path)

    detections: List[Dict[str, Any]] = []
    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].detach().cpu().tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            score = float(box.conf[0].detach().cpu().item())
            cls_en = names[cls_id]
            cls_zh = CLASS_EN2ZH.get(cls_en, cls_en)
            detections.append(
                {
                    "idx": i + 1,
                    "xyxy": xyxy,
                    "cls_id": cls_id,
                    "cls_en": cls_en,
                    "yolo_type": cls_zh,
                    "yolo_conf": score,
                }
            )

    return {
        "output_dir": str(output_dir),
        "crop_dir": str(crop_dir),
        "annotated_image_path": str(annotated_path),
        "detections": detections,
        "messages": _append_msg(state, f"DetectionAgent: detected {len(detections)} region(s)."),
    }


def visual_diagnosis_agent(state: PCBGraphState) -> PCBGraphState:
    detections = state.get("detections", [])
    image_path = Path(state["image_path"])
    crop_dir = Path(state["crop_dir"])
    prefer_yolo_conf = float(state.get("prefer_yolo_conf", 0.65))

    if not detections:
        return {
            "crops": [],
            "visual_diagnoses": [],
            "vlm_results": [],
            "qwen_call_stats": {"called": 0, "skipped": 0, "target_indices": []},
            "messages": _append_msg(state, "VisualDiagnosisAgent: no detections, skipped."),
        }

    qwen_mode = state.get("qwen_mode", "auto")
    low_conf_threshold = float(state.get("low_conf_threshold", 0.65))
    representative_per_type = int(state.get("representative_per_type", 1))
    high_risk_types = state.get("high_risk_types", HIGH_RISK_DEFAULT)

    target_indices = select_qwen_targets(
        detections=detections,
        qwen_mode=qwen_mode,
        low_conf_threshold=low_conf_threshold,
        representative_per_type=representative_per_type,
        high_risk_types=high_risk_types,
    )

    original_img = Image.open(image_path).convert("RGB")
    crops: List[Dict[str, Any]] = []
    visual_diagnoses: List[Dict[str, Any]] = []
    legacy_vlm_results: List[Dict[str, Any]] = []

    for det in detections:
        crop_img, crop_xyxy = crop_with_margin(original_img, det["xyxy"])
        crop_path = crop_dir / f"crop_{det['idx']:02d}_yolo_{det['cls_en']}_{det['yolo_conf']:.3f}.jpg"
        crop_img.save(crop_path)
        crops.append(
            {
                "idx": det["idx"],
                "crop_path": str(crop_path),
                "crop_xyxy": crop_xyxy,
            }
        )

        if int(det["idx"]) in target_indices:
            try:
                qwen_output, diagnosis = vlm_visual_diagnose_crop(
                    crop_path=str(crop_path),
                    yolo_type=det["yolo_type"],
                    yolo_conf=det["yolo_conf"],
                    bbox=det["xyxy"],
                )
                diagnosis["qwen_called"] = True
                diagnosis["qwen_output"] = qwen_output
                diagnosis["qwen_skip_reason"] = ""
            except Exception as e:
                qwen_output = f"Qwen 调用失败：{repr(e)}"
                diagnosis = build_yolo_only_visual_diagnosis(
                    det,
                    prefer_yolo_conf=prefer_yolo_conf,
                    skip_reason=qwen_output,
                )
                diagnosis["defect_confirmed"] = "uncertain"
                diagnosis["need_human_review"] = True
                diagnosis["review_reason"] = "Qwen 调用失败，建议人工复核。"
                diagnosis["qwen_output"] = qwen_output
        else:
            qwen_output = ""
            diagnosis = build_yolo_only_visual_diagnosis(
                det,
                prefer_yolo_conf=prefer_yolo_conf,
                skip_reason="当前检测框不满足 Qwen 调用策略；采用 YOLO 初判和 RAG 模板生成报告。",
            )
            diagnosis["qwen_output"] = ""

        visual_diagnoses.append(
            {
                "idx": det["idx"],
                "visual_diagnosis": diagnosis,
            }
        )
        legacy_vlm_results.append(
            {
                "idx": det["idx"],
                "vlm_output": qwen_output,
                "vlm_type": diagnosis.get("final_type", "未知"),
            }
        )

    called = len(target_indices)
    skipped = max(0, len(detections) - called)
    return {
        "crops": crops,
        "visual_diagnoses": visual_diagnoses,
        "vlm_results": legacy_vlm_results,
        "qwen_call_stats": {
            "called": called,
            "skipped": skipped,
            "total": len(detections),
            "target_indices": sorted(target_indices),
            "qwen_mode": qwen_mode,
            "low_conf_threshold": low_conf_threshold,
            "representative_per_type": representative_per_type,
            "high_risk_types": high_risk_types,
        },
        "messages": _append_msg(
            state,
            f"VisualDiagnosisAgent: cropped {len(crops)} region(s), called Qwen for {called}/{len(detections)} region(s).",
        ),
    }


def decision_agent(state: PCBGraphState) -> PCBGraphState:
    detections = state.get("detections", [])
    crops = {x["idx"]: x for x in state.get("crops", [])}
    diagnoses = {x["idx"]: x["visual_diagnosis"] for x in state.get("visual_diagnoses", [])}
    prefer_yolo_conf = float(state.get("prefer_yolo_conf", 0.65))

    decisions: List[Dict[str, Any]] = []
    for det in detections:
        idx = det["idx"]
        crop = crops.get(idx, {"crop_path": "", "crop_xyxy": []})
        diagnosis = diagnoses.get(
            idx,
            build_yolo_only_visual_diagnosis(
                det,
                prefer_yolo_conf=prefer_yolo_conf,
                skip_reason="缺少 VisualDiagnosisAgent 输出，采用 YOLO 初判。",
            ),
        )

        final_type, confirmed_bool, decision_note, decision_confidence = decide_with_visual_diagnosis(
            det=det,
            diagnosis=diagnosis,
            prefer_yolo_conf=prefer_yolo_conf,
        )

        qwen_type = diagnosis.get("final_type", "未知")
        yolo_type = det.get("yolo_type", "未知")
        type_conflict = bool(diagnosis.get("qwen_called")) and qwen_type not in {"未知", yolo_type}
        need_review = bool(diagnosis.get("need_human_review", False)) or type_conflict or not confirmed_bool
        review_reason = diagnosis.get("review_reason", "")
        if type_conflict:
            review_reason = review_reason or "Qwen 诊断类型与 YOLO 初判不一致，需要人工复核。"
        if not confirmed_bool and not review_reason:
            review_reason = "缺陷确认状态不充分，需要人工复核。"

        decisions.append(
            {
                **det,
                **crop,
                "visual_diagnosis": diagnosis,
                "final_type": final_type,
                "defect_confirmed": confirmed_bool,
                "decision_confidence": decision_confidence,
                "type_conflict": type_conflict,
                "need_human_review": need_review,
                "review_reason": review_reason,
                "decision_note": decision_note,
            }
        )

    return {
        "decisions": decisions,
        "messages": _append_msg(state, f"DecisionAgent: fused YOLO confidence and visual diagnosis for {len(decisions)} region(s)."),
    }


def rag_agent(state: PCBGraphState) -> PCBGraphState:
    decisions = state.get("decisions", [])
    top_k = int(state.get("top_k", 3))
    rag_backend = state.get("rag_backend", "local")

    if not decisions:
        return {
            "rag_reports": {},
            "messages": _append_msg(state, "RAGAgent: no decisions, skipped."),
        }

    # 每个最终缺陷类型只检索一次。
    # 代表区域选择优先级：
    # 1. 缺陷已确认
    # 2. 调用过 Qwen
    # 3. YOLO 置信度最高
    representatives: Dict[str, Dict[str, Any]] = {}

    for d in decisions:
        defect_type = d.get("final_type", "未知")
        if defect_type == "未知":
            continue

        current = representatives.get(defect_type)
        if current is None:
            representatives[defect_type] = d
            continue

        score_new = (
            int(bool(d.get("defect_confirmed"))),
            int(bool(d.get("visual_diagnosis", {}).get("qwen_called"))),
            float(d.get("yolo_conf", 0.0)),
        )
        score_old = (
            int(bool(current.get("defect_confirmed"))),
            int(bool(current.get("visual_diagnosis", {}).get("qwen_called"))),
            float(current.get("yolo_conf", 0.0)),
        )

        if score_new > score_old:
            representatives[defect_type] = d

    rag_reports: Dict[str, str] = {}

    for defect_type, decision in representatives.items():
        query = build_rag_query_from_decision(decision)

        if rag_backend == "mcp":
            mcp_result = call_mcp_search(
                query=query,
                defect_type=defect_type,
                top_k=top_k,
            )

            mcp_error = mcp_result.get("_mcp_error", "")
            results = mcp_result.get("results", [])

            if mcp_error or not results:
                reason = mcp_error or "MCP 未返回有效 results。"
                rag_reports[defect_type] = _run_local_rag_fallback(
                    decision=decision,
                    defect_type=defect_type,
                    top_k=top_k,
                    reason=reason,
                )
            else:
                try:
                    rag_reports[defect_type] = format_mcp_rag_report(
                        decision=decision,
                        mcp_result=mcp_result,
                    )
                except Exception as e:
                    rag_reports[defect_type] = _run_local_rag_fallback(
                        decision=decision,
                        defect_type=defect_type,
                        top_k=top_k,
                        reason=f"MCP 返回格式异常，format_mcp_rag_report 失败：{repr(e)}",
                    )

        else:
            # local 模式下直接调用本地 RAG，不显示“失败降级”提示。
            try:
                visual_diagnosis = dict(decision.get("visual_diagnosis", {}))
                visual_diagnosis["final_type"] = defect_type

                rag_result = run_diagnosis_pipeline(
                    visual_diagnosis,
                    top_k=top_k,
                )

                if isinstance(rag_result, dict) and "report" in rag_result:
                    rag_reports[defect_type] = rag_result["report"]
                else:
                    rag_reports[defect_type] = str(rag_result)

            except Exception as e:
                rag_reports[defect_type] = (
                    "Local RAG 检索失败。\n\n"
                    f"- 失败原因：{repr(e)}"
                )

    return {
        "rag_reports": rag_reports,
        "messages": _append_msg(
            state,
            f"RAGAgent: generated {len(rag_reports)} RAG report(s) via {rag_backend}.",
        ),
    }


def report_agent(state: PCBGraphState) -> PCBGraphState:
    image_path = state.get("image_path", "")
    output_dir = Path(state.get("output_dir", PROJECT_ROOT / "output" / "langgraph_runs" / "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = state.get("detections", [])
    decisions = state.get("decisions", [])
    rag_reports = state.get("rag_reports", {})
    errors = state.get("errors", [])
    messages = state.get("messages", [])
    qwen_stats = state.get("qwen_call_stats", {})

    lines: List[str] = []
    lines.append("# PCB LangGraph 多 Agent 视觉诊断与维修报告\n")
    lines.append("## 0. Graph 执行轨迹")
    for msg in messages:
        lines.append(f"- {msg}")
    if errors:
        lines.append("\n## 错误信息")
        for e in errors:
            lines.append(f"- {e}")

    lines.append("\n## 1. 输入与 YOLO 检测概况")
    lines.append(f"- 输入图像：`{image_path}`")
    lines.append(f"- YOLO 检测可视化图：`{state.get('annotated_image_path', '')}`")
    lines.append(f"- 检测区域数量：**{len(detections)}**")
    lines.append(f"- Qwen 调用策略：`{qwen_stats.get('qwen_mode', state.get('qwen_mode', 'auto'))}`")
    lines.append(f"- Qwen 调用次数：**{qwen_stats.get('called', 0)} / {qwen_stats.get('total', len(detections))}**")
    lines.append(f"- Qwen 目标区域 idx：`{qwen_stats.get('target_indices', [])}`")

    if not detections:
        lines.append("\nYOLO 未检测到缺陷区域。建议降低检测置信度阈值后复测。")
    else:
        lines.append("\n## 2. 局部视觉诊断与融合决策")
        for d in decisions:
            vd = d.get("visual_diagnosis", {})
            lines.append(f"\n### 缺陷区域 {d['idx']}")
            lines.append(f"- YOLO 检测类别：**{d['yolo_type']}**")
            lines.append(f"- YOLO 置信度：**{d['yolo_conf']:.3f}**")
            lines.append(f"- YOLO 原始框 xyxy：`{[round(x, 1) for x in d['xyxy']]}`")
            lines.append(f"- 自动裁剪图：`{d.get('crop_path', '')}`")
            lines.append(f"- 是否调用 Qwen：**{'是' if vd.get('qwen_called') else '否'}**")
            if not vd.get("qwen_called"):
                lines.append(f"- 跳过原因：{vd.get('qwen_skip_reason', '')}")
            lines.append(f"- 缺陷确认状态：**{vd.get('defect_confirmed', 'uncertain')}**")
            lines.append(f"- Qwen/视觉诊断类型：**{vd.get('final_type', '未知')}**")
            lines.append(f"- 细分形态：**{vd.get('subtype', '未知')}**")
            lines.append(f"- 视觉证据：{vd.get('visual_evidence', '')}")
            lines.append(f"- 可返修性：**{vd.get('repairability', '未知')}**")
            lines.append(f"- 直接维修建议：{vd.get('direct_repair_suggestion', '')}")
            lines.append(f"- 风险等级：**{vd.get('risk_level', '未知')}**")
            lines.append(f"- 最终融合类别：**{d['final_type']}**")
            lines.append(f"- 决策置信度：**{d.get('decision_confidence', '未知')}**")
            lines.append(f"- 是否需要人工复核：**{'是' if d['need_human_review'] else '否'}**")
            if d["need_human_review"]:
                lines.append(f"- 人工复核原因：{d.get('review_reason', '')}")
            lines.append(f"- 决策说明：{d['decision_note']}")
            if vd.get("qwen_output"):
                lines.append(f"- Qwen 原始输出：`{vd.get('qwen_output')}`")

    lines.append("\n## 3. RAG 维修知识诊断")
    if not rag_reports:
        lines.append("未生成 RAG 维修知识。")
    else:
        for defect_type, report in rag_reports.items():
            lines.append(f"\n### {defect_type}：维修建议与风险分析\n")
            lines.append(report)

    lines.append("\n## 4. 结论")
    lines.append("本流程将 YOLO 定位结果、Qwen 视觉诊断解释和 RAG 维修知识分层使用：YOLO 负责发现问题，Qwen 负责解释问题，RAG 负责提供维修依据，ReportAgent 负责形成可交付报告。")

    report = "\n".join(lines)
    report_path = output_dir / "langgraph_visual_diagnosis_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "report": report,
        "report_path": str(report_path),
        "messages": _append_msg(state, f"ReportAgent: saved report to {report_path}."),
    }


def build_pcb_graph():
    builder = StateGraph(PCBGraphState)
    builder.add_node("DetectionAgent", detection_agent)
    builder.add_node("VisualDiagnosisAgent", visual_diagnosis_agent)
    builder.add_node("DecisionAgent", decision_agent)
    builder.add_node("RAGAgent", rag_agent)
    builder.add_node("ReportAgent", report_agent)

    builder.add_edge(START, "DetectionAgent")
    builder.add_edge("DetectionAgent", "VisualDiagnosisAgent")
    builder.add_edge("VisualDiagnosisAgent", "DecisionAgent")
    builder.add_edge("DecisionAgent", "RAGAgent")
    builder.add_edge("RAGAgent", "ReportAgent")
    builder.add_edge("ReportAgent", END)
    return builder.compile()


def run_pcb_langgraph(
    image_path: str,
    conf: float = 0.50,
    imgsz: int = 1024,
    max_det: int = 20,
    top_k: int = 3,
    prefer_yolo_conf: float = 0.65,
    rag_backend: str = "local",
    output_dir: str = "",
    qwen_mode: str = "auto",
    low_conf_threshold: float = 0.65,
    representative_per_type: int = 1,
    high_risk_types: List[str] = None,
) -> PCBGraphState:
    graph = build_pcb_graph()
    init_state: PCBGraphState = {
        "image_path": image_path,
        "conf": conf,
        "imgsz": imgsz,
        "max_det": max_det,
        "top_k": top_k,
        "prefer_yolo_conf": prefer_yolo_conf,
        "rag_backend": rag_backend,
        "qwen_mode": qwen_mode,
        "low_conf_threshold": low_conf_threshold,
        "representative_per_type": representative_per_type,
        "high_risk_types": high_risk_types or HIGH_RISK_DEFAULT,
        "messages": [],
        "errors": [],
    }
    if output_dir:
        init_state["output_dir"] = output_dir
        init_state["crop_dir"] = str(Path(output_dir) / "crops")
    return graph.invoke(init_state)
