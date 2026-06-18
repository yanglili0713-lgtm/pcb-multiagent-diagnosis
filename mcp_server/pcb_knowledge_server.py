import sys
import contextlib
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from rag.knowledge_search import PCBKnowledgeSearcher


mcp = FastMCP("pcb-knowledge-server")

_searcher: Optional[PCBKnowledgeSearcher] = None


def get_searcher() -> PCBKnowledgeSearcher:
    global _searcher
    if _searcher is None:
        # MCP stdio 模式下 stdout 只能用于 JSON-RPC。
        # 因此把已有 RAG 模块中的 print 输出重定向到 stderr，避免污染协议消息。
        with contextlib.redirect_stdout(sys.stderr):
            _searcher = PCBKnowledgeSearcher()
    return _searcher


def normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
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


@mcp.tool()
def pcb_knowledge_search(
    query: str,
    defect_type: str = "",
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Search PCB fault-maintenance knowledge.

    Args:
        query: Defect description or maintenance question.
        defect_type: Optional defect type filter, such as 短路、开路、鼠咬、毛刺、多余铜、漏孔.
        top_k: Number of knowledge entries to return.
    """
    if not query and not defect_type:
        query = "PCB 缺陷检测方法、风险分析和维修建议"

    if top_k <= 0:
        top_k = 3
    if top_k > 10:
        top_k = 10

    defect_filter = defect_type.strip() or None

    searcher = get_searcher()

    # 这里也重定向 stdout，防止 encode/search 过程中的普通输出污染 MCP stdio。
    with contextlib.redirect_stdout(sys.stderr):
        results = searcher.search(
            query=query,
            defect_type=defect_filter,
            top_k=top_k,
        )

    normalized = [normalize_result(x) for x in results]

    return {
        "tool": "pcb_knowledge_search",
        "query": query,
        "defect_type": defect_type,
        "top_k": top_k,
        "num_results": len(normalized),
        "results": normalized,
    }


if __name__ == "__main__":
    mcp.run()
