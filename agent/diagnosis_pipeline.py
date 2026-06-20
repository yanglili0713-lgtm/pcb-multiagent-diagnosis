from typing import Any, Dict, List, Optional

from rag.knowledge_search import pcb_knowledge_search


def build_retrieval_query(visual_diagnosis: Dict[str, Any]) -> str:
    """Build a RAG query from structured visual diagnosis instead of only class name."""
    defect_type = visual_diagnosis.get("final_type", "未知")
    subtype = visual_diagnosis.get("subtype", "")
    evidence = visual_diagnosis.get("visual_evidence", "")
    repairability = visual_diagnosis.get("repairability", "未知")
    direct_suggestion = visual_diagnosis.get("direct_repair_suggestion", "")

    return (
        f"PCB 图像疑似存在{defect_type}缺陷。"
        f"子类型/形态：{subtype}。"
        f"视觉证据：{evidence}。"
        f"可返修性判断：{repairability}。"
        f"初步维修建议：{direct_suggestion}。"
        "请给出标准检测方法、风险分析、维修建议、复测方法和预防措施。"
    )


def generate_diagnosis_report(
    visual_diagnosis: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
) -> str:
    defect_type = visual_diagnosis.get("final_type", "未知")
    subtype = visual_diagnosis.get("subtype", "未知")
    confirmed = visual_diagnosis.get("defect_confirmed", "uncertain")
    evidence = visual_diagnosis.get("visual_evidence", "图像证据不足。")
    repairability = visual_diagnosis.get("repairability", "未知")
    direct_suggestion = visual_diagnosis.get("direct_repair_suggestion", "建议人工复核。")
    risk_level = visual_diagnosis.get("risk_level", "未知")
    need_review = visual_diagnosis.get("need_human_review", True)
    review_reason = visual_diagnosis.get("review_reason", "建议结合 AOI 局部放大图和电测结果复核。")

    lines: List[str] = []
    lines.append("# PCB 多模态视觉诊断与维修知识报告")
    lines.append("")
    lines.append("## 1. Qwen 结构化视觉诊断")
    lines.append(f"- 缺陷确认状态：{confirmed}")
    lines.append(f"- 最终缺陷类型：{defect_type}")
    lines.append(f"- 细分形态/子类型：{subtype}")
    lines.append(f"- 视觉证据：{evidence}")
    lines.append(f"- 可返修性：{repairability}")
    lines.append(f"- 直接维修建议：{direct_suggestion}")
    lines.append(f"- 风险等级：{risk_level}")
    lines.append(f"- 是否需要人工复核：{'是' if need_review else '否'}")
    if need_review:
        lines.append(f"- 复核原因：{review_reason}")
    lines.append("")
    lines.append("注意：视觉诊断只能说明图像形态，是否已经造成电气失效仍需结合通断测试、绝缘测试或功能测试确认。")
    lines.append("")

    lines.append("## 2. 知识库检索结果")
    if not rag_results:
        lines.append("未检索到相关知识。")
    else:
        for i, item in enumerate(rag_results, 1):
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

    lines.append("## 3. 综合处理建议")
    if rag_results:
        top = rag_results[0]
        lines.append(f"- 优先参考知识条目：{top.get('title')}")
        lines.append(f"- 建议措施：{top.get('repair_suggestions')}")
    else:
        lines.append("- 建议人工复核图像并补充电气测试结果后重新检索知识库。")
    lines.append("- 使用显微镜或 AOI 局部放大确认缺陷边界。")
    lines.append("- 维修后重新进行外观检查和电气测试。")

    return "\n".join(lines)


def run_diagnosis_pipeline(
    visual_diagnosis: Dict[str, Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    defect_type: Optional[str] = visual_diagnosis.get("final_type")
    query = build_retrieval_query(visual_diagnosis)
    rag_results = pcb_knowledge_search(
        query=query,
        defect_type=defect_type,
        top_k=top_k,
    )
    report = generate_diagnosis_report(
        visual_diagnosis=visual_diagnosis,
        rag_results=rag_results,
    )
    return {
        "visual_diagnosis": visual_diagnosis,
        "retrieval_query": query,
        "rag_results": rag_results,
        "report": report,
    }
