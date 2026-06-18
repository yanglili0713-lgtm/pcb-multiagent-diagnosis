from typing import Dict, Any, List, Optional

from rag.knowledge_search import pcb_knowledge_search


def build_retrieval_query(vlm_result: Dict[str, Any]) -> str:
    defect_type = vlm_result.get("defect_type", "未知缺陷")
    description = vlm_result.get("description", "")
    return f"PCB图像疑似存在{defect_type}缺陷。{description}。请给出检测方法、风险分析和维修建议。"


def generate_diagnosis_report(
    vlm_result: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
) -> str:
    defect_type = vlm_result.get("defect_type", "未知")
    confidence = vlm_result.get("confidence", "中")
    description = vlm_result.get("description", "图像中存在疑似缺陷区域。")

    lines = []

    lines.append("# PCB 多模态故障诊断报告")
    lines.append("")
    lines.append("## 1. 图像诊断结果")
    lines.append(f"- 疑似缺陷类型：{defect_type}")
    lines.append(f"- 诊断置信度：{confidence}")
    lines.append(f"- 图像观察说明：{description}")
    lines.append("")
    lines.append("注意：该结果由视觉模型自动生成，建议结合 AOI 检测框、局部放大图和人工复核进一步确认。")
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
            lines.append(f"- 维修建议：{item.get('repair_suggestions')}")
            lines.append("")

    lines.append("## 3. 综合风险分析")
    if rag_results:
        high_risk_items = [x for x in rag_results if x.get("severity") == "高"]
        if high_risk_items:
            lines.append(f"该缺陷在知识库中存在高风险记录，可能影响电气连接、绝缘可靠性或长期稳定性。")
        else:
            lines.append("该缺陷风险等级为中低风险或可变风险，但仍需结合实际线路位置判断。")
    else:
        lines.append("由于缺少知识库结果，无法给出可靠风险分析。")
    lines.append("")

    lines.append("## 4. 建议处理措施")
    if rag_results:
        first = rag_results[0]
        lines.append(f"优先参考知识条目：{first.get('title')}")
        lines.append(f"建议措施：{first.get('repair_suggestions')}")
        lines.append("")
        lines.append("进一步建议：")
        lines.append("- 使用显微镜或 AOI 局部放大确认缺陷位置。")
        lines.append("- 使用万用表、绝缘电阻测试或通断测试验证电气影响。")
        lines.append("- 维修后重新进行外观检查和电气测试。")
    else:
        lines.append("- 建议人工复核图像。")
        lines.append("- 补充更多缺陷描述后重新检索知识库。")

    return "\n".join(lines)


def run_diagnosis_pipeline(
    vlm_result: Dict[str, Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    defect_type: Optional[str] = vlm_result.get("defect_type")
    query = build_retrieval_query(vlm_result)

    rag_results = pcb_knowledge_search(
        query=query,
        defect_type=defect_type,
        top_k=top_k,
    )

    report = generate_diagnosis_report(
        vlm_result=vlm_result,
        rag_results=rag_results,
    )

    return {
        "vlm_result": vlm_result,
        "retrieval_query": query,
        "rag_results": rag_results,
        "report": report,
    }