import csv
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")

random_eval_dir = Path("outputs/eval_langgraph_e2e/eval_20260617_230809")
val_eval_dir = Path("outputs/eval_langgraph_yolo_val/yolo_val_20260618_083918")

out_dir = Path("outputs/final_materials")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "final_evaluation_summary.md"


def read_summary(path: Path):
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def read_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def per_class_table(rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["gt_type"]].append(r)

    lines = []
    lines.append("| 缺陷类型 | 样本数 | 图像级准确率 | YOLO检出率 | RAG命中率 | 人工复核触发率 |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for cls in ["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"]:
        items = by_class.get(cls, [])
        if not items:
            continue
        total = len(items)
        correct = sum(r["is_correct"] == "1" for r in items)
        detected = sum(int(r["num_detections"]) > 0 for r in items)
        rag_hit = sum(r["rag_hit"] == "1" for r in items)
        review = sum(r["need_human_review"] == "1" for r in items)

        lines.append(
            f"| {cls} | {total} | {correct / total:.4f} | {detected / total:.4f} | "
            f"{rag_hit / total:.4f} | {review / total:.4f} |"
        )

    return "\n".join(lines)


def confusion_table(rows):
    pairs = Counter((r["gt_type"], r["image_pred_type"]) for r in rows)
    lines = []
    lines.append("| 真实类别 | 预测类别 | 数量 |")
    lines.append("|---|---|---:|")
    for (gt, pred), n in sorted(pairs.items()):
        lines.append(f"| {gt} | {pred} | {n} |")
    return "\n".join(lines)


random_summary = read_summary(random_eval_dir / "summary.txt")
val_summary = read_summary(val_eval_dir / "summary.txt")
val_rows = read_rows(val_eval_dir / "eval_results.csv")

bad_cases = [r for r in val_rows if r["is_correct"] != "1" or r["flow_success"] != "1"]

md = []
md.append("# PCB LangGraph + MCP 多 Agent 系统最终评测总结\n")

md.append("## 1. 系统组成\n")
md.append(
    "系统采用 LangGraph 构建多 Agent 编排流程，包括 DetectionAgent、VisionAgent、"
    "DecisionAgent、RAGAgent 和 ReportAgent。DetectionAgent 调用 YOLO11n 对整张 PCB 图像进行缺陷定位；"
    "VisionAgent 根据检测框裁剪局部图像，并调用 Qwen2.5-VL LoRA 进行缺陷类型复核；"
    "DecisionAgent 融合 YOLO 与 VLM 结果，并在二者不一致时触发人工复核标记；"
    "RAGAgent 通过 MCP 工具服务 pcb_knowledge_search 检索 PCB 维修知识库；"
    "ReportAgent 汇总生成结构化诊断报告。"
)

md.append("\n## 2. 60 张随机端到端流程测试结果\n")
md.append("| 指标 | 结果 |")
md.append("|---|---:|")
for k in [
    "Total samples",
    "Flow success rate",
    "YOLO detection rate",
    "Image-level defect accuracy",
    "Human review trigger rate",
    "RAG hit rate",
]:
    md.append(f"| {k} | {random_summary.get(k, '')} |")

md.append("\n## 3. YOLO 验证集 69 张端到端评测结果\n")
md.append("| 指标 | 结果 |")
md.append("|---|---:|")
for k in [
    "Total samples",
    "Flow success rate",
    "YOLO detection rate",
    "Image-level defect accuracy",
    "Human review trigger rate",
    "RAG hit rate",
]:
    md.append(f"| {k} | {val_summary.get(k, '')} |")

md.append("\n## 4. YOLO 验证集分类别结果\n")
md.append(per_class_table(val_rows))

md.append("\n## 5. 混淆矩阵统计\n")
md.append(confusion_table(val_rows))

md.append("\n## 6. 失败案例统计\n")
md.append(f"- Bad cases: {len(bad_cases)}")
if bad_cases:
    for r in bad_cases:
        md.append(
            f"- {r['image_path']}：GT={r['gt_type']}，Pred={r['image_pred_type']}，"
            f"YOLO={r['yolo_top_type']}，VLM={r['vlm_types']}，Final={r['final_types']}"
        )
else:
    md.append("- 本次 YOLO 验证集端到端评测中未出现图像级错误样本。")

md.append("\n## 7. 报告推荐表述\n")
md.append(
    "在 YOLO 验证集 69 张 PCB 图像上，LangGraph 多 Agent 系统完成了从整图输入、"
    "YOLO 缺陷定位、Qwen2.5-VL 局部复核、MCP 知识检索到报告生成的完整流程。"
    "系统流程成功率为 100%，YOLO 缺陷检出率为 100%，图像级缺陷诊断准确率为 100%，"
    "RAG 命中率为 100%，人工复核触发率为 30.43%。实验结果表明，该系统能够稳定完成"
    "PCB 缺陷端到端自动诊断，并能在 YOLO 与 VLM 结果不一致时触发人工复核机制。"
)

out_path.write_text("\n".join(md), encoding="utf-8")
print(out_path)
