import csv
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from agent.pcb_langgraph import run_pcb_langgraph


IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images"

CLASS_DIR_TO_ZH = {
    "Short": "短路",
    "Open_circuit": "开路",
    "Mouse_bite": "鼠咬",
    "Spur": "毛刺",
    "Spurious_copper": "多余铜",
    "Missing_hole": "漏孔",
}


def collect_images(per_class: int, seed: int):
    random.seed(seed)

    samples = []

    for class_dir, gt_zh in CLASS_DIR_TO_ZH.items():
        d = IMAGE_ROOT / class_dir
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            files.extend(d.glob(ext))

        files = sorted(files)
        random.shuffle(files)

        selected = files[:per_class]

        for p in selected:
            samples.append(
                {
                    "image_path": str(p),
                    "class_dir": class_dir,
                    "gt_type": gt_zh,
                }
            )

    random.shuffle(samples)
    return samples


def choose_image_level_prediction(decisions):
    if not decisions:
        return "未检出"

    # 先按最终类别投票；票数相同时按 YOLO 置信度总和决定
    count = Counter()
    conf_sum = defaultdict(float)

    for d in decisions:
        t = d.get("final_type", "未知")
        count[t] += 1
        conf_sum[t] += float(d.get("yolo_conf", 0.0))

    candidates = list(count.keys())
    candidates.sort(key=lambda x: (count[x], conf_sum[x]), reverse=True)

    return candidates[0]


def summarize_yolo_top(detections):
    if not detections:
        return "未检出", 0.0

    top = max(detections, key=lambda x: float(x.get("yolo_conf", 0.0)))
    return top.get("yolo_type", "未知"), float(top.get("yolo_conf", 0.0))


def safe_join(values):
    return "|".join(str(x) for x in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max_det", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--prefer_yolo_conf", type=float, default=0.65)
    parser.add_argument("--rag_backend", choices=["local", "mcp"], default="mcp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_name = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    eval_dir = PROJECT_ROOT / "outputs" / "eval_langgraph_e2e" / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_images(per_class=args.per_class, seed=args.seed)

    csv_path = eval_dir / "eval_results.csv"

    rows = []

    print(f"[INFO] Eval dir: {eval_dir}")
    print(f"[INFO] Total samples: {len(samples)}")
    print(f"[INFO] RAG backend: {args.rag_backend}")

    for idx, sample in enumerate(samples, 1):
        image_path = sample["image_path"]
        gt_type = sample["gt_type"]
        stem = Path(image_path).stem

        output_dir = eval_dir / f"{idx:03d}_{sample['class_dir']}_{stem}"

        print("=" * 80)
        print(f"[{idx}/{len(samples)}] {image_path}")
        print(f"[GT] {gt_type}")

        row = {
            "idx": idx,
            "image_path": image_path,
            "class_dir": sample["class_dir"],
            "gt_type": gt_type,
            "flow_success": 0,
            "num_detections": 0,
            "num_decisions": 0,
            "yolo_top_type": "",
            "yolo_top_conf": 0.0,
            "vlm_types": "",
            "final_types": "",
            "image_pred_type": "",
            "is_correct": 0,
            "need_human_review": 0,
            "rag_hit": 0,
            "report_path": "",
            "annotated_image_path": "",
            "output_dir": str(output_dir),
            "error": "",
        }

        try:
            state = run_pcb_langgraph(
                image_path=image_path,
                conf=args.conf,
                imgsz=args.imgsz,
                max_det=args.max_det,
                top_k=args.top_k,
                prefer_yolo_conf=args.prefer_yolo_conf,
                rag_backend=args.rag_backend,
                output_dir=str(output_dir),
            )

            detections = state.get("detections", [])
            decisions = state.get("decisions", [])
            rag_reports = state.get("rag_reports", {})

            yolo_top_type, yolo_top_conf = summarize_yolo_top(detections)
            image_pred_type = choose_image_level_prediction(decisions)

            vlm_types = [d.get("vlm_type", "未知") for d in decisions]
            final_types = [d.get("final_type", "未知") for d in decisions]

            need_review = any(bool(d.get("need_human_review", False)) for d in decisions)

            row.update(
                {
                    "flow_success": 1 if not state.get("errors") else 0,
                    "num_detections": len(detections),
                    "num_decisions": len(decisions),
                    "yolo_top_type": yolo_top_type,
                    "yolo_top_conf": round(yolo_top_conf, 4),
                    "vlm_types": safe_join(vlm_types),
                    "final_types": safe_join(final_types),
                    "image_pred_type": image_pred_type,
                    "is_correct": 1 if image_pred_type == gt_type else 0,
                    "need_human_review": 1 if need_review else 0,
                    "rag_hit": 1 if gt_type in rag_reports else 0,
                    "report_path": state.get("report_path", ""),
                    "annotated_image_path": state.get("annotated_image_path", ""),
                    "error": safe_join(state.get("errors", [])),
                }
            )

            print(
                f"[PRED] image_pred={image_pred_type}, "
                f"yolo_top={yolo_top_type}({yolo_top_conf:.3f}), "
                f"correct={row['is_correct']}, review={row['need_human_review']}, rag_hit={row['rag_hit']}"
            )

        except Exception as e:
            row["error"] = repr(e)
            print("[ERROR]", repr(e))

        rows.append(row)

        # 每跑一张就写一次，防止中途断了丢结果
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    total = len(rows)
    success = sum(r["flow_success"] for r in rows)
    correct = sum(r["is_correct"] for r in rows)
    detected = sum(1 for r in rows if int(r["num_detections"]) > 0)
    review = sum(r["need_human_review"] for r in rows)
    rag_hit = sum(r["rag_hit"] for r in rows)

    summary_path = eval_dir / "summary.txt"

    summary = []
    summary.append(f"Total samples: {total}")
    summary.append(f"Flow success rate: {success / total:.4f}")
    summary.append(f"YOLO detection rate: {detected / total:.4f}")
    summary.append(f"Image-level defect accuracy: {correct / total:.4f}")
    summary.append(f"Human review trigger rate: {review / total:.4f}")
    summary.append(f"RAG hit rate: {rag_hit / total:.4f}")
    summary.append(f"CSV path: {csv_path}")

    summary_text = "\n".join(summary)
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("[DONE] End-to-end evaluation finished.")
    print(summary_text)
    print(f"Summary path: {summary_path}")


if __name__ == "__main__":
    main()
