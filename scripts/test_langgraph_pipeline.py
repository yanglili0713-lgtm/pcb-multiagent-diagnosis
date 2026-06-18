import argparse
from pathlib import Path

from agent.pcb_langgraph import run_pcb_langgraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max_det", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--prefer_yolo_conf", type=float, default=0.65)
    parser.add_argument("--rag_backend", choices=["local", "mcp"], default="local")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    state = run_pcb_langgraph(
        image_path=args.image,
        conf=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        top_k=args.top_k,
        prefer_yolo_conf=args.prefer_yolo_conf,
        rag_backend=args.rag_backend,
        output_dir=args.output_dir,
    )

    print("\n[DONE] LangGraph pipeline finished.")
    print("Report:", state.get("report_path"))
    print("Annotated image:", state.get("annotated_image_path"))
    print("Num detections:", len(state.get("detections", [])))
    print("Num decisions:", len(state.get("decisions", [])))
    print("RAG backend:", args.rag_backend)

    print("\n=== Messages ===")
    for msg in state.get("messages", []):
        print("-", msg)

    print("\n=== Decisions ===")
    for d in state.get("decisions", []):
        print(
            f"region={d['idx']} "
            f"yolo={d['yolo_type']}({d['yolo_conf']:.3f}) "
            f"vlm={d.get('vlm_type')} "
            f"final={d['final_type']} "
            f"review={d['need_human_review']}"
        )

    if state.get("errors"):
        print("\n=== Errors ===")
        for e in state["errors"]:
            print("-", e)


if __name__ == "__main__":
    main()
