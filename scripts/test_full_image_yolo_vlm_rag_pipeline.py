import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info


PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")

YOLO_MODEL = PROJECT_ROOT / "output/yolo_pcb_detect/yolo11n_pcb_1024_pretrained_noamp/weights/best.pt"
BASE_MODEL = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"

LORA_CANDIDATES = [
    PROJECT_ROOT / "LLaMA-Factory/output/qwen25vl_7b_pcb_crop_cls_full",
    PROJECT_ROOT / "output/qwen25vl_7b_pcb_crop_cls_full",
]

CLASS_EN2ZH = {
    "short": "短路",
    "open_circuit": "开路",
    "mouse_bite": "鼠咬",
    "spur": "毛刺",
    "spurious_copper": "多余铜",
    "missing_hole": "漏孔",
}

ZH_TYPES = ["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"]


def find_lora_dir() -> Path:
    for p in LORA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "没有找到 crop LoRA 目录，请检查是否存在：\n"
        + "\n".join(str(p) for p in LORA_CANDIDATES)
    )


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


def extract_defect_type(text: str):
    for t in ZH_TYPES:
        if t in text:
            return t
    return None


def load_vlm():
    lora_dir = find_lora_dir()
    print(f"[INFO] Loading base model: {BASE_MODEL}")
    print(f"[INFO] Loading LoRA adapter: {lora_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL),
        trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=dtype,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.to("cuda")
    model.eval()

    return model, processor


def vlm_classify_crop(model, processor, crop_path: Path):
    prompt = (
        "请判断这张 PCB 缺陷局部图像的缺陷类型。"
        "只从以下六类中选择一个：短路、开路、鼠咬、毛刺、多余铜、漏孔。"
        "只输出“缺陷类型：xxx”。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(crop_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="整张 PCB 图像路径")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_det", type=int, default=20)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在：{image_path}")

    out_dir = PROJECT_ROOT / "output/full_image_yolo_vlm_rag" / image_path.stem
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input image: {image_path}")
    print(f"[INFO] Output dir: {out_dir}")

    # 1. YOLO 检测
    print("[INFO] Running YOLO detection...")
    yolo = YOLO(str(YOLO_MODEL))
    results = yolo.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        save=False,
        verbose=False,
        device=0,
    )

    r = results[0]
    names = r.names

    # 保存 YOLO 可视化图
    annotated = r.plot()
    annotated_img = Image.fromarray(annotated[..., ::-1])
    annotated_path = out_dir / "yolo_detect_annotated.jpg"
    annotated_img.save(annotated_path)

    detections = []
    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].detach().cpu().tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            conf = float(box.conf[0].detach().cpu().item())
            cls_en = names[cls_id]
            cls_zh = CLASS_EN2ZH.get(cls_en, cls_en)

            detections.append({
                "idx": i + 1,
                "xyxy": xyxy,
                "cls_id": cls_id,
                "cls_en": cls_en,
                "cls_zh": cls_zh,
                "conf": conf,
            })

    del yolo
    torch.cuda.empty_cache()

    if not detections:
        report_path = out_dir / "diagnosis_report.md"
        report_path.write_text(
            f"# PCB 整图自动诊断报告\n\n"
            f"输入图像：`{image_path}`\n\n"
            f"YOLO 未检测到缺陷区域。建议降低 `--conf` 后重新测试，例如 `--conf 0.10`。\n",
            encoding="utf-8"
        )
        print(f"[WARN] No detections. Report saved to: {report_path}")
        print(f"[INFO] Annotated image saved to: {annotated_path}")
        return

    print(f"[INFO] YOLO detected {len(detections)} defect region(s).")

    # 2. 自动裁剪
    original_img = Image.open(image_path).convert("RGB")

    for det in detections:
        crop_img, crop_xyxy = crop_with_margin(original_img, det["xyxy"])
        crop_path = crop_dir / f"crop_{det['idx']:02d}_yolo_{det['cls_en']}_{det['conf']:.3f}.jpg"
        crop_img.save(crop_path)

        det["crop_path"] = crop_path
        det["crop_xyxy"] = crop_xyxy

    # 3. 加载 VLM crop 分类模型
    model, processor = load_vlm()

    # 4. 加载 RAG pipeline
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from agent.diagnosis_pipeline import run_diagnosis_pipeline
        rag_available = True
    except Exception as e:
        print(f"[WARN] RAG pipeline import failed: {e}")
        rag_available = False
        run_diagnosis_pipeline = None

    # 5. VLM 分类 + RAG 报告
    report_lines = []
    report_lines.append("# PCB 整图自动诊断报告\n")
    report_lines.append(f"- 输入图像：`{image_path}`")
    report_lines.append(f"- YOLO 模型：`{YOLO_MODEL}`")
    report_lines.append(f"- 检测可视化图：`{annotated_path}`")
    report_lines.append(f"- 检测区域数量：**{len(detections)}**\n")

    for det in detections:
        print(f"[INFO] Classifying crop {det['idx']} with VLM: {det['crop_path']}")

        vlm_output = vlm_classify_crop(model, processor, det["crop_path"])
        vlm_type = extract_defect_type(vlm_output)

        yolo_type = det["cls_zh"]

        # 决策策略：
        # 1. YOLO 与 VLM 一致：直接采用该类别
        # 2. VLM 未解析：采用 YOLO 类别
        # 3. 二者不一致时：若 YOLO 置信度较高，优先采用 YOLO，但标记人工复核
        # 4. 若 YOLO 置信度较低，则采用 VLM 结果，但同样标记人工复核
        agree = (vlm_type == yolo_type) if vlm_type else False

        if agree:
            final_type = yolo_type
            decision_note = "YOLO 与 VLM 分类一致，可信度较高。"
        elif not vlm_type:
            final_type = yolo_type
            decision_note = "VLM 未能解析类别，采用 YOLO 检测类别，建议人工复核。"
        elif det["conf"] >= 0.65:
            final_type = yolo_type
            decision_note = "YOLO 置信度较高但与 VLM 不一致，优先采用 YOLO 类别，并建议人工复核。"
        else:
            final_type = vlm_type
            decision_note = "YOLO 置信度较低且与 VLM 不一致，暂采用 VLM 类别，并建议人工复核。"

        det["vlm_output"] = vlm_output
        det["vlm_type"] = vlm_type
        det["final_type"] = final_type
        det["agree"] = agree

        report_lines.append(f"\n## 缺陷区域 {det['idx']}\n")
        report_lines.append(f"- YOLO 检测类别：**{yolo_type}**")
        report_lines.append(f"- YOLO 置信度：**{det['conf']:.3f}**")
        report_lines.append(f"- YOLO 原始框 xyxy：`{[round(x, 1) for x in det['xyxy']]}`")
        report_lines.append(f"- 自动裁剪图：`{det['crop_path']}`")
        report_lines.append(f"- VLM 输出：`{vlm_output}`")
        report_lines.append(f"- VLM 分类：**{vlm_type if vlm_type else '未解析'}**")
        report_lines.append(f"- 最终用于检索的缺陷类型：**{final_type}**")

        report_lines.append(f"- 一致性判断：{decision_note}")

        if rag_available:
            vlm_result = {
                "defect_type": final_type,
                "defect_type_zh": final_type,
                "yolo_defect_type": yolo_type,
                "vlm_output": vlm_output,
                "location": f"YOLO 检测框 xyxy={det['xyxy']}",
                "crop_path": str(det["crop_path"]),
                "confidence": det["conf"],
            }

            try:
                rag_report = run_diagnosis_pipeline(vlm_result, top_k=args.top_k)
                report_lines.append("\n### RAG 维修知识诊断\n")

                if isinstance(rag_report, dict) and "report" in rag_report:
                    report_lines.append(rag_report["report"])
                else:
                    report_lines.append(str(rag_report))
            except Exception as e:
                report_lines.append("\n### RAG 维修知识诊断\n")
                report_lines.append(f"RAG 检索失败：`{repr(e)}`")
                report_lines.append("请确认 Elasticsearch 是否已启动：`curl http://localhost:9200`")
        else:
            report_lines.append("\n### RAG 维修知识诊断\n")
            report_lines.append("RAG pipeline 未成功导入，跳过知识库检索。")

    report_path = out_dir / "diagnosis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n[DONE] Full image YOLO + VLM + RAG pipeline finished.")
    print(f"[DONE] Report: {report_path}")
    print(f"[DONE] Annotated image: {annotated_path}")
    print(f"[DONE] Crops dir: {crop_dir}")


if __name__ == "__main__":
    main()
