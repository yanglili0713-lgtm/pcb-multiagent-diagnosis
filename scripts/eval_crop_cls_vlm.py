import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


BASE_MODEL = "/extra/caochunhong/gm/pcb_multi_agent/models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/extra/caochunhong/gm/pcb_multi_agent/output/qwen25vl_7b_pcb_crop_cls_full"
VAL_JSON = "/extra/caochunhong/gm/pcb_multi_agent/LLaMA-Factory/data/pcb_crop_cls_val.json"

DEFECT_TYPES = ["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"]

PROMPT = "请判断该 PCB 局部缺陷图像的缺陷类型。只从以下六类中选择一个：短路、开路、鼠咬、毛刺、多余铜、漏孔。只输出“缺陷类型：xxx”。"


def extract_defect_type(text: str) -> str:
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


def load_model():
    print("加载 processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("加载 base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("移动 base model 到 cuda...")
    model = model.to("cuda")

    print("加载 crop LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        device_map=None,
    )
    model = model.to("cuda")
    model.eval()

    print("模型加载完成")
    return model, processor


def infer_one(model, processor, image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
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
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_csv", type=str, default="outputs/eval_crop_cls_val_results.csv")
    args = parser.parse_args()

    data = json.loads(Path(VAL_JSON).read_text(encoding="utf-8"))

    if args.max_samples is not None:
        data = data[:args.max_samples]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()

    rows = []
    total = 0
    correct = 0
    confusion = defaultdict(Counter)

    for idx, item in enumerate(data, 1):
        image_path = item["images"][0]
        true_text = item["messages"][1]["content"]
        true_label = extract_defect_type(true_text)

        pred_text = infer_one(model, processor, image_path)
        pred_label = extract_defect_type(pred_text)

        is_correct = true_label == pred_label
        total += 1
        correct += int(is_correct)
        confusion[true_label][pred_label] += 1

        print(f"[{idx}/{len(data)}] true={true_label} pred={pred_label} correct={is_correct} image={Path(image_path).name}")

        rows.append({
            "idx": idx,
            "image": image_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(is_correct),
            "raw_output": pred_text,
        })

    acc = correct / total if total else 0.0

    print("=" * 100)
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {acc:.4f}")

    print("=" * 100)
    print("混淆矩阵 true -> pred:")
    for true_label in DEFECT_TYPES:
        print(true_label, dict(confusion[true_label]))

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "image", "true_label", "pred_label", "correct", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("结果已保存:", args.out_csv)


if __name__ == "__main__":
    main()
