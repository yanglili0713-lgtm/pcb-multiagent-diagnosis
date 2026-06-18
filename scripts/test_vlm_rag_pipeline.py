import argparse
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.diagnosis_pipeline import run_diagnosis_pipeline


BASE_MODEL = "/extra/caochunhong/gm/pcb_multi_agent/models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/extra/caochunhong/gm/pcb_multi_agent/output/qwen25vl_7b_pcb_crop_cls_full"


DEFECT_TYPES = [
    "短路",
    "开路",
    "鼠咬",
    "毛刺",
    "多余铜",
    "漏孔",
]


def extract_defect_type(text: str) -> str:
    """
    从 VLM 输出中提取缺陷类型。
    先处理“无缺陷/未发现缺陷”类否定表达，避免把“没有短路”误判成短路。
    """
    no_defect_patterns = [
        "没有明显",
        "未发现明显",
        "无明显",
        "未检测到明显",
        "没有可见缺陷",
        "未见明显缺陷",
        "正常",
    ]

    if any(p in text for p in no_defect_patterns):
        # 如果同时没有明确的“缺陷类型：xxx”，则判为未知
        if not re.search(r"缺陷类型[:：]", text):
            return "未知"

    m = re.search(r"缺陷类型[:：]\s*([^\n。；;，,]+)", text)
    if m:
        candidate = m.group(1).strip()
        for d in DEFECT_TYPES:
            if d in candidate:
                return d
        if "无" in candidate or "未" in candidate or "正常" in candidate:
            return "未知"

    # 只有在没有明显否定表达时，才全文关键词匹配
    if not any(p in text for p in no_defect_patterns):
        for d in DEFECT_TYPES:
            if d in text:
                return d

    return "未知"


def load_vlm():
    print("加载 processor...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    print("加载 base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("移动 base model 到 cuda...")
    model = model.to("cuda")

    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        device_map=None,
    )
    model = model.to("cuda")
    model.eval()

    print("VLM 加载完成")
    return model, processor


def run_vlm_inference(model, processor, image_path: str) -> str:
    prompt = "请判断这张 PCB 图像的缺陷类型。只从以下六类中选择一个：短路、开路、鼠咬、毛刺、多余铜、漏孔。只输出“缺陷类型：xxx”。"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
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
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
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
    parser.add_argument(
        "--image",
        type=str,
        default="/extra/caochunhong/gm/pcb_multi_agent/data/raw/PCB_DATASET/images/Missing_hole/01_missing_hole_09.jpg",
        help="PCB image path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="RAG top-k",
    )
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    print("=" * 100)
    print("输入图片:")
    print(image_path)

    model, processor = load_vlm()

    print("=" * 100)
    print("开始 VLM 图像诊断...")
    vlm_text = run_vlm_inference(model, processor, image_path)

    print("=" * 100)
    print("VLM 原始输出:")
    print(vlm_text)

    defect_type = extract_defect_type(vlm_text)

    if defect_type == "未知":
        confidence = "低"
        description = vlm_text
    else:
        confidence = "中"
        description = vlm_text

    vlm_result = {
        "defect_type": defect_type,
        "confidence": confidence,
        "description": description,
    }

    print("=" * 100)
    print("结构化 VLM 结果:")
    print(vlm_result)

    print("=" * 100)
    print("开始 RAG 检索并生成报告...")
    result = run_diagnosis_pipeline(vlm_result, top_k=args.top_k)

    print("=" * 100)
    print("最终诊断报告:")
    print(result["report"])


if __name__ == "__main__":
    main()
