import json
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")

base_model = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
adapter_path = PROJECT_ROOT / "output/qwen25vl_7b_pcb_diag_full"
val_json = PROJECT_ROOT / "LLaMA-Factory/data/pcb_real_val.json"

with open(val_json, "r", encoding="utf-8") as f:
    val_data = json.load(f)

sample = val_data[0]
image_path = sample["images"][0]

print("测试图片:", image_path)
print("参考答案:")
print(sample["messages"][1]["content"])
print("=" * 80)

processor = AutoProcessor.from_pretrained(
    str(base_model),
    trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(base_model),
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, str(adapter_path))
model.eval()

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
                "text": "请判断这张 PCB 图像中的缺陷类型，并给出简要位置描述和维修建议。不要输出具体坐标。"
            }
        ]
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
    return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        repetition_penalty=1.15
    )

generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("模型输出:")
print(output_text)
