import json
from pathlib import Path
import random

project_root = Path(__file__).resolve().parents[1]
raw_dirs = [
    project_root / "data/raw/DeepPCB-master",
    project_root / "data/raw/PCB_DATASET",
]

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
images = []

for d in raw_dirs:
    if d.exists():
        for p in d.rglob("*"):
            if p.suffix.lower() in image_exts:
                images.append(p.resolve())

if len(images) == 0:
    raise RuntimeError("没有找到任何图片，请检查 data/raw 目录。")

random.seed(42)
images = random.sample(images, min(20, len(images)))

records = []
for img in images:
    records.append({
        "messages": [
            {
                "role": "user",
                "content": "<image>请观察这张 PCB 图像，判断是否可能存在缺陷，并给出简要诊断。"
            },
            {
                "role": "assistant",
                "content": "该 PCB 图像需要重点检查是否存在短路、开路、缺口、毛刺、漏孔、错位等缺陷。建议结合局部放大图和检测结果进一步确认故障类型。"
            }
        ],
        "images": [str(img)]
    })

out_dir = project_root / "LLaMA-Factory/data"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "pcb_smoke.json"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"生成完成: {out_file}")
print(f"样本数量: {len(records)}")
print("第一条样本:")
print(json.dumps(records[0], ensure_ascii=False, indent=2))
