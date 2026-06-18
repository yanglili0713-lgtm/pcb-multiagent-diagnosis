import json
import re
from pathlib import Path

ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent/LLaMA-Factory/data")

INPUTS = {
    "pcb_real_train.json": "pcb_cls_train.json",
    "pcb_real_val.json": "pcb_cls_val.json",
}

DEFECT_TYPES = ["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"]


def extract_type(answer: str) -> str:
    m = re.search(r"缺陷类型[:：]\s*([^\n。；;，,]+)", answer)
    if m:
        cand = m.group(1).strip()
        for d in DEFECT_TYPES:
            if d in cand:
                return d

    for d in DEFECT_TYPES:
        if d in answer:
            return d

    return "未知"


for src_name, out_name in INPUTS.items():
    src = ROOT / src_name
    out = ROOT / out_name

    data = json.loads(src.read_text(encoding="utf-8"))
    new_data = []

    for item in data:
        old_answer = item["messages"][1]["content"]
        defect_type = extract_type(old_answer)

        if defect_type == "未知":
            continue

        new_item = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>请判断这张 PCB 图像的缺陷类型。只从以下六类中选择一个：短路、开路、鼠咬、毛刺、多余铜、漏孔。只输出“缺陷类型：xxx”。"
                },
                {
                    "role": "assistant",
                    "content": f"缺陷类型：{defect_type}"
                }
            ],
            "images": item["images"]
        }
        new_data.append(new_item)

    out.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out, len(new_data))
