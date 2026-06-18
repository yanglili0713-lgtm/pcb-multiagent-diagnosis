import json
import re
from pathlib import Path

ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
DATA_DIR = ROOT / "LLaMA-Factory/data"

SRC_FILES = {
    "train": DATA_DIR / "pcb_real_train.json",
    "val": DATA_DIR / "pcb_real_val.json",
}

OUT_FILES = {
    "train": DATA_DIR / "pcb_diag_train.json",
    "val": DATA_DIR / "pcb_diag_val.json",
}

REPAIR_MAP = {
    "短路": "建议检查相邻线路或焊盘之间是否存在铜残留、锡桥或异物，并进行清理、修铜或重新焊接。",
    "开路": "建议检查断线位置，必要时进行飞线、补铜或重新制作线路连接。",
    "鼠咬": "建议检查铜箔边缘缺损区域，评估是否影响导通，必要时补铜修复。",
    "毛刺": "建议清除多余突起铜箔，避免其继续扩展形成短路风险。",
    "多余铜": "建议去除非设计区域的残铜或异物，防止造成线路间异常连接。",
    "漏孔": "建议检查钻孔或过孔缺失位置，必要时重新钻孔、补孔或更换板件。",
}

def extract_defects(answer):
    m = re.search(r"缺陷类型[:：](.*?)[。.\n]", answer)
    if not m:
        return ["未知缺陷"]

    s = m.group(1)
    parts = re.split(r"[、,，;；\s]+", s)
    defects = []
    for p in parts:
        p = p.strip()
        if p:
            defects.append(p)

    return defects or ["未知缺陷"]

def convert_one(item):
    old_answer = item["messages"][1]["content"]
    defects = extract_defects(old_answer)

    repair_lines = []
    for d in defects:
        repair = REPAIR_MAP.get(d, "建议结合显微检查、电气测试和人工复核进一步确认，并进行相应返修。")
        repair_lines.append(f"{d}：{repair}")

    new_answer = (
        "该 PCB 图像中检测到疑似缺陷。\n"
        f"缺陷类型：{'、'.join(defects)}。\n"
        "缺陷位置：图中存在疑似缺陷区域，具体位置建议结合检测框、局部放大图或人工复核进一步确认。\n"
        f"维修建议：{' '.join(repair_lines)}"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>请判断这张 PCB 图像中的缺陷类型，并给出简要位置描述和维修建议。不要输出具体坐标。"
            },
            {
                "role": "assistant",
                "content": new_answer
            }
        ],
        "images": item["images"]
    }

for split, src in SRC_FILES.items():
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = [convert_one(x) for x in data]

    with open(OUT_FILES[split], "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(split, len(new_data), "->", OUT_FILES[split])
    print(json.dumps(new_data[0], ensure_ascii=False, indent=2)[:1000])
