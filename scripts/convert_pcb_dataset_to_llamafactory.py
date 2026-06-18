import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/raw/PCB_DATASET"
OUT_DIR = PROJECT_ROOT / "LLaMA-Factory/data"

TRAIN_OUT = OUT_DIR / "pcb_real_train.json"
VAL_OUT = OUT_DIR / "pcb_real_val.json"

random.seed(42)

CLASS_ZH = {
    "short": "短路",
    "open_circuit": "开路",
    "open": "开路",
    "mouse_bite": "鼠咬",
    "mousebite": "鼠咬",
    "spur": "毛刺",
    "spurious_copper": "多余铜",
    "spurious": "多余铜",
    "missing_hole": "漏孔",
    "missing": "漏孔",
}

SOLUTION = {
    "短路": "建议检查相邻线路或焊盘之间是否存在铜残留、锡桥或异物，并进行清理、修铜或重新焊接。",
    "开路": "建议检查断线位置，必要时进行飞线、补铜或重新制作线路连接。",
    "鼠咬": "建议检查铜箔边缘缺损区域，评估是否影响导通，必要时补铜修复。",
    "毛刺": "建议清除多余突起铜箔，避免其继续扩展形成短路风险。",
    "多余铜": "建议去除非设计区域的残铜或异物，防止造成线路间异常连接。",
    "漏孔": "建议检查钻孔或过孔缺失位置，必要时重新钻孔、补孔或更换板件。",
}

def norm_name(name: str) -> str:
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    return CLASS_ZH.get(s, s)

def build_image_index():
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    index = {}
    for p in DATA_ROOT.rglob("*"):
        if p.suffix.lower() in image_exts:
            index[p.stem] = p.resolve()
    return index

def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    width, height = None, None
    if size_node is not None:
        w = size_node.findtext("width")
        h = size_node.findtext("height")
        if w and h:
            width, height = int(float(w)), int(float(h))

    objects = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", default="unknown")
        defect = norm_name(raw_name)

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))

        objects.append({
            "defect": defect,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return width, height, objects

def make_answer(objects):
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj["defect"]].append(obj["bbox"])

    defect_types = list(grouped.keys())

    lines = []
    lines.append("该 PCB 图像中检测到疑似缺陷。")
    lines.append("缺陷类型：" + "、".join(defect_types) + "。")

    loc_parts = []
    for defect, boxes in grouped.items():
        box_str = "；".join([f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]" for b in boxes])
        loc_parts.append(f"{defect}位于 {box_str}")

    lines.append("缺陷位置：" + "；".join(loc_parts) + "。")

    repair_parts = []
    for defect in defect_types:
        repair_parts.append(f"{defect}：{SOLUTION.get(defect, '建议结合显微检查和电气测试进一步确认，并进行相应返修。')}")

    lines.append("维修建议：" + " ".join(repair_parts))

    return "\n".join(lines)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index()
    xml_paths = sorted((DATA_ROOT / "Annotations").rglob("*.xml"))

    if not xml_paths:
        raise RuntimeError(f"没有找到 XML 标注文件: {DATA_ROOT / 'Annotations'}")

    records = []
    missing_images = []

    for xml_path in xml_paths:
        stem = xml_path.stem
        img_path = image_index.get(stem)

        if img_path is None:
            missing_images.append(str(xml_path))
            continue

        width, height, objects = parse_xml(xml_path)
        if not objects:
            continue

        answer = make_answer(objects)

        record = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>请检测这张 PCB 图像中的缺陷，说明缺陷类型、位置，并给出维修建议。"
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "images": [str(img_path)]
        }
        records.append(record)

    random.shuffle(records)

    n_total = len(records)
    n_val = max(1, int(n_total * 0.1))

    val_records = records[:n_val]
    train_records = records[n_val:]

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(VAL_OUT, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    print(f"XML 数量: {len(xml_paths)}")
    print(f"成功生成样本: {len(records)}")
    print(f"训练集: {len(train_records)} -> {TRAIN_OUT}")
    print(f"验证集: {len(val_records)} -> {VAL_OUT}")
    print(f"缺失图片的 XML 数量: {len(missing_images)}")

    if records:
        print("第一条样本:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2)[:1500])

if __name__ == "__main__":
    main()
