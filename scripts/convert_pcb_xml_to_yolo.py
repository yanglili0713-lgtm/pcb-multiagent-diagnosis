import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from PIL import Image


PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET"
ANN_ROOT = RAW_ROOT / "Annotations"

LF_DATA_DIR = PROJECT_ROOT / "LLaMA-Factory" / "data"

SPLIT_JSON = {
    "train": LF_DATA_DIR / "pcb_cls_train.json",
    "val": LF_DATA_DIR / "pcb_cls_val.json",
}

OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "pcb_yolo_detect"

CLASS_NAMES = [
    "short",
    "open_circuit",
    "mouse_bite",
    "spur",
    "spurious_copper",
    "missing_hole",
]

CLASS_ZH = {
    "short": "短路",
    "open_circuit": "开路",
    "mouse_bite": "鼠咬",
    "spur": "毛刺",
    "spurious_copper": "多余铜",
    "missing_hole": "漏孔",
}

NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

XML_NAME_MAP = {
    "short": "short",
    "open_circuit": "open_circuit",
    "open": "open_circuit",
    "mouse_bite": "mouse_bite",
    "mousebite": "mouse_bite",
    "spur": "spur",
    "spurious_copper": "spurious_copper",
    "spurious": "spurious_copper",
    "missing_hole": "missing_hole",
    "missing": "missing_hole",
}


def norm_name(name: str):
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return XML_NAME_MAP.get(key)


def find_xml(image_path: Path):
    candidates = list(ANN_ROOT.rglob(image_path.stem + ".xml"))
    return candidates[0] if candidates else None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_xml_to_yolo(xml_path: Path, image_path: Path):
    img = Image.open(image_path)
    w, h = img.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []

    for obj in root.findall(".//object"):
        name_node = obj.find("name")
        bndbox = obj.find("bndbox")

        if name_node is None or bndbox is None:
            continue

        cls_name = norm_name(name_node.text or "")
        if cls_name is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except Exception:
            continue

        xmin = clamp(xmin, 0, w - 1)
        xmax = clamp(xmax, 0, w - 1)
        ymin = clamp(ymin, 0, h - 1)
        ymax = clamp(ymax, 0, h - 1)

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        cls_id = NAME_TO_ID[cls_name]

        labels.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    return labels


def make_split(split: str):
    image_out = OUT_ROOT / "images" / split
    label_out = OUT_ROOT / "labels" / split

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    data = json.loads(SPLIT_JSON[split].read_text(encoding="utf-8"))

    image_count = 0
    box_count = 0
    missing_xml = 0
    empty_label = 0
    counter = Counter()

    for item in data:
        image_path = Path(item["images"][0])

        if not image_path.exists():
            continue

        xml_path = find_xml(image_path)
        if xml_path is None:
            missing_xml += 1
            continue

        labels = parse_xml_to_yolo(xml_path, image_path)

        if not labels:
            empty_label += 1
            continue

        dst_img = image_out / image_path.name
        dst_txt = label_out / f"{image_path.stem}.txt"

        shutil.copy2(image_path, dst_img)
        dst_txt.write_text("\n".join(labels) + "\n", encoding="utf-8")

        image_count += 1
        box_count += len(labels)

        for line in labels:
            cls_id = int(line.split()[0])
            counter[CLASS_NAMES[cls_id]] += 1

    print("=" * 100)
    print("split:", split)
    print("images:", image_count)
    print("boxes:", box_count)
    print("missing_xml:", missing_xml)
    print("empty_label:", empty_label)
    print("class counter:", counter)


def write_yaml():
    yaml_path = OUT_ROOT / "pcb.yaml"

    names_lines = "\n".join(
        [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    )

    content = f"""path: {OUT_ROOT}
train: images/train
val: images/val

names:
{names_lines}
"""

    yaml_path.write_text(content, encoding="utf-8")
    print("=" * 100)
    print("YOLO yaml:", yaml_path)
    print(content)


def main():
    for split in ["train", "val"]:
        make_split(split)

    write_yaml()


if __name__ == "__main__":
    main()
