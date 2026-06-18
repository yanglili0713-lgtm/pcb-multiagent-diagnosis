import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from collections import Counter

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
LF_DATA_DIR = PROJECT_ROOT / "LLaMA-Factory" / "data"

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET"
ANN_ROOT = RAW_ROOT / "Annotations"

OUT_IMG_ROOT = PROJECT_ROOT / "data" / "processed" / "pcb_crop_cls"

SPLITS = {
    "train": LF_DATA_DIR / "pcb_cls_train.json",
    "val": LF_DATA_DIR / "pcb_cls_val.json",
}

OUT_JSON = {
    "train": LF_DATA_DIR / "pcb_crop_cls_train.json",
    "val": LF_DATA_DIR / "pcb_crop_cls_val.json",
}

DEFECT_MAP = {
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

PROMPT = "<image>请判断该 PCB 局部缺陷图像的缺陷类型。只从以下六类中选择一个：短路、开路、鼠咬、毛刺、多余铜、漏孔。只输出“缺陷类型：xxx”。"


def normalize_label(name: str):
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return DEFECT_MAP.get(key)


def find_xml_for_image(image_path: Path):
    stem = image_path.stem
    candidates = list(ANN_ROOT.rglob(stem + ".xml"))
    if candidates:
        return candidates[0]
    return None


def parse_objects(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall(".//object"):
        name_node = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_node is None or bndbox is None:
            continue

        label = normalize_label(name_node.text or "")
        if label is None:
            continue

        try:
            xmin = int(float(bndbox.findtext("xmin")))
            ymin = int(float(bndbox.findtext("ymin")))
            xmax = int(float(bndbox.findtext("xmax")))
            ymax = int(float(bndbox.findtext("ymax")))
        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({
            "label": label,
            "bbox": [xmin, ymin, xmax, ymax],
        })

    return objects


def expand_box(box, w, h, margin_ratio=0.25, min_size=96):
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    new_w = max(bw * (1 + margin_ratio * 2), min_size)
    new_h = max(bh * (1 + margin_ratio * 2), min_size)

    # 做成接近正方形，避免缺陷太窄
    side = max(new_w, new_h)

    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = int(round(cx + side / 2))
    ny2 = int(round(cy + side / 2))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return [nx1, ny1, nx2, ny2]


def make_split(split_name: str):
    input_json = SPLITS[split_name]
    output_json = OUT_JSON[split_name]
    out_img_dir = OUT_IMG_ROOT / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_json.read_text(encoding="utf-8"))
    new_data = []
    counter = Counter()

    missing_xml = 0
    crop_id = 0

    for item in data:
        image_path = Path(item["images"][0])
        if not image_path.exists():
            continue

        xml_path = find_xml_for_image(image_path)
        if xml_path is None:
            missing_xml += 1
            continue

        objects = parse_objects(xml_path)
        if not objects:
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size

        for obj_idx, obj in enumerate(objects):
            label = obj["label"]
            box = expand_box(obj["bbox"], w=w, h=h)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            crop = img.crop((x1, y1, x2, y2))

            crop_name = f"{image_path.stem}_obj{obj_idx}_{label}.jpg"
            crop_path = out_img_dir / crop_name
            crop.save(crop_path, quality=95)

            new_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT,
                    },
                    {
                        "role": "assistant",
                        "content": f"缺陷类型：{label}",
                    },
                ],
                "images": [
                    str(crop_path)
                ],
            }

            new_data.append(new_item)
            counter[label] += 1
            crop_id += 1

    output_json.write_text(
        json.dumps(new_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("split:", split_name)
    print("原始样本数:", len(data))
    print("crop 样本数:", len(new_data))
    print("缺失 XML 数:", missing_xml)
    print("类别统计:", counter)
    print("输出 JSON:", output_json)
    print("输出图片目录:", out_img_dir)


def main():
    for split in ["train", "val"]:
        make_split(split)


if __name__ == "__main__":
    main()
