import sys
import re
import uuid
import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from agent.diagnosis_pipeline import run_diagnosis_pipeline


YOLO_MODEL = PROJECT_ROOT / "output/yolo_pcb_detect/yolo11n_pcb_1024_pretrained_noamp/weights/best.pt"
BASE_MODEL = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "output/qwen25vl_7b_pcb_crop_cls_full"

DEFECT_TYPES = ["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"]

CLASS_EN2ZH = {
    "short": "短路",
    "open_circuit": "开路",
    "mouse_bite": "鼠咬",
    "spur": "毛刺",
    "spurious_copper": "多余铜",
    "missing_hole": "漏孔",
}

PROMPT = (
    "请判断这张 PCB 缺陷局部图像的缺陷类型。"
    "只从以下六类中选择一个：短路、开路、鼠咬、毛刺、多余铜、漏孔。"
    "只输出“缺陷类型：xxx”。"
)


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


@st.cache_resource
def load_yolo():
    return YOLO(str(YOLO_MODEL))


@st.cache_resource
def load_vlm():
    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL),
        trust_remote_code=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = model.to("cuda")

    model = PeftModel.from_pretrained(
        model,
        str(ADAPTER_PATH),
        device_map=None,
    )

    model = model.to("cuda")
    model.eval()

    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    except Exception:
        pass

    return model, processor


def vlm_classify_crop(crop_path: str):
    model, processor = load_vlm()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_path},
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
    ).to("cuda")

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
    )[0].strip()

    return output_text, extract_defect_type(output_text)


def decide_final_type(yolo_type: str, yolo_conf: float, vlm_type: str, prefer_yolo_conf: float):
    if vlm_type == yolo_type:
        return yolo_type, True, "YOLO 与 VLM 分类一致，可信度较高。"

    if vlm_type == "未知":
        return yolo_type, False, "VLM 未能解析类别，采用 YOLO 检测类别，建议人工复核。"

    if yolo_conf >= prefer_yolo_conf:
        return yolo_type, False, "YOLO 置信度较高但与 VLM 不一致，优先采用 YOLO 类别，并建议人工复核。"

    return vlm_type, False, "YOLO 置信度较低且与 VLM 不一致，暂采用 VLM 类别，并建议人工复核。"


def list_demo_images():
    demo_root = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images"
    if not demo_root.exists():
        return []

    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(demo_root.rglob(ext))

    return sorted(files)


def run_yolo_detection(image_path: Path, out_dir: Path, conf: float, imgsz: int, max_det: int):
    yolo = load_yolo()

    results = yolo.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        max_det=max_det,
        save=False,
        verbose=False,
        device=0,
    )

    r = results[0]
    names = r.names

    annotated = r.plot()
    annotated_rgb = annotated[..., ::-1]
    annotated_path = out_dir / "yolo_detect_annotated.jpg"
    Image.fromarray(annotated_rgb).save(annotated_path)

    detections = []

    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].detach().cpu().tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            score = float(box.conf[0].detach().cpu().item())

            cls_en = names[cls_id]
            cls_zh = CLASS_EN2ZH.get(cls_en, cls_en)

            detections.append(
                {
                    "idx": i + 1,
                    "xyxy": xyxy,
                    "cls_id": cls_id,
                    "cls_en": cls_en,
                    "yolo_type": cls_zh,
                    "yolo_conf": score,
                }
            )

    return detections, annotated_path


def save_uploaded_image(uploaded_file):
    tmp_dir = PROJECT_ROOT / "outputs" / "streamlit_uploaded"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".bmp"]:
        suffix = ".jpg"

    save_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    image = Image.open(uploaded_file).convert("RGB")
    image.save(save_path)

    return save_path


def build_rag_for_type(defect_type: str, confidence: float, crop_path: str, top_k: int):
    vlm_result = {
        "defect_type": defect_type,
        "defect_type_zh": defect_type,
        "confidence": confidence,
        "location": f"局部裁剪图：{crop_path}",
        "crop_path": crop_path,
    }

    rag_result = run_diagnosis_pipeline(vlm_result, top_k=top_k)

    if isinstance(rag_result, dict) and "report" in rag_result:
        return rag_result["report"]

    return str(rag_result)


st.set_page_config(
    page_title="PCB 多模态故障诊断系统",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 PCB 多模态故障诊断系统")
st.caption("YOLO11n 缺陷定位 + Qwen2.5-VL crop-level LoRA 复核 + Elasticsearch/bge-m3 RAG 维修知识库")

with st.sidebar:
    st.header("系统配置")

    conf = st.slider(
        "YOLO 检测置信度阈值",
        min_value=0.05,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    prefer_yolo_conf = st.slider(
        "YOLO 高置信优先阈值",
        min_value=0.50,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="当 YOLO 与 VLM 不一致时，若 YOLO 置信度高于该值，则最终类别优先采用 YOLO，但会标记人工复核。",
    )

    imgsz = st.select_slider(
        "YOLO 输入尺寸",
        options=[640, 768, 1024, 1280],
        value=1024,
    )

    max_det = st.slider("最大检测框数量", min_value=1, max_value=30, value=20)
    top_k = st.slider("RAG 检索 Top-K", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown(f"**YOLO 模型：** `{YOLO_MODEL.name}`")
    st.markdown("**VLM：** Qwen2.5-VL-7B + crop LoRA")
    st.markdown("**类别：** 短路、开路、鼠咬、毛刺、多余铜、漏孔")

tab_upload, tab_demo = st.tabs(["上传整张 PCB 图", "使用验证集整图样例"])

image_path = None
image_for_display = None

with tab_upload:
    uploaded = st.file_uploader(
        "上传整张 PCB 图像",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is not None:
        image_path = save_uploaded_image(uploaded)
        image_for_display = Image.open(image_path).convert("RGB")

with tab_demo:
    demo_files = list_demo_images()

    if not demo_files:
        st.info("未找到 PCB_DATASET 整图样例。")
    else:
        selected = st.selectbox(
            "选择一张 PCB_DATASET 整图样例",
            demo_files,
            format_func=lambda p: str(p.relative_to(PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images")),
        )

        if selected is not None:
            image_path = selected
            image_for_display = Image.open(selected).convert("RGB")

if image_path and image_for_display:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("输入整图")
        st.image(image_for_display, caption=str(image_path), use_container_width=True)

    with col_right:
        st.subheader("诊断操作")
        run_button = st.button("开始整图诊断", type="primary")

    if run_button:
        run_id = f"{Path(image_path).stem}_{uuid.uuid4().hex[:8]}"
        out_dir = PROJECT_ROOT / "output" / "streamlit_full_image" / run_id
        crop_dir = out_dir / "crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("正在运行 YOLO 检测..."):
            detections, annotated_path = run_yolo_detection(
                image_path=Path(image_path),
                out_dir=out_dir,
                conf=conf,
                imgsz=imgsz,
                max_det=max_det,
            )

        st.subheader("YOLO 检测结果")
        st.image(Image.open(annotated_path), caption="YOLO 检测可视化", use_container_width=True)

        if not detections:
            st.warning("YOLO 未检测到缺陷区域。可以尝试降低置信度阈值，例如 0.30 或 0.20。")
            st.stop()

        st.success(f"检测到 {len(detections)} 个疑似缺陷区域。")

        original_img = Image.open(image_path).convert("RGB")

        with st.spinner("正在裁剪缺陷区域并调用 VLM 复核..."):
            for det in detections:
                crop_img, crop_xyxy = crop_with_margin(original_img, det["xyxy"])
                crop_path = crop_dir / f"crop_{det['idx']:02d}_yolo_{det['cls_en']}_{det['yolo_conf']:.3f}.jpg"
                crop_img.save(crop_path)

                vlm_output, vlm_type = vlm_classify_crop(str(crop_path))

                final_type, agree, decision_note = decide_final_type(
                    yolo_type=det["yolo_type"],
                    yolo_conf=det["yolo_conf"],
                    vlm_type=vlm_type,
                    prefer_yolo_conf=prefer_yolo_conf,
                )

                det["crop_path"] = crop_path
                det["crop_xyxy"] = crop_xyxy
                det["vlm_output"] = vlm_output
                det["vlm_type"] = vlm_type
                det["final_type"] = final_type
                det["agree"] = agree
                det["decision_note"] = decision_note

        st.subheader("局部缺陷复核结果")

        for det in detections:
            with st.expander(
                f"缺陷区域 {det['idx']}｜最终类别：{det['final_type']}｜YOLO置信度：{det['yolo_conf']:.3f}",
                expanded=True,
            ):
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.image(Image.open(det["crop_path"]), caption=Path(det["crop_path"]).name, use_container_width=True)

                with c2:
                    st.markdown(f"- **YOLO 检测类别：** {det['yolo_type']}")
                    st.markdown(f"- **YOLO 置信度：** {det['yolo_conf']:.3f}")
                    st.markdown(f"- **YOLO 原始框：** `{[round(x, 1) for x in det['xyxy']]}`")
                    st.markdown(f"- **VLM 输出：** `{det['vlm_output']}`")
                    st.markdown(f"- **VLM 分类：** {det['vlm_type']}")
                    st.markdown(f"- **最终类别：** {det['final_type']}")

                    if det["agree"]:
                        st.success(det["decision_note"])
                    else:
                        st.warning(det["decision_note"])

        st.subheader("RAG 维修知识诊断")

        final_types = {}
        for det in detections:
            t = det["final_type"]
            if t not in final_types or det["yolo_conf"] > final_types[t]["yolo_conf"]:
                final_types[t] = det

        for defect_type, det in final_types.items():
            with st.expander(f"{defect_type}：维修建议与风险分析", expanded=True):
                with st.spinner(f"正在检索 {defect_type} 的维修知识..."):
                    try:
                        rag_report = build_rag_for_type(
                            defect_type=defect_type,
                            confidence=det["yolo_conf"],
                            crop_path=str(det["crop_path"]),
                            top_k=top_k,
                        )
                        st.markdown(rag_report)
                    except Exception as e:
                        st.error(f"RAG 检索失败：{repr(e)}")
                        st.info("请确认 Elasticsearch 是否已启动：curl http://localhost:9200")

        st.markdown("---")
        st.info(f"本次结果已保存到：`{out_dir}`")

else:
    st.info("请上传一张整张 PCB 图像，或在样例页选择一张图片。")
