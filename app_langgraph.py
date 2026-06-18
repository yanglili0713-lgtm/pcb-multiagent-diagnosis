import uuid
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from agent.pcb_langgraph import run_pcb_langgraph


PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")

DEMO_ROOT = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images"


def save_uploaded_image(uploaded_file):
    tmp_dir = PROJECT_ROOT / "outputs" / "streamlit_langgraph_uploaded"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".bmp"]:
        suffix = ".jpg"

    save_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    image = Image.open(uploaded_file).convert("RGB")
    image.save(save_path)
    return save_path


def list_demo_images():
    if not DEMO_ROOT.exists():
        return []

    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(DEMO_ROOT.rglob(ext))
    return sorted(files)


st.set_page_config(
    page_title="PCB LangGraph 多 Agent 故障诊断系统",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 PCB LangGraph 多 Agent 故障诊断系统")
st.caption(
    "DetectionAgent + VisionAgent + DecisionAgent + RAGAgent(MCP) + ReportAgent"
)

with st.sidebar:
    st.header("多 Agent 配置")

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
    )

    imgsz = st.select_slider(
        "YOLO 输入尺寸",
        options=[640, 768, 1024, 1280],
        value=1024,
    )

    max_det = st.slider("最大检测框数量", min_value=1, max_value=30, value=20)
    top_k = st.slider("RAG 检索 Top-K", min_value=1, max_value=5, value=3)

    rag_backend = st.radio(
        "RAG 调用方式",
        options=["mcp", "local"],
        index=0,
        help="mcp 表示 RAGAgent 通过 MCP 工具服务 pcb_knowledge_search 检索知识库；local 表示直接调用本地 Python RAG。",
    )

    st.markdown("---")
    st.markdown("**Agent 节点：**")
    st.markdown("- DetectionAgent：YOLO 缺陷定位")
    st.markdown("- VisionAgent：crop + Qwen2.5-VL 复核")
    st.markdown("- DecisionAgent：融合判断")
    st.markdown("- RAGAgent：MCP/RAG 知识检索")
    st.markdown("- ReportAgent：报告生成")


tab_upload, tab_demo = st.tabs(["上传整张 PCB 图", "使用 PCB_DATASET 样例"])

uploaded_image_path = None
demo_image_path = None

with tab_upload:
    uploaded = st.file_uploader(
        "上传整张 PCB 图像",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is not None:
        uploaded_image_path = save_uploaded_image(uploaded)

with tab_demo:
    demo_files = list_demo_images()

    if not demo_files:
        st.info("未找到 PCB_DATASET 整图样例。")
    else:
        selected = st.selectbox(
            "选择一张整图样例",
            demo_files,
            format_func=lambda p: str(p.relative_to(DEMO_ROOT)),
        )

        if selected is not None:
            demo_image_path = selected


# 关键修复：
# Streamlit 的两个 tab 会同时执行。
# 因此必须显式规定优先级：上传图优先；没有上传图时才使用 demo 样例。
image_path = None
image_for_display = None
image_source = None

if uploaded_image_path is not None:
    image_path = uploaded_image_path
    image_source = "用户上传图像"
elif demo_image_path is not None:
    image_path = demo_image_path
    image_source = "PCB_DATASET 样例"

if image_path is not None:
    image_for_display = Image.open(image_path).convert("RGB")
    st.info(f"当前诊断图像来源：**{image_source}**，路径：`{image_path}`")


if image_path and image_for_display:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("输入整图")
        st.image(image_for_display, caption=str(image_path), use_container_width=True)

    with col_right:
        st.subheader("执行多 Agent 诊断")
        st.markdown("点击后将依次执行：")
        st.markdown("`DetectionAgent → VisionAgent → DecisionAgent → RAGAgent → ReportAgent`")
        run_button = st.button("开始 LangGraph 多 Agent 诊断", type="primary")

    if run_button:
        run_id = f"{Path(image_path).stem}_{uuid.uuid4().hex[:8]}"
        out_dir = PROJECT_ROOT / "output" / "streamlit_langgraph" / run_id

        with st.spinner("正在执行 LangGraph 多 Agent 流程..."):
            state = run_pcb_langgraph(
                image_path=str(image_path),
                conf=conf,
                imgsz=imgsz,
                max_det=max_det,
                top_k=top_k,
                prefer_yolo_conf=prefer_yolo_conf,
                rag_backend=rag_backend,
                output_dir=str(out_dir),
            )

        st.success("LangGraph 多 Agent 诊断完成。")

        st.subheader("Graph 执行轨迹")
        for msg in state.get("messages", []):
            st.markdown(f"- {msg}")

        if state.get("errors"):
            st.error("流程中出现错误：")
            for e in state["errors"]:
                st.markdown(f"- `{e}`")

        annotated_path = state.get("annotated_image_path")
        if annotated_path and Path(annotated_path).exists():
            st.subheader("DetectionAgent：YOLO 检测结果")
            st.image(Image.open(annotated_path), caption=annotated_path, use_container_width=True)

        decisions = state.get("decisions", [])

        st.subheader("VisionAgent + DecisionAgent：局部复核与融合决策")

        if not decisions:
            st.warning("未检测到缺陷区域。可以尝试降低 YOLO 检测置信度阈值。")
        else:
            for d in decisions:
                title = (
                    f"缺陷区域 {d['idx']}｜最终类别：{d['final_type']}｜"
                    f"YOLO：{d['yolo_type']}({d['yolo_conf']:.3f})｜"
                    f"VLM：{d.get('vlm_type', '未知')}"
                )

                with st.expander(title, expanded=True):
                    c1, c2 = st.columns([1, 2])

                    with c1:
                        crop_path = d.get("crop_path", "")
                        if crop_path and Path(crop_path).exists():
                            st.image(Image.open(crop_path), caption=Path(crop_path).name, use_container_width=True)
                        else:
                            st.info("未找到 crop 图像。")

                    with c2:
                        st.markdown(f"- **YOLO 检测类别：** {d['yolo_type']}")
                        st.markdown(f"- **YOLO 置信度：** {d['yolo_conf']:.3f}")
                        st.markdown(f"- **YOLO 原始框：** `{[round(x, 1) for x in d['xyxy']]}`")
                        st.markdown(f"- **VLM 输出：** `{d.get('vlm_output', '')}`")
                        st.markdown(f"- **VLM 分类：** {d.get('vlm_type', '未知')}")
                        st.markdown(f"- **最终类别：** {d['final_type']}")
                        st.markdown(f"- **是否需要人工复核：** {'是' if d.get('need_human_review') else '否'}")

                        if d.get("need_human_review"):
                            st.warning(d.get("decision_note", "建议人工复核。"))
                        else:
                            st.success(d.get("decision_note", "分类一致，可信度较高。"))

        st.subheader("RAGAgent：维修知识检索结果")

        rag_reports = state.get("rag_reports", {})
        if not rag_reports:
            st.info("未生成 RAG 维修知识。")
        else:
            for defect_type, report in rag_reports.items():
                with st.expander(f"{defect_type}：维修建议与风险分析", expanded=True):
                    st.markdown(report)

        st.subheader("ReportAgent：最终诊断报告")

        report_path = state.get("report_path")
        if report_path and Path(report_path).exists():
            st.markdown(f"报告路径：`{report_path}`")

            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()

            st.download_button(
                label="下载 Markdown 诊断报告",
                data=report_text,
                file_name="pcb_langgraph_diagnosis_report.md",
                mime="text/markdown",
            )

            with st.expander("查看完整 Markdown 报告", expanded=False):
                st.markdown(report_text)
        else:
            st.info("未找到最终报告文件。")

        st.info(f"本次运行目录：`{out_dir}`")

else:
    st.info("请上传一张 PCB 整图，或选择一张 PCB_DATASET 样例。")
