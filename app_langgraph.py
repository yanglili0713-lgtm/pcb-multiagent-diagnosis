import os
import sys
import uuid
from pathlib import Path

# ============================================================
# GPU 固定配置
# ============================================================
# 物理 GPU 2 在当前进程中映射为 cuda:0。
# 如果你不想固定 GPU，可删除 CUDA_VISIBLE_DEVICES 这一行，并在外部启动命令里控制。
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PCB_DEVICE"] = "cuda:0"
os.environ["PCB_VLM_DEVICE"] = "cuda:0"
os.environ["PCB_YOLO_DEVICE"] = "0"
os.environ.setdefault("PCB_MCP_DEVICE", "auto")
os.environ.setdefault("PCB_MCP_EXCLUDE_GPUS", "0,1")
os.environ.setdefault("PCB_MCP_MIN_FREE_MIB", "2048")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from agent.pcb_langgraph import run_pcb_langgraph


DEMO_ROOT = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images"
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "streamlit_langgraph_uploaded"


def init_session_state():
    st.session_state.setdefault("upload_clear_counter", 0)
    st.session_state.setdefault("uploaded_image_path", None)
    st.session_state.setdefault("uploaded_file_signature", None)
    st.session_state.setdefault("demo_image_path", None)
    st.session_state.setdefault("active_source", None)  # upload / demo / None


def reset_uploaded_image():
    st.session_state.uploaded_image_path = None
    st.session_state.uploaded_file_signature = None
    if st.session_state.active_source == "upload":
        st.session_state.active_source = None
    st.session_state.upload_clear_counter += 1


def reset_demo_image():
    st.session_state.demo_image_path = None
    if st.session_state.active_source == "demo":
        st.session_state.active_source = None


def reset_all_images():
    st.session_state.uploaded_image_path = None
    st.session_state.uploaded_file_signature = None
    st.session_state.demo_image_path = None
    st.session_state.active_source = None
    st.session_state.upload_clear_counter += 1


def save_uploaded_image(uploaded_file):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".bmp"]:
        suffix = ".jpg"

    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
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


def get_current_image():
    active_source = st.session_state.get("active_source")

    if active_source == "upload":
        p = st.session_state.get("uploaded_image_path")
        if p and Path(p).exists():
            return Path(p), "用户上传图像"
        reset_uploaded_image()
        return None, None

    if active_source == "demo":
        p = st.session_state.get("demo_image_path")
        if p and Path(p).exists():
            return Path(p), "PCB_DATASET 样例"
        reset_demo_image()
        return None, None

    return None, None


def get_qwen_status(vd: dict, idx: int, target_indices: list):
    qwen_called = bool(vd.get("qwen_called", False))
    qwen_output = str(vd.get("qwen_output", "") or "")
    skip_reason = str(vd.get("qwen_skip_reason", "") or "")

    planned = idx in set(target_indices)

    failed_keywords = [
        "Qwen 调用失败",
        "OutOfMemory",
        "CUDA out of memory",
        "RuntimeError",
        "Traceback",
        "Invalid CUDA device index",
    ]

    failed = any(k in qwen_output for k in failed_keywords) or any(
        k in skip_reason for k in failed_keywords
    )

    if failed:
        return "调用失败", planned, qwen_output or skip_reason

    if qwen_called:
        return "调用成功", planned, ""

    return "未调用", planned, skip_reason


init_session_state()

st.set_page_config(
    page_title="PCB LangGraph 多 Agent 视觉诊断系统",
    page_icon="🧠",
    layout="wide",
)

st.title("PCB LangGraph 多 Agent 视觉诊断系统")
st.caption(
    "DetectionAgent + VisualDiagnosisAgent(Qwen) + DecisionAgent + RAGAgent(MCP) + ReportAgent"
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
        help=(
            "mcp 表示 RAGAgent 通过 MCP 工具服务 pcb_knowledge_search 检索知识库；"
            "local 表示直接调用本地 Python RAG。"
        ),
    )

    st.markdown("---")
    st.subheader("Qwen 视觉诊断调用策略")

    qwen_mode = st.radio(
        "Qwen 调用模式",
        options=["auto", "always", "never"],
        index=0,
        help=(
            "auto：低置信、高风险、代表性区域调用 Qwen；"
            "always：每个检测框都调用 Qwen；"
            "never：完全不调用 Qwen，只用 YOLO + RAG 模板。"
        ),
    )

    low_conf_threshold = st.slider(
        "低置信触发 Qwen 阈值",
        min_value=0.30,
        max_value=0.90,
        value=0.65,
        step=0.05,
    )

    representative_per_type = st.slider(
        "每类代表性区域调用数量",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
        help="一张图中同类缺陷较多时，只调用最高置信的若干个代表性区域，控制 Qwen 成本。",
    )

    high_risk_types = st.multiselect(
        "高风险缺陷类型",
        options=["短路", "开路", "鼠咬", "毛刺", "多余铜", "漏孔"],
        default=["短路", "开路", "漏孔"],
    )

    st.markdown("---")
    st.subheader("GPU 设置")
    st.markdown("- 主流程物理 GPU：`2`")
    st.markdown("- 主流程内部设备：`cuda:0`")
    st.markdown("- MCP RAG：`auto`，默认排除物理 GPU 0/1")
    st.markdown(f"- `CUDA_VISIBLE_DEVICES = {os.getenv('CUDA_VISIBLE_DEVICES')}`")
    st.markdown(f"- `PCB_DEVICE = {os.getenv('PCB_DEVICE')}`")
    st.markdown(f"- `PCB_MCP_DEVICE = {os.getenv('PCB_MCP_DEVICE')}`")
    st.markdown(f"- `PCB_MCP_EXCLUDE_GPUS = {os.getenv('PCB_MCP_EXCLUDE_GPUS')}`")

    st.markdown("---")
    st.markdown("**Agent 节点：**")
    st.markdown("- DetectionAgent：YOLO 缺陷定位")
    st.markdown("- VisualDiagnosisAgent：基于 YOLO 初判做 Qwen 视觉诊断")
    st.markdown("- DecisionAgent：融合 YOLO 置信度与 Qwen 诊断")
    st.markdown("- RAGAgent：MCP/RAG 维修知识检索")
    st.markdown("- ReportAgent：生成可交付诊断报告")


tab_upload, tab_demo = st.tabs(["上传整张 PCB 图", "使用 PCB_DATASET 样例"])

with tab_upload:
    st.subheader("上传整张 PCB 图像")

    upload_key = f"uploaded_file_{st.session_state.upload_clear_counter}"
    uploaded = st.file_uploader(
        "选择 JPG、PNG 或 BMP 图像",
        type=["jpg", "jpeg", "png", "bmp"],
        key=upload_key,
        label_visibility="collapsed",
    )

    # 原生 file_uploader 点 x 取消后会返回 None，这里同步清空当前显示。
    if uploaded is None and st.session_state.active_source == "upload":
        reset_uploaded_image()
        st.rerun()

    if uploaded is not None:
        signature = f"{uploaded.name}-{uploaded.size}-{uploaded.type}"
        if signature != st.session_state.uploaded_file_signature:
            saved_path = save_uploaded_image(uploaded)
            st.session_state.uploaded_image_path = str(saved_path)
            st.session_state.uploaded_file_signature = signature
            st.session_state.demo_image_path = None
            st.session_state.active_source = "upload"
            st.rerun()

    if st.session_state.active_source == "upload" and st.session_state.uploaded_image_path:
        st.markdown(f"当前已选择上传图像：`{st.session_state.uploaded_image_path}`")
        if st.button("清空上传图像", key="clear_upload_btn"):
            reset_uploaded_image()
            st.rerun()
    else:
        st.markdown("尚未上传图像。")

with tab_demo:
    st.subheader("使用 PCB_DATASET 样例")

    demo_files = list_demo_images()

    if not demo_files:
        st.markdown("未找到 PCB_DATASET 整图样例。")
    else:
        demo_options = [None] + demo_files

        selected_demo = st.selectbox(
            "选择一张整图样例",
            demo_options,
            index=0,
            format_func=lambda p: "请选择样例图像" if p is None else str(p.relative_to(DEMO_ROOT)),
        )

        c_load, c_clear = st.columns([1, 1])
        with c_load:
            load_demo = st.button(
                "加载这张样例",
                disabled=selected_demo is None,
                key="load_demo_btn",
            )
        with c_clear:
            clear_demo = st.button("清空样例选择", key="clear_demo_btn")

        if load_demo and selected_demo is not None:
            st.session_state.demo_image_path = str(selected_demo)
            st.session_state.uploaded_image_path = None
            st.session_state.uploaded_file_signature = None
            st.session_state.upload_clear_counter += 1
            st.session_state.active_source = "demo"
            st.rerun()

        if clear_demo:
            reset_demo_image()
            st.rerun()

        if st.session_state.active_source == "demo" and st.session_state.demo_image_path:
            st.markdown(f"当前已加载样例图像：`{st.session_state.demo_image_path}`")
        else:
            st.markdown("尚未加载样例图像。")


image_path, image_source = get_current_image()

if image_path is None:
    st.markdown("请先上传一张 PCB 整图，或在 PCB_DATASET 样例页中选择并加载一张样例。")
else:
    image_for_display = Image.open(image_path).convert("RGB")
    st.markdown(f"当前诊断图像来源：**{image_source}**")
    st.markdown(f"图像路径：`{image_path}`")

    if st.button("清空当前图像", key="clear_current_image"):
        reset_all_images()
        st.rerun()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("输入整图")
        st.image(
            image_for_display,
            caption=str(image_path),
            use_container_width=True,
        )

    with col_right:
        st.subheader("执行多 Agent 视觉诊断")
        st.markdown("点击后将依次执行：")
        st.markdown(
            "`DetectionAgent → VisualDiagnosisAgent → DecisionAgent → RAGAgent → ReportAgent`"
        )
        st.markdown(
            f"""
            **本次配置：**

            - Qwen 调用模式：`{qwen_mode}`
            - 低置信触发阈值：`{low_conf_threshold}`
            - 每类代表性区域数量：`{representative_per_type}`
            - 高风险缺陷：`{high_risk_types}`
            - 主流程物理 GPU：`2`
            - 主流程内部设备：`cuda:0`
            - MCP RAG：`auto`
            """
        )

        run_button = st.button(
            "开始 LangGraph 多 Agent 诊断",
            type="primary",
        )

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
                qwen_mode=qwen_mode,
                low_conf_threshold=low_conf_threshold,
                representative_per_type=representative_per_type,
                high_risk_types=high_risk_types,
            )

        st.success("LangGraph 多 Agent 诊断完成。")

        st.subheader("Graph 执行轨迹")
        for msg in state.get("messages", []):
            st.markdown(f"- {msg}")

        if state.get("errors"):
            st.error("流程中出现错误：")
            for e in state["errors"]:
                st.markdown(f"- `{e}`")

        qwen_stats = state.get("qwen_call_stats", {})
        if qwen_stats:
            st.subheader("Qwen 调用统计")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("检测框总数", qwen_stats.get("total", 0))
            c2.metric("Qwen 目标框数", qwen_stats.get("called", 0))
            c3.metric("跳过框数", qwen_stats.get("skipped", 0))
            c4.metric("调用模式", qwen_stats.get("qwen_mode", qwen_mode))

            st.markdown(f"- Qwen 目标区域 idx：`{qwen_stats.get('target_indices', [])}`")
            st.markdown(f"- 低置信触发阈值：`{qwen_stats.get('low_conf_threshold', low_conf_threshold)}`")
            st.markdown(f"- 每类代表性区域数量：`{qwen_stats.get('representative_per_type', representative_per_type)}`")

        annotated_path = state.get("annotated_image_path")
        if annotated_path and Path(annotated_path).exists():
            st.subheader("DetectionAgent：YOLO 检测结果")
            st.image(
                Image.open(annotated_path),
                caption=annotated_path,
                use_container_width=True,
            )

        decisions = state.get("decisions", [])

        st.subheader("VisualDiagnosisAgent + DecisionAgent：视觉诊断与融合决策")

        if not decisions:
            st.warning("未检测到缺陷区域。可以尝试降低 YOLO 检测置信度阈值。")
        else:
            target_indices = qwen_stats.get("target_indices", [])

            for d in decisions:
                vd = d.get("visual_diagnosis", {}) or {}

                idx = int(d.get("idx", 0))
                yolo_type = d.get("yolo_type", "未知")
                yolo_conf = float(d.get("yolo_conf", 0.0))
                final_type = d.get("final_type", "未知")

                qwen_status, qwen_planned, qwen_reason = get_qwen_status(
                    vd=vd,
                    idx=idx,
                    target_indices=target_indices,
                )

                title = (
                    f"缺陷区域 {idx}｜最终类别：{final_type}｜"
                    f"YOLO：{yolo_type}({yolo_conf:.3f})｜"
                    f"Qwen：{qwen_status}｜"
                    f"复核：{'是' if d.get('need_human_review') else '否'}"
                )

                with st.expander(title, expanded=True):
                    c1, c2 = st.columns([1, 2])

                    with c1:
                        crop_path = d.get("crop_path", "")
                        if crop_path and Path(crop_path).exists():
                            st.image(
                                Image.open(crop_path),
                                caption=Path(crop_path).name,
                                use_container_width=True,
                            )
                        else:
                            st.markdown("未找到 crop 图像。")

                    with c2:
                        st.markdown(f"- **YOLO 检测类别：** {yolo_type}")
                        st.markdown(f"- **YOLO 置信度：** {yolo_conf:.3f}")
                        st.markdown(
                            f"- **YOLO 原始框：** "
                            f"`{[round(float(x), 1) for x in d.get('xyxy', [])]}`"
                        )

                        st.markdown("---")
                        st.markdown(f"- **是否计划调用 Qwen：** {'是' if qwen_planned else '否'}")
                        st.markdown(f"- **Qwen 诊断状态：** {qwen_status}")

                        if qwen_status == "调用失败":
                            st.error(qwen_reason or "Qwen 调用失败。")
                        elif qwen_status == "未调用":
                            st.markdown(qwen_reason or "当前检测框未满足 Qwen 调用策略。")
                        else:
                            st.success("Qwen 已完成视觉诊断。")

                        st.markdown("---")
                        st.markdown(f"- **缺陷确认状态：** {vd.get('defect_confirmed', 'uncertain')}")
                        st.markdown(f"- **视觉诊断类型：** {vd.get('final_type', '未知')}")
                        st.markdown(f"- **细分形态：** {vd.get('subtype', '未知')}")
                        st.markdown(f"- **视觉证据：** {vd.get('visual_evidence', '')}")
                        st.markdown(f"- **可返修性：** {vd.get('repairability', '未知')}")
                        st.markdown(f"- **直接维修建议：** {vd.get('direct_repair_suggestion', '')}")
                        st.markdown(f"- **风险等级：** {vd.get('risk_level', '未知')}")

                        st.markdown("---")
                        st.markdown(f"- **最终融合类别：** {final_type}")
                        st.markdown(f"- **决策置信度：** {d.get('decision_confidence', '未知')}")
                        st.markdown(f"- **类型冲突：** {'是' if d.get('type_conflict') else '否'}")
                        st.markdown(f"- **是否需要人工复核：** {'是' if d.get('need_human_review') else '否'}")

                        if d.get("need_human_review"):
                            st.warning(
                                d.get("review_reason")
                                or d.get("decision_note", "建议人工复核。")
                            )
                        else:
                            st.success(d.get("decision_note", "融合决策可信度较高。"))

                        st.markdown(f"- **决策说明：** {d.get('decision_note', '')}")

                        qwen_output = vd.get("qwen_output", "")
                        if qwen_output:
                            with st.expander("查看 Qwen 原始输出", expanded=False):
                                st.code(qwen_output, language="text")

        st.subheader("RAGAgent：维修知识检索结果")

        rag_reports = state.get("rag_reports", {})
        if not rag_reports:
            st.markdown("未生成 RAG 维修知识。")
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
                file_name="pcb_langgraph_visual_diagnosis_report.md",
                mime="text/markdown",
            )

            with st.expander("查看完整 Markdown 报告", expanded=False):
                st.markdown(report_text)
        else:
            st.markdown("未找到最终报告文件。")

        st.markdown(f"本次运行目录：`{out_dir}`")
