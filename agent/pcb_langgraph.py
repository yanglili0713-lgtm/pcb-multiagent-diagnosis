import sys
import re
import json
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from ultralytics import YOLO
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from langgraph.graph import StateGraph, START, END

PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
sys.path.insert(0, str(PROJECT_ROOT))

from agent.pcb_graph_state import PCBGraphState
from agent.diagnosis_pipeline import run_diagnosis_pipeline


YOLO_MODEL = PROJECT_ROOT / "output/yolo_pcb_detect/yolo11n_pcb_1024_pretrained_noamp/weights/best.pt"
BASE_MODEL = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "output/qwen25vl_7b_pcb_crop_cls_full"
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server" / "pcb_knowledge_server.py"

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

_yolo_model = None
_vlm_model = None
_vlm_processor = None


def _append_msg(state: PCBGraphState, msg: str) -> List[str]:
    return list(state.get("messages", [])) + [msg]


def _append_error(state: PCBGraphState, err: str) -> List[str]:
    return list(state.get("errors", [])) + [err]


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_MODEL))
    return _yolo_model


def get_vlm():
    global _vlm_model, _vlm_processor

    if _vlm_model is not None and _vlm_processor is not None:
        return _vlm_model, _vlm_processor

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

    _vlm_model = model
    _vlm_processor = processor

    return _vlm_model, _vlm_processor


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


def vlm_classify_crop(crop_path: str) -> Tuple[str, str]:
    model, processor = get_vlm()

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


async def _call_mcp_search_async(query: str, defect_type: str, top_k: int) -> Dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "pcb_knowledge_search",
                arguments={
                    "query": query,
                    "defect_type": defect_type,
                    "top_k": top_k,
                },
            )

            text_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                else:
                    text_parts.append(str(content))

            text = "\n".join(text_parts).strip()
            return json.loads(text)


def call_mcp_search(query: str, defect_type: str, top_k: int) -> Dict[str, Any]:
    return asyncio.run(_call_mcp_search_async(query, defect_type, top_k))


def format_mcp_rag_report(defect_type: str, confidence: float, mcp_result: Dict[str, Any]) -> str:
    results = mcp_result.get("results", [])

    lines = []
    lines.append("# PCB 多模态故障诊断报告\n")
    lines.append("## 1. 图像诊断结果")
    lines.append(f"- 疑似缺陷类型：{defect_type}")
    lines.append(f"- 诊断置信度：{confidence:.3f}")
    lines.append("- 图像观察说明：图像中存在疑似缺陷区域。")
    lines.append("\n注意：该结果由视觉模型和检测模型自动生成，建议结合 AOI 检测框、局部放大图和人工复核进一步确认。\n")

    lines.append("## 2. MCP 知识库检索结果")
    if not results:
        lines.append("未检索到相关知识条目。")
        return "\n".join(lines)

    for i, item in enumerate(results, 1):
        lines.append(f"### 知识条目 {i}")
        lines.append(f"- 文档编号：{item.get('doc_id')}")
        lines.append(f"- 缺陷类型：{item.get('defect_type')}")
        lines.append(f"- 标题：{item.get('title')}")
        lines.append(f"- 风险等级：{item.get('severity')}")
        lines.append(f"- 检测方法：{item.get('detection_methods')}")
        lines.append(f"- 维修建议：{item.get('repair_suggestions')}")
        lines.append("")

    top = results[0]
    lines.append("## 3. 综合风险分析")
    lines.append(f"该缺陷风险等级为：{top.get('severity')}。主要风险：{top.get('risk')}。")

    lines.append("\n## 4. 建议处理措施")
    lines.append(f"优先参考知识条目：{top.get('title')}")
    lines.append(f"建议措施：{top.get('repair_suggestions')}")
    lines.append("\n进一步建议：")
    lines.append("- 使用显微镜或 AOI 局部放大确认缺陷位置。")
    lines.append("- 使用万用表、绝缘电阻测试或通断测试验证电气影响。")
    lines.append("- 维修后重新进行外观检查和电气测试。")

    return "\n".join(lines)


# -------------------------
# LangGraph Nodes
# -------------------------

def detection_agent(state: PCBGraphState) -> PCBGraphState:
    image_path = Path(state["image_path"])
    conf = float(state.get("conf", 0.50))
    imgsz = int(state.get("imgsz", 1024))
    max_det = int(state.get("max_det", 20))

    output_dir = Path(state.get("output_dir") or PROJECT_ROOT / "output" / "langgraph_runs" / f"{image_path.stem}_{uuid.uuid4().hex[:8]}")
    crop_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        err = f"输入图像不存在：{image_path}"
        return {
            "output_dir": str(output_dir),
            "crop_dir": str(crop_dir),
            "detections": [],
            "messages": _append_msg(state, "DetectionAgent failed."),
            "errors": _append_error(state, err),
        }

    yolo = get_yolo()

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
    annotated_path = output_dir / "yolo_detect_annotated.jpg"
    Image.fromarray(annotated_rgb).save(annotated_path)

    detections = []
    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].detach().cpu().tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            score = float(box.conf[0].detach().cpu().item())

            cls_en = names[cls_id]
            cls_zh = CLASS_EN2ZH.get(cls_en, cls_en)

            detections.append({
                "idx": i + 1,
                "xyxy": xyxy,
                "cls_id": cls_id,
                "cls_en": cls_en,
                "yolo_type": cls_zh,
                "yolo_conf": score,
            })

    return {
        "output_dir": str(output_dir),
        "crop_dir": str(crop_dir),
        "annotated_image_path": str(annotated_path),
        "detections": detections,
        "messages": _append_msg(state, f"DetectionAgent: detected {len(detections)} region(s)."),
    }


def vision_agent(state: PCBGraphState) -> PCBGraphState:
    detections = state.get("detections", [])
    image_path = Path(state["image_path"])
    crop_dir = Path(state["crop_dir"])

    if not detections:
        return {
            "crops": [],
            "vlm_results": [],
            "messages": _append_msg(state, "VisionAgent: no detections, skipped."),
        }

    original_img = Image.open(image_path).convert("RGB")

    crops = []
    vlm_results = []

    for det in detections:
        crop_img, crop_xyxy = crop_with_margin(original_img, det["xyxy"])
        crop_path = crop_dir / f"crop_{det['idx']:02d}_yolo_{det['cls_en']}_{det['yolo_conf']:.3f}.jpg"
        crop_img.save(crop_path)

        vlm_output, vlm_type = vlm_classify_crop(str(crop_path))

        crops.append({
            "idx": det["idx"],
            "crop_path": str(crop_path),
            "crop_xyxy": crop_xyxy,
        })

        vlm_results.append({
            "idx": det["idx"],
            "vlm_output": vlm_output,
            "vlm_type": vlm_type,
        })

    return {
        "crops": crops,
        "vlm_results": vlm_results,
        "messages": _append_msg(state, f"VisionAgent: classified {len(vlm_results)} crop(s)."),
    }


def decision_agent(state: PCBGraphState) -> PCBGraphState:
    detections = state.get("detections", [])
    crops = {x["idx"]: x for x in state.get("crops", [])}
    vlm_results = {x["idx"]: x for x in state.get("vlm_results", [])}
    prefer_yolo_conf = float(state.get("prefer_yolo_conf", 0.65))

    decisions = []

    for det in detections:
        idx = det["idx"]
        vlm = vlm_results.get(idx, {"vlm_output": "", "vlm_type": "未知"})
        crop = crops.get(idx, {"crop_path": "", "crop_xyxy": []})

        final_type, agree, decision_note = decide_final_type(
            yolo_type=det["yolo_type"],
            yolo_conf=det["yolo_conf"],
            vlm_type=vlm["vlm_type"],
            prefer_yolo_conf=prefer_yolo_conf,
        )

        decisions.append({
            **det,
            **crop,
            **vlm,
            "final_type": final_type,
            "agree": agree,
            "need_human_review": not agree,
            "decision_note": decision_note,
        })

    return {
        "decisions": decisions,
        "messages": _append_msg(state, f"DecisionAgent: made {len(decisions)} decision(s)."),
    }


def rag_agent(state: PCBGraphState) -> PCBGraphState:
    decisions = state.get("decisions", [])
    top_k = int(state.get("top_k", 3))
    rag_backend = state.get("rag_backend", "local")

    if not decisions:
        return {
            "rag_reports": {},
            "messages": _append_msg(state, "RAGAgent: no decisions, skipped."),
        }

    # 每个最终类别只检索一次，选该类别中 YOLO 置信度最高的区域作为代表。
    representatives: Dict[str, Dict[str, Any]] = {}
    for d in decisions:
        t = d["final_type"]
        if t not in representatives or d["yolo_conf"] > representatives[t]["yolo_conf"]:
            representatives[t] = d

    rag_reports = {}

    for defect_type, det in representatives.items():
        try:
            if rag_backend == "mcp":
                query = f"PCB图像疑似存在{defect_type}缺陷，请给出检测方法、风险分析和维修建议。"
                mcp_result = call_mcp_search(query=query, defect_type=defect_type, top_k=top_k)
                report = format_mcp_rag_report(defect_type, det["yolo_conf"], mcp_result)
            else:
                vlm_result = {
                    "defect_type": defect_type,
                    "defect_type_zh": defect_type,
                    "confidence": det["yolo_conf"],
                    "location": f"YOLO 检测框 xyxy={det.get('xyxy')}",
                    "crop_path": det.get("crop_path", ""),
                }
                rag_result = run_diagnosis_pipeline(vlm_result, top_k=top_k)
                if isinstance(rag_result, dict) and "report" in rag_result:
                    report = rag_result["report"]
                else:
                    report = str(rag_result)

            rag_reports[defect_type] = report

        except Exception as e:
            rag_reports[defect_type] = f"RAG 检索失败：{repr(e)}"

    return {
        "rag_reports": rag_reports,
        "messages": _append_msg(state, f"RAGAgent: generated {len(rag_reports)} RAG report(s) via {rag_backend}."),
    }


def report_agent(state: PCBGraphState) -> PCBGraphState:
    image_path = state.get("image_path", "")
    output_dir = Path(state.get("output_dir", PROJECT_ROOT / "output" / "langgraph_runs" / "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = state.get("detections", [])
    decisions = state.get("decisions", [])
    rag_reports = state.get("rag_reports", {})
    errors = state.get("errors", [])
    messages = state.get("messages", [])

    lines = []
    lines.append("# PCB LangGraph 多 Agent 故障诊断报告\n")
    lines.append("## 0. Graph 执行轨迹")
    for msg in messages:
        lines.append(f"- {msg}")

    if errors:
        lines.append("\n## 错误信息")
        for e in errors:
            lines.append(f"- {e}")

    lines.append("\n## 1. 输入与检测概况")
    lines.append(f"- 输入图像：`{image_path}`")
    lines.append(f"- YOLO 检测可视化图：`{state.get('annotated_image_path', '')}`")
    lines.append(f"- 检测区域数量：**{len(detections)}**")

    if not detections:
        lines.append("\nYOLO 未检测到缺陷区域。建议降低检测置信度阈值后复测。")
    else:
        lines.append("\n## 2. 多 Agent 局部诊断结果")
        for d in decisions:
            lines.append(f"\n### 缺陷区域 {d['idx']}")
            lines.append(f"- YOLO 检测类别：**{d['yolo_type']}**")
            lines.append(f"- YOLO 置信度：**{d['yolo_conf']:.3f}**")
            lines.append(f"- YOLO 原始框 xyxy：`{[round(x, 1) for x in d['xyxy']]}`")
            lines.append(f"- 自动裁剪图：`{d.get('crop_path', '')}`")
            lines.append(f"- VLM 输出：`{d.get('vlm_output', '')}`")
            lines.append(f"- VLM 分类：**{d.get('vlm_type', '未知')}**")
            lines.append(f"- 最终类别：**{d['final_type']}**")
            lines.append(f"- 是否需要人工复核：**{'是' if d['need_human_review'] else '否'}**")
            lines.append(f"- 决策说明：{d['decision_note']}")

    lines.append("\n## 3. RAG 维修知识诊断")
    if not rag_reports:
        lines.append("未生成 RAG 维修知识。")
    else:
        for defect_type, report in rag_reports.items():
            lines.append(f"\n### {defect_type}：维修建议与风险分析\n")
            lines.append(report)

    report = "\n".join(lines)
    report_path = output_dir / "langgraph_diagnosis_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "report": report,
        "report_path": str(report_path),
        "messages": _append_msg(state, f"ReportAgent: saved report to {report_path}."),
    }


def build_pcb_graph():
    builder = StateGraph(PCBGraphState)

    builder.add_node("DetectionAgent", detection_agent)
    builder.add_node("VisionAgent", vision_agent)
    builder.add_node("DecisionAgent", decision_agent)
    builder.add_node("RAGAgent", rag_agent)
    builder.add_node("ReportAgent", report_agent)

    builder.add_edge(START, "DetectionAgent")
    builder.add_edge("DetectionAgent", "VisionAgent")
    builder.add_edge("VisionAgent", "DecisionAgent")
    builder.add_edge("DecisionAgent", "RAGAgent")
    builder.add_edge("RAGAgent", "ReportAgent")
    builder.add_edge("ReportAgent", END)

    return builder.compile()


def run_pcb_langgraph(
    image_path: str,
    conf: float = 0.50,
    imgsz: int = 1024,
    max_det: int = 20,
    top_k: int = 3,
    prefer_yolo_conf: float = 0.65,
    rag_backend: str = "local",
    output_dir: str = "",
) -> PCBGraphState:
    graph = build_pcb_graph()

    init_state: PCBGraphState = {
        "image_path": image_path,
        "conf": conf,
        "imgsz": imgsz,
        "max_det": max_det,
        "top_k": top_k,
        "prefer_yolo_conf": prefer_yolo_conf,
        "rag_backend": rag_backend,
        "messages": [],
        "errors": [],
    }

    if output_dir:
        init_state["output_dir"] = output_dir
        init_state["crop_dir"] = str(Path(output_dir) / "crops")

    return graph.invoke(init_state)
