# PCB-MultiAgent: 面向工业 PCB 缺陷检测的多模态诊断与维修知识推理系统

本项目构建了一个面向 PCB 质检与维修场景的多模态多 Agent 诊断系统，支持整张 PCB 图像输入，自动完成缺陷定位、局部视觉复核、维修知识检索与结构化诊断报告生成。

## Features

- YOLO11n 整图 PCB 缺陷定位
- Qwen2.5-VL-7B-Instruct LoRA 局部 crop 缺陷复核
- Elasticsearch + bge-m3 PCB 维修知识库 RAG
- MCP 工具服务 `pcb_knowledge_search`
- LangGraph 多 Agent 编排
- Streamlit 可视化诊断界面
- YOLO/VLM 一致性判断与人工复核机制

## Supported Defect Types

- Short / 短路
- Open Circuit / 开路
- Mouse Bite / 鼠咬
- Spur / 毛刺
- Spurious Copper / 多余铜
- Missing Hole / 漏孔

## System Architecture

```text
PCB image
  ↓
DetectionAgent: YOLO11n defect detection
  ↓
VisionAgent: crop generation + Qwen2.5-VL LoRA classification
  ↓
DecisionAgent: YOLO/VLM consistency check and final decision
  ↓
RAGAgent: MCP tool call + PCB maintenance knowledge retrieval
  ↓
ReportAgent: structured diagnosis report generation
Tech Stack

Python, PyTorch, YOLO11n, Qwen2.5-VL, LoRA, LLaMA-Factory, Elasticsearch, bge-m3, MCP, LangGraph, Streamlit

Experimental Results
YOLO Detection
MetricValue
Precision0.953
Recall0.927
mAP500.968
mAP50-950.534
Qwen2.5-VL Crop Classification
MetricValue
Validation crops280
Accuracy90.71%
End-to-End LangGraph + MCP Evaluation

On 69 images from the YOLO validation split:

MetricValue
Flow success rate100%
YOLO detection rate100%
Image-level defect accuracy100%
RAG hit rate100%
Human review trigger rate30.43%
Bad cases0
Run Streamlit Demo
streamlit run app_langgraph.py --server.address 0.0.0.0 --server.port 8502
Run MCP Tool Test
python mcp_server/test_mcp_client.py
Run LangGraph Pipeline
python scripts/test_langgraph_pipeline.py \
  --image path/to/pcb_image.jpg \
  --conf 0.50 \
  --imgsz 1024 \
  --rag_backend mcp
Notes

Large files such as model weights, raw datasets, training outputs and Elasticsearch runtime files are not included in this repository. Please prepare the required models and datasets according to your local environment.
