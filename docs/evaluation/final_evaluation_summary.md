# PCB LangGraph + MCP 多 Agent 系统最终评测总结

## 1. 系统组成

系统采用 LangGraph 构建多 Agent 编排流程，包括 DetectionAgent、VisionAgent、DecisionAgent、RAGAgent 和 ReportAgent。DetectionAgent 调用 YOLO11n 对整张 PCB 图像进行缺陷定位；VisionAgent 根据检测框裁剪局部图像，并调用 Qwen2.5-VL LoRA 进行缺陷类型复核；DecisionAgent 融合 YOLO 与 VLM 结果，并在二者不一致时触发人工复核标记；RAGAgent 通过 MCP 工具服务 pcb_knowledge_search 检索 PCB 维修知识库；ReportAgent 汇总生成结构化诊断报告。

## 2. 60 张随机端到端流程测试结果

| 指标 | 结果 |
|---|---:|
| Total samples | 60 |
| Flow success rate | 1.0000 |
| YOLO detection rate | 1.0000 |
| Image-level defect accuracy | 1.0000 |
| Human review trigger rate | 0.3500 |
| RAG hit rate | 1.0000 |

## 3. YOLO 验证集 69 张端到端评测结果

| 指标 | 结果 |
|---|---:|
| Total samples | 69 |
| Flow success rate | 1.0000 |
| YOLO detection rate | 1.0000 |
| Image-level defect accuracy | 1.0000 |
| Human review trigger rate | 0.3043 |
| RAG hit rate | 1.0000 |

## 4. YOLO 验证集分类别结果

| 缺陷类型 | 样本数 | 图像级准确率 | YOLO检出率 | RAG命中率 | 人工复核触发率 |
|---|---:|---:|---:|---:|---:|
| 短路 | 15 | 1.0000 | 1.0000 | 1.0000 | 0.6000 |
| 开路 | 9 | 1.0000 | 1.0000 | 1.0000 | 0.3333 |
| 鼠咬 | 13 | 1.0000 | 1.0000 | 1.0000 | 0.2308 |
| 毛刺 | 12 | 1.0000 | 1.0000 | 1.0000 | 0.3333 |
| 多余铜 | 9 | 1.0000 | 1.0000 | 1.0000 | 0.2222 |
| 漏孔 | 11 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## 5. 混淆矩阵统计

| 真实类别 | 预测类别 | 数量 |
|---|---|---:|
| 多余铜 | 多余铜 | 9 |
| 开路 | 开路 | 9 |
| 毛刺 | 毛刺 | 12 |
| 漏孔 | 漏孔 | 11 |
| 短路 | 短路 | 15 |
| 鼠咬 | 鼠咬 | 13 |

## 6. 失败案例统计

- Bad cases: 0
- 本次 YOLO 验证集端到端评测中未出现图像级错误样本。

## 7. 报告推荐表述

在 YOLO 验证集 69 张 PCB 图像上，LangGraph 多 Agent 系统完成了从整图输入、YOLO 缺陷定位、Qwen2.5-VL 局部复核、MCP 知识检索到报告生成的完整流程。系统流程成功率为 100%，YOLO 缺陷检出率为 100%，图像级缺陷诊断准确率为 100%，RAG 命中率为 100%，人工复核触发率为 30.43%。实验结果表明，该系统能够稳定完成PCB 缺陷端到端自动诊断，并能在 YOLO 与 VLM 结果不一致时触发人工复核机制。