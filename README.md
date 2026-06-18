# PCB-MultiAgent: 面向工业 PCB 缺陷检测的多模态诊断与维修知识推理系统

本项目构建了一个面向 PCB 质检与维修场景的多模态多 Agent 故障诊断系统。系统支持输入整张 PCB 图像，自动完成缺陷定位、局部缺陷复核、维修知识检索、融合决策和结构化诊断报告生成。

项目覆盖六类典型 PCB 缺陷：

- Short / 短路
- Open Circuit / 开路
- Mouse Bite / 鼠咬
- Spur / 毛刺
- Spurious Copper / 多余铜
- Missing Hole / 漏孔

## 1. 项目亮点

- 构建 YOLO11n + Qwen2.5-VL LoRA 两阶段视觉诊断方案，解决 PCB 小缺陷在整图中容易被背景淹没的问题。
- 使用 LLaMA-Factory 对 Qwen2.5-VL-7B-Instruct 进行 crop-level LoRA 微调，实现 PCB 局部缺陷分类。
- 基于 Elasticsearch + bge-m3 构建 PCB 维修知识库，实现缺陷成因、风险等级、检测方法和维修建议的语义检索。
- 封装 MCP 工具服务 `pcb_knowledge_search`，使知识检索能力可以被 Agent 通过标准工具接口调用。
- 使用 LangGraph 构建多 Agent 工作流，实现 DetectionAgent、VisionAgent、DecisionAgent、RAGAgent、ReportAgent 的端到端编排。
- 设计 YOLO/VLM 一致性判断机制，当检测模型和视觉语言模型结果冲突时自动标记人工复核。
- 使用 Streamlit 实现可视化诊断界面，支持整图上传、检测框展示、局部 crop 展示、RAG 维修建议展示和 Markdown 报告下载。

![PCB-MultiAgent Demo](./docs/assets/demo.png)

## 2. 系统架构

整体流程如下：

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
```

系统由五个核心 Agent 组成：

| Agent          | 功能                                                         |
| -------------- | ------------------------------------------------------------ |
| DetectionAgent | 调用 YOLO11n 对整张 PCB 图像进行缺陷定位                     |
| VisionAgent    | 根据 YOLO 检测框自动裁剪局部 crop，并调用 Qwen2.5-VL LoRA 复核缺陷类型 |
| DecisionAgent  | 融合 YOLO 与 VLM 输出，判断最终缺陷类别，并标记是否需要人工复核 |
| RAGAgent       | 通过 MCP 工具服务 `pcb_knowledge_search` 检索 PCB 维修知识库 |
| ReportAgent    | 汇总检测结果、复核结果、知识库内容，生成结构化诊断报告       |

## 3. 技术栈

| 模块           | 技术                   |
| -------------- | ---------------------- |
| 目标检测       | YOLO11n, Ultralytics   |
| 多模态视觉模型 | Qwen2.5-VL-7B-Instruct |
| 微调方法       | LoRA, LLaMA-Factory    |
| 向量检索       | bge-m3                 |
| 知识库检索     | Elasticsearch          |
| 工具服务       | MCP                    |
| Agent 编排     | LangGraph              |
| 前端展示       | Streamlit              |
| 深度学习框架   | PyTorch                |
| 主要语言       | Python                 |

## 4. 数据处理

项目使用 PCB_DATASET 数据集，包含六类 PCB 缺陷图像和 XML 标注文件。

原始 XML 标注被转换为两类训练数据：

### 4.1 YOLO 检测数据

将 XML 中的缺陷框转换为 YOLO 格式：

```text
class_id x_center y_center width height
```

用于训练 YOLO11n 缺陷定位模型。

### 4.2 Qwen2.5-VL crop-level 多模态数据

根据 XML 标注框裁剪局部缺陷图像，构造 Qwen2.5-VL 多模态指令数据：

```text
输入：PCB 局部缺陷 crop 图像
指令：请判断该 PCB 局部缺陷图像的缺陷类型
输出：缺陷类型：xxx
```

这种设计避免了整图背景信息过多导致视觉语言模型无法聚焦小缺陷的问题。

## 5. 模型训练与评测结果

### 5.1 YOLO11n 缺陷定位模型

YOLO11n 在 PCB_DATASET 验证集上的检测结果：

| Metric    | Value |
| --------- | ----- |
| Precision | 0.953 |
| Recall    | 0.927 |
| mAP50     | 0.968 |
| mAP50-95  | 0.534 |

### 5.2 Qwen2.5-VL crop 分类模型

使用 LLaMA-Factory 对 Qwen2.5-VL-7B-Instruct 进行 LoRA 微调。

| Metric           | Value  |
| ---------------- | ------ |
| Validation crops | 280    |
| Accuracy         | 90.71% |

各类缺陷均能在局部 crop 输入下获得较稳定的分类效果，明显优于直接使用整图输入的方式。

### 5.3 LangGraph + MCP 端到端系统评测

在 YOLO 验证集 69 张 PCB 图像上进行端到端评测：

| Metric                      | Value  |
| --------------------------- | ------ |
| Total samples               | 69     |
| Flow success rate           | 100%   |
| YOLO detection rate         | 100%   |
| Image-level defect accuracy | 100%   |
| RAG hit rate                | 100%   |
| Human review trigger rate   | 30.43% |
| Bad cases                   | 0      |

说明系统能够稳定完成：

```text
整图输入 → 缺陷定位 → 局部复核 → 融合决策 → 知识检索 → 报告生成
```

的完整诊断流程。

## 6. 项目实现中的关键问题与解决方案

### 6.1 问题一：整图直接输入 Qwen2.5-VL 效果不稳定

早期尝试直接将整张 PCB 图像输入 Qwen2.5-VL 进行缺陷分类，但由于 PCB 缺陷区域通常只占整图很小比例，大量线路背景会干扰模型判断，导致模型容易出现类别塌缩，例如多张图都输出同一类缺陷。

#### 解决方案

将视觉诊断流程改为两阶段结构：

```text
YOLO11n 整图定位缺陷区域
↓
根据检测框裁剪局部 crop
↓
Qwen2.5-VL LoRA 对局部 crop 进行复核分类
```

这样模型输入从“整张复杂 PCB 图像”变成“局部缺陷区域”，显著降低背景干扰，提升分类稳定性。

------

### 6.2 问题二：YOLO 低置信检测框容易产生误检

在较低置信度阈值下，YOLO 会检出一些低置信误检框。例如在 Mouse_bite 或 Spurious_copper 图像中，低置信框可能被误判为其他类别。

#### 解决方案

将默认检测阈值调整为：

```text
conf = 0.50
```

并在系统中提供置信度滑块，方便根据场景调整。同时引入 DecisionAgent，对 YOLO 与 VLM 结果进行一致性判断，避免单模型误判直接影响最终结果。

------

### 6.3 问题三：VLM 与 YOLO 有时会出现分类冲突

在部分样本中，YOLO 能正确定位并给出较高置信类别，但 Qwen2.5-VL crop 复核结果可能与 YOLO 不一致。例如鼠咬样本中曾出现：

```text
YOLO: 鼠咬
VLM: 毛刺
```

#### 解决方案

设计 DecisionAgent 融合策略：

```text
YOLO 与 VLM 一致 → 直接采用该类别，可信度较高
VLM 未解析 → 采用 YOLO 类别，标记人工复核
YOLO 与 VLM 不一致且 YOLO 置信度较高 → 采用 YOLO 类别，标记人工复核
YOLO 与 VLM 不一致且 YOLO 置信度较低 → 参考 VLM 类别，标记人工复核
```

该机制使系统不会盲目相信单一模型，而是将冲突样本显式标记为“需人工复核”，更符合工业检测场景的可靠性要求。

------

### 6.4 问题四：MCP stdio 模式下普通 print 输出污染协议

在封装 MCP 工具服务时，RAG 模块中的普通输出，例如：

```text
Elasticsearch 已连接
加载向量模型
向量模型加载完成
```

会被打印到 stdout。MCP stdio transport 要求 stdout 只能传输 JSON-RPC 消息，因此 client 会报 JSON 解析错误。

#### 解决方案

在 MCP Server 中使用：

```python
contextlib.redirect_stdout(sys.stderr)
```

将普通日志输出重定向到 stderr，避免污染 MCP 的 JSON-RPC 通信。

------

### 6.5 问题五：Streamlit tab 同时执行导致上传图被样例图覆盖

Streamlit 中多个 tab 的代码会同时执行。早期版本中，用户上传图片后，demo tab 的默认 selectbox 仍然会执行，并覆盖 `image_path`，导致系统实际诊断的是样例图而不是上传图。

#### 解决方案

显式设置图像来源优先级：

```text
上传图像优先
如果没有上传图像，才使用 demo 样例
```

并在页面上显示当前诊断图像来源和路径，避免误判。

------

### 6.6 问题六：服务器无法联网，模型和项目上传受限

服务器无法直接访问 GitHub 或下载模型权重，导致：

- YOLO 权重无法直接从服务器下载
- GitHub 无法从服务器直接 push
- 部分自动下载检查会卡住

#### 解决方案

- YOLO 权重在本地下载后上传到服务器。
- 训练 YOLO 时关闭 AMP 自动检查，避免联网下载额外权重：

```bash
amp=False
```

- GitHub 发布时不直接从服务器 push，而是在服务器创建干净发布目录并打包，再通过本地电脑上传 GitHub。
- 仓库中不包含模型权重、原始数据集、训练输出和 Elasticsearch 运行目录，避免超出 GitHub 文件限制。

## 7. MCP 工具服务

本项目封装了 MCP 工具：

```text
pcb_knowledge_search
```

输入：

```json
{
  "query": "漏孔缺陷会导致什么风险，如何维修？",
  "defect_type": "漏孔",
  "top_k": 3
}
```

输出包括：

- 缺陷类型
- 文档编号
- 维修知识标题
- 风险等级
- 视觉特征
- 可能成因
- 检测方法
- 维修建议
- 预防措施

测试命令：

```bash
python mcp_server/test_mcp_client.py
```

## 8. LangGraph 多 Agent 流程

运行单张图像诊断：

```bash
python scripts/test_langgraph_pipeline.py \
  --image path/to/pcb_image.jpg \
  --conf 0.50 \
  --imgsz 1024 \
  --rag_backend mcp
```

输出包括：

- YOLO 检测可视化图
- crop 图像
- VLM 复核结果
- DecisionAgent 最终决策
- MCP/RAG 维修建议
- Markdown 诊断报告

## 9. Streamlit 可视化 Demo

启动 LangGraph + MCP 版本前端：

```bash
streamlit run app_langgraph.py \
  --server.address 0.0.0.0 \
  --server.port 8502
```

功能包括：

- 上传整张 PCB 图像
- 显示 YOLO 检测框
- 显示局部 crop 图像
- 显示 YOLO 类别和置信度
- 显示 Qwen2.5-VL 复核结果
- 显示是否需要人工复核
- 显示 RAG 维修建议
- 下载 Markdown 诊断报告

## 10. 项目目录结构

```text
.
├── app.py
├── app_langgraph.py
├── agent
│   ├── diagnosis_pipeline.py
│   ├── pcb_graph_state.py
│   └── pcb_langgraph.py
├── rag
│   └── knowledge_search.py
├── mcp_server
│   ├── pcb_knowledge_server.py
│   └── test_mcp_client.py
├── scripts
│   ├── convert_pcb_xml_to_yolo.py
│   ├── make_pcb_crop_cls_dataset.py
│   ├── eval_langgraph_end_to_end.py
│   ├── eval_langgraph_yolo_val.py
│   └── test_langgraph_pipeline.py
├── configs
├── data
│   └── knowledge_base
│       └── processed
│           └── pcb_fault_knowledge.jsonl
├── docs
│   └── evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

## 11. 注意事项

本仓库不包含以下大文件：

- Qwen2.5-VL 模型权重
- bge-m3 模型权重
- YOLO 训练权重
- PCB 原始数据集
- Elasticsearch 运行目录
- 训练输出目录
- Streamlit 临时输出目录

请根据自己的环境准备模型和数据，并修改代码中的路径配置。

关键路径示例：

```text
models/Qwen2.5-VL-7B-Instruct
models/bge-m3
output/yolo_pcb_detect/.../best.pt
output/qwen25vl_7b_pcb_crop_cls_full
```

## 12. 简历描述参考

```text
构建基于 YOLO11n + Qwen2.5-VL LoRA + LangGraph + MCP 的 PCB 多模态故障诊断系统，实现整图缺陷定位、局部视觉复核、维修知识检索和结构化报告生成。YOLO11n 验证集 mAP50 达到 0.968，Qwen2.5-VL crop 分类准确率达到 90.71%；在 69 张 YOLO 验证集图像上，端到端流程成功率、缺陷检出率、图像级诊断准确率和 RAG 命中率均达到 100%。
```

## 13. License

This project is for research and learning purposes only.
