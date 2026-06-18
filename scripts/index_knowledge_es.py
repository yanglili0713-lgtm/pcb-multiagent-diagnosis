import json
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

KB_PATH = ROOT / "data" / "knowledge_base" / "processed" / "pcb_fault_knowledge.jsonl"
INDEX_NAME = "pcb_fault_knowledge"
MODEL_NAME = "/extra/caochunhong/gm/pcb_multi_agent/models/bge-m3"
ES_URL = "http://localhost:9200"


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"第 {line_no} 行 JSON 解析失败: {e}")
    return records


def build_text(record):
    tags = record.get("tags", [])
    if isinstance(tags, list):
        tags = "、".join(tags)

    return "\n".join([
        f"缺陷类型：{record.get('defect_type_zh', '')}",
        f"英文类型：{record.get('defect_type_en', '')}",
        f"标题：{record.get('title', '')}",
        f"视觉特征：{record.get('visual_features', '')}",
        f"可能原因：{record.get('possible_causes', '')}",
        f"电气症状：{record.get('electrical_symptoms', '')}",
        f"检测方法：{record.get('detection_methods', '')}",
        f"维修建议：{record.get('repair_suggestions', '')}",
        f"风险等级：{record.get('severity', '')}",
        f"风险：{record.get('risk', '')}",
        f"预防措施：{record.get('prevention', '')}",
        f"标签：{tags}",
        f"正文：{record.get('content', '')}",
    ])


def main():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"找不到知识库文件: {KB_PATH}")

    records = load_jsonl(KB_PATH)
    print(f"知识条目数量: {len(records)}")

    es = Elasticsearch(
        ES_URL,
        request_timeout=120,
        retry_on_timeout=True,
         max_retries=3
    )

    try:
        info = es.info()
        print(f"Elasticsearch 已连接: {info['version']['number']}")
    except Exception as e:
        raise RuntimeError(f"Elasticsearch 连接失败: {e}")

    print(f"加载向量模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    sample_vec = model.encode("PCB短路缺陷检测与维修", normalize_embeddings=True)
    dim = len(sample_vec)
    print(f"向量维度: {dim}")

    if es.indices.exists(index=INDEX_NAME):
        print(f"删除旧索引: {INDEX_NAME}")
        es.indices.delete(index=INDEX_NAME)

    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "defect_type_en": {"type": "keyword"},
                "defect_type_zh": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "tags": {"type": "keyword"},
                "severity": {"type": "keyword"},
                "text_for_embedding": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    print(f"创建索引: {INDEX_NAME}")
    es.indices.create(index=INDEX_NAME, body=mapping)

    actions = []
    for record in tqdm(records, desc="向量化并写入 ES"):
        text = build_text(record)
        emb = model.encode(text, normalize_embeddings=True).tolist()

        doc = {
            **record,
            "text_for_embedding": text,
            "embedding": emb
        }

        actions.append({
            "_index": INDEX_NAME,
            "_id": record["doc_id"],
            "_source": doc
        })

    helpers.bulk(es, actions)
    es.indices.refresh(index=INDEX_NAME)

    count = es.count(index=INDEX_NAME)["count"]
    print(f"索引完成，当前文档数: {count}")


if __name__ == "__main__":
    main()