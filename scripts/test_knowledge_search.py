from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX_NAME = "pcb_fault_knowledge"
MODEL_NAME = "/extra/caochunhong/gm/pcb_multi_agent/models/bge-m3"


def search(query, top_k=3):
    es = Elasticsearch(ES_URL)
    model = SentenceTransformer(MODEL_NAME)

    qvec = model.encode(query, normalize_embeddings=True).tolist()

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "title^3",
                            "content^2",
                            "text_for_embedding"
                        ]
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": qvec
                    }
                }
            }
        }
    }

    res = es.search(index=INDEX_NAME, body=body)

    print("=" * 100)
    print("查询:", query)
    print("=" * 100)

    for i, hit in enumerate(res["hits"]["hits"], 1):
        src = hit["_source"]
        print(f"\nTop {i} | score={hit['_score']:.4f}")
        print("doc_id:", src.get("doc_id"))
        print("缺陷类型:", src.get("defect_type_zh"))
        print("标题:", src.get("title"))
        print("风险等级:", src.get("severity"))
        print("内容:", src.get("content")[:300])


if __name__ == "__main__":
    tests = [
        "PCB图像显示漏孔，应该怎么检测和维修？",
        "短路缺陷导致电源和地阻值很低，怎么处理？",
        "开路断线可以怎么修复？",
        "鼠咬缺陷会有什么风险？",
        "多余铜可能造成什么问题？",
        "毛刺靠近相邻线路，应该怎么处理？"
    ]

    for q in tests:
        search(q, top_k=3)