from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


class PCBKnowledgeSearcher:
    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "pcb_fault_knowledge",
        model_name: str = "/extra/caochunhong/gm/pcb_multi_agent/models/bge-m3",
    ):
        self.es_url = es_url
        self.index_name = index_name
        self.model_name = model_name

        self.es = Elasticsearch(
            es_url,
            request_timeout=60,
            retry_on_timeout=True,
            max_retries=3,
        )

        info = self.es.info()
        print(f"Elasticsearch 已连接: {info['version']['number']}")

        print(f"加载向量模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("向量模型加载完成")

    def search(
        self,
        query: str,
        defect_type: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        PCB 知识库检索工具。

        Args:
            query: 用户问题或模型诊断文本
            defect_type: 可选缺陷类型，如 短路、开路、漏孔、毛刺、多余铜、鼠咬
            top_k: 返回条数

        Returns:
            检索结果列表
        """
        qvec = self.model.encode(query, normalize_embeddings=True).tolist()

        base_query: Dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": [
                    "defect_type_zh^4",
                    "title^3",
                    "content^2",
                    "text_for_embedding",
                ],
            }
        }

        if defect_type:
            es_query = {
                "bool": {
                    "must": [base_query],
                    "filter": [
                        {
                            "term": {
                                "defect_type_zh": defect_type
                            }
                        }
                    ],
                }
            }
        else:
            es_query = base_query

        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": es_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": qvec
                        },
                    },
                }
            },
        }

        response = self.es.search(index=self.index_name, body=body)

        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                {
                    "score": hit["_score"],
                    "doc_id": src.get("doc_id"),
                    "defect_type": src.get("defect_type_zh"),
                    "title": src.get("title"),
                    "severity": src.get("severity"),
                    "visual_features": src.get("visual_features"),
                    "possible_causes": src.get("possible_causes"),
                    "electrical_symptoms": src.get("electrical_symptoms"),
                    "detection_methods": src.get("detection_methods"),
                    "repair_suggestions": src.get("repair_suggestions"),
                    "risk": src.get("risk"),
                    "prevention": src.get("prevention"),
                    "content": src.get("content"),
                    "tags": src.get("tags", []),
                }
            )

        return results


_searcher: Optional[PCBKnowledgeSearcher] = None


def get_searcher() -> PCBKnowledgeSearcher:
    global _searcher
    if _searcher is None:
        _searcher = PCBKnowledgeSearcher()
    return _searcher


def pcb_knowledge_search(
    query: str,
    defect_type: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    searcher = get_searcher()
    return searcher.search(query=query, defect_type=defect_type, top_k=top_k)