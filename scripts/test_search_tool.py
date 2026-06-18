import sys
from pathlib import Path
# 将项目根目录加入Python检索路径
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


from rag.knowledge_search import pcb_knowledge_search


def print_results(query, defect_type=None):
    print("=" * 100)
    print("Query:", query)
    print("Defect filter:", defect_type)
    print("=" * 100)

    results = pcb_knowledge_search(
        query=query,
        defect_type=defect_type,
        top_k=3,
    )

    for i, item in enumerate(results, 1):
        print(f"\nTop {i}")
        print("score:", round(item["score"], 4))
        print("doc_id:", item["doc_id"])
        print("defect_type:", item["defect_type"])
        print("title:", item["title"])
        print("severity:", item["severity"])
        print("repair:", item["repair_suggestions"])
        print("content:", item["content"][:200])


if __name__ == "__main__":
    print_results("PCB图像显示漏孔，应该怎么检测和维修？")
    print_results("模型判断为毛刺，靠近相邻线路，应该如何处理？", defect_type="毛刺")
    print_results("电源和地之间阻值很低，怀疑短路，如何维修？", defect_type="短路")