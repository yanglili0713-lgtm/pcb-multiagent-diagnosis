from agent.diagnosis_pipeline import run_diagnosis_pipeline


if __name__ == "__main__":
    fake_vlm_result = {
        "defect_type": "短路",
        "confidence": "中",
        "description": "图像中疑似存在相邻铜箔或焊点异常连接，可能导致电源与地之间阻值偏低。",
    }

    result = run_diagnosis_pipeline(fake_vlm_result, top_k=3)

    print("=" * 100)
    print("检索 query:")
    print(result["retrieval_query"])

    print("=" * 100)
    print("最终诊断报告:")
    print(result["report"])