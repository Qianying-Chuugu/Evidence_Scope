"""
BM25、Dense 和 Hybrid 检索效果评测。

使用方法：
1. 在 data/retrieval_eval.json 中维护问题、参考答案和正确 Chunk ID。
2. 在 evidence_scope 目录运行：python -m retrieval.evaluate_retrieval
3. 如需调整参数，修改文件底部 run_evaluation() 的 top_k、candidate_k 和 rrf_k。

输出指标：
- Hit@K：前 K 条是否至少命中一个正确 Chunk。
- MRR：所有问题第一个正确结果倒数排名的平均值。
- Recall@K：前 K 条命中的正确 Chunk 占全部正确 Chunk 的比例。
"""

import json
from pathlib import Path

from .hybrid_search import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_DENSE_WEIGHT,
    HybridRetriever,
    reciprocal_rank_fusion,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "retrieval_eval.json"


def load_evaluation_cases():
    """读取评测集，并在加载模型前检查必要字段。"""
    with open(EVAL_PATH, "r", encoding="utf-8") as file:
        evaluation_cases = json.load(file)

    if not isinstance(evaluation_cases, list) or not evaluation_cases:
        raise ValueError("评测文件最外层必须是非空列表")

    for case_number, case in enumerate(evaluation_cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {case_number} 条评测数据必须是对象")

        query = case.get("query")
        relevant_chunk_ids = case.get("relevant_chunk_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"第 {case_number} 条评测数据的 query 不能为空")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(
                f"第 {case_number} 条评测数据的 relevant_chunk_ids 必须是非空列表"
            )

    return evaluation_cases


def calculate_query_metrics(retrieval_results, relevant_chunk_ids, top_k):
    """计算单个问题的 Hit@K、RR 和 Recall@K。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids 不能为空")

    top_results = retrieval_results[:top_k]
    relevant_id_set = set(relevant_chunk_ids)
    first_relevant_rank = None
    retrieved_relevant_ids = []

    for rank, result in enumerate(top_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id in relevant_id_set:
            retrieved_relevant_ids.append(chunk_id)
            if first_relevant_rank is None:
                first_relevant_rank = rank

    hit_at_k = int(first_relevant_rank is not None)
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1 / first_relevant_rank
    recall_at_k = len(retrieved_relevant_ids) / len(relevant_id_set)

    return {
        "hit_at_k": hit_at_k,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": reciprocal_rank,
        "recall_at_k": recall_at_k,
        "retrieved_relevant_ids": retrieved_relevant_ids,
    }


def run_evaluation(
    top_k=5,
    candidate_k=20,
    rrf_k=60,
    bm25_weight=DEFAULT_BM25_WEIGHT,
    dense_weight=DEFAULT_DENSE_WEIGHT,
):
    """运行三个检索器的逐题评测，并输出平均指标。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if candidate_k < top_k:
        raise ValueError("candidate_k 不能小于 top_k")

    evaluation_cases = load_evaluation_cases()
    retriever = HybridRetriever()
    summary = {
        "BM25": {"hit_sum": 0, "rr_sum": 0.0, "recall_sum": 0.0},
        "Dense": {"hit_sum": 0, "rr_sum": 0.0, "recall_sum": 0.0},
        "Hybrid": {"hit_sum": 0, "rr_sum": 0.0, "recall_sum": 0.0},
    }

    for case_number, case in enumerate(evaluation_cases, start=1):
        query = case["query"].strip()
        relevant_chunk_ids = case["relevant_chunk_ids"]

        bm25_candidates = retriever.bm25.search(query=query, top_k=candidate_k)
        dense_candidates = retriever.dense.search(query=query, top_k=candidate_k)
        hybrid_candidates = reciprocal_rank_fusion(
            bm25_results=bm25_candidates,
            dense_results=dense_candidates,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

        results_by_retriever = {
            "BM25": bm25_candidates[:top_k],
            "Dense": dense_candidates[:top_k],
            "Hybrid": hybrid_candidates[:top_k],
        }

        print("\n" + "#" * 70)
        print("评测问题编号：", case_number)
        print("查询：", query)
        print("参考答案：", case.get("expected_answer", "未提供"))
        print("正确 Chunk：", relevant_chunk_ids)

        for retriever_name, retrieval_results in results_by_retriever.items():
            metrics = calculate_query_metrics(
                retrieval_results=retrieval_results,
                relevant_chunk_ids=relevant_chunk_ids,
                top_k=top_k,
            )
            retrieved_chunk_ids = [
                result["chunk_id"] for result in retrieval_results
            ]

            print("\n检索器：", retriever_name)
            print(f"Top-{top_k} Chunk：", retrieved_chunk_ids)
            print(f"Hit@{top_k}：", metrics["hit_at_k"])
            print("第一个正确结果排名：", metrics["first_relevant_rank"])
            print("RR：", round(metrics["reciprocal_rank"], 4))
            print(f"Recall@{top_k}：", round(metrics["recall_at_k"], 4))
            print("命中的正确 Chunk：", metrics["retrieved_relevant_ids"])

            summary[retriever_name]["hit_sum"] += metrics["hit_at_k"]
            summary[retriever_name]["rr_sum"] += metrics["reciprocal_rank"]
            summary[retriever_name]["recall_sum"] += metrics["recall_at_k"]

    question_count = len(evaluation_cases)
    print("\n" + "=" * 70)
    print("总体评测结果")
    print("问题数量：", question_count)
    print("融合权重：", f"BM25={bm25_weight}, Dense={dense_weight}")

    for retriever_name, totals in summary.items():
        hit_rate = totals["hit_sum"] / question_count
        mrr = totals["rr_sum"] / question_count
        average_recall = totals["recall_sum"] / question_count

        print("\n检索器：", retriever_name)
        print(f"Hit Rate@{top_k}：", round(hit_rate, 4))
        print("MRR：", round(mrr, 4))
        print(f"平均 Recall@{top_k}：", round(average_recall, 4))


if __name__ == "__main__":
    run_evaluation(top_k=5, candidate_k=20, rrf_k=60)
