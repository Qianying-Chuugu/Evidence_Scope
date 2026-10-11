"""
BM25、Dense、Hybrid 和 Hybrid + Reranker 检索效果评测。

使用方法：
1. 开发时可修改 data/retrieval_eval_dev.json，冻结测试集
   data/retrieval_eval_test.json 只用于最后检查泛化效果。
2. 在 evidence_scope 目录运行开发集：
   python -m retrieval.evaluate_retrieval --dataset dev
3. 运行冻结测试集：python -m retrieval.evaluate_retrieval --dataset test
4. 两组依次运行：python -m retrieval.evaluate_retrieval --dataset all

输出指标：
- Hit@K：前 K 条是否至少命中一个正确 Chunk。
- MRR：所有问题第一个正确结果倒数排名的平均值。
- Recall@K：前 K 条命中的正确 Chunk 占全部正确 Chunk 的比例。
- 平均查询耗时：不包含模型初始化，每道题从查询到得到结果的平均时间。
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from .hybrid_search import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_DENSE_WEIGHT,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from .reranker import Reranker


ROOT = Path(__file__).resolve().parents[1]
DEV_EVAL_PATH = ROOT / "data" / "retrieval_eval_dev.json"
TEST_EVAL_PATH = ROOT / "data" / "retrieval_eval_test.json"


def load_evaluation_cases(evaluation_path=DEV_EVAL_PATH):
    """读取评测集，并在加载模型前检查必要字段。"""
    with open(evaluation_path, "r", encoding="utf-8") as file:
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
    evaluation_path=DEV_EVAL_PATH,
    top_k=5,
    candidate_k=20,
    rrf_k=60,
    bm25_weight=DEFAULT_BM25_WEIGHT,
    dense_weight=DEFAULT_DENSE_WEIGHT,
    retriever=None,
    reranker=None,
):
    """运行四种检索流程的逐题评测，并输出平均指标。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if candidate_k < top_k:
        raise ValueError("candidate_k 不能小于 top_k")

    evaluation_cases = load_evaluation_cases(evaluation_path)
    retriever = retriever or HybridRetriever()
    reranker = reranker or Reranker()
    summary = {
        "BM25": {
            "hit_sum": 0,
            "rr_sum": 0.0,
            "recall_sum": 0.0,
            "latency_sum": 0.0,
        },
        "Dense": {
            "hit_sum": 0,
            "rr_sum": 0.0,
            "recall_sum": 0.0,
            "latency_sum": 0.0,
        },
        "Hybrid": {
            "hit_sum": 0,
            "rr_sum": 0.0,
            "recall_sum": 0.0,
            "latency_sum": 0.0,
        },
        "Hybrid + Reranker": {
            "hit_sum": 0,
            "rr_sum": 0.0,
            "recall_sum": 0.0,
            "latency_sum": 0.0,
        },
    }

    for case_number, case in enumerate(evaluation_cases, start=1):
        query = case["query"].strip()
        relevant_chunk_ids = case["relevant_chunk_ids"]

        start_time = perf_counter()
        bm25_candidates = retriever.bm25.search(query=query, top_k=candidate_k)
        bm25_latency = perf_counter() - start_time

        start_time = perf_counter()
        dense_candidates = retriever.dense.search(query=query, top_k=candidate_k)
        dense_latency = perf_counter() - start_time

        start_time = perf_counter()
        hybrid_candidates = reciprocal_rank_fusion(
            bm25_results=bm25_candidates,
            dense_results=dense_candidates,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        fusion_latency = perf_counter() - start_time
        hybrid_latency = bm25_latency + dense_latency + fusion_latency

        start_time = perf_counter()
        reranked_candidates = reranker.rerank(
            query=query,
            candidates=hybrid_candidates[:candidate_k],
            top_k=top_k,
        )
        reranker_latency = perf_counter() - start_time
        hybrid_reranker_latency = hybrid_latency + reranker_latency

        results_by_retriever = {
            "BM25": bm25_candidates[:top_k],
            "Dense": dense_candidates[:top_k],
            "Hybrid": hybrid_candidates[:top_k],
            "Hybrid + Reranker": reranked_candidates,
        }
        latency_by_retriever = {
            "BM25": bm25_latency,
            "Dense": dense_latency,
            "Hybrid": hybrid_latency,
            "Hybrid + Reranker": hybrid_reranker_latency,
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
            print(
                "查询耗时（毫秒）：",
                round(latency_by_retriever[retriever_name] * 1000, 2),
            )

            summary[retriever_name]["hit_sum"] += metrics["hit_at_k"]
            summary[retriever_name]["rr_sum"] += metrics["reciprocal_rank"]
            summary[retriever_name]["recall_sum"] += metrics["recall_at_k"]
            summary[retriever_name]["latency_sum"] += latency_by_retriever[
                retriever_name
            ]

    question_count = len(evaluation_cases)
    print("\n" + "=" * 70)
    print("总体评测结果")
    print("评测集：", evaluation_path.name)
    print("问题数量：", question_count)
    print("融合权重：", f"BM25={bm25_weight}, Dense={dense_weight}")

    for retriever_name, totals in summary.items():
        hit_rate = totals["hit_sum"] / question_count
        mrr = totals["rr_sum"] / question_count
        average_recall = totals["recall_sum"] / question_count
        average_latency_ms = totals["latency_sum"] / question_count * 1000

        print("\n检索器：", retriever_name)
        print(f"Hit Rate@{top_k}：", round(hit_rate, 4))
        print("MRR：", round(mrr, 4))
        print(f"平均 Recall@{top_k}：", round(average_recall, 4))
        print("平均查询耗时（毫秒）：", round(average_latency_ms, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行检索效果评测")
    parser.add_argument(
        "--dataset",
        choices=("dev", "test", "all"),
        default="test",
        help="选择开发集、冻结测试集或依次运行两者，默认运行 test",
    )
    arguments = parser.parse_args()

    selected_datasets = {
        "dev": [("dev", DEV_EVAL_PATH)],
        "test": [("test", TEST_EVAL_PATH)],
        "all": [("dev", DEV_EVAL_PATH), ("test", TEST_EVAL_PATH)],
    }[arguments.dataset]

    # 运行 all 时共用同一组模型，避免在两个评测集之间重复加载。
    shared_retriever = HybridRetriever()
    shared_reranker = Reranker()
    for dataset_name, dataset_path in selected_datasets:
        print("\n" + "*" * 70)
        print("开始运行评测集：", dataset_name)
        run_evaluation(
            evaluation_path=dataset_path,
            top_k=5,
            candidate_k=20,
            rrf_k=60,
            retriever=shared_retriever,
            reranker=shared_reranker,
        )
