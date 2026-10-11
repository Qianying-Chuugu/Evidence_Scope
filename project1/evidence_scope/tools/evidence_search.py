"""供 Agent 调用的 RAG 检索工具。"""

import json

from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker

DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 5

_hybrid_retriever = None
_reranker = None

evidence_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": (
            "使用 BM25、Dense、RRF Hybrid 和 Reranker 检索宁德时代 "
            "2025 年年度报告，返回可用于回答问题的原文、PDF 页码和 Chunk ID。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "完整的检索查询。尽量包含指标名称、年份、业务名称、"
                        "地区或统计口径，例如：2025年研发投入金额及占营业收入比例"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "最终返回的证据数量，建议 3 到 5，默认 5",
                    "minimum": 1,
                    "maximum": 10,
                },
                "source": {
                    "type": "string",
                    "description": (
                        "可选的 PDF 文件名，例如 CATL_2025.pdf。"
                        "用户没有指定文件时不要填写，将在全部本地文件中检索。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def _get_retrieval_models():
    global _hybrid_retriever, _reranker
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    if _reranker is None:
        _reranker = Reranker()
    return _hybrid_retriever, _reranker

def search_evidence(query: str, top_k: int = DEFAULT_TOP_K, source=None) -> str:

    query = query.strip()
    if not query:
        return json.dumps({"error": "检索查询不能为空"}, ensure_ascii=False)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        return json.dumps({"error": "top_k 必须是整数"}, ensure_ascii=False)
    if not 1 <= top_k <= 10:
        return json.dumps({"error": "top_k 必须在 1 到 10 之间"}, ensure_ascii=False)

    hybrid_retriever, reranker = _get_retrieval_models()
    available_sources = sorted(
        {chunk.get("source") for chunk in hybrid_retriever.bm25.chunks}
    )

    if source is not None:
        if not isinstance(source, str) or not source.strip():
            return json.dumps(
                {"error": "source 必须是非空文件名"}, ensure_ascii=False
            )
        source = source.strip()
        matched_source = next(
            (
                available_source
                for available_source in available_sources
                if available_source.casefold() == source.casefold()
            ),
            None,
        )
        if matched_source is None:
            return json.dumps(
                {
                    "error": f"本地知识库中没有文件：{source}",
                    "available_sources": available_sources,
                },
                ensure_ascii=False,
            )
        source = matched_source

    # HybridRetriever 内部会分别运行 BM25 和 Dense，再用加权 RRF 融合。
    hybrid_candidates = hybrid_retriever.search(
        query=query,
        top_k=DEFAULT_CANDIDATE_K,
        candidate_k=DEFAULT_CANDIDATE_K,
        source=source,
    )

    # Cross-Encoder 阅读查询和每个候选全文，输出最终证据排序。
    reranked_results = reranker.rerank(
        query=query,
        candidates=hybrid_candidates,
        top_k=top_k,
    )

    results = []
    for final_rank, item in enumerate(reranked_results, start=1):
        results.append(
            {
                "rank": final_rank,
                "chunk_id": item["chunk_id"],
                "source": item.get("source"),
                "pdf_page": item["page"],
                "text": item["text"],
                "reranker_score": item["reranker_score"],
                "hybrid_rank": item["original_rank"],
                "bm25_rank": item.get("bm25_rank"),
                "dense_rank": item.get("dense_rank"),
            }
        )

    return json.dumps(
        {
            "query": query,
            "retrieval_method": "Hybrid (BM25 + Dense + weighted RRF) + Reranker",
            "source_filter": source,
            "available_sources": available_sources,
            "results": results,
        },
        ensure_ascii=False,
    )
