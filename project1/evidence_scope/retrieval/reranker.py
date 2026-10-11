#使用 Cross-Encoder 对第一阶段检索结果进行rerank

from sentence_transformers import CrossEncoder
from .hybrid_search import HybridRetriever

MODEL_NAME = "BAAI/bge-reranker-base"

def load_reranker_model():
    try:
        return CrossEncoder(
            MODEL_NAME,
            local_files_only=True,
            max_length=512,
        )
    except OSError as error:
        raise RuntimeError(
            "本地没有可用的 Reranker 模型，请先下载 BAAI/bge-reranker-base"
        ) from error

class Reranker:
    def __init__(self):
        self.model = load_reranker_model()

    def rerank(self, query, candidates, top_k=5):
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not candidates:
            return []

        # Cross-Encoder 的每个输入都是一个“问题、候选文本”对。
        # 它会同时阅读两段文字，然后直接输出相关性分数。
        query_chunk_pairs = [(query, candidate["text"])for candidate in candidates]

        # batch_size=8 表示每次处理 8 个文本对。
        scores = self.model.predict(
            query_chunk_pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        reranked_results = []
        for original_rank, (candidate, score) in enumerate(
            zip(candidates, scores),
            start=1,
        ):
            reranked_item = dict(candidate)
            reranked_item["original_rank"] = original_rank
            reranked_item["reranker_score"] = float(score)
            reranked_results.append(reranked_item)

        # Reranker 分数越大，说明问题与 Chunk 越相关，因此按降序排列。
        reranked_results.sort(
            key=lambda item: (-item["reranker_score"], item["chunk_id"])
        )
        return reranked_results[:top_k]


if __name__ == "__main__":
    query = "宁德时代2025年的营业收入是多少？"

    # 第一阶段：Hybrid Search 从整个知识库中召回 20 个候选 Chunk。
    hybrid_retriever = HybridRetriever()
    candidates = hybrid_retriever.search(
        query=query,
        top_k=20,
        candidate_k=20,
        rrf_k=60,
    )

    # 第二阶段：Reranker 阅读问题和候选全文，重新排列前 20 个候选。
    reranker = Reranker()
    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=10,
    )

    for final_rank, result in enumerate(results, start=1):
        print("=" * 60)
        print("重排后排名：", final_rank)
        print("重排前排名：", result["original_rank"])
        print("Chunk ID：", result["chunk_id"])
        print("PDF 页码：", result["page"])
        print("Reranker 分数：", result["reranker_score"])
        print("BM25 排名：", result["bm25_rank"])
        print("Dense 排名：", result["dense_rank"])
        print("文本：")
        print(result["text"][:300])
