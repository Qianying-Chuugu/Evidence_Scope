# 混合检索

# BM25Retriever 负责关键词检索；
# DenseRetriever 负责向量语义检索。
from .bm25_search import BM25Retriever
from .dense_search import DenseRetriever

DEFAULT_BM25_WEIGHT = 1.5
DEFAULT_DENSE_WEIGHT = 1.0

def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    rrf_k=60,
    bm25_weight=DEFAULT_BM25_WEIGHT,
    dense_weight=DEFAULT_DENSE_WEIGHT,
):
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")
    if bm25_weight <= 0 or dense_weight <= 0:
        raise ValueError("检索器权重必须大于 0")

    fused_results = {}
    for rank, item in enumerate(bm25_results, start=1):
        # 从当前结果字典中取出 chunk_id。
        chunk_id = item["chunk_id"]
        # RRF 公式：
        # 1 / (rrf_k + rank)
        fused_results[chunk_id] = {
            "chunk_id": chunk_id,
            "source": item.get("source"),
            "source_path": item.get("source_path", item.get("source")),
            "page": item["page"],
            "text": item["text"],
            # 权重越大，这一路检索结果对最终排名的影响越大。
            "rrf_score": bm25_weight / (rrf_k + rank),
            # 记录它在 BM25 中的排名
            "bm25_rank": rank,
            "dense_rank": None,
        }
    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "source": item.get("source"),
                "source_path": item.get("source_path", item.get("source")),
                "page": item["page"],
                "text": item["text"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "dense_rank": None,
            }
        fused_results[chunk_id]["rrf_score"] += dense_weight / (rrf_k + rank)
        fused_results[chunk_id]["dense_rank"] = rank


    result_list = list(fused_results.values())

    def get_sort_key(result):
        # Python 默认从小到大排序。
        # 给 rrf_score 加负号后：
        # 原来分数越大的结果，负数反而越小，
        # 因此可以实现按照分数从高到低排列。
        # chunk_id 是第二排序条件：
        # 如果两个结果分数完全相同，就按 chunk_id 排序，
        return -result["rrf_score"], result["chunk_id"]

    # sorted返回一个排好序的新列表。
    sorted_results = sorted(result_list, key=get_sort_key)
    return sorted_results



class HybridRetriever:

    def __init__(self):
        # self.bm25 保存一个 BM25Retriever 对象。
        # 创建 BM25Retriever 时，它会：
        # 1. 读取 chunks.json；
        # 2. 对所有 Chunk 进行中文分词；
        # 3. 在内存中建立 BM25 索引。
        self.bm25 = BM25Retriever()
        # self.dense 保存一个 DenseRetriever 对象。
        # 创建 DenseRetriever 时，它会：
        # 1. 读取 chunks.json；
        # 2. 从本地缓存加载 Embedding 模型；
        # 3. 读取 dense_vectors.npy。
        self.dense = DenseRetriever()

    def search(
        self,
        query,
        top_k=10,
        candidate_k=20,
        rrf_k=60,
        bm25_weight=DEFAULT_BM25_WEIGHT,
        dense_weight=DEFAULT_DENSE_WEIGHT,
        source=None,
    ):
        """
        执行一次混合检索。
        参数：
        query：用户的查询文本。
        top_k：最终返回多少个融合结果。
        candidate_k：BM25 和 Dense 各自召回多少个候选。
        rrf_k：RRF 公式中的平滑常数。
        bm25_weight：BM25 排名分数的权重。
        dense_weight：Dense 排名分数的权重。
        source：可选文件名；提供后只在该文件中检索。
        """
        query = query.strip()
        # 如果删除空格后字符串为空，就拒绝执行检索。
        if not query:
            raise ValueError("query 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if candidate_k < top_k:
            raise ValueError("candidate_k 不能小于 top_k")

        # 调用 BM25Retriever.search()，得到稀疏排序的召回结果。
        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
            source=source,
        )
        # 调用 DenseRetriever.search()，得到稠密排序的召回结果。
        dense_results = self.dense.search(
            query=query,
            top_k=candidate_k,
            source=source,
        )

        # 把两路候选交给已经测试通过的 RRF 函数。
        # fused_results 已经按照 rrf_score 从高到低排序。
        fused_results = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

        return fused_results[:top_k]




if __name__ == "__main__":
    # 创建混合检索器。
    # 初始化过程中会加载 BM25 索引和 Dense 模型。
    retriever = HybridRetriever()

    # 使用项目中的真实问题进行测试。
    query = "宁德时代2025年的营业收入是多少？"

    # 每个检索器先召回 20 个候选，
    # RRF 融合后最终返回前 10 个。
    results = retriever.search(
        query=query,
        top_k=10,
        candidate_k=20,
        rrf_k=60,
    )

    # enumerate(..., start=1) 让最终排名从 1 开始。
    for final_rank, result in enumerate(results, start=1):
        print("=" * 60)
        print("融合排名：", final_rank)
        print("Chunk ID：", result["chunk_id"])
        print("PDF 页码：", result["page"])
        print("RRF 分数：", result["rrf_score"])
        print("BM25 排名：", result["bm25_rank"])
        print("Dense 排名：", result["dense_rank"])

        # 只打印前 300 个字符，避免终端输出过长。
        print("文本：")
        print(result["text"][:300])
