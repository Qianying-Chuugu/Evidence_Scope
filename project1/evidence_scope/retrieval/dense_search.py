# 稠密检索

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

CHUNK_PATH = ROOT / "data" / "chunks.json"
VECTOR_PATH = ROOT / "data" / "dense_vectors.npy"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_embedding_model():
    #优先从本地缓存加载模型，避免每次启动都访问 Hugging Face。
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError:
        print("本地未找到 Embedding 模型，首次运行将从 Hugging Face 下载。")
        return SentenceTransformer(MODEL_NAME)


class DenseRetriever:
    def __init__(self):
        # 读取文档分块
        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        # 加载 Embedding 模型
        self.model = load_embedding_model()
        # 读取已有向量，或首次生成向量
        if VECTOR_PATH.exists() and VECTOR_PATH.stat().st_mtime >= CHUNK_PATH.stat().st_mtime:
            self.vectors = np.load(VECTOR_PATH)
            if len(self.vectors) != len(self.chunks):
                self.build_index()
        else:
            self.build_index()

    def build_index(self):
        texts = [chunk["text"] for chunk in self.chunks]
        # 批量生成文档向量
        self.vectors = self.model.encode(
            texts,
            batch_size=16,  #每批处理16个chunk
            normalize_embeddings=True,  #向量归一化
            show_progress_bar=True  #显示处理进度
        )
        # 保存向量文档，避免每次启动都重新计算
        np.save(VECTOR_PATH, self.vectors)

    def search(self, query, top_k=10, source=None):
        # 用户问题转换为向量
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True
        )
        # 计算余弦相似度
        scores = self.vectors @ query_vector
        # 取分数最高的 Top-K
        indices = np.argsort(scores)[::-1]
        results = []
        for index in indices:
            chunk = self.chunks[index]
            if source and chunk.get("source", "").casefold() != source.casefold():
                continue
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk.get("source"),
                "source_path": chunk.get("source_path", chunk.get("source")),
                "page": chunk["page"],
                "score": float(scores[index]),
                "text": chunk["text"]
            })
            if len(results) >= top_k:
                break
        return results






if __name__ == "__main__":
    retriever = DenseRetriever()
    print("向量矩阵形状：", retriever.vectors.shape)
    query = "宁德时代2025年的营业收入是多少？"
    results = retriever.search(query, top_k=10)

    for result in results:
        print("=" * 50)
        print("页码：", result["page"])
        print("相似度：", result["score"])
        print(result["text"][:500])

