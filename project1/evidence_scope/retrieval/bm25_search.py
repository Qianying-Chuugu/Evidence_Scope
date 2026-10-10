import json
import jieba
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CHUNK_PATH = ROOT / "data" / "chunks.json"

#添加一般的分词词语，防止进一步被拆分
jieba.add_word("营业收入")
jieba.add_word("宁德时代")

def tokenize(text):
    #中文分词，分词中去除空格
    return [word for word in jieba.lcut(text) if word.strip()]

class BM25Retriever:
    def __init__(self):
        with open(CHUNK_PATH,"r",encoding="utf-8") as f:
            self.chunks = json.load(f)
        # 中文分词
        self.documents = [tokenize(chunk["text"]) for chunk in self.chunks]

        self.bm25 = BM25Okapi(
            self.documents
        )

    def search(self,query,top_k=5):

        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)

        results = []
        for index in scores.argsort()[::-1][:top_k]:
            results.append({
                "score": float(scores[index]),
                "page":self.chunks[index]["page"],
                "text":self.chunks[index]["text"]
            })
        return results

if __name__ == "__main__":

    retriever = BM25Retriever()
    query = "营业收入 2025 2024 同比增长"
    print("查询分词：", tokenize(query))
    results = retriever.search(query)

    for r in results:
        print("="*50)
        print("页码:",r["page"])
        print("得分:",r["score"])
        print(r["text"])