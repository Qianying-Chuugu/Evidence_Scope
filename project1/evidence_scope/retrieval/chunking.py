# 原始文本分块
# 分块大小为 500 个字符，相邻分块重叠 150 个字符。
# 较大的重叠区域用于减少“小节标题”和“具体数据”被分到两个 Chunk 的情况。

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "CATL_2025.pdf"
OUTPUT_PATH = ROOT / "data" / "chunks.json"

CHUNK_SIZE = 500
OVERLAP = 150

def chunk_pdf():
    chunks = []
    with pymupdf.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()
            #如果页面字符小于100，则认为是无效页面
            if len(text) < 100:
                continue

            start = 0

            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunk_text = text[start:end]

                chunks.append({
                    "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                    "source": PDF_PATH.name,
                    "page": page_number,
                    "text": chunk_text
                })

                if end == len(text):
                    break

                start = end - OVERLAP

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"分块完成，共生成 {len(chunks)} 个 Chunk")

    if chunks:
        print("\n第一个 Chunk：")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    chunk_pdf()
