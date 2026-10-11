"""扫描本地 PDF 文件并生成统一的 chunks.json。

使用方法：
1. 把新的 PDF 放入 data/documents 目录；可以继续保留原来的 data/CATL_2025.pdf。
2. 在 evidence_scope 目录运行：python -m retrieval.chunking
3. 下次启动 Dense 检索时，会发现 chunks.json 已更新并自动重建向量。
"""

import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
LEGACY_PDF_PATH = DATA_DIR / "CATL_2025.pdf"
OUTPUT_PATH = DATA_DIR / "chunks.json"

CHUNK_SIZE = 500
OVERLAP = 150

def discover_pdf_files():
    """返回需要建立索引的 PDF；排序保证重复运行结果顺序一致。"""
    pdf_files = []
    if LEGACY_PDF_PATH.exists():
        pdf_files.append(LEGACY_PDF_PATH)
    if DOCUMENTS_DIR.exists():
        pdf_files.extend(DOCUMENTS_DIR.rglob("*.pdf"))

    # resolve() 后去重，避免同一个文件被重复加入。
    unique_files = {path.resolve() for path in pdf_files}
    return sorted(unique_files, key=lambda path: str(path).casefold())


def make_document_id(pdf_path):
    """根据相对路径生成稳定的短 ID，用于区分不同文件的 Chunk。"""
    relative_path = pdf_path.relative_to(DATA_DIR).as_posix().casefold()
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]


def chunk_documents():
    pdf_files = discover_pdf_files()
    if not pdf_files:
        raise FileNotFoundError(
            f"没有找到 PDF。请把文件放入 {DOCUMENTS_DIR}"
        )

    chunks = []
    for pdf_path in pdf_files:
        document_id = make_document_id(pdf_path)
        source_path = pdf_path.relative_to(DATA_DIR).as_posix()
        document_chunk_number = 0

        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True).strip()
                # 字符过少的封面或空白页不进入检索索引。
                if len(text) < 100:
                    continue

                start = 0
                while start < len(text):
                    end = min(start + CHUNK_SIZE, len(text))
                    chunk_text = text[start:end]
                    document_chunk_number += 1

                    # 旧年报继续沿用原有 ID，保证已有 30 道评测题仍然有效。
                    if pdf_path == LEGACY_PDF_PATH.resolve():
                        chunk_id = f"chunk_{document_chunk_number:04d}"
                    else:
                        chunk_id = (
                            f"doc_{document_id}_chunk_{document_chunk_number:04d}"
                        )

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "source": pdf_path.name,
                            "source_path": source_path,
                            "file_type": "pdf",
                            "page": page_number,
                            "text": chunk_text,
                        }
                    )

                    if end == len(text):
                        break
                    start = end - OVERLAP

        print(f"已处理：{source_path}，生成 {document_chunk_number} 个 Chunk")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"分块完成：{len(pdf_files)} 个 PDF，共 {len(chunks)} 个 Chunk")

    if chunks:
        print("\n第一个 Chunk：")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


def chunk_pdf():
    """保留旧函数名，已有调用代码无需修改。"""
    chunk_documents()


if __name__ == "__main__":
    chunk_documents()
