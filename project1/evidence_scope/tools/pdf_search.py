import json
from pathlib import Path

import pymupdf

PDF_PATH = (Path(__file__).resolve().parent.parent/ "data"/ "CATL_2025.pdf")

pdf_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_annual_report",
        "description": (
            "检索宁德时代2025年年度报告PDF，"
            "返回包含关键词的原文片段及PDF页码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "需要在财报中检索的关键词，例如营业收入"
                }
            },
            "required": ["keyword"]
        }
    }
}


def search_annual_report(keyword: str) -> str:

    keyword = keyword.strip()

    if not keyword:
        return json.dumps({"error": "关键词不能为空"}, ensure_ascii=False)

    results = []

    with pymupdf.open(PDF_PATH) as doc:

        for page_index, page in enumerate(doc):

            text = page.get_text(sort=True)
            pos = text.find(keyword)

            if pos == -1:
                continue

            # 提取关键词附近的原文
            start = max(0, pos - 300)
            end = min(len(text), pos + 600)

            results.append({
                "pdf_page": page_index + 1,
                "snippet": text[start:end]
            })

            # 暂时最多返回前5处匹配
            if len(results) >= 5:
                break

    return json.dumps({"keyword": keyword, "results": results,"source": "宁德时代2025年年度报告"}, ensure_ascii=False)