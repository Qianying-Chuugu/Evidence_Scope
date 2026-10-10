import json
from pathlib import Path


# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "financial_facts.json"


# 提供给 LLM 的工具定义
tool_schema = {
    "type": "function",
    "function": {
        "name": "compare_revenue",
        "description": "查询上市公司2024年与2025年的营业收入，并计算同比增长率",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "公司名称，例如宁德时代"
                }
            },
            "required": ["company"]
        }
    }
}


def compare_revenue(company):

    # 读取本地财务资料
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data["companies"]:

        aliases = [name.lower()for name in record["aliases"]]

        if company.strip().lower() not in aliases:
            continue

        # 获取两个年度的收入
        revenue_2024 = record["revenue"]["2024"]
        revenue_2025 = record["revenue"]["2025"]

        # 计算同比增长率
        growth_rate = ((revenue_2025 - revenue_2024)/ revenue_2024* 100)

        result = {
            "company": record["company"],
            "metric": record["metric"],
            "scope": record["scope"],
            "revenue_2024": round(revenue_2024 / 100000, 2),
            "revenue_2025": round(revenue_2025 / 100000, 2),
            "unit": "亿元",
            "growth_rate": round(growth_rate, 2),
            "source_title": record["source_title"],
            "source_url": record["source_url"]
        }

        return json.dumps(result, ensure_ascii=False)

    return json.dumps({"error": "当前资料库没有收录这家公司的数据"}, ensure_ascii=False)