"""集中注册 Agent 可以调用的工具。"""

import json
from tools.evidence_search import evidence_tool_schema, search_evidence


# 这份列表会原样传给大模型，让模型知道有哪些工具和参数。
TOOL_SCHEMAS = [evidence_tool_schema]


def execute_tool(tool_name, arguments):
    """根据大模型返回的工具名称，调用对应的 Python 函数。"""
    if tool_name == "search_evidence":
        return search_evidence(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
            source=arguments.get("source"),
        )

    return json.dumps(
        {"error": f"未知工具：{tool_name}"},
        ensure_ascii=False,
    )
