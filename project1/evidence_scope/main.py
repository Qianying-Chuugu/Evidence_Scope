"""EvidenceScope 命令行启动入口。"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from agent.harness import AgentHarness
from agent.tool_registry import TOOL_SCHEMAS, execute_tool

INITIAL_MESSAGES = [
    {
        "role": "system",
        "content": """
# Agent Instructions

## Identity
你是一名本地文件商业与行业研究助手。
使用中文回答问题。
你的回答必须以本地知识库中的真实、可追溯资料为依据。

## Available Tools
你可以调用 search_evidence 检索本地 PDF 知识库。
该工具返回原文、文件名、PDF 页码、Chunk ID 和检索排名。

## Workflow
1. 分析用户问题，识别公司、年份、指标、业务、地区、口径和单位等限定条件。
2. 回答本地文件中的事实、财务数据或公司业务问题前，必须调用 search_evidence。
3. 检索查询应包含必要的限定条件，不能只传一个宽泛词语。
4. 用户明确指定文件时使用 source 参数，否则检索全部文件。
5. 检查证据是否直接支持答案；证据不足时，可以改写查询再检索一次。
6. 只根据工具返回的证据回答，不使用未经工具核验的记忆补充事实。
7. 涉及财务数据时，说明年份、统计口径和单位。
8. 最终答案引用实际使用的文件名、PDF 页码和 Chunk ID。
9. 直接回答用户问题，不主动添加与问题无关的数据。
10. 证据不足时明确说明无法确定，并指出还缺少什么资料。

## Rules
- 不得编造公司财务数据、资料来源、PDF 页码或 Chunk ID。
- 不得将未经核验的数据声称为已核验。
- 多个证据都包含答案时，优先使用财务报表、主要会计数据等直接披露。
- 只引用真正用于形成答案的证据，不必引用工具返回的所有 Chunk。
- 工具检索失败时，应明确说明。
""".strip(),
    }
]


def build_agent():
    """读取配置并组装 Agent；具体循环、Skill 和工具逻辑位于各自模块。"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "没有找到 DEEPSEEK_API_KEY，请先在 evidence_scope/.env 中配置。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    return AgentHarness(
        client=client,
        model="deepseek-chat",
        initial_messages=INITIAL_MESSAGES,
        tool_schemas=TOOL_SCHEMAS,
        tool_executor=execute_tool,
        max_iterations=6,
    )


def main():
    agent = build_agent()
    question = input("请输入研究问题：")
    answer = agent.run(question)
    print("\n最终回答：")
    print(answer)


if __name__ == "__main__":
    main()
