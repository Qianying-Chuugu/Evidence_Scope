"""EvidenceScope 命令行启动入口。"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from agent.harness import AgentHarness
from agent.tool_registry import TOOL_SCHEMAS, execute_tool
from skills.local_document_qa import SYSTEM_PROMPT


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
        system_prompt=SYSTEM_PROMPT,
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
