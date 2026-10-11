from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from tools.evidence_search import evidence_tool_schema, search_evidence

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [
    {
        "role": "system",
        "content": """
# Agent Instructions

## Identity
<identity>
你是一名本地文件商业与行业研究助手。
使用中文回答问题。
你的回答应以真实、可追溯的资料为依据。
</identity>

## Available Tools
<tools>
1. search_evidence
   依次使用 BM25、Dense、Hybrid 和 Reranker 检索本地 PDF 知识库，
   可以检索全部文件或用户指定的文件，返回相关原文、文件名、PDF页码、Chunk ID和检索排名。
</tools>

## Workflow
<workflow>
1. 分析用户的问题，判断需要哪些信息。
2. 回答本地文件中的事实、财务数据或公司业务问题前，必须调用 search_evidence。
3. 检索查询应包含关键指标、年份、业务或地区等限定条件，不能只传一个宽泛词语。
4. 如果第一次证据不足，可以更换查询表述再检索一次。
5. 只根据工具返回的证据组织回答。
6. 涉及财务数据时，应说明年份、统计口径和单位。
7. 最终答案必须标明所依据的文件名、PDF页码和Chunk ID。
8. 如果证据不足，应明确说明，不得编造结论。
</workflow>

## Rules
<rules>
- 不得编造公司财务数据。
- 不得伪造资料来源、PDF页码或Chunk ID。
- 不得将未经核验的数据声称为已核验。
- 工具检索失败时，应明确说明。
</rules>
"""
    }
]
tools = [
    evidence_tool_schema
]

def execute_tool(tool_name, arguments):

    if tool_name == "search_evidence":
        return search_evidence(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
            source=arguments.get("source"),
        )

    return json.dumps({"error": "未知工具"}, ensure_ascii=False)


question = input("请输入研究问题：")
messages.append({
    "role": "user",
    "content": question
})

# Agent Loop
for iteration in range(6):

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )

    assistant_message = response.choices[0].message

    messages.append(assistant_message.model_dump(exclude_none=True))

    if not assistant_message.tool_calls:
        print("\n最终回答：")
        print(assistant_message.content)
        break

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
            print("\n模型请求调用：", tool_name)
            result = execute_tool(tool_name, arguments)
            if tool_name == "search_evidence":

                data = json.loads(result)

                print("\n===== RAG 检索结果 =====")

                for item in data.get("results", []):
                    print(
                        f"\n排名 {item['rank']} | {item['source']} | "
                        f"{item['chunk_id']} | "
                        f"PDF 第 {item['pdf_page']} 页"
                    )
                    print(item["text"])

                print("======================")

        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            result = json.dumps({"error": str(e)}, ensure_ascii=False)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
else:
    print("Agent 达到最大迭代次数，任务终止。")
