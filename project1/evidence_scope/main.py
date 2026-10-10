from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from tools.finance import tool_schema, compare_revenue
from tools.pdf_search import pdf_tool_schema, search_annual_report

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
你是一名商业与行业研究助手。
使用中文回答问题。
你的回答应以真实、可追溯的资料为依据。
</identity>

## Available Tools
<tools>
1. compare_revenue
   查询已整理的公司营业收入数据，
   并计算2024年至2025年的同比增长率。

2. search_annual_report
   检索宁德时代2025年年度报告PDF，
   返回匹配关键词的原文片段及PDF页码。
</tools>

## Workflow
<workflow>
1. 分析用户的问题，判断需要哪些信息。
2. 如果用户需要查询营业收入或计算同比增长率，
   可以调用 compare_revenue。
3. 如果用户需要查看财报原文、查找相关内容
   或核实数据来源，调用 search_annual_report。
4. 根据工具返回的资料组织回答。
5. 涉及财务数据时，应说明年份、统计口径和单位。
6. 如果引用PDF内容，应标明来源及对应PDF页码。
7. 如果证据不足，应明确说明，不得编造结论。
</workflow>

## Rules
<rules>
- 不得编造公司财务数据。
- 不得伪造资料来源或PDF页码。
- 不得擅自修改工具计算的增长率。
- 不得将未经核验的数据声称为已核验。
- JSON中的数据不能自动视为经过PDF原文核验。
- 工具检索失败时，应明确说明。
</rules>
"""
    }
]
tools = [
    tool_schema,
    pdf_tool_schema
]

def execute_tool(tool_name, arguments):

    if tool_name == "compare_revenue":
        return compare_revenue(arguments["company"])
    if tool_name == "search_annual_report":
        return search_annual_report(arguments["keyword"])

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
            if tool_name == "search_annual_report":

                data = json.loads(result)

                print("\n===== PDF 检索结果 =====")

                for item in data.get("results", []):
                    print(f"\nPDF 第 {item['pdf_page']} 页")
                    print(item["snippet"])

                print("======================")

        except (ValueError, KeyError, TypeError) as e:
            result = json.dumps({"error": str(e)}, ensure_ascii=False)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
else:    print("Agent 达到最大迭代次数，任务终止。")