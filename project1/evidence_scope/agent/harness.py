"""通用 Agent Harness：负责模型调用、循环控制和工具执行。"""

import json
from openai import OpenAIError


class AgentHarness:
    def __init__(self,client, model,initial_messages,tool_schemas,tool_executor,max_iterations=6,):
        self.client = client
        self.model = model
        self.initial_messages = initial_messages
        self.tool_schemas = tool_schemas
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations

    def run(self, question):
        """处理一个用户问题，返回 Agent 的最终文本答案。"""
        question = question.strip()
        if not question:
            return "问题不能为空。"

        # 每次调用 run() 都创建独立消息历史，避免不同问题互相污染。
        messages = [dict(message) for message in self.initial_messages]
        messages.append({"role": "user", "content": question})

        for i in range(self.max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                )
            except OpenAIError as error:
                return f"模型调用失败：{error}"

            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump(exclude_none=True))

            # 没有工具调用表示模型已经形成最终答案。
            if not assistant_message.tool_calls:
                return assistant_message.content or "模型没有返回文本答案。"

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                print("\n模型请求调用：", tool_name)

                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = self.tool_executor(tool_name, arguments)
                except (json.JSONDecodeError,ValueError,KeyError,TypeError,RuntimeError,OSError,) as error:
                    # 把错误作为工具结果交还给模型，让模型可以解释或改用其他参数。
                    result = json.dumps(
                        {"error": str(error)},
                        ensure_ascii=False,
                    )

                self._display_tool_result(tool_name, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        return "Agent 达到最大迭代次数，任务终止。"

    @staticmethod
    def _display_tool_result(tool_name, result):
        """把关键工具结果显示在终端，便于观察 Agent 实际使用了哪些证据。"""
        if tool_name != "search_evidence":
            print(result)
            return

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            print(result)
            return

        if "error" in data:
            print("工具调用失败：", data["error"])
            return

        print("\n===== RAG 检索结果 =====")
        for item in data.get("results", []):
            print(
                f"\n排名 {item['rank']} | {item['source']} | "
                f"{item['chunk_id']} | PDF 第 {item['pdf_page']} 页"
            )
            print(item["text"])
        print("======================")
