# core/worker.py
from core.llm_engine import LLMEngine
from core.parser import parse_tool_call
from core.skill_registry import Skill
from typing import Optional


class SkillWorker:
    def __init__(self, llm_engine: LLMEngine, max_steps: int):
        self.llm = llm_engine
        self.max_steps = max_steps

    async def execute(self, user_query: str, skill: Optional[Skill], mcp_session, tools: list) -> str:
        """Execute the standard ReAct/Tool-calling loop using the Skill's context."""

        # 1. Context Injection: Convert the Skill's Markdown into the System Prompt
        if skill:
            system_prompt = f"You are a specialized AI Agent. You MUST strictly follow these rules and output formats:\n\n{skill.content}"
        else:
            system_prompt = "You are a helpful AI Agent."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # 2. Execution Loop
        for step in range(self.max_steps):
            print(f"\n=== Worker Step {step + 1} ===")

            response_text = self.llm.generate(messages, tools)
            print("Raw LLM Output:\n", response_text)

            func_name, arguments = parse_tool_call(response_text)

            if func_name:
                print("\n[Function Calling Triggered]")
                print(f"Function: {func_name} | Params: {arguments}")

                # Execute tool via MCP
                res = await mcp_session.call_tool(name=func_name, arguments=arguments)
                result = res.content[0].text if res.content else "No result from MCP"
                print(f"Tool Result: {result}")

                # Update context with the tool execution result
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "name": func_name, "content": result})
            else:
                # If no tool is called, the LLM has synthesized the final answer
                print("\n[Final Synthesis Reached]")
                return response_text

        print("\n[Warning]: Worker terminated due to max steps limit.")
        return "Error: Maximum execution steps reached."