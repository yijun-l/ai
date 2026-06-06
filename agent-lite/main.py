# main.py
import asyncio
from config import MODEL_PATH, MCP_SERVER_URL, MAX_AGENT_STEPS, DEVICE_MAP
from core.llm_engine import LLMEngine
from core.mcp_client import MCPSessionManager
from core.parser import parse_tool_call


async def agent_loop():
    # 1. Initialize Engine & MCP Manager
    llm = LLMEngine(MODEL_PATH, DEVICE_MAP)
    mcp_manager = MCPSessionManager(MCP_SERVER_URL)

    # 2. Start Agent Session
    async with mcp_manager.connect() as session:
        # Fetch and format tools
        tools_res = await session.list_tools()
        tools = mcp_manager.format_tools(tools_res)
        print("Successfully loaded tools from MCP:", [t["function"]["name"] for t in tools])

        # Initialize conversation
        messages = [
            {"role": "system", "content": "You are a helpful AI Agent."},
            {"role": "user", "content": "What is the weather today in Beijing?"}
        ]

        # 3. Execution Loop
        for step in range(MAX_AGENT_STEPS):
            print(f"\n=== Agent Step {step + 1} ===")

            response_text = llm.generate(messages, tools)
            print("Raw LLM Output:\n", response_text)

            func_name, arguments = parse_tool_call(response_text)

            if func_name:
                print("\n[Function Calling Triggered]")
                print(f"Function: {func_name}")
                print(f"Params: {arguments}")

                # Execute tool via MCP
                res = await session.call_tool(name=func_name, arguments=arguments)
                result = res.content[0].text if res.content else "No result"
                print(f"Tool Result: {result}")

                # Update context
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "name": func_name, "content": result})
            else:
                print("\n[Final Answer]:\n", response_text)
                break


if __name__ == "__main__":
    asyncio.run(agent_loop())