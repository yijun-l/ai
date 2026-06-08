# main.py
import asyncio
from config import MODEL_PATH, MCP_SERVER_URL, MAX_AGENT_STEPS, DEVICE_MAP, SKILLS_DIR
from core.llm_engine import LLMEngine
from core.mcp_client import MCPSessionManager
from core.skill_registry import SkillRegistry
from core.router import AgentRouter
from core.worker import SkillWorker


async def main():
    print("=== Initializing Agent-Lite (Skill-Driven) ===")

    # 1. Initialize Engines and Registries
    llm = LLMEngine(MODEL_PATH, DEVICE_MAP)
    mcp_manager = MCPSessionManager(MCP_SERVER_URL)

    registry = SkillRegistry(SKILLS_DIR)
    router = AgentRouter(llm)
    worker = SkillWorker(llm, MAX_AGENT_STEPS)

    user_query = "What is the weather today in Beijing?"
    print(f"\n[User Query]: {user_query}")

    # 2. Start Session and Load Tools
    async with mcp_manager.connect() as session:
        tools_res = await session.list_tools()
        tools = mcp_manager.format_tools(tools_res)
        print(f"Successfully loaded {len(tools)} tools from MCP Server.")

        # ==========================================
        # STAGE 1: Routing (Intent Classification)
        # ==========================================
        print("\n--- STAGE 1: ROUTING ---")
        skills_metadata = registry.get_metadata_prompt()
        selected_skill_name = router.route(user_query, skills_metadata)

        print(f"Router Decision: Assigned to '{selected_skill_name}'")
        selected_skill = registry.get_skill(selected_skill_name)

        # ==========================================
        # STAGE 2 & 3 & 4: Execution & Synthesis
        # ==========================================
        print("\n--- STAGE 2: EXECUTION ---")
        final_answer = await worker.execute(user_query, selected_skill, session, tools)

        print("\n==========================================")
        print("[FINAL ANSWER DELIVERED TO USER]")
        print("==========================================")
        print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())