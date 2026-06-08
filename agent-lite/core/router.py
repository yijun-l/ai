# core/router.py
from core.llm_engine import LLMEngine


class AgentRouter:
    def __init__(self, llm_engine: LLMEngine):
        self.llm = llm_engine

    def route(self, user_query: str, skills_metadata: str) -> str:
        """Use the LLM to classify the user's intent and select the appropriate skill."""

        prompt = f"""You are a routing agent. Analyze the user query and select the exact name of the best skill from the list below. 
Return ONLY the exact skill name. If no skill matches the query, return 'DEFAULT'.

Available Skills:
{skills_metadata}

User Query: "{user_query}"
"""
        messages = [{"role": "user", "content": prompt}]

        # We pass an empty tools list here because the router's only job is classification
        response = self.llm.generate(messages, tools=[])

        # Clean up the response in case the LLM outputs extra spaces or markdown backticks
        selected_skill = response.replace("`", "").strip()
        return selected_skill