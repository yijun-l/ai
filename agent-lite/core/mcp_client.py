# core/mcp_client.py
import contextlib
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

class MCPSessionManager:
    def __init__(self, url: str):
        self.url = url

    @contextlib.asynccontextmanager
    async def connect(self):
        """Async context manager for handling MCP connection lifecycle"""
        async with streamable_http_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @staticmethod
    def format_tools(tools_res) -> list:
        """Format MCP tools into the standard schema required by LLMs"""
        return [{"type": "function", "function": t.model_dump()} for t in tools_res.tools]