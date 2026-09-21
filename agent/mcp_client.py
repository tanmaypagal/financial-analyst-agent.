"""MCP client used by the agent: spawns/connects to the MCP server, lists tools, calls tools.

The agent never imports mcp_server or touches SQLite - everything crosses the MCP protocol boundary.
Set MCP_SERVER_URL (e.g. http://127.0.0.1:8765/mcp) to use streamable HTTP; otherwise the server is
spawned as a stdio child process.
"""
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP Tool (name/description/inputSchema) to an OpenAI function-calling tool. Automatic - no
    hand-written duplicate of the schemas."""
    return {"type": "function", "function": {"name": tool.name, "description": tool.description or "",
                                             "parameters": tool.inputSchema}}


def parse_result(result) -> dict:
    """Extract the JSON dict from a CallToolResult."""
    sc = getattr(result, "structuredContent", None)
    if sc:
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"found": False, "reason": f"tool error: {text[:300]}"}


SERVER_ENV_VARS = ("PATH", "FINANCE_DB", "AGENT_TODAY")      # + PYTHONPATH. The server never needs OPENAI_API_KEY or any other secret.


def server_params() -> StdioServerParameters:
    """How the MCP server child process is started. Only the variables it needs are passed (the SDK adds its own small
    list of safe defaults such as SYSTEMROOT/TEMP); in particular OPENAI_API_KEY is NOT inherited."""
    env = {k: os.environ[k] for k in SERVER_ENV_VARS if k in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT), env=env)


class MCPToolClient:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("MCP_SERVER_URL")
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools = []

    async def __aenter__(self):
        if self.url:
            from mcp.client.streamable_http import streamablehttp_client
            read, write, _ = await self._stack.enter_async_context(streamablehttp_client(self.url))
        else:
            read, write = await self._stack.enter_async_context(stdio_client(server_params()))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self.tools = (await self.session.list_tools()).tools
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    def openai_tools(self) -> list[dict]:
        return [mcp_tool_to_openai(t) for t in self.tools]

    async def call(self, name: str, args: dict) -> tuple[dict, float]:
        t0 = time.perf_counter()
        res = await self.session.call_tool(name, args)
        return parse_result(res), (time.perf_counter() - t0) * 1000
