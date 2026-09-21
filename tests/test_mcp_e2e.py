import asyncio

from agent.mcp_client import MCPToolClient, mcp_tool_to_openai


def test_real_stdio_session_lists_and_calls_tools():
    async def go():
        async with MCPToolClient() as c:
            names = {t.name for t in c.tools}
            out, ms = await c.call("get_valuation", {"sector": "defense", "company": "Thales"})
            bad, _ = await c.call("get_valuation", {"sector": "defense", "company": "Tesla"})
            return names, c.openai_tools(), out, bad
    names, oa, out, bad = asyncio.run(go())
    assert {"list_sectors", "list_companies", "get_company_profile", "get_financials", "get_valuation", "get_signals",
            "get_sector_stats", "get_defense_metrics", "get_data_quality", "get_schema"} <= names
    assert out["found"] and out["ticker"] == "HO.PA" and bad["found"] is False
    fn = next(t for t in oa if t["function"]["name"] == "get_sector_stats")["function"]
    assert "enum" in fn["parameters"]["properties"]["metric"]        # schema auto-converted from MCP, incl. enum


def test_conversion_is_lossless():
    class T:
        name, description = "x", "d"
        inputSchema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert mcp_tool_to_openai(T)["function"]["parameters"] == T.inputSchema
