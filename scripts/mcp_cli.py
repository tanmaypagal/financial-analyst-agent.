"""Tiny MCP test client.  Examples:
  python scripts/mcp_cli.py tools
  python scripts/mcp_cli.py call get_financials '{"sector":"defense","company":"Saab","periods":2}'
  python scripts/mcp_cli.py smoke          # calls every tool once + negative tests
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.mcp_client import MCPToolClient  # noqa: E402


async def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tools"
    async with MCPToolClient() as c:
        if cmd == "tools":
            for t in c.tools:
                print(f"- {t.name}: {(t.description or '').splitlines()[0]}\n    schema: {json.dumps(t.inputSchema.get('properties'))}")
        elif cmd == "call":
            out, ms = await c.call(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
            print(json.dumps(out, indent=2, default=str)); print(f"\n[{ms:.0f} ms]")
        elif cmd == "smoke":
            cases = [("list_sectors", {}), ("list_companies", {"sector": "defense"}),
                     ("get_company_profile", {"sector": "defense", "company": "thales"}),
                     ("get_financials", {"sector": "defense", "company": "LMT", "periods": 2}),
                     ("get_valuation", {"sector": "defense", "company": "Saab"}),
                     ("get_signals", {"sector": "defense", "company": "Rheinmetall", "signal_type": "headcount"}),
                     ("get_sector_stats", {"sector": "defense", "metric": "ev_ebitda"}),
                     ("get_defense_metrics", {"company": "Lockheed"}),
                     ("get_data_quality", {"sector": "defense", "company": "Kongsberg"}),
                     ("get_schema", {}),
                     ("get_company_profile", {"sector": "defense", "company": "Tesla"}),        # not in DB
                     ("get_company_profile", {"sector": "tech", "company": "Lockheed Martin"}), # wrong sector
                     ("get_company_profile", {"sector": "defense", "company": "rtx"})]
            for name, args in cases:
                out, ms = await c.call(name, args)
                keys = {k: out.get(k) for k in ("found", "reason", "source_id", "as_of_date", "stale") if k in out}
                print(f"{name}({json.dumps(args)}) -> {ms:.0f}ms {keys}")


asyncio.run(main())
