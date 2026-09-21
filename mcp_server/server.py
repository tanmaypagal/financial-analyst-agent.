"""MCP server exposing typed, read-only, parameterized finance tools. No free-form SQL is exposed.

Transport:
  stdio (default)         python -m mcp_server.server
  streamable HTTP         python -m mcp_server.server --transport http --port 8765

Why stdio by default: the agent spawns the server as a child process, so there is no port, no auth
surface and no lifecycle to manage - and only that child ever opens the SQLite file. HTTP is offered for
running the server as a shared/remote service (several agents, other MCP clients).
"""
import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server import queries as q  # noqa: E402

mcp = FastMCP("finance-data", instructions=(
    "Read-only financial data for defense/tech/logistics companies. Every result has source_id, as_of_date and "
    "a stale flag. {'found': false} means the company/sector is not in the database - do not guess."))

SignalType = Literal["headcount", "hiring", "news", "contract_award"]
Metric = Literal[tuple(q.SECTOR_METRICS)]  # type: ignore[valid-type]


@mcp.tool()
def list_sectors() -> dict:
    """List the sectors available in the database and how many companies each has."""
    return q.list_sectors()


@mcp.tool()
def list_companies(sector: str) -> dict:
    """List all companies (ticker, name, country, exchange, currency) in a sector."""
    return q.list_companies(sector)


@mcp.tool()
def get_company_profile(sector: str, company: str) -> dict:
    """Profile of one company: country, exchange, fiscal year end, accounting standard (US GAAP/IFRS),
    ownership notes (state/family stakes, relevant to take-private feasibility) and description.
    `company` accepts a name or ticker. Returns found=false if the company is not in the given sector."""
    return q.get_company_profile(sector, company)


@mcp.tool()
def get_financials(sector: str, company: str, periods: int = 3) -> dict:
    """Annual financials, newest first (native currency in full units, plus revenue_usd and the FX rate/date).
    Each period includes server-computed `derived` ratios: revenue_growth_yoy, ebitda_margin_change,
    net_debt_to_ebitda, fcf_conversion (FCF/EBITDA), capex_to_revenue. Use these; do not compute your own."""
    return q.get_financials(sector, company, periods)


@mcp.tool()
def get_valuation(sector: str, company: str) -> dict:
    """Latest market valuation: market cap, EV, EV/EBITDA, P/E, EV/sales, dividend yield, plus each
    multiple's premium/discount to the sector median (`vs_sector`)."""
    return q.get_valuation(sector, company)


@mcp.tool()
def get_company_snapshot(sector: str, companies: list[str]) -> dict:
    """ONE call for up to 12 companies: latest annual financials (+ server-computed ratios and per-field verification status), latest
    valuation (+ premium to sector median, enterprise_value_usd) and data-quality flags. Each company carries its own source_id, as_of_date,
    stale and verified. Use it instead of separate get_financials/get_valuation calls when covering several companies; use get_financials
    for multi-year history. Companies not in the sector are listed under `not_found` with a reason_code."""
    return q.get_company_snapshot(sector, companies)


@mcp.tool()
def get_signals(sector: str, company: str, signal_type: SignalType | None = None) -> dict:
    """Recent signals for a company, newest first: headcount, hiring, news headlines, contract awards.
    An empty list means a data gap, not that nothing happened."""
    return q.get_signals(sector, company, signal_type)


@mcp.tool()
def get_sector_stats(sector: str, metric: Metric) -> dict:
    """Distribution of one metric across all companies in a sector: median, q1, q3, min, max, the best company,
    and the per-company values (use this for rankings and 'vs sector' comparisons in a single call)."""
    return q.get_sector_stats(sector, metric)


@mcp.tool()
def get_defense_metrics(company: str) -> dict:
    """Defense-specific metrics (order backlog, book-to-bill, government and export revenue share) for a company
    in the defense sector. NULL means undisclosed / not sourced; order_backlog for US filers is SEC RPO, a proxy."""
    return q.get_defense_metrics(company)


@mcp.tool()
def get_data_quality(sector: str, company: str | None = None) -> dict:
    """Data-quality report: staleness flags per data type (from data_policy.yaml), verified share, known data gaps
    and caveats. Omit `company` for a sector-wide summary. Call before stating conclusions to calibrate confidence."""
    return q.get_data_quality(sector, company)


@mcp.tool()
def get_schema() -> dict:
    """Describe the database tables and columns (static metadata, not data)."""
    return q.get_schema()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.transport == "http":
        mcp.settings.host, mcp.settings.port = a.host, a.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
