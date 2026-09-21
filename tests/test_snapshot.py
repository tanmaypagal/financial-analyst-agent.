"""Stage 9: get_company_snapshot - one typed, read-only call for up to 12 companies, without opening loopholes in the guardrails."""
import asyncio
import json

import pytest

from agent.confidence import Ctx, failures
from agent.core import run_agent
from agent.grounding import GroundingPool
from agent.mcp_client import MCPToolClient
from agent.models import ToolCall
from eval.judge import digest_tool_outputs
from mcp_server import queries as q
from tests.fake_llm import ScriptedClient, final_msg, tool_msg

LOG = [c["ticker"] for c in q.list_companies("logistics")["companies"]]


def test_snapshot_returns_latest_financials_valuation_and_flags_with_per_company_provenance():
    r = q.get_company_snapshot("logistics", ["XPO", "Hub Group", "ArcBest"])
    assert r["found"] and [e["company"] for e in r["companies"]] == ["XPO", "Hub Group", "ArcBest"] and r["not_found"] == []
    for e in r["companies"]:
        assert {"source_id", "as_of_date", "stale", "verified"} <= set(e)                          # per company
        assert e["financials"]["derived"]["fcf_conversion"] is not None and e["valuation"]["enterprise_value_usd"] > 0
        assert set(e["data_quality"]) >= {"financials_stale", "valuation_stale", "n_data_gaps", "caveats"}
        assert e["financials"]["verified_fields"] is not None and e["financials"]["unverified_fields"] is not None and e["valuation"]["vs_sector"]
    hub = next(e for e in r["companies"] if e["company"] == "Hub Group")
    assert hub["stale"] is True and hub["financials"]["period_end"] == "2024-12-31"
    assert r["stale"] is True and r["stale_companies"] == ["Hub Group"] and r["n_stale"] == 1               # top level aggregates
    assert r["sources"] and all(sid in r["sources"] for sid in r["source_ids"])


def test_snapshot_values_are_identical_to_the_per_company_tools():
    snap = {e["ticker"]: e for e in q.get_company_snapshot("logistics", LOG)["companies"]}
    for t in ("XPO", "SAIA", "DSV.CO"):
        f = q.get_financials("logistics", t, 2)["financials"][0]
        v = q.get_valuation("logistics", t)
        e = snap[t]
        assert e["financials"]["revenue"] == f["revenue"] and e["financials"]["fcf"] == f["fcf"] and e["financials"]["derived"] == f["derived"]
        assert e["financials"]["verified_fields"] == {x["field"]: x["status"] for x in f["verified_fields"]}
        assert e["financials"]["unverified_fields"] == f["unverified_fields"]
        assert e["valuation"]["ev_ebitda"] == v["valuation"]["ev_ebitda"] and e["valuation"]["vs_sector"] == v["vs_sector"]
        assert e["valuation"]["enterprise_value_usd"] == v["valuation"]["enterprise_value_usd"]


def test_snapshot_enforces_sector_uses_the_same_lookup_rules_and_reports_partial_failures():
    r = q.get_company_snapshot("logistics", ["XPO", "Tesla", "Oracle Corp", "xpo"])
    assert [e["company"] for e in r["companies"]] == ["XPO"]                                     # duplicate 'xpo' collapsed
    assert {m["requested"]: m["reason_code"] for m in r["not_found"]} == {"Tesla": "NOT_IN_DB", "Oracle Corp": "WRONG_SECTOR"}
    d = q.get_company_snapshot("defense", ["General Dynamics versus Northrop Grumman"])
    assert d["found"] is False and d["not_found"][0]["reason_code"] == "AMBIGUOUS"                  # ambiguity refused, not guessed
    assert q.get_company_snapshot("defense", ["Tesla", "Boeing"])["reason_code"] == "NOT_IN_DB"     # nothing found -> found:false


@pytest.mark.parametrize("bad", [[], "XPO", None, [""], [1, 2]])
def test_snapshot_rejects_bad_arguments(bad):
    r = q.get_company_snapshot("logistics", bad)
    assert r["found"] is False and r["reason_code"] == "BAD_ARGS"
    assert q.get_company_snapshot("space", ["XPO"])["reason_code"] == "UNKNOWN_SECTOR"


def test_snapshot_is_capped_at_twelve_and_says_what_it_ignored():
    r = q.get_company_snapshot("logistics", LOG + ["Extra Co", "Another Co"])
    assert len(r["companies"]) == 12 and r["ignored_over_limit"] == ["Extra Co", "Another Co"]


def test_snapshot_is_much_smaller_than_the_equivalent_separate_calls():
    snap = len(json.dumps(q.get_company_snapshot("logistics", LOG)))
    sep = sum(len(json.dumps(q.get_financials("logistics", t, 3))) + len(json.dumps(q.get_valuation("logistics", t))) for t in LOG)
    assert snap < 0.6 * sep, (snap, sep)


def test_snapshot_is_a_typed_readonly_tool_over_a_real_mcp_session():
    async def go():
        async with MCPToolClient() as c:
            tool = next(t for t in c.tools if t.name == "get_company_snapshot")
            out, _ = await c.call("get_company_snapshot", {"sector": "defense", "companies": ["Saab", "Rheinmetall", "Tesla"]})
            return tool, out
    tool, out = asyncio.run(go())
    props = tool.inputSchema["properties"]
    assert props["companies"]["type"] == "array" and props["companies"]["items"]["type"] == "string" and set(tool.inputSchema["required"]) == {"sector", "companies"}
    assert out["found"] and {e["ticker"] for e in out["companies"]} == {"SAAB-B.ST", "RHM.DE"} and out["not_found"][0]["requested"] == "Tesla"


# ----------------------------------------------------------------------------- guardrails still bite
def test_agent_sector_guard_covers_the_snapshot_tool():
    c = ScriptedClient([tool_msg([("get_company_snapshot", {"sector": "tech", "companies": ["Microsoft"]})]),
                        final_msg({"answer": "no data", "companies_referenced": [], "data_points": [], "confidence": "high",
                                   "persona_output": {"stance": "NEUTRAL", "verdicts": [], "portfolio_fit_notes": "n/a"},
                                   "confidence_reasons": [], "data_gaps": []})])
    trace = []
    asyncio.run(run_agent("q", "mf_analyst", "defense", client=c, model="fake", trace=trace))
    assert trace[0]["result"]["reason_code"] == "WRONG_SELECTED_SECTOR"


def test_grounding_is_scoped_per_company_for_snapshot_entries():
    pool = GroundingPool()
    pool.add_result(q.get_company_snapshot("logistics", ["XPO", "Hub Group"]))
    xpo = q.get_financials("logistics", "XPO", 1)["financials"][0]["revenue"]
    text = f"${xpo / 1e9:.1f}B"
    assert pool.check_text(text, company="XPO", scoped=True) and not pool.check_text(text, company="Hub Group", scoped=True)
    hub_ev = q.get_valuation("logistics", "Hub Group")["valuation"]["ev_ebitda"]
    assert pool.check_text(f"{hub_ev:.1f}x", company="Hub Group", scoped=True)


def test_missing_snapshot_companies_lower_confidence_unless_a_later_call_resolves_them():
    snap = q.get_company_snapshot("logistics", ["XPO", "Tesla"])
    call = ToolCall(name="get_company_snapshot", args={"sector": "logistics", "companies": ["XPO", "Tesla"]}, latency_ms=1.0, found=True)
    unresolved, resolved = failures(Ctx([snap], [call], "mf_analyst"))
    assert [u["arg"] for u in unresolved] == ["Tesla"] and resolved == []
    later = q.get_company_snapshot("logistics", ["XPO", "Hub Group"])
    call2 = ToolCall(name="get_company_snapshot", args={"sector": "logistics", "companies": ["XPO", "Hub Grp"]}, latency_ms=1.0, found=True)
    typo = q.get_company_snapshot("logistics", ["XPO", "Hub Grp"])                                # typo: NOT_IN_DB...
    unresolved, resolved = failures(Ctx([typo, later], [call2, call], "mf_analyst"))               # ...resolved by a later snapshot
    assert unresolved == [] and resolved and resolved[0]["resolved_to"] == "Hub Group"


def test_judge_digest_splits_a_snapshot_into_per_company_entities():
    snap = q.get_company_snapshot("logistics", ["XPO", "Hub Group"])
    text, stats = digest_tool_outputs([{"tool": "get_company_snapshot", "args": {}, "result": snap}])
    d = json.loads(text)
    assert {"XPO", "Hub Group", "_sector"} <= set(d) and "get_company_snapshot" in d["XPO"] and stats["level"] == 0


def test_prompt_prefers_the_snapshot_and_the_tool_reaches_the_model():
    from agent.config import build_system_prompt
    assert "get_company_snapshot" in build_system_prompt("pe_analyst", "logistics")
    c = ScriptedClient([final_msg({"answer": "x", "companies_referenced": [], "data_points": [], "confidence": "low",
                                   "persona_output": {"stance": "NEUTRAL", "verdicts": [], "portfolio_fit_notes": "n/a"},
                                   "confidence_reasons": [], "data_gaps": []})])
    asyncio.run(run_agent("q", "mf_analyst", "defense", client=c, model="fake"))
    assert "get_company_snapshot" in {t["function"]["name"] for t in c.requests[0]["tools"]}


def test_key_metric_missing_also_sees_snapshot_entries():
    """Regression found by the eval: RXO's null P/E stopped lowering confidence once the model used the snapshot instead of sector stats."""
    from agent import confidence as conf
    snap = q.get_company_snapshot("logistics", ["RXO", "XPO"])
    call = ToolCall(name="get_company_snapshot", args={"sector": "logistics", "companies": ["RXO", "XPO"]}, latency_ms=1.0, found=True)
    ctx = Ctx([snap], [call], "equity_analyst", ["RXO", "XPO"])                    # equity priority metrics include pe
    out = conf.key_metric_missing(ctx, None)
    assert out and "RXO" in out[0] and "pe" in out[0] and "XPO" not in out[0]
    ctx.companies = ["XPO"]
    assert conf.key_metric_missing(ctx, None) == []
