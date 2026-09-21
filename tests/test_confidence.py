"""Stage 5: reason codes, the confidence rule registry, resolved vs unresolved failures."""
import asyncio
import shutil
import sqlite3

import pytest

from agent import confidence as conf
from agent.confidence import Ctx, RULES, policy_confidence, rule_names
from agent.core import run_agent
from agent.models import ToolCall
from mcp_server import queries as q
from tests.fake_llm import ScriptedClient, final_msg, tool_msg

MF_OUT = {"stance": "NEUTRAL", "verdicts": [], "portfolio_fit_notes": "n/a"}


def final(companies=(), confidence="high", persona_output=None):
    return final_msg({"answer": "ok", "companies_referenced": list(companies), "data_points": [], "persona_output": persona_output or MF_OUT,
                      "confidence": confidence, "confidence_reasons": [], "data_gaps": []})


def run(steps, persona="mf_analyst", sector="defense"):
    return asyncio.run(run_agent("q", persona, sector, client=ScriptedClient(steps), model="fake"))


# ----------------------------------------------------------------------------- reason codes
def test_every_not_found_has_a_machine_readable_reason_code(tmp_path, monkeypatch):
    assert q.get_company_profile("space", "x")["reason_code"] == "UNKNOWN_SECTOR"
    assert q.get_company_profile("defense", "systems")["reason_code"] == "AMBIGUOUS"
    assert q.get_company_profile("tech", "Lockheed Martin")["reason_code"] == "WRONG_SECTOR"
    assert q.get_company_profile("defense", "Tesla")["reason_code"] == "NOT_IN_DB"
    assert q.get_sector_stats("defense", "nonsense")["reason_code"] == "BAD_ARGS"
    db = tmp_path / "t.db"
    shutil.copy(q.DB_PATH, db)
    c = sqlite3.connect(db)
    c.execute("DELETE FROM financials WHERE company_id=(SELECT id FROM companies WHERE ticker='SAIA')")
    c.commit()
    c.close()
    monkeypatch.setattr(q, "DB_PATH", db)
    assert q.get_financials("logistics", "Saia")["reason_code"] == "NO_ROWS"


# ----------------------------------------------------------------------------- resolved vs unresolved failures
def test_typo_then_recovery_is_not_forced_low():
    r = run([tool_msg([("get_company_profile", {"sector": "defense", "company": "Lockheed Martn"})]),          # typo -> NOT_IN_DB
             tool_msg([("get_valuation", {"sector": "defense", "company": "Lockheed Martin"})]),              # recovered
             final(["Lockheed Martin"])])
    assert r.tools_called[0].found is False and r.tools_called[1].found is True
    assert r.confidence != "low"
    assert any("resolved it to 'Lockheed Martin'" in x and "no penalty" in x for x in r.confidence_reasons)


def test_pure_out_of_scope_is_low():
    r = run([tool_msg([("get_company_profile", {"sector": "defense", "company": "Tesla"})]), final()])
    assert r.confidence == "low" and any("company_not_found" in x for x in r.confidence_reasons)


def test_failure_resolved_only_by_a_matching_LATER_success():
    fail = ToolCall(name="get_company_profile", args={"company": "Tesla"}, latency_ms=1.0, found=False)
    ok = ToolCall(name="get_valuation", args={"company": "Saab"}, latency_ms=1.0, found=True)
    nf = {"found": False, "reason_code": "NOT_IN_DB"}
    saab = {"found": True, "company": "Saab", "ticker": "SAAB-B.ST"}
    assert conf.failures(Ctx([nf, saab], [fail, ok], "mf_analyst"))[0][0]["arg"] == "Tesla"          # Saab does not resolve Tesla
    fail_saab = ToolCall(name="get_company_profile", args={"company": "Saab AB"}, latency_ms=1.0, found=False)
    assert conf.failures(Ctx([nf, saab], [fail_saab, ok], "mf_analyst"))[0] == []                  # suffix variant resolved
    assert conf.failures(Ctx([saab, nf], [ok, fail_saab], "mf_analyst"))[0] != []                   # success came BEFORE the failure: not resolved


def test_model_errors_are_not_treated_as_absent_data():
    r = run([tool_msg([("get_sector_stats", {"sector": "defense", "metric": "ev_ebitda"}),
                       ("get_sector_stats", {"sector": "tech", "metric": "ev_ebitda"})]),                  # WRONG_SELECTED_SECTOR (agent-side)
             final(["Saab"])])
    assert r.tools_called[1].found is False and r.confidence != "low"


# ----------------------------------------------------------------------------- YAML drives behaviour
def test_yaml_rule_names_and_code_registry_are_identical():
    assert set(rule_names()) == set(RULES), (set(rule_names()) ^ set(RULES))


def test_unknown_rule_in_yaml_fails_loudly(monkeypatch):
    bad = {"confidence_rules": {"start": "high", "downgrade_to_low_if": ["a_rule_nobody_wrote"], "downgrade_one_level_if": []}}
    monkeypatch.setattr(conf, "load_policy", lambda: bad)
    with pytest.raises(KeyError, match="a_rule_nobody_wrote"):
        policy_confidence([], [], "mf_analyst", [])


def test_editing_the_yaml_lists_changes_behaviour(monkeypatch):
    steps = lambda: [tool_msg([("get_company_profile", {"sector": "defense", "company": "Tesla"})]), final()]     # noqa: E731
    assert run(steps()).confidence == "low"                                                                   # default policy
    monkeypatch.setattr(conf, "load_policy", lambda: {"confidence_rules": {"start": "high", "downgrade_to_low_if": [], "downgrade_one_level_if": []}})
    assert run(steps()).confidence == "high"                                                                  # rules removed -> model's own confidence stands
    monkeypatch.setattr(conf, "load_policy", lambda: {"confidence_rules": {"start": "high", "downgrade_to_low_if": ["company_not_found"],
                                                                           "downgrade_one_level_if": []}})
    assert run(steps()).confidence == "low"                                                                   # only that rule listed


# ----------------------------------------------------------------------------- individual rules on real tool output
def _ctx(persona, sector, metric, companies):
    res = [q.get_sector_stats(sector, metric)]
    calls = [ToolCall(name="get_sector_stats", args={"sector": sector, "metric": metric}, latency_ms=1.0, found=True)]
    return Ctx(res, calls, persona, companies)


def test_key_metric_missing_and_more_than_half_rules():
    ctx = _ctx("equity_analyst", "logistics", "pe", ["RXO", "XPO"])                 # pe is a priority metric; RXO's P/E is null
    assert conf.key_metric_missing(ctx, None) and "RXO" in conf.key_metric_missing(ctx, None)[0]
    assert conf.more_than_half_gaps(ctx, None) == []                                # 1 of 2 is not MORE than half
    ctx.companies = ["RXO"]
    assert conf.more_than_half_gaps(ctx, None)                                      # 1 of 1
    ctx.companies = ["XPO"]
    assert conf.key_metric_missing(ctx, None) == []


def test_stale_rule_sees_a_stale_contributor_to_a_median(monkeypatch):
    monkeypatch.setenv("AGENT_TODAY", "2026-09-21")
    ctx = _ctx("equity_analyst", "logistics", "ebitda_margin", ["XPO"])
    out = conf.any_data_point_stale(ctx, None)
    assert out and "Hub Group" in out[0]


def test_non_dividend_payers_are_zero_not_a_gap():
    assert q.get_valuation("tech", "Adobe")["valuation"]["dividend_yield"] == 0.0


# ----------------------------------------------------------------------------- sector stats must expose verification (eval finding)
def test_sector_stats_report_how_many_contributing_rows_are_unverified():
    v = q.get_sector_stats("tech", "ev_ebitda")                        # market data is never filing-verified
    assert v["n_contributors"] == v["n"] == 12 and v["n_unverified_contributors"] == 12
    m = q.get_sector_stats("tech", "ebitda_margin")                    # all 12 tech companies are SEC filers -> verified rows
    assert m["n_contributors"] == 12 and m["n_unverified_contributors"] == 0
    d = q.get_sector_stats("defense", "ebitda_margin")                 # 8 of 13 defense companies are not SEC registrants
    assert d["n_contributors"] == 13 and d["n_unverified_contributors"] == 7


def test_a_sector_level_answer_on_unverified_data_can_no_longer_reach_high_confidence():
    """The eval showed money_tech_* answers rated HIGH: sector-stat results carried no verification info, so the unverified-share rule was blind."""
    r = run([tool_msg([("get_sector_stats", {"sector": "tech", "metric": "ev_ebitda"})]), final(confidence="high")], sector="tech")
    assert r.confidence == "medium" and any("unverified" in x for x in r.confidence_reasons)
    r2 = run([tool_msg([("get_sector_stats", {"sector": "tech", "metric": "ebitda_margin"})]), final(confidence="high")], sector="tech")
    assert r2.confidence == "high"                                     # all contributing rows are filing-verified: nothing to downgrade
