"""Negative control for numeric grounding: a check that accepts random numbers is not a check.

Builds a REALISTIC pool by calling the query layer for a full PE screen (sector stats for 6 metrics + financials, valuation, profile and
data-quality for 5 companies), then measures (a) how often random FABRICATED numbers are accepted and (b) how often genuine
tool-derived numbers, formatted the ways an LLM writes them, are accepted. Run with `pytest -s tests/test_grounding.py` to see the rates.
"""
import asyncio
import re

import pytest

from agent.grounding import (GroundingPool, false_accept_rate, parse_display, scoped_false_accept_rate, synthetic_fabrications, _leaves)
from mcp_server import queries as q
from tests.fake_llm import ScriptedClient, final_msg, tool_msg

SECTOR = "logistics"
METRICS = ["ev_ebitda", "net_debt_to_ebitda", "total_debt_to_ebitda", "fcf_conversion", "capex_to_revenue", "ebitda_margin"]
COMPANIES = ["XPO", "Hub Group", "ArcBest", "Saia", "Werner"]


# ----------------------------------------------------------------------------- the OLD algorithm, kept only to report the BEFORE rate
_OLD_SCALES = (1, 100, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 1e12, 1e-12)


def _old_numbers(obj, pool):
    if isinstance(obj, bool) or obj is None:
        return pool
    if isinstance(obj, (int, float)):
        pool.append(float(obj))
    elif isinstance(obj, str):
        pool.extend(float(m.replace(",", "")) for m in re.findall(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*", obj) if m.replace(",", "").replace(".", "", 1).isdigit())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _old_numbers(v, pool)
            if isinstance(k, str) and k.isdigit():
                pool.append(float(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _old_numbers(v, pool)
    return pool


def _old_grounded(x, pool, rel=0.02):
    if x == 0:
        return True
    return any(abs(p * s - x) <= rel * abs(x) + 1e-9 or abs(abs(p * s) - abs(x)) <= rel * abs(x) + 1e-9 for p in pool if p != 0 for s in _OLD_SCALES)


# ----------------------------------------------------------------------------- realistic PE-screen pool
@pytest.fixture(scope="module")
def results():
    res = [q.get_sector_stats(SECTOR, m) for m in METRICS]
    for c in COMPANIES:
        res += [q.get_financials(SECTOR, c, 3), q.get_valuation(SECTOR, c), q.get_company_profile(SECTOR, c), q.get_data_quality(SECTOR, c)]
    assert all(r["found"] for r in res)
    return res


@pytest.fixture(scope="module")
def pool(results):
    p = GroundingPool()
    for r in results:
        p.add_result(r)
    return p


def fmt(v: float) -> list[str]:
    """The ways an LLM might present a tool number."""
    a = abs(v)
    if a >= 1e9:
        return [f"${v / 1e9:.1f}B", f"{v / 1e6:,.0f} million"]
    if a >= 1e6:
        return [f"{v / 1e6:,.0f} million", f"${v / 1e6:.1f}M"]
    if a <= 1.0:
        return [f"{v * 100:.1f}%", f"{v:.3f}"]
    return [f"{v:.1f}x", f"{v:.2f}x"]


def test_negative_control_false_accept_and_genuine_accept(results, pool):
    seeds = range(6)                                                 # 6 x 2000 random fabricated numbers: a single draw varies by ~+-0.6pt
    fabs = [synthetic_fabrications(2000, seed=sd) for sd in seeds]
    fab = fabs[0]
    # BEFORE: old algorithm, old pool (numbers parsed from strings, ids, dates included)
    old_pool: list[float] = []
    for r in results:
        _old_numbers(r, old_pool)
    old_rate = sum(_old_grounded(parse_display(t)[0][0], old_pool) for t in fab) / len(fab)
    # AFTER
    per_seed = [scoped_false_accept_rate(pool, f) for f in fabs]     # number claimed for one company (data_point / one-company line)
    scoped = sum(per_seed) / len(per_seed)
    global_ = false_accept_rate(pool, fab)                          # no company context at all (weakest path)
    genuine = [(c, s) for c, vals in pool.by_company.items() for v in vals if v != 0 for s in fmt(v)]
    accepted = sum(bool(pool.check_text(s, company=c, scoped=True)) for c, s in genuine) / len(genuine)
    print(f"\n[grounding negative control] pool: {len(pool.shared)} shared + {sum(map(len, pool.by_company.values()))} company numbers"
          f"\n  BEFORE (old algorithm)                 false-accept {old_rate:.1%}"
          f"\n  AFTER  scoped to one company           false-accept {scoped:.1%}"
          f"\n  AFTER  no company context              false-accept {global_:.1%}   (weakest path; prose lines naming one company use the scoped rule)"
          f"\n  AFTER  genuine numbers accepted        {accepted:.1%}  (n={len(genuine)}: 14.4x, 12.3%, $75.0B, 75,048 million, ...)")
    assert old_rate > 0.90                                          # documents why the old check was uninformative
    assert scoped < 0.05                                            # (a) required (borderline: see README / final report)
    assert accepted >= 0.98                                         # (b) required
    assert global_ < 0.20                                           # regression guard for the weak path, not a claim of strength


def test_pool_excludes_ids_years_dates_and_string_numbers():
    got = list(_leaves({"source_id": 27, "id": 5, "source_ids": [1, 2], "verified": 1, "note": "raised 9999.5 in 2024 on 2026-09-21",
                        "period_end": "2025-12-31", "ev_ebitda": 14.447, "nested": {"revenue": 75048000000, "flag": True}}))
    assert sorted(got) == [14.447, 75048000000.0]


def test_scoping_rejects_another_companys_number(pool):
    xpo_rev = q.get_financials(SECTOR, "XPO", 1)["financials"][0]["revenue"]
    text = f"${xpo_rev / 1e9:.1f}B"
    assert pool.check_text(text, company="XPO", scoped=True)
    assert not pool.check_text(text, company="Hub Group", scoped=True)          # XPO's revenue is not Hub Group's number
    assert not pool.check_text(text, company="Tesla", scoped=True)              # unknown company: only sector-level numbers count


def test_unit_aware_transformations(pool):
    hg = q.get_valuation(SECTOR, "Hub Group")["valuation"]["ev_ebitda"]
    assert pool.check_text(f"{hg:.1f}x", "Hub Group", True)
    assert not pool.check_text(f"{hg * 100:.1f}x", "Hub Group", True)           # 'x' never rescales
    frac = q.get_financials(SECTOR, "Hub Group", 1)["financials"][0]["ebitda_margin"]
    assert pool.check_text(f"{frac * 100:.1f}%", "Hub Group", True)             # fraction -> percent
    assert not pool.check_text(f"{frac * 100:.1f}B", "Hub Group", True)         # ... but not billions
    assert not pool.check_text(f"-{hg:.1f}x", "Hub Group", True)                # signs are not flipped


def test_prose_is_checked_against_the_company_named_on_the_line(pool):
    hg = q.get_valuation(SECTOR, "Hub Group")["valuation"]["ev_ebitda"]
    xpo = q.get_valuation(SECTOR, "XPO")["valuation"]["ev_ebitda"]
    assert pool.ungrounded_in(f"- Hub Group: EV/EBITDA {hg:.1f}x") == []
    bad = pool.ungrounded_in(f"- Hub Group: EV/EBITDA {xpo:.1f}x")               # XPO's multiple attributed to Hub Group
    assert len(bad) == 1


# ----------------------------------------------------------------------------- runtime behaviour in the agent
def _run(final_obj, calls):
    from agent.core import run_agent
    c = ScriptedClient([tool_msg(calls), final_msg(final_obj)])
    return asyncio.run(run_agent("q", "pe_analyst", "logistics", client=c, model="fake"))


PE_OUT = {"no_clean_target": False, "top_pick": "Hub Group",
          "candidates": [{"company": "Hub Group", "lbo_score": 6, "operational_thesis": "t", "key_blockers": []}]}


def test_runtime_grounds_string_data_points_and_prose_numbers():
    v = q.get_valuation("logistics", "Hub Group")["valuation"]
    good = f"{v['ev_ebitda']:.1f}x"
    obj = {"answer": f"Thesis\n- Hub Group trades at {good} and 987.6x sales.\n", "companies_referenced": ["Hub Group"],
           "data_points": [{"company": "Hub Group", "metric": "ev_ebitda", "value": good},
                           {"company": "Hub Group", "metric": "ev_sales", "value": "$999.9B"},
                           {"company": "Hub Group", "metric": "trend", "value": "improving"}],
           "persona_output": PE_OUT, "confidence": "high", "confidence_reasons": [], "data_gaps": []}
    r = _run(obj, [("get_valuation", {"sector": "logistics", "company": "Hub Group"})])
    assert [d.metric for d in r.data_points] == ["ev_ebitda"]                     # fabricated + non-numeric strings dropped
    gaps = " | ".join(r.data_gaps)
    assert "Removed ungrounded data points" in gaps and "Removed non-numeric" in gaps
    assert "987.6x" in gaps and "not found in any tool result" in gaps               # prose number reported, prose not rewritten
    assert "987.6x" in r.answer and r.confidence == "medium"
    assert any("could not be traced" in x for x in r.confidence_reasons)


def test_prose_parser_regressions():
    """Cases found by re-grading real model answers: these must not be treated as data numbers, and decimals must not be eaten."""
    from agent.grounding import answer_numbers_detail as nums
    assert nums("- 2025‑12‑31: $268B (source 232), sources 27, 30 and 31") == [(268.0, 0, "B")]   # non-breaking hyphens, citations
    assert nums("Oracle (+7.997 pp)") == [(7.997, 3, "%")]                                                     # percentage points
    assert nums("converts at 1.6–2.3x") == [(1.6, 1, None), (2.3, 1, "x")]                              # decimal range not eaten
    assert nums("scored 7/10; range 15-20") == []                                                             # scores and integer ranges skipped
