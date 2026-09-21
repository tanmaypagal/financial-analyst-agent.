"""Stage 8: the judge sees an ordered per-company digest, not a raw dump cut at 40,000 characters."""
import json

from eval.judge import OLD_CUT, build_messages, digest_tool_outputs, judge_input_report
from mcp_server import queries as q


def trace_of(*results):
    return [{"tool": t, "args": a, "result": r} for t, a, r in results]


def big_result(company, tool="get_financials", n=60):
    return {"found": True, "company": company, "financials": [{"period_end": f"20{10 + i}-12-31", "revenue": 1.0e9 + i, "note": "x" * 300}
                                                              for i in range(n)], "sources": {"1": {"url": "u" * 500}}}


def test_late_tool_results_are_not_lost_when_the_old_cut_would_have_dropped_them():
    calls = [("get_financials", {"company": f"Co{i}"}, big_result(f"Co{i}")) for i in range(30)]
    calls.append(("get_data_quality", {}, {"found": True, "companies": {"Co29": {"marker": "LAST-MARKER-XYZ"}}, "policy_staleness_days": {}}))
    tr = trace_of(*calls)
    raw = json.dumps([t["result"] for t in tr], default=str)
    assert len(raw) > OLD_CUT and "LAST-MARKER-XYZ" in raw and "LAST-MARKER-XYZ" not in raw[:OLD_CUT]      # the old cut lost it
    text, stats = digest_tool_outputs(tr)
    assert "LAST-MARKER-XYZ" in text and stats["truncated"] is False
    order = [text.index(f'"Co{i}"') for i in range(30)]
    assert order == sorted(order)                                                                            # call order preserved


def test_every_company_survives_even_the_last_resort_truncation():
    tr = trace_of(*[("get_financials", {}, big_result(f"Company{i}")) for i in range(20)])
    text, stats = digest_tool_outputs(tr, max_chars=6000)
    assert stats["truncated"] is True and stats["level"] == 3 and stats["digest_chars"] < 6000 * 1.3
    assert all(f'"Company{i}"' in text for i in range(20))                                                   # nothing disappears entirely


def test_small_runs_are_untouched_and_numbers_are_not_altered_at_level_zero():
    r = q.get_valuation("logistics", "Hub Group")
    text, stats = digest_tool_outputs(trace_of(("get_valuation", {"company": "Hub Group"}, r)))
    d = json.loads(text)
    assert stats["level"] == 0 and stats["truncated"] is False and "Hub Group" in d and "sources" not in d["Hub Group"]["get_valuation"]
    assert d["Hub Group"]["get_valuation"]["valuation"]["ev_ebitda"] == float(f"{r['valuation']['ev_ebitda']:.6g}")


def test_realistic_pe_screen_groups_by_company_and_metric():
    res = [("get_sector_stats", {"metric": m}, q.get_sector_stats("logistics", m)) for m in ("ev_ebitda", "fcf_conversion")]
    for c in ("XPO", "ArcBest", "Saia"):
        res += [("get_financials", {"company": c}, q.get_financials("logistics", c, 3)), ("get_valuation", {"company": c}, q.get_valuation("logistics", c))]
    text, stats = digest_tool_outputs(trace_of(*res))
    d = json.loads(text)
    assert {"XPO", "ArcBest", "Saia", "_sector"} <= set(d) and set(d["XPO"]) == {"get_financials", "get_valuation"}
    assert "get_sector_stats(ev_ebitda)" in d["_sector"] and stats["digest_chars"] < stats["raw_chars"]


def test_build_messages_reports_digest_level_to_the_judge_and_accepts_plain_results():
    rubric = {"judge_instructions": "be strict", "scale": {"min": 1, "max": 5}, "criteria": [{"id": "g", "description": "d"}]}
    msgs, stats = build_messages("q", "pe_analyst", "answer", [{"found": True, "company": "X", "v": 1.5}], rubric)
    assert "digest level 0" in msgs[1]["content"] and '"X"' in msgs[1]["content"] and stats["level"] == 0


def test_judge_input_report_counts_how_often_the_old_cut_would_have_bitten():
    small = trace_of(("get_valuation", {}, {"found": True, "company": "A", "v": 1}))
    large = trace_of(*[("get_financials", {}, big_result(f"C{i}")) for i in range(30)])
    rep = judge_input_report([small, large, small])
    assert rep["runs"] == 3 and rep["old_cut_would_have_dropped_data"] == 1 and rep["digest_truncated_level_3"] == 0
