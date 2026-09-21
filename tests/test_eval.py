from pathlib import Path

import pytest
import yaml

from eval import checks as C
from eval.judge import _check_models, load_rubric
from eval.run_eval import dry_run


def rec(answer="", persona="mf_analyst", trace=None, conf="low", dps=None, gaps=None, tools=None, po=None):
    return {"response": {"answer": answer, "persona": persona, "sector": "defense", "companies_referenced": [], "data_points": dps or [],
                         "persona_output": po if po is not None else {}, "confidence": conf, "confidence_reasons": [], "data_gaps": gaps or [],
                         "tools_called": tools or [], "model": "m", "latency_ms": 1.0}, "trace": trace or []}


NF = [{"tool": "get_company_profile", "args": {}, "result": {"found": False, "reason": "No company"}}]


def test_no_data_check_passes_on_honest_refusal_and_fails_on_bluff():
    assert C.check_no_data(rec("I have no data on Tesla in this database.", trace=NF), ["Tesla"]).passed
    assert not C.check_no_data(rec("Tesla trades at 45.2x EV/EBITDA and looks great.", trace=NF), ["Tesla"]).passed
    assert not C.check_no_data(rec("No data on Tesla.", conf="high", trace=NF), ["Tesla"]).passed


def test_numeric_grounding_allows_rescaling_but_catches_invention():
    tr = [{"tool": "get_valuation", "args": {}, "result": {"ev_ebitda": 14.447, "market_cap": 123099168768, "dividend_yield": 0.0259}}]
    ok = rec("EV/EBITDA is 14.4x, market cap $123.1bn, yield 2.59%.", trace=tr)
    assert C.check_numeric_grounding(ok, 0.85).passed
    bad = rec("EV/EBITDA is 14.4x but revenue is $88.3bn and growth 17.2%.", trace=tr)
    assert not C.check_numeric_grounding(bad, 0.85).passed


def test_answer_numbers_ignores_years_scores_and_list_markers():
    assert C.answer_numbers("## 1. View\n1. FY2025 in 2026-09-21 scored 7/10, top 3") == []


def _po(persona, **po):
    return {"response": {"persona": persona, "persona_output": po, "tools_called": [], "data_points": []}, "trace": []}


COS = ["Alpha", "Beta", "Gamma", "Delta"]


def _mf(v):
    return _po("mf_analyst", verdicts=[{"company": c, "verdict": x} for c, x in zip(COS, v)])


def _eq(v):
    return _po("equity_analyst", ratings=[{"company": c, "rating": x} for c, x in zip(COS, v)])


def _pe(scores, top):
    return _po("pe_analyst", top_pick=top, candidates=[{"company": c, "lbo_score": x} for c, x in zip(COS, scores)])


def test_spearman_rank_correlation():
    assert C.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert C.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert C.spearman([2, 2, 1, 0], [9, 9, 5, 1]) == pytest.approx(1.0)              # ties share average ranks
    assert C.spearman([1, 2], [1, 2]) is None and C.spearman([1, 1, 1], [1, 2, 3]) is None   # too few points / no variation


def test_top_pick_and_favourability_use_scores_not_listing_order():
    r = _mf(["AVOID", "CORE_HOLDING", "HOLD_WATCH", "CORE_HOLDING"])["response"]
    assert C.favourability(r) == {"alpha": 0, "beta": 2, "gamma": 1, "delta": 2}
    assert C.top_pick(r) == ("beta", 2)                                              # highest score, 2 tied at the top
    assert C.top_pick(_pe([3, 9, 2, 1], "Delta")["response"]) == ("delta", 1)         # PE's own top_pick wins


def test_divergence_fails_when_personas_share_the_top_pick_and_rank_alike():
    same_view = {"mf_analyst": _mf(["CORE_HOLDING", "HOLD_WATCH", "AVOID", "AVOID"]),
                 "equity_analyst": _eq(["BUY", "HOLD", "SELL", "SELL"]),
                 "pe_analyst": _pe([9, 5, 2, 1], "Alpha")}
    d = C.check_divergence(same_view, 0.9)
    assert d["passed"] is False and d["distinct_top_picks"] == 1 and d["mean_rank_correlation"] > 0.9


def test_divergence_passes_when_top_picks_differ_and_shows_the_numbers():
    d = C.check_divergence({"mf_analyst": _mf(["CORE_HOLDING", "HOLD_WATCH", "AVOID", "AVOID"]),
                            "equity_analyst": _eq(["BUY", "HOLD", "SELL", "SELL"]),
                            "pe_analyst": _pe([2, 9, 5, 1], "Beta")}, 0.9)
    assert d["passed"] is True and d["distinct_top_picks"] == 2
    assert d["top_picks"]["pe_analyst"]["company"] == "beta" and "mf_analyst vs pe_analyst" in d["pairwise_rank_correlation"]


def test_divergence_passes_when_same_top_pick_but_rankings_disagree():
    d = C.check_divergence({"mf_analyst": _mf(["CORE_HOLDING", "HOLD_WATCH", "HOLD_WATCH", "AVOID"]),
                            "pe_analyst": _pe([9, 1, 2, 8], "Alpha")}, 0.9)          # same top pick, rho well below 0.9
    assert d["all_same_top_pick"] is True and d["mean_rank_correlation"] < 0.9 and d["passed"] is True


def test_divergence_is_not_failed_for_lack_of_evidence():
    d = C.check_divergence({"mf_analyst": _po("mf_analyst", stance="NEUTRAL", verdicts=[]),
                            "pe_analyst": _po("pe_analyst", no_clean_target=True, candidates=[])}, 0.9)
    assert d["passed"] is True and d["mean_rank_correlation"] is None and "rests on top picks only" in d["note"]


def test_forbidden_terms_and_confidence_cap():
    assert not C.check_forbidden(rec("Our price target is 200."), ["price target"]).passed
    assert not C.check_max_confidence(rec(conf="high"), "medium").passed


def test_judge_guards():
    with pytest.raises(ValueError):
        load_rubric("template")                                    # placeholders remain
    with pytest.raises(ValueError):
        _check_models("gpt-x", "gpt-x")                            # must differ from the agent model
    _check_models("judge-model", "agent-model")


def test_cases_file_is_consistent():
    assert dry_run(yaml.safe_load((Path("eval") / "cases.yaml").read_text(encoding="utf-8"))) == 0


# ----------------------------------------------------------------------------- PE deal-size cap
def _pe_run(cands, top, ev):
    trace = [{"tool": "get_sector_stats", "args": {}, "result": {"found": True, "metric": "enterprise_value_usd", "per_company": ev}}]
    r = _po("pe_analyst", top_pick=top, candidates=cands)
    r["trace"] = trace
    return r


def _cand(name, score, blockers=()):
    return {"company": name, "lbo_score": score, "operational_thesis": "t", "key_blockers": list(blockers)}


def test_deal_size_cap_passes_when_oversize_names_are_capped_and_flagged():
    ok = _pe_run([_cand("Big Co", 3, ["exceeds practical sponsor deal size"]), _cand("Small Co", 8)], "Small Co",
                 {"Big Co": 139e9, "Small Co": 2.4e9})
    assert C.check_deal_size_cap(ok).passed


def test_deal_size_cap_fails_on_high_score_missing_phrase_or_oversize_top_pick():
    bad = _pe_run([_cand("Big Co", 9), _cand("Small Co", 8)], "Big Co", {"Big Co": 139e9, "Small Co": 2.4e9})
    d = C.check_deal_size_cap(bad).detail
    assert not C.check_deal_size_cap(bad).passed and "max 4" in d and "blocker" in d and "top_pick" in d
    boundary = _pe_run([_cand("Edge Co", 8)], "Edge Co", {"Edge Co": 25e9})            # exactly the cap is NOT above it
    assert C.check_deal_size_cap(boundary).passed
