"""Programmatic (non-LLM) eval checks. Each check returns Check(name, passed, detail).

A `record` is {"response": AgentResponse-as-dict, "trace": [{"tool","args","result"}...]} for one agent run.
"""
import re
from dataclasses import asdict, dataclass
from typing import Any

import jsonschema

from agent.config import load_personas
from agent.grounding import (GroundingPool, answer_numbers, answer_numbers_detail, decimals_of, false_accept_rate, parse_numbers,
                             persona_numbers, scoped_false_accept_rate, synthetic_fabrications, unit_of)
from agent.models import AgentResponse
from common.naming import normalize_company

NO_DATA = re.compile(r"no (data|information|record|coverage)|not (in|covered|included|present in) (the |my |this |our )?(database|dataset|data)|"
                     r"(do(es)?n['o]t|do not|does not|cannot|can't|unable to)[^.]{0,40}(have|find|provide|contain|cover|locate)|"
                     r"not (available|found)|outside (the|of) (the )?(selected )?sector|isn't in|is not in|no such company", re.I)
STALE_WORDS = re.compile(r"stale|out[- ]of[- ]date|dated|older|not (current|recent)|as of|fiscal year 20\d\d|FY ?20\d\d|months? old|aged", re.I)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def run_pool(record) -> GroundingPool:
    """The same GroundingPool the agent uses at runtime (numeric leaves of this run's tool outputs + persona thresholds)."""
    pool = GroundingPool(extra=persona_numbers(load_personas()[record["response"]["persona"]]["thresholds"]))
    for t in record["trace"]:
        pool.add_result(t["result"])
    return pool


def fabrication_control(record, n: int = 300, seed: int = 0) -> tuple[float, float]:
    """Negative control: share of random fabricated numbers this run's pool would wrongly accept.
    Returns (no company context, scoped to one company). Both should be small; the first is the weaker path."""
    pool, fab = run_pool(record), synthetic_fabrications(n, seed)
    return false_accept_rate(pool, fab), scoped_false_accept_rate(pool, fab)


# --------------------------------------------------------------------------- checks
def check_schema(record) -> Check:
    r = record["response"]
    try:
        m = AgentResponse.model_validate(r)
        jsonschema.validate(m.persona_output, load_personas()[m.persona]["output_schema_extension"])
        need = {"answer", "persona", "sector", "companies_referenced", "data_points", "persona_output", "confidence",
                "confidence_reasons", "data_gaps", "tools_called", "model", "latency_ms"}
        missing = need - set(r)
        if missing:
            return Check("schema_valid", False, f"missing keys {missing}")
        return Check("schema_valid", True)
    except Exception as e:                                          # noqa: BLE001
        return Check("schema_valid", False, str(e)[:300])


def check_numeric_grounding(record, min_rate: float) -> Check:
    r = record["response"]
    pool = run_pool(record)
    total = len(pool.prose_numbers(r["answer"]))
    bad = [v for v, _, _ in pool.ungrounded_in(r["answer"])]
    for dp in r["data_points"]:
        v = dp.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total += 1
            if not pool.check(float(v), decimals_of(v), unit_of(dp.get("unit")), company=dp.get("company"), scoped=True):
                bad.append(float(v))
    if not total:
        return Check("numeric_grounding", True, "no numbers to ground")
    rate = 1 - len(bad) / total
    fa_all, fa_scoped = fabrication_control(record)
    return Check("numeric_grounding", rate >= min_rate,
                 f"{rate:.0%} grounded ({total - len(bad)}/{total}); ungrounded: {bad[:12]}; negative control: this run's pool would accept "
                 f"{fa_all:.0%} of random fabricated numbers with no company context, {fa_scoped:.0%} when tied to one company")


def check_no_data(record, names: list[str]) -> Check:
    r = record["response"]
    text = r["answer"] + " " + " ".join(r["data_gaps"])
    why = []
    if not NO_DATA.search(text):
        why.append("answer does not clearly say there is no data")
    if r["confidence"] != "low":
        why.append(f"confidence={r['confidence']} (want low)")
    if not any(t["result"].get("found") is False or t["tool"] == "list_companies" for t in record["trace"]):
        why.append("no tool call established absence (no found=false / list_companies)")
    for n in names:
        if any(n.lower() in (d["company"] or "").lower() for d in r["data_points"]):
            why.append(f"data_points cite {n}")
    pool = run_pool(record)
    bad = [v for v, _, _ in pool.ungrounded_in(r["answer"])]
    if bad:
        why.append(f"answer contains ungrounded numbers {bad[:6]} (possible fabrication)")
    return Check("out_of_scope_refusal", not why, "; ".join(why) or "clear no-data answer, low confidence, no fabricated numbers")


def check_tools(record, must_call=None, must_call_any=None) -> list[Check]:
    used = {t["name"] for t in record["response"]["tools_called"]}
    out = []
    if must_call:
        miss = set(must_call) - used
        out.append(Check("must_call", not miss, f"missing {sorted(miss)}; used {sorted(used)}" if miss else f"used {sorted(used)}"))
    if must_call_any:
        out.append(Check("must_call_any", bool(used & set(must_call_any)), f"want any of {must_call_any}; used {sorted(used)}"))
    return out


def check_gap_terms(record, patterns) -> Check:
    r = record["response"]
    text = r["answer"] + " " + " ".join(r["data_gaps"]) + " " + " ".join(r["confidence_reasons"])
    miss = [p for p in patterns if not re.search(p, text, re.I)]
    return Check("gap_acknowledged", not miss, f"missing pattern(s): {miss}" if miss else "gap acknowledged")


def check_max_confidence(record, cap) -> Check:
    order = ["low", "medium", "high"]
    c = record["response"]["confidence"]
    return Check("max_confidence", order.index(c) <= order.index(cap), f"confidence={c}, cap={cap}")


def check_stale_mention(record) -> Check:
    r = record["response"]
    text = r["answer"] + " " + " ".join(r["data_gaps"]) + " " + " ".join(r["confidence_reasons"])
    return Check("stale_mentioned", bool(STALE_WORDS.search(text)), "staleness/age acknowledged" if STALE_WORDS.search(text) else "no mention of age/staleness")


def check_must_reference(record, names) -> Check:
    have = " | ".join(record["response"]["companies_referenced"]).lower()
    miss = [n for n in names if n.lower() not in have]
    return Check("must_reference", not miss, f"missing {miss}" if miss else "ok")


def check_cite_headcount(record) -> Check:
    vals = [s["value_num"] for t in record["trace"] if t["tool"] == "get_signals"
            for s in t["result"].get("signals", []) if s.get("signal_type") == "headcount" and s.get("value_num")]
    if not vals:
        return Check("cite_headcount", False, "no headcount value came back from a get_signals call")
    ans = record["response"]["answer"]
    nums = parse_numbers(ans)
    hit = any(abs(n - v) < 1 or abs(n * 1000 - v) < 1 for v in vals for n in nums)
    return Check("cite_headcount", hit, f"DB headcount {vals[0]:,.0f} {'found' if hit else 'NOT found'} in the answer")


CAP_PHRASE = "exceeds practical sponsor deal size"


def check_deal_size_cap(record) -> Check:
    """PE hard cap: a candidate whose enterprise_value_usd (from this run's tool output) is above the persona's
    largest_practical_deal_ev_usd must have lbo_score <= 4, carry the blocker phrase, and not be top_pick."""
    r = record["response"]
    cap = load_personas()["pe_analyst"]["thresholds"]["largest_practical_deal_ev_usd"]
    ev: dict[str, float] = {}
    for t in record["trace"]:
        res = t["result"]
        if res.get("found") is not True:
            continue
        if isinstance(res.get("valuation"), dict) and res["valuation"].get("enterprise_value_usd") is not None:
            ev[normalize_company(res["company"])] = res["valuation"]["enterprise_value_usd"]
        if res.get("metric") == "enterprise_value_usd":
            ev.update({normalize_company(n): v for n, v in res["per_company"].items() if v is not None})
    po, why, unknown = r["persona_output"], [], []
    for c in po.get("candidates") or []:
        key = normalize_company(c["company"])
        if key not in ev:
            unknown.append(c["company"])
            continue
        if ev[key] > cap:
            if c["lbo_score"] > 4:
                why.append(f"{c['company']}: EV ${ev[key] / 1e9:.1f}B > cap but lbo_score={c['lbo_score']} (max 4)")
            if CAP_PHRASE not in " ".join(c.get("key_blockers") or []).lower():
                why.append(f"{c['company']}: EV above cap but blocker '{CAP_PHRASE}' missing")
            if po.get("top_pick") and normalize_company(po["top_pick"]) == key:
                why.append(f"{c['company']}: above the deal-size cap yet top_pick")
    note = f"; deal size not retrieved for {unknown}" if unknown else ""
    return Check("deal_size_cap", not why, ("; ".join(why) or f"no candidate above the ${cap / 1e9:.0f}B cap breaks the rule") + note)


def check_forbidden(record, patterns) -> Check:
    hits = [p for p in patterns if re.search(p, record["response"]["answer"], re.I)]
    return Check("forbidden_reasoning", not hits, f"forbidden terms present: {hits}" if hits else "none used")


# --------------------------------------------------------------------------- persona divergence
# The structural dimensions (tools, sector-stat metrics, schema fields) come straight from each persona's YAML, so they differ by
# construction and prove nothing about the ANSWERS. They are kept as information. The pass/fail test is about the answers:
# do the personas pick the same company, and do they rank the shared companies alike?
FAVOUR = {"CORE_HOLDING": 2, "HOLD_WATCH": 1, "AVOID": 0, "BUY": 2, "HOLD": 1, "SELL": 0}


def favourability(r) -> dict[str, float]:
    """{normalised company: score}: MF CORE_HOLDING=2/HOLD_WATCH=1/AVOID=0, Equity BUY/HOLD/SELL=2/1/0, PE lbo_score. Listed order is kept."""
    po, p = r["persona_output"], r["persona"]
    try:
        if p == "mf_analyst":
            return {normalize_company(v["company"]): FAVOUR[v["verdict"]] for v in po.get("verdicts", [])}
        if p == "equity_analyst":
            return {normalize_company(v["company"]): FAVOUR[v["rating"]] for v in po.get("ratings", [])}
        if p == "pe_analyst":
            return {normalize_company(v["company"]): float(v["lbo_score"]) for v in po.get("candidates", [])}
    except (KeyError, TypeError):
        pass
    return {}


def top_pick(r) -> tuple[str | None, int]:
    """(top company, number tied at the top). PE's own `top_pick` wins; otherwise the highest score (ties: first listed)."""
    po, fav = r["persona_output"], favourability(r)
    if r["persona"] == "pe_analyst" and po.get("top_pick"):
        return normalize_company(po["top_pick"]), 1
    if not fav:
        return None, 0
    best = max(fav.values())
    leaders = [c for c, s in fav.items() if s == best]
    return leaders[0], len(leaders)


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation with tie handling; None if fewer than 3 points or either side has no variation."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def _dims(rec) -> dict[str, Any]:
    r = rec["response"]
    return {"tools": {t["name"] for t in r["tools_called"]},
            "sector_stat_metrics": {t["args"].get("metric") for t in r["tools_called"] if t["name"] == "get_sector_stats"},
            "metrics_cited": {d["metric"].lower() for d in r["data_points"]},
            "schema_fields": set(r["persona_output"].keys())}


def jaccard(a: set, b: set) -> float:
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def check_divergence(records: dict[str, dict], max_rank_corr: float = 0.9) -> dict:
    """records: persona -> record for the SAME question.

    FAILS only if every persona gives the same top pick AND the mean pairwise Spearman rank correlation of their per-company
    favourability is above `max_rank_corr` (the personas are the same analyst in different clothes). Otherwise passes, and the numbers
    are reported. Rank correlation is undefined for sector-level answers with fewer than 3 shared companies: then only the top-pick
    test can speak, and a group is never failed for lack of evidence."""
    ps = list(records)
    dims = {p: _dims(records[p]) for p in ps}
    fav = {p: favourability(records[p]["response"]) for p in ps}
    tops = {p: top_pick(records[p]["response"]) for p in ps}
    pairs = {}
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            shared = [c for c in fav[a] if c in fav[b]]
            pairs[f"{a} vs {b}"] = {"n_shared": len(shared), "spearman": (None if len(shared) < 3 else
                                    spearman([fav[a][c] for c in shared], [fav[b][c] for c in shared]))}
    rhos = [v["spearman"] for v in pairs.values() if v["spearman"] is not None]
    mean_rho = sum(rhos) / len(rhos) if rhos else None
    picks = [t[0] for t in tops.values()]
    distinct = len({x for x in picks if x is not None})
    same_top = None not in picks and len(set(picks)) == 1
    failed = bool(same_top and mean_rho is not None and mean_rho > max_rank_corr)
    out = {"personas": ps, "dimensions": {}, "distinct_top_picks": distinct,
           "top_picks": {p: {"company": t[0], "tied_at_top": t[1]} for p, t in tops.items()},
           "pairwise_rank_correlation": pairs, "mean_rank_correlation": mean_rho, "all_same_top_pick": same_top,
           "fail_rule": f"all personas share the top pick AND mean rank correlation > {max_rank_corr}",
           "passed": not failed,
           "note": ("no rank correlation (fewer than 3 shared companies): verdict rests on top picks only" if mean_rho is None and len(ps) > 1 else "")}
    for d in ("tools", "sector_stat_metrics", "metrics_cited", "schema_fields"):        # informational only
        sims = [jaccard(dims[a][d], dims[b][d]) for i, a in enumerate(ps) for b in ps[i + 1:]]
        out["dimensions"][d] = {"mean_jaccard": round(sum(sims) / len(sims), 2) if sims else 1.0,
                                "values": {p: sorted(map(str, dims[p][d])) for p in ps}}
    return out


# --------------------------------------------------------------------------- entry point
def run_case_checks(case: dict, record: dict, cfg: dict, forbidden: dict) -> list[Check]:
    e = case.get("expect", {})
    checks = [check_schema(record)]
    if record["response"]["persona"] != case["persona"] or record["response"]["sector"] != case["sector"]:
        checks.append(Check("persona_sector_echo", False, "response persona/sector do not match request"))
    checks.append(check_numeric_grounding(record, cfg["grounding_min_rate"]))
    if e.get("no_data_for"):
        checks.append(check_no_data(record, e["no_data_for"]))
    checks += check_tools(record, e.get("must_call"), e.get("must_call_any"))
    if e.get("expect_gap_terms"):
        checks.append(check_gap_terms(record, e["expect_gap_terms"]))
    if e.get("max_confidence"):
        checks.append(check_max_confidence(record, e["max_confidence"]))
    if e.get("expect_stale_mention"):
        checks.append(check_stale_mention(record))
    if e.get("must_reference"):
        checks.append(check_must_reference(record, e["must_reference"]))
    if e.get("cite_headcount"):
        checks.append(check_cite_headcount(record))
    if case["persona"] == "pe_analyst":
        checks.append(check_deal_size_cap(record))
    if forbidden.get(case["persona"]):
        checks.append(check_forbidden(record, forbidden[case["persona"]]))
    return checks
