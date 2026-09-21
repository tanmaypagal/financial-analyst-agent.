"""Confidence policy: every rule named in config/data_policy.yaml is a registered function; the YAML lists decide which run.

  confidence_rules:
    start: high
    downgrade_to_low_if:        rules run in order; ANY that triggers -> confidence "low"
    downgrade_one_level_if:     otherwise, ANY that triggers -> one level below `start`
A list item is either a rule name or {rule_name: parameter}. tests/test_confidence.py fails if the YAML names a rule with no
implementation here, or this module registers a rule the YAML does not list.
"""
import difflib
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.config import load_personas, load_policy
from agent.models import ToolCall
from common.naming import normalize_company

LEVELS = ["low", "medium", "high"]
META_TOOLS = {"get_schema", "list_sectors", "list_companies"}
# found:false reasons that mean "what you asked about is not available". Others (BAD_ARGS, UNKNOWN_SECTOR, WRONG_SELECTED_SECTOR) are
# the model's own mistakes: listed as reasons, but they say nothing about the data.
ABSENCE_CODES = {"NOT_IN_DB", "WRONG_SECTOR", "NO_ROWS", "AMBIGUOUS"}
RULES: dict[str, Callable[["Ctx", Any], list[str]]] = {}


def rule(name: str):
    def deco(fn):
        RULES[name] = fn
        return fn
    return deco


@dataclass
class Ctx:
    results: list[dict]
    calls: list[ToolCall]
    persona: str
    companies: list[str] = field(default_factory=list)          # companies_referenced by the answer

    @property
    def data_results(self) -> list[dict]:
        return [r for c, r in zip(self.calls, self.results) if c.name not in META_TOOLS and c.name != "get_data_quality"]

    @property
    def priority_metrics(self) -> list[str]:
        return list(load_personas()[self.persona]["priority_metrics"].get("sector_stats") or [])


# ------------------------------------------------------------------------------------------------ helpers
def _returned_names(r: dict) -> list[str]:
    prof = r.get("profile") or {}
    names = [x for x in (r.get("company"), r.get("ticker"), prof.get("name"), prof.get("ticker")) if isinstance(x, str)]
    for e in r.get("companies") if isinstance(r.get("companies"), list) else []:          # get_company_snapshot entries
        names += [x for x in (e.get("company"), e.get("ticker")) if isinstance(x, str)]
    return names


def _same_company(a: str, b: str) -> bool:
    """name/ticker contains, or is contained in, the other (after dropping corporate suffixes); a close spelling (typo) also counts."""
    x, y = normalize_company(a), normalize_company(b)
    if len(x) < 3 or len(y) < 3:
        return x == y
    return x in y or y in x or difflib.SequenceMatcher(None, x, y).ratio() >= 0.8


def failures(ctx: Ctx) -> tuple[list[dict], list[dict]]:
    """(unresolved, resolved) absence failures. A NOT_IN_DB / WRONG_SECTOR / NO_ROWS / AMBIGUOUS failure is RESOLVED if a LATER successful
    call in the same run returned a company matching the failed argument."""
    unresolved, resolved = [], []
    for i, (c, r) in enumerate(zip(ctx.calls, ctx.results)):
        if isinstance(r.get("not_found"), list) and r["not_found"]:                        # snapshot: some requested companies missing
            items = [{"tool": c.name, "arg": n.get("requested"), "code": n.get("reason_code")} for n in r["not_found"]]
        elif r.get("found") is False:
            items = [{"tool": c.name, "arg": c.args.get("company"), "code": r.get("reason_code")}]
        else:
            items = []
        later = [x for x in ctx.results[i + 1:] if x.get("found") is True]
        for item in items:
            if item["code"] not in ABSENCE_CODES:
                continue
            arg = item["arg"]
            match = next((n for x in later for n in _returned_names(x) if arg and _same_company(arg, n)), None)
            (resolved if match else unresolved).append({**item, "resolved_to": match})
    return unresolved, resolved


def _null_priority(ctx: Ctx) -> dict[str, list[str]]:
    """{referenced company: [priority metrics that are null for it]} from this run's get_sector_stats results."""
    out: dict[str, list[str]] = {}
    for r in ctx.results:
        if r.get("found") is True and r.get("metric") in ctx.priority_metrics and isinstance(r.get("per_company"), dict):
            for name, v in r["per_company"].items():
                if v is None and any(_same_company(name, c) for c in ctx.companies):
                    out.setdefault(name, []).append(r["metric"])
        for e in r.get("companies") if isinstance(r.get("companies"), list) else []:      # get_company_snapshot entries
            name = e.get("company")
            if not name or not any(_same_company(name, c) for c in ctx.companies):
                continue
            fin = e.get("financials") or {}
            for m in ctx.priority_metrics:
                for src in (e.get("valuation") or {}, fin.get("derived") or {}, fin):     # where a sector-stat metric lives in an entry
                    if m in src:
                        if src[m] is None and m not in out.get(name, []):
                            out.setdefault(name, []).append(m)
                        break
    return out


def _walk(obj, fn):
    if isinstance(obj, dict):
        fn(obj)
        for v in obj.values():
            _walk(v, fn)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, fn)


# ------------------------------------------------------------------------------------------------ the rules (names = YAML keys)
@rule("any_data_point_stale")
def any_data_point_stale(ctx: Ctx, _):
    tools = sorted({c.name for c, r in zip(ctx.calls, ctx.results) if r.get("stale") is True and c.name not in META_TOOLS | {"get_data_quality"}})
    stale_cos = sorted({n for r in ctx.data_results for n in (r.get("stale_companies") or [])})
    if not tools:
        return []
    return ["some data used is stale per data_policy.yaml (" + ", ".join(tools) + (f"; stale contributors: {', '.join(stale_cos)}" if stale_cos else "") + ")"]


@rule("any_data_point_unverified_share_above")
def unverified_share(ctx: Ctx, threshold):
    stats = {"n": 0, "unver": 0}

    def scan(d):
        if isinstance(d.get("verified"), int) and not isinstance(d.get("verified"), bool):
            stats["n"] += 1
            stats["unver"] += d["verified"] == 0
        if isinstance(d.get("n_contributors"), int):                      # get_sector_stats: contributing rows and how many are unverified
            stats["n"] += d["n_contributors"]
            stats["unver"] += d.get("n_unverified_contributors") or 0
    for r in ctx.data_results:
        _walk(r, scan)
    if stats["n"] and stats["unver"] / stats["n"] > float(threshold if threshold is not None else 0.5):
        return [f"{stats['unver']}/{stats['n']} data rows are unverified (verified=0), above the {threshold} share limit"]
    return []


@rule("key_metric_missing")
def key_metric_missing(ctx: Ctx, _):
    miss = _null_priority(ctx)
    return [f"priority metric(s) missing for cited companies: " + "; ".join(f"{n}: {', '.join(m)}" for n, m in miss.items())] if miss else []


@rule("company_not_found")
def company_not_found(ctx: Ctx, _):
    unresolved, _resolved = failures(ctx)
    return [f"{u['code']}: '{u['arg']}' was not found and no later call resolved it ({u['tool']})" for u in unresolved]


@rule("no_tool_returned_data")
def no_tool_returned_data(ctx: Ctx, _):
    data_calls = [c for c in ctx.calls if c.name not in META_TOOLS]
    return [] if any(c.found is True for c in data_calls) else ["no tool returned data for the question"]


@rule("more_than_half_of_cited_companies_have_gaps_in_priority_metrics")
def more_than_half_gaps(ctx: Ctx, _):
    cited = {normalize_company(c) for c in ctx.companies if c}
    if not cited:
        return []
    gappy = {normalize_company(n) for n in _null_priority(ctx)}
    n_gappy = sum(1 for c in cited if any(_same_company(c, g) for g in gappy))
    return [f"{n_gappy} of {len(cited)} cited companies have gaps in priority metrics"] if n_gappy / len(cited) > 0.5 else []


# ------------------------------------------------------------------------------------------------ driver
def rule_names(policy: dict | None = None) -> list[str]:
    """Every rule name mentioned in the YAML lists (used by the consistency test)."""
    rules = (policy or load_policy())["confidence_rules"]
    names = []
    for key in ("downgrade_to_low_if", "downgrade_one_level_if"):
        for item in rules.get(key) or []:
            names.append(next(iter(item)) if isinstance(item, dict) else item)
    return names


def _run(ctx: Ctx, items: list) -> list[str]:
    reasons = []
    for item in items or []:
        name, param = (next(iter(item.items())) if isinstance(item, dict) else (item, None))
        if name not in RULES:
            raise KeyError(f"data_policy.yaml names confidence rule '{name}' but agent/confidence.py has no implementation")
        reasons += [f"policy [{name}]: {r}" for r in RULES[name](ctx, param)]
    return reasons


def policy_confidence(results: list[dict], calls: list[ToolCall], persona: str, companies: list[str]) -> tuple[str, list[str]]:
    """(confidence cap, reasons). Resolved failures are reported but never lower confidence."""
    rules = load_policy()["confidence_rules"]
    ctx = Ctx(results, calls, persona, companies)
    info = [f"policy: '{f['arg']}' was not found ({f['code']}) but a later call resolved it to '{f['resolved_to']}' - no penalty"
            for f in failures(ctx)[1]]
    low = _run(ctx, rules.get("downgrade_to_low_if"))
    if low:
        return "low", low + info
    one = _run(ctx, rules.get("downgrade_one_level_if"))
    level = LEVELS.index(rules.get("start", "high")) - (1 if one else 0)
    return LEVELS[max(level, 0)], one + info
