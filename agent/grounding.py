"""Numeric grounding: is a number the model wrote really a number some tool returned?

Design (each rule exists because the previous, looser version accepted ~98% of random fabricated numbers):
* The pool holds only NUMERIC LEAF values of tool outputs. Numbers parsed out of strings, ids (source_id ...), counts and flags are excluded.
* The pool is scoped per company when the tool result names one, so a number cited for company A is checked against A's results
  (plus sector-level numbers such as medians), not against everyone's.
* A displayed number may be a rescaled tool number, but only in ways its own unit allows:
    no unit -> identity, /1e3, /1e6, /1e9      '%' -> x100 (fraction shown as percent) or identity (already percent points)
    'x'     -> identity                        'B' -> /1e9        'M' -> /1e6        'K' -> /1e3
  and only where the rescale makes sense: /1e3.. applies to big amounts (result >= 0.1), x100 only to fractions (|value| <= 10).
* Tolerance = half a unit of the LAST DISPLAYED DECIMAL (the exact rounding bound): "14.4x" is compared at +-0.05, "75,048" at +-0.5.
  (The brief said min(that, 0.5% relative); measured, that rejects correctly rounded numbers - see the note above _tolerance.)
* Signs are NOT flipped: "-0.225" must match a negative tool number.
* Prose is checked line by line: a line that names exactly one company is checked against that company's numbers (plus sector-level
  ones); any other line is checked against everything (weaker).
"""
import math
import random
import re
from typing import Any

from common.naming import normalize_company

# Numeric leaves under these keys are identifiers / flags / counts, not data values.
SKIP_KEYS = {"source_id", "source_ids", "id", "company_id", "row_id", "rowid", "verified", "n", "n_companies", "n_stale", "fy"}
UNIT_MULTS: dict[str | None, tuple[float, ...]] = {
    None: (1, 1e-3, 1e-6, 1e-9), "%": (100, 1), "x": (1,), "B": (1e-9,), "M": (1e-6,), "K": (1e-3,)}
_UNIT_ALIASES = {"%": "%", "pp": "%", "pt": "%", "pts": "%", "percentage point": "%", "percentage points": "%", "x": "x", "×": "x",
                 "bn": "B", "billion": "B", "b": "B", "mn": "M", "million": "M", "m": "M", "k": "K", "thousand": "K"}
_NUM_RE = re.compile(r"(?<![\w.])([-+−]?)[$€£]?\s?(\d[\d,]*)(\.\d+)?(?:\s?(%|×|x\b|bn\b|billion\b|mn\b|million\b|thousand\b|percentage points?\b|pp\b|pts?\b|[bBmMkK]\b))?")


# ------------------------------------------------------------------------------------------------ parsing
def unit_of(text: str | None) -> str | None:
    """Map a free-text unit ('x', '%', 'USD bn', 'million') to one of None, %, x, B, M, K."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if t in _UNIT_ALIASES:
        return _UNIT_ALIASES[t]
    if re.search(r"\b(bn|billion)\b|\bb$", t):
        return "B"
    if re.search(r"\b(mn|million)\b|\bm$", t):
        return "M"
    if "%" in t or "percent" in t or "pct" in t:
        return "%"
    return None


def parse_display(text: str) -> list[tuple[float, int, str | None]]:
    """All numbers in `text` as (value, displayed_decimals, unit). '$41.7B' -> (41.7, 1, 'B'); '75,048 million' -> (75048, 0, 'M')."""
    out = []
    for m in _NUM_RE.finditer(text):
        sign, whole, frac, suf = m.groups()
        try:
            v = float((whole + (frac or "")).replace(",", ""))
        except ValueError:
            continue
        if sign in ("-", "−"):
            v = -v
        out.append((v, len(frac) - 1 if frac else 0, _UNIT_ALIASES.get(suf.lower()) if suf else None))
    return out


def parse_numbers(text: str) -> list[float]:
    return [v for v, _, _ in parse_display(text)]


def decimals_of(x: float) -> int:
    s = repr(float(x))
    return len(s.split(".")[1].rstrip("0")) if "." in s and "e" not in s else 0


def answer_numbers_detail(text: str) -> list[tuple[float, int, str | None]]:
    """Numbers in prose that need grounding: skips dates, FY/quarter tokens, list numbering, 'x/10' scores and ranges,
    years, and trivial integers <= 10."""
    t = re.sub(r"[‐‑‒–—]", "-", text)                                   # non-breaking hyphens / en dashes
    t = re.sub(r"\bsources?(?:[ _]ids?)?\s*[:#]?\s*\d+(?:\s*(?:,|and|&)\s*\d+)*", " ", t, flags=re.I)   # "(source 232)" citations
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " ", t)
    t = re.sub(r"\b(FY|H[12]|Q[1-4])\s?\d{2,4}\b", " ", t)
    t = re.sub(r"(?m)^\s*(#+\s+)?((\d+[.)]|[-*])\s+)?", " ", t)
    t = re.sub(r"\b\d{1,2}\s*/\s*10\b|(?<![\d.])\d{1,2}\s*-\s*\d{1,2}(?![\d.])", " ", t)      # x/10 scores; whole integer ranges only
    out = []
    for v, dec, unit in parse_display(t):
        if dec == 0 and abs(v) == int(abs(v)) and (1900 <= abs(v) <= 2100):
            continue
        if dec == 0 and abs(v) <= 10:
            continue
        out.append((v, dec, unit))
    return out


def answer_numbers(text: str) -> list[float]:
    return [v for v, _, _ in answer_numbers_detail(text)]


def persona_numbers(cfg: Any) -> list[float]:
    """Numbers legitimately present in a persona's own thresholds (e.g. the 4.5x leverage ceiling)."""
    out: list[float] = []

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            out.append(float(o))
        elif isinstance(o, str):
            out.extend(parse_numbers(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                out.extend(parse_numbers(str(k)))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(cfg)
    return out


# ------------------------------------------------------------------------------------------------ pool
# DEVIATION from the brief, which asked for min(half a unit of the last displayed decimal, 0.5% relative). Measured on a realistic
# pool (tests/test_grounding.py) the 0.5% cap rejects correctly ROUNDED numbers ("3.2%" from 3.244%: error 0.044 > 0.016) and
# accepted only ~85% of genuine numbers, while barely changing the false-accept rate. The tolerance is therefore the exact rounding
# bound: half a unit of the last displayed decimal. Coarse displays ("12%") are inherently loose; prose skips integers <= 10.
def _tolerance(x: float, decimals: int) -> float:
    return 0.5 * 10 ** (-decimals) + 1e-12


def _leaves(obj: Any, key: str | None = None):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))) and key not in SKIP_KEYS:
            yield float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, k if isinstance(k, str) else None)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v, key)


class GroundingPool:
    """Numeric leaves of tool results, scoped per company; sector-level numbers are shared."""

    def __init__(self, extra: list[float] | None = None):
        self.by_company: dict[str, list[float]] = {}
        self.aliases: dict[str, str] = {}
        self.shared: list[float] = list(extra or [])

    # -- building
    def _canon(self, name: str, ticker: str | None = None) -> str:
        c = normalize_company(name)
        self.aliases[c] = c
        if ticker:
            self.aliases[normalize_company(ticker)] = c
        return c

    def _add(self, name: str, obj: Any, ticker: str | None = None):
        self.by_company.setdefault(self._canon(name, ticker), []).extend(_leaves(obj))

    def add_result(self, r: Any):
        if not isinstance(r, dict):
            return
        rest = {k: v for k, v in r.items() if k != "sources"}
        if isinstance(r.get("per_company"), dict):                              # get_sector_stats: per-company values are company-scoped
            for name, v in r["per_company"].items():
                self._add(name, v)
            for name, v in (r.get("gap_to_max") or {}).items():
                self._add(name, v)
            self.shared += list(_leaves({k: v for k, v in rest.items() if k not in ("per_company", "gap_to_max")}))
            return
        if isinstance(r.get("companies"), dict) and "policy_staleness_days" in r:  # get_data_quality
            for name, rep in r["companies"].items():
                self._add(name, rep)
            self.shared += list(_leaves({k: v for k, v in rest.items() if k != "companies"}))
            return
        if isinstance(r.get("companies"), list):                                # get_company_snapshot: one entry per company
            for e in r["companies"]:
                if isinstance(e, dict) and e.get("company"):
                    self._add(e["company"], e, e.get("ticker"))
                    self.shared += list(_leaves((e.get("valuation") or {}).get("vs_sector")))     # sector medians are quotable for any company
            self.shared += list(_leaves({k: v for k, v in rest.items() if k not in ("companies", "not_found")}))
            return
        name = r.get("company") or (r.get("profile") or {}).get("name")
        if name:
            self._add(name, {k: v for k, v in rest.items() if k != "vs_sector"}, r.get("ticker") or (r.get("profile") or {}).get("ticker"))
            if r.get("vs_sector"):                                               # sector medians are quotable for any company
                self._add(name, r["vs_sector"])
                self.shared += list(_leaves(r["vs_sector"]))
            return
        self.shared += list(_leaves(rest))

    # -- querying
    def resolve(self, company: str | None) -> str | None:
        if not company:
            return None
        n = normalize_company(company)
        if n in self.aliases:
            return self.aliases[n]
        hits = {c for a, c in self.aliases.items() if len(a) >= 3 and len(n) >= 3 and (a in n or n in a)}
        return hits.pop() if len(hits) == 1 else None

    def _pools(self, company: str | None, scoped: bool):
        if scoped:
            key = self.resolve(company)
            return self.shared + (self.by_company.get(key, []) if key else [])
        return self.shared + [v for vals in self.by_company.values() for v in vals]

    def check(self, x: float, decimals: int, unit: str | None = None, company: str | None = None, scoped: bool = False) -> bool:
        """True if x (as displayed, with `decimals` decimals and `unit`) equals some pool number under the allowed rescalings.
        scoped=True checks only the named company's numbers + shared ones; scoped=False (prose) checks everyone's."""
        if x == 0:
            return True
        tol = _tolerance(x, decimals)
        mults = UNIT_MULTS.get(unit, UNIT_MULTS[None])
        for p in self._pools(company, scoped):
            for m in mults:
                if m < 1 and abs(p * m) < 0.1:      # display-unit rescaling only makes sense for big amounts, not ratios
                    continue
                if m > 1 and abs(p) > 10:           # fraction -> percent only makes sense for fractions
                    continue
                if abs(p * m - x) <= tol:
                    return True
        return False

    def company_in_line(self, line: str) -> str | None:
        """The single company a line is about (by name or ticker), or None if it names none or several."""
        flat = re.sub(r"\s+", " ", re.sub(r"[.,]", " ", line.lower()))
        hits = {c for a, c in self.aliases.items() if len(a) >= 3 and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", flat)}
        return hits.pop() if len(hits) == 1 else None

    def prose_numbers(self, text: str) -> list[tuple[float, int, str | None]]:
        return [n for line in text.splitlines() for n in answer_numbers_detail(line)]

    def ungrounded_in(self, text: str) -> list[tuple[float, int, str | None]]:
        """Numbers in prose that no tool result supports, checking each line against the company it names (if exactly one)."""
        bad = []
        for line in text.splitlines():
            nums = answer_numbers_detail(line)
            if nums:
                comp = self.company_in_line(line)
                bad += [n for n in nums if not self.check(*n, company=comp, scoped=comp is not None)]
        return bad

    def check_text(self, s: str, company: str | None = None, scoped: bool = False) -> bool | None:
        """Ground the first number in a displayed string like '$41.7B' / '14.4x' / '12%'. None if there is no number."""
        nums = parse_display(s)
        if not nums:
            return None
        v, dec, unit = nums[0]
        return self.check(v, dec, unit, company, scoped)


# ------------------------------------------------------------------------------------------------ negative control
def synthetic_fabrications(n: int, seed: int = 0) -> list[str]:
    """Random numbers formatted the way an LLM would write them, across several magnitudes. None comes from any tool."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.random()
        if k < 0.30:
            out.append(f"{rnd.uniform(1, 60):.1f}x")
        elif k < 0.55:
            out.append(f"{rnd.uniform(0.1, 60):.1f}%")
        elif k < 0.70:
            out.append(f"${rnd.uniform(0.1, 400):.1f}B")
        elif k < 0.80:
            out.append(f"{int(rnd.uniform(50, 300000)):,} million")
        elif k < 0.90:
            out.append(f"{rnd.uniform(0.001, 1):.3f}")
        else:
            out.append(f"{int(10 ** rnd.uniform(5, 12)):,}")
    return out


def false_accept_rate(pool: GroundingPool, texts: list[str], company: str | None = None) -> float:
    """Share of `texts` the pool accepts. company=None: no company context (whole pool). company=X: scoped to X (+ shared numbers)."""
    return sum(bool(pool.check_text(t, company=company, scoped=company is not None)) for t in texts) / len(texts)


def scoped_false_accept_rate(pool: GroundingPool, texts: list[str]) -> float:
    """Mean false-accept rate when each fabricated number is claimed for one company (as a data_point or a one-company sentence)."""
    comps = [c for c, v in pool.by_company.items() if len(v) >= 20] or list(pool.by_company)
    if not comps:                                   # no company-scoped data in this pool: only the whole-pool rate exists
        return false_accept_rate(pool, texts)
    return sum(false_accept_rate(pool, texts, c) for c in comps) / len(comps)
