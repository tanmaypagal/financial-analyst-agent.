"""Read-only, parameterized query layer used by the MCP server. Only this process opens the SQLite file.

Every result carries source_id / as_of_date / stale. Derived ratios are computed here (server side)
so the LLM never does arithmetic and every number it cites appears in a tool output.
"""
import os
import re
import sqlite3
import statistics
from datetime import date
from pathlib import Path

import yaml

from common.naming import normalize_company

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("FINANCE_DB", ROOT / "data" / "finance.db"))
POLICY = yaml.safe_load((ROOT / "config" / "data_policy.yaml").read_text(encoding="utf-8"))
SECTOR_CFG = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]

# metric -> (kind, description). kind decides where the value is read from.
SECTOR_METRICS = {
    "ev_ebitda": ("valuation", "EV / EBITDA (x)"),
    "pe": ("valuation", "trailing P/E (x)"),
    "ev_sales": ("valuation", "EV / sales (x)"),
    "dividend_yield": ("valuation", "dividend yield (fraction)"),
    "market_cap_usd": ("valuation", "market cap in USD"),
    "enterprise_value_usd": ("valuation", "enterprise value in USD (EV x FX rate) - deal size"),
    "revenue_usd": ("financial", "latest FY revenue in USD"),
    "revenue_growth_yoy": ("derived", "latest FY revenue growth vs prior FY (fraction, native currency)"),
    "ebitda_margin": ("financial", "latest FY EBITDA margin (fraction)"),
    "ebitda_margin_change": ("derived", "change in EBITDA margin vs prior FY (fraction points)"),
    "gross_margin": ("financial", "latest FY gross margin (fraction)"),
    "net_debt_to_ebitda": ("derived", "net debt / EBITDA (x), latest FY"),
    "total_debt_to_ebitda": ("derived", "total debt / EBITDA (x), latest FY"),
    "fcf_conversion": ("derived", "FCF / EBITDA (fraction), latest FY"),
    "capex_to_revenue": ("derived", "capex / revenue (fraction), latest FY"),
    "order_backlog_to_revenue": ("derived", "order backlog (RPO) / latest FY revenue (x); defense only"),
}


# Valuation metrics that are computed in SQL instead of stored: metric -> expression (NULL if either input is NULL)
VALUATION_EXPR = {"enterprise_value_usd": "enterprise_value * fx_rate"}

# Key financial fields a conclusion typically rests on; a field is 'verified' only if it appears in field_verification as match/sec_*.
KEY_FIELDS = ("revenue", "ebitda", "net_income", "net_debt", "fcf", "capex")


def today() -> date:
    return date.fromisoformat(os.environ["AGENT_TODAY"]) if os.environ.get("AGENT_TODAY") else date.today()


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def is_stale(kind: str, ref_date: str | None) -> bool | None:
    if not ref_date:
        return None
    limit = POLICY["staleness_days"][kind]
    return (today() - date.fromisoformat(ref_date[:10])).days > limit


def _sources(con, ids) -> dict:
    ids = sorted({i for i in ids if i})
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    rows = con.execute(f"SELECT id,url,publisher,doc_type,retrieved_at FROM sources WHERE id IN ({q})", ids).fetchall()
    return {r["id"]: dict(r) for r in rows}


def _div(a, b):
    return None if a is None or b in (None, 0) else a / b


# ---------------------------------------------------------------- resolution
def _candidates(rows, company: str) -> list:
    """Companies in `rows` that `company` refers to. Tried in order; the first rule with any hit decides:
    1. exact ticker or name (case-insensitive), also after dropping corporate suffixes ("Oracle Corp" -> "oracle")
    2. the query is a substring of a name/ticker (>= 3 chars), e.g. "lockheed"
    3. a company NAME appears as whole words inside the query ("What do you think about Saab?"). Names only (>= 4 chars): short
       tickers such as NOW, BOX or DT are ordinary words and would match by accident."""
    q = company.strip().lower()
    nq = normalize_company(company)
    exact = [r for r in rows if r["ticker"].lower() == q or r["name"].lower() == q or normalize_company(r["name"]) == nq]
    if exact:
        return exact
    part = [r for r in rows if len(q) >= 3 and (q in r["name"].lower() or q in r["ticker"].lower()
                                                or (len(nq) >= 3 and nq in normalize_company(r["name"])))]
    if part:
        return part
    flat = re.sub(r"[.,]", " ", q)
    return [r for r in rows if len(normalize_company(r["name"])) >= 4
            and re.search(r"(?<!\w)" + re.escape(normalize_company(r["name"])) + r"(?!\w)", flat)]


def resolve(con, sector: str, company: str | None = None):
    """Return (sector_row, company_row, error_dict). See _candidates for the matching rules; ambiguity is refused, never guessed."""
    sec = con.execute("SELECT * FROM sectors WHERE lower(name)=lower(?)", (sector.strip(),)).fetchone()
    if not sec:
        valid = [r["name"] for r in con.execute("SELECT name FROM sectors")]
        return None, None, {"found": False, "reason_code": "UNKNOWN_SECTOR", "reason": f"Unknown sector '{sector}'. Valid sectors: {valid}"}
    if company is None:
        return sec, None, None
    rows = con.execute("SELECT * FROM companies WHERE sector_id=?", (sec["id"],)).fetchall()
    hits = _candidates(rows, company)
    if len(hits) == 1:
        return sec, hits[0], None
    if len(hits) > 1:
        return sec, None, {"found": False, "reason_code": "AMBIGUOUS", "reason": f"'{company}' is ambiguous in sector {sec['name']}; "
                           f"candidates: {[r['name'] for r in hits]}. No guess made."}
    every = con.execute("SELECT c.*, s.name AS sector_name FROM companies c JOIN sectors s ON s.id=c.sector_id").fetchall()
    other = _candidates(every, company)
    if other:
        o = other[0]
        return sec, None, {"found": False, "reason_code": "WRONG_SECTOR", "reason": f"'{company}' ({o['name']}) is in the '{o['sector_name']}' sector, "
                           f"not '{sec['name']}'. No data returned for this sector."}
    return sec, None, {"found": False, "reason_code": "NOT_IN_DB", "reason": f"No company matching '{company}' exists in the database "
                       f"(sector {sec['name']}). No data available; do not answer from general knowledge."}


# Machine-readable `reason_code` on every {"found": false}: NOT_IN_DB, WRONG_SECTOR, AMBIGUOUS, UNKNOWN_SECTOR, BAD_ARGS, NO_ROWS
# (the agent adds WRONG_SELECTED_SECTOR when a call names a sector other than the user's selection).
# ---------------------------------------------------------------- tools
def list_sectors():
    with connect() as con:
        out = []
        for r in con.execute("SELECT s.id,s.name,COUNT(c.id) n FROM sectors s LEFT JOIN companies c ON c.sector_id=s.id GROUP BY s.id"):
            out.append({"sector": r["name"], "display_name": SECTOR_CFG.get(r["name"], {}).get("display_name", r["name"]),
                        "n_companies": r["n"]})
        return {"found": True, "sectors": out, "as_of_date": today().isoformat()}


def list_companies(sector):
    with connect() as con:
        sec, _, err = resolve(con, sector)
        if err:
            return err
        rows = con.execute("SELECT ticker,name,country,exchange,currency FROM companies WHERE sector_id=? ORDER BY name",
                           (sec["id"],)).fetchall()
        as_of = con.execute("SELECT MAX(as_of_date) FROM companies WHERE sector_id=?", (sec["id"],)).fetchone()[0]
        return {"found": True, "sector": sec["name"], "companies": [dict(r) for r in rows],
                "as_of_date": as_of, "stale": is_stale("company_profile", as_of)}


def get_company_profile(sector, company):
    with connect() as con:
        sec, co, err = resolve(con, sector, company)
        if err:
            return err
        d = {k: co[k] for k in co.keys() if k not in ("id", "sector_id")}
        return {"found": True, "sector": sec["name"], "profile": d, "source_id": co["source_id"],
                "as_of_date": co["as_of_date"], "stale": is_stale("company_profile", co["as_of_date"]),
                "sources": _sources(con, [co["source_id"]])}


def _fin_rows(con, cid, n):
    return con.execute("SELECT * FROM financials WHERE company_id=? ORDER BY period_end DESC LIMIT ?", (cid, n)).fetchall()


def _derived(rows):
    """rows: newest first. Returns list of derived dicts aligned to rows."""
    out = []
    for i, r in enumerate(rows):
        prev = rows[i + 1] if i + 1 < len(rows) else None
        em, pem = r["ebitda_margin"], (prev["ebitda_margin"] if prev else None)
        out.append({
            "revenue_growth_yoy": _div(r["revenue"] - prev["revenue"], prev["revenue"]) if prev and r["revenue"] is not None and prev["revenue"] else None,
            "ebitda_margin_change": em - pem if em is not None and pem is not None else None,
            "net_debt_to_ebitda": _div(r["net_debt"], r["ebitda"]),
            "total_debt_to_ebitda": _div(r["total_debt"], r["ebitda"]),
            "fcf_conversion": _div(r["fcf"], r["ebitda"]),
            "capex_to_revenue": _div(r["capex"], r["revenue"]),
        })
    return out


def get_financials(sector, company, periods=3):
    periods = max(1, min(int(periods), 5))
    with connect() as con:
        sec, co, err = resolve(con, sector, company)
        if err:
            return err
        rows = _fin_rows(con, co["id"], periods + 1)          # +1 so the oldest requested period has a growth base
        if not rows:
            return {"found": False, "reason_code": "NO_ROWS", "reason": f"{co['name']} is in the database but has no financial rows."}
        der = _derived(rows)
        out, extra_sources = [], []
        for r, d in list(zip(rows, der))[:periods]:
            row = {k: r[k] for k in r.keys() if k not in ("id", "company_id")}
            row["derived"] = d
            row["stale"] = is_stale("financials", r["period_end"])
            fv = con.execute("SELECT field,status,db_value,other_value,source_id,note FROM field_verification WHERE company_id=? AND period_end=? ORDER BY field",
                             (co["id"], r["period_end"])).fetchall()
            row["verified_fields"] = [dict(x) for x in fv]                     # what was checked against a filing, and how it came out
            row["unverified_fields"] = [f for f in KEY_FIELDS if f not in {x["field"] for x in fv}]   # key fields still only from Yahoo
            extra_sources += [x["source_id"] for x in fv]
            out.append(row)
        return {"found": True, "company": co["name"], "ticker": co["ticker"], "sector": sec["name"],
                "currency": co["currency"], "units": "full units of native currency (not millions); margins/ratios are fractions",
                "financials": out, "source_id": out[0]["source_id"], "as_of_date": out[0]["as_of_date"],
                "stale": out[0]["stale"], "sources": _sources(con, [r["source_id"] for r in out] + extra_sources)}


MEDIAN_METRICS = ("ev_ebitda", "pe", "ev_sales", "dividend_yield")


def _sector_medians(con, sector_id) -> dict:
    """{metric: median across the sector (needs >= 3 values) or None}."""
    out = {}
    for m in MEDIAN_METRICS:
        vals = [x for x in _sector_values(con, sector_id, m).values() if x is not None]
        out[m] = statistics.median(vals) if len(vals) >= 3 else None
    return out


def _valuation_view(v, medians: dict) -> tuple[dict, dict]:
    """(valuation dict incl. enterprise_value_usd, vs_sector premiums) for one valuation row."""
    d = {k: v[k] for k in v.keys() if k not in ("id", "company_id")}
    d["enterprise_value_usd"] = v["enterprise_value"] * v["fx_rate"] if v["enterprise_value"] is not None and v["fx_rate"] else None
    vs = {m: {"sector_median": medians[m], "premium_to_median_pct": (v[m] / medians[m] - 1) * 100 if medians[m] else None}
          for m in MEDIAN_METRICS if v[m] is not None and medians[m] is not None}
    return d, vs


def get_valuation(sector, company):
    with connect() as con:
        sec, co, err = resolve(con, sector, company)
        if err:
            return err
        v = con.execute("SELECT * FROM valuations WHERE company_id=? ORDER BY as_of_date DESC LIMIT 1", (co["id"],)).fetchone()
        if not v:
            return {"found": False, "reason_code": "NO_ROWS", "reason": f"{co['name']} has no valuation row."}
        d, vs = _valuation_view(v, _sector_medians(con, sec["id"]))
        return {"found": True, "company": co["name"], "ticker": co["ticker"], "sector": sec["name"], "valuation": d,
                "vs_sector": vs, "source_id": v["source_id"], "as_of_date": v["as_of_date"],
                "stale": is_stale("market_data", v["as_of_date"]), "sources": _sources(con, [v["source_id"]])}


SNAPSHOT_MAX = 12
NOTE_CHARS = 300                 # ownership notes are cut to this many characters in a snapshot (full text: get_company_profile)
SNAPSHOT_FIN_FIELDS = ("period_end", "revenue", "ebitda", "ebit", "net_income", "gross_margin", "ebitda_margin", "net_debt", "total_debt",
                       "cash", "capex", "fcf", "currency", "revenue_usd")


def _snapshot_entry(con, sec, co, medians) -> dict:
    """One company: latest FY financials + derived ratios, latest valuation, data-quality flags. Provenance kept per company."""
    own = co["ownership_notes"]
    e = {"company": co["name"], "ticker": co["ticker"], "country": co["country"], "currency": co["currency"],
         "fiscal_year_end": co["fiscal_year_end"], "accounting_standard": co["accounting_standard"],
         "ownership_notes": (own[:NOTE_CHARS] + "...") if own and len(own) > NOTE_CHARS else own,
         "ownership_notes_truncated": bool(own and len(own) > NOTE_CHARS)}
    rows = _fin_rows(con, co["id"], 2)
    fin_stale = val_stale = None
    e["financials"], src_ids = None, []
    if rows:
        r = rows[0]
        fv = {x["field"]: x["status"] for x in con.execute("SELECT field,status FROM field_verification WHERE company_id=? AND period_end=?",
                                                          (co["id"], r["period_end"]))}
        fin_stale = is_stale("financials", r["period_end"])
        e["financials"] = {**{k: r[k] for k in SNAPSHOT_FIN_FIELDS}, "derived": _derived(rows)[0], "verified": r["verified"],
                           "verified_fields": fv, "unverified_fields": [f for f in KEY_FIELDS if f not in fv],
                           "stale": fin_stale, "source_id": r["source_id"], "as_of_date": r["as_of_date"]}
        src_ids.append(r["source_id"])
    v = con.execute("SELECT * FROM valuations WHERE company_id=? ORDER BY as_of_date DESC LIMIT 1", (co["id"],)).fetchone()
    e["valuation"] = None
    if v:
        d, vs = _valuation_view(v, medians)
        val_stale = is_stale("market_data", v["as_of_date"])
        e["valuation"] = {**{k: d[k] for k in ("as_of_date", "market_cap", "enterprise_value", "enterprise_value_usd", "ev_ebitda", "pe", "ev_sales",
                                                "dividend_yield", "currency", "market_cap_usd")}, "vs_sector": vs, "verified": v["verified"],
                          "stale": val_stale, "source_id": v["source_id"]}
        src_ids.append(v["source_id"])
    gaps = con.execute("SELECT reason FROM data_gaps WHERE company_id=?", (co["id"],)).fetchall()
    e["data_quality"] = {"financials_stale": fin_stale, "valuation_stale": val_stale, "n_data_gaps": len(gaps),
                         "caveats": [g["reason"][:200] for g in gaps if g["reason"].startswith("CAVEAT")]}
    # per-company provenance (the tool-level fields aggregate these)
    e["source_id"] = src_ids[0] if src_ids else None
    e["as_of_date"] = (e["financials"] or {}).get("as_of_date") or (e["valuation"] or {}).get("as_of_date")
    e["stale"] = bool(fin_stale or val_stale) if (fin_stale is not None or val_stale is not None) else None
    e["verified"] = (e["financials"] or {}).get("verified", 0)
    e["_source_ids"] = src_ids
    return e


def get_company_snapshot(sector, companies):
    """Latest financials, valuation and data-quality flags for up to SNAPSHOT_MAX companies in ONE call (read-only)."""
    if not isinstance(companies, list) or not companies or not all(isinstance(c, str) and c.strip() for c in companies):
        return {"found": False, "reason_code": "BAD_ARGS", "reason": "`companies` must be a non-empty list of company names or tickers"}
    asked, ignored = companies[:SNAPSHOT_MAX], companies[SNAPSHOT_MAX:]
    with connect() as con:
        sec, _, err = resolve(con, sector)
        if err:
            return err
        medians = _sector_medians(con, sec["id"])
        entries, missing, seen = [], [], set()
        for name in asked:
            _, co, e = resolve(con, sector, name)          # same lookup rules (and sector enforcement) as every other tool
            if e:
                missing.append({"requested": name, "reason_code": e["reason_code"], "reason": e["reason"]})
            elif co["id"] not in seen:
                seen.add(co["id"])
                entries.append(_snapshot_entry(con, sec, co, medians))
        if not entries:
            return {"found": False, "reason_code": missing[0]["reason_code"] if missing else "NOT_IN_DB", "not_found": missing,
                    "reason": "; ".join(m["reason"] for m in missing)[:600]}
        sids = sorted({i for e in entries for i in e.pop("_source_ids")})
        dates = [e["as_of_date"] for e in entries if e["as_of_date"]]
        stale_cos = [e["company"] for e in entries if e["stale"]]
        out = {"found": True, "sector": sec["name"], "companies": entries, "not_found": missing,
               "source_id": sids[0] if sids else None, "source_ids": sids, "as_of_date": max(dates) if dates else None,
               "stale": bool(stale_cos), "stale_companies": stale_cos, "n_stale": len(stale_cos), "sources": _sources(con, sids),
               "note": "one entry per company; multi-year history: get_financials; full ownership text: get_company_profile"}
        if ignored:
            out["ignored_over_limit"] = ignored
        return out


def get_signals(sector, company, signal_type=None, limit=10):
    with connect() as con:
        sec, co, err = resolve(con, sector, company)
        if err:
            return err
        sql = "SELECT * FROM signals WHERE company_id=?"
        args = [co["id"]]
        if signal_type:
            sql += " AND signal_type=?"
            args.append(signal_type)
        sql += " ORDER BY signal_date DESC, id LIMIT ?"
        args.append(max(1, min(int(limit), 20)))
        rows = con.execute(sql, args).fetchall()
        if not rows:
            return {"found": True, "company": co["name"], "sector": sec["name"], "signals": [],
                    "note": f"No '{signal_type or 'any'}' signals stored for {co['name']} - this is a data gap, not evidence of absence.",
                    "source_id": None, "as_of_date": None, "stale": None}
        sigs = []
        for r in rows:
            s = {k: r[k] for k in r.keys() if k not in ("id", "company_id")}
            s["stale"] = is_stale("signals", r["signal_date"])
            sigs.append(s)
        return {"found": True, "company": co["name"], "sector": sec["name"], "signals": sigs,
                "source_id": sigs[0]["source_id"], "as_of_date": sigs[0]["as_of_date"], "stale": sigs[0]["stale"],
                "sources": _sources(con, [s["source_id"] for s in sigs])}


def _latest_fin(con, cid):
    rows = _fin_rows(con, cid, 2)
    return rows


def _sector_values(con, sector_id, metric, dates=None, verified=None):
    """{company_name: value} for a metric using each company's latest data.
    If `dates` (a dict) is given it is filled with {company_name: reference date of the value} (valuation as_of_date, or the
    financial period_end; for backlog ratios the OLDER of the two) so callers can judge staleness per contributing company.
    If `verified` (a dict) is given it is filled with {company_name: 0/1}: was the row the value came from checked against a filing?"""
    kind = SECTOR_METRICS[metric][0]
    out = {}
    for co in con.execute("SELECT id,name FROM companies WHERE sector_id=?", (sector_id,)).fetchall():
        val = None
        if kind == "valuation":
            expr = VALUATION_EXPR.get(metric, metric)
            r = con.execute(f"SELECT {expr} v, as_of_date d, verified vf FROM valuations WHERE company_id=? ORDER BY as_of_date DESC LIMIT 1",
                            (co["id"],)).fetchone()
            val = r["v"] if r else None
            if dates is not None and r:
                dates[co["name"]] = r["d"]
            if verified is not None and r:
                verified[co["name"]] = r["vf"]
        else:
            rows = _fin_rows(con, co["id"], 2)
            if rows:
                ref = rows[0]["period_end"]
                if kind == "financial":
                    val = rows[0][metric]
                elif metric == "order_backlog_to_revenue":
                    b = con.execute("SELECT order_backlog, period_end FROM sector_metrics_defense WHERE company_id=? "
                                    "AND period_end<=? ORDER BY period_end DESC LIMIT 1", (co["id"], rows[0]["period_end"])).fetchone()
                    val = _div(b["order_backlog"], rows[0]["revenue"]) if b else None
                    if b:
                        ref = min(ref, b["period_end"])
                else:
                    val = _derived(rows)[0][metric]
                if dates is not None:
                    dates[co["name"]] = ref
                if verified is not None:
                    verified[co["name"]] = rows[0]["verified"]
        out[co["name"]] = val
    return out


def get_sector_stats(sector, metric):
    if metric not in SECTOR_METRICS:
        return {"found": False, "reason_code": "BAD_ARGS", "reason": f"Unknown metric '{metric}'. Valid: {sorted(SECTOR_METRICS)}"}
    with connect() as con:
        sec, _, err = resolve(con, sector)
        if err:
            return err
        dates: dict = {}
        ver: dict = {}
        vals = _sector_values(con, sec["id"], metric, dates, ver)
        nums = sorted(v for v in vals.values() if v is not None)
        if not nums:
            return {"found": True, "sector": sec["name"], "metric": metric, "n": 0, "note": "no company has this metric (data gap)",
                    "source_id": None, "as_of_date": None, "stale": None, "stale_companies": [], "n_stale": 0}
        q = statistics.quantiles(nums, n=4, method="inclusive") if len(nums) >= 2 else [nums[0]] * 3
        kind = SECTOR_METRICS[metric][0]
        as_of = con.execute("SELECT MAX(as_of_date) FROM valuations WHERE company_id IN (SELECT id FROM companies WHERE sector_id=?)"
                            if kind == "valuation" else
                            "SELECT MAX(as_of_date) FROM financials WHERE company_id IN (SELECT id FROM companies WHERE sector_id=?)",
                            (sec["id"],)).fetchone()[0]
        tbl = "valuations" if kind == "valuation" else "financials"
        # staleness is judged per CONTRIBUTING company from its own reference date (period_end / valuation date), not from the
        # retrieval date: a FY2024 value inside a median is stale even though it was retrieved today.
        rule = "market_data" if kind == "valuation" else "financials"
        stale_cos = sorted(n for n, v in vals.items() if v is not None and is_stale(rule, dates.get(n)))
        sids = [r[0] for r in con.execute(f"SELECT DISTINCT source_id FROM {tbl} WHERE company_id IN "
                                          "(SELECT id FROM companies WHERE sector_id=?)", (sec["id"],))]
        return {"found": True, "sector": sec["name"], "metric": metric, "description": SECTOR_METRICS[metric][1], "n": len(nums),
                "median": statistics.median(nums), "q1": q[0], "q3": q[2], "min": nums[0], "max": nums[-1],
                "best_company_by_max": max((k for k, v in vals.items() if v is not None), key=lambda k: vals[k]),
                "per_company": vals, "gap_to_max": {k: nums[-1] - v for k, v in vals.items() if v is not None},
                "note": "per-company values may be null where the source had no data; nulls are excluded from stats",
                "source_id": min(sids) if sids else None, "source_ids": sids,
                "as_of_date": as_of, "stale": bool(stale_cos), "stale_companies": stale_cos, "n_stale": len(stale_cos),
                # how much of the distribution rests on rows checked against a filing (market data never is)
                "n_contributors": len(nums), "n_unverified_contributors": sum(1 for n, v in vals.items() if v is not None and not ver.get(n)),
                "data_date_per_company": {n: dates.get(n) for n, v in vals.items() if v is not None},
                "sources": _sources(con, sids)}


def get_defense_metrics(company):
    with connect() as con:
        sec, co, err = resolve(con, "defense", company)
        if err:
            return err
        rows = con.execute("SELECT * FROM sector_metrics_defense WHERE company_id=? ORDER BY period_end DESC LIMIT 4", (co["id"],)).fetchall()
        gaps = [dict(r) for r in con.execute("SELECT field,reason FROM data_gaps WHERE company_id=? AND field LIKE 'sector_metrics_defense.%'", (co["id"],))]
        if not rows:
            return {"found": True, "company": co["name"], "metrics": [], "data_gaps": gaps,
                    "note": "No defense-specific metrics stored (backlog etc. undisclosed in machine-readable sources) - data gap.",
                    "source_id": None, "as_of_date": None, "stale": None}
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys() if k != "company_id"}
            d["stale"] = is_stale("financials", r["period_end"])
            out.append(d)
        fin = _fin_rows(con, co["id"], 1)
        # backlog is USD (SEC RPO) - cover ratio only meaningful when revenue is in the same currency
        cover = None
        if fin and out[0]["order_backlog"] and fin[0]["currency"] == "USD" and fin[0]["revenue"]:
            cover = out[0]["order_backlog"] / fin[0]["revenue"]
        # implied book-to-bill = (backlog_t - backlog_t-1 + revenue_t) / revenue_t on consecutive fiscal years. DERIVED, not disclosed.
        fy = sorted((m for m in out if m["order_backlog"] and m["notes"] and "10-K" in m["notes"] or m["order_backlog"] and m["notes"] and "20-F" in m["notes"]),
                    key=lambda m: m["period_end"])
        implied = []
        for prev, cur in zip(fy, fy[1:]):
            days = (date.fromisoformat(cur["period_end"]) - date.fromisoformat(prev["period_end"])).days
            rev = con.execute("SELECT revenue,currency FROM financials WHERE company_id=? AND ABS(julianday(period_end)-julianday(?))<=7",
                              (co["id"], cur["period_end"])).fetchone()
            if 350 <= days <= 380 and rev and rev["revenue"] and rev["currency"] == "USD":
                implied.append({"period_end": cur["period_end"], "implied_book_to_bill":
                                (cur["order_backlog"] - prev["order_backlog"] + rev["revenue"]) / rev["revenue"]})
        return {"found": True, "company": co["name"], "sector": "defense", "metrics": out,
                "backlog_to_latest_fy_revenue_x": cover, "implied_book_to_bill": implied,
                "implied_book_to_bill_method": "DERIVED (not company-disclosed): (RPO backlog change over the fiscal year + revenue) / revenue; "
                "backlog is SEC RPO and revenue is yfinance, so treat as indicative" if implied else None, "data_gaps": gaps,
                "source_id": out[0]["source_id"], "as_of_date": out[0]["as_of_date"], "stale": out[0]["stale"],
                "sources": _sources(con, [m["source_id"] for m in out])}


def get_data_quality(sector, company=None):
    with connect() as con:
        sec, co, err = resolve(con, sector, company)
        if err:
            return err
        ids = [co["id"]] if co else [r["id"] for r in con.execute("SELECT id FROM companies WHERE sector_id=?", (sec["id"],))]
        q = ",".join("?" * len(ids))
        report = {}
        for cid in ids:
            c = con.execute("SELECT name FROM companies WHERE id=?", (cid,)).fetchone()["name"]
            lf = con.execute("SELECT MAX(period_end) d, MIN(verified) mv, COUNT(*) n, SUM(verified) sv FROM financials WHERE company_id=?", (cid,)).fetchone()
            lv = con.execute("SELECT MAX(as_of_date) d, SUM(verified) sv, COUNT(*) n FROM valuations WHERE company_id=?", (cid,)).fetchone()
            ls = con.execute("SELECT MAX(signal_date) d, SUM(verified) sv, COUNT(*) n FROM signals WHERE company_id=?", (cid,)).fetchone()
            gaps = con.execute("SELECT field,reason FROM data_gaps WHERE company_id=?", (cid,)).fetchall()
            fv = dict(con.execute("SELECT status, COUNT(*) FROM field_verification WHERE company_id=? GROUP BY status", (cid,)).fetchall())
            total = (lf["n"] or 0) + (lv["n"] or 0) + (ls["n"] or 0)
            ver = (lf["sv"] or 0) + (lv["sv"] or 0) + (ls["sv"] or 0)
            report[c] = {
                "latest_financials_period": lf["d"], "financials_stale": is_stale("financials", lf["d"]),
                "latest_valuation_date": lv["d"], "valuation_stale": is_stale("market_data", lv["d"]),
                "latest_signal_date": ls["d"], "signals_stale": is_stale("signals", ls["d"]),
                "rows_total": total, "rows_verified": ver, "unverified_share": (1 - ver / total) if total else None,
                "n_data_gaps": len(gaps),
                "field_checks_vs_filings": {"match": fv.get("match", 0), "mismatch": fv.get("mismatch", 0),
                                            "value_taken_from_filing": fv.get("sec_replaced", 0) + fv.get("sec_filled", 0)},
                "caveats": [g["reason"] for g in gaps if g["reason"].startswith("CAVEAT")],
                "gaps": [{"field": g["field"], "reason": g["reason"]} for g in gaps][: (60 if co else 0)],
            }
        as_of = today().isoformat()
        return {"found": True, "sector": sec["name"], "policy_staleness_days": POLICY["staleness_days"],
                "companies": report, "source_id": None, "as_of_date": as_of, "stale": False,
                "note": "gaps are listed per company only when a single company is requested"}


def get_schema():
    with connect() as con:
        tables = {}
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            tables[r["name"]] = [{"column": c["name"], "type": c["type"]} for c in con.execute(f"PRAGMA table_info({r['name']})")]
        return {"found": True, "tables": tables, "source_id": None, "as_of_date": today().isoformat(), "stale": False}
