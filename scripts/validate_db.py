"""Data-quality report for data/finance.db.  Usage: python scripts/validate_db.py [--strict]

Checks: NULLs, outliers, unit/currency mismatches, duplicate rows, orphaned source_ids, fiscal-year alignment.
Thresholds come from config/data_policy.yaml (outlier_checks). Prints a report; exits 1 with --strict if any ERROR.
"""
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("FINANCE_DB", ROOT / "data" / "finance.db"))
POL = yaml.safe_load((ROOT / "config" / "data_policy.yaml").read_text(encoding="utf-8"))["outlier_checks"]
findings = defaultdict(list)          # section -> [(level, msg)]


def add(section, level, msg):
    findings[section].append((level, msg))


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    name = {r["id"]: f"{r['ticker']} ({r['sector']})" for r in con.execute(
        "SELECT c.id,c.ticker,s.name sector FROM companies c JOIN sectors s ON s.id=c.sector_id")}

    # 1. NULLs --------------------------------------------------------------
    for col in ("revenue", "ebitda", "net_income", "net_debt", "fcf", "gross_margin", "revenue_usd"):
        n = con.execute(f"SELECT COUNT(*) FROM financials WHERE {col} IS NULL").fetchone()[0]
        tot = con.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
        if n:
            add("NULLs", "INFO" if n / tot < 0.1 else "WARN", f"financials.{col}: {n}/{tot} rows NULL")
    for col in ("ev_ebitda", "pe", "dividend_yield", "market_cap"):
        rows = [name[r[0]] for r in con.execute(f"SELECT company_id FROM valuations WHERE {col} IS NULL")]
        if rows:
            add("NULLs", "INFO", f"valuations.{col} NULL for: {', '.join(rows)}")
    for cid, n in con.execute("SELECT company_id, COUNT(*) FROM data_gaps GROUP BY 1 HAVING COUNT(*)>=6"):
        add("NULLs", "INFO", f"{name[cid]} has {n} recorded data_gaps")
    for f in ("accounting_standard", "ownership_notes", "fiscal_year_end"):
        miss = [r["ticker"] for r in con.execute(f"SELECT ticker FROM companies WHERE {f} IS NULL")]
        if miss:
            add("NULLs", "WARN" if f != "ownership_notes" else "INFO", f"companies.{f} NULL for {len(miss)}: {', '.join(miss)}")

    # 2. Outliers -----------------------------------------------------------
    def rng(section, table, col, lo, hi):
        for r in con.execute(f"SELECT company_id, {col} v FROM {table} WHERE {col} IS NOT NULL AND ({col}<? OR {col}>?)", (lo, hi)):
            add(section, "WARN", f"{name[r['company_id']]}: {table}.{col}={r['v']:.3g} outside [{lo}, {hi}]")
    lo, hi = POL["ebitda_margin_range"]; rng("Outliers", "financials", "ebitda_margin", lo, hi)
    lo, hi = POL["gross_margin_range"]; rng("Outliers", "financials", "gross_margin", lo, hi)
    lo, hi = POL["ev_ebitda_range"]; rng("Outliers", "valuations", "ev_ebitda", lo, hi)
    lo, hi = POL["pe_range"]; rng("Outliers", "valuations", "pe", lo, hi)
    rng("Outliers", "valuations", "dividend_yield", 0, POL["dividend_yield_max"])
    prev = {}
    for r in con.execute("SELECT company_id,period_end,revenue FROM financials WHERE revenue IS NOT NULL ORDER BY company_id,period_end"):
        p = prev.get(r["company_id"])
        if p and p > 0 and abs(r["revenue"] / p - 1) > POL["revenue_yoy_abs_change_max"]:
            add("Outliers", "WARN", f"{name[r['company_id']]}: revenue change {r['revenue']/p-1:+.0%} into {r['period_end']}")
        prev[r["company_id"]] = r["revenue"]

    # 3. Unit / currency mismatches ------------------------------------------
    for r in con.execute("SELECT f.company_id,f.period_end,f.currency fc,c.currency cc,f.revenue,f.revenue_usd,f.fx_rate FROM financials f JOIN companies c ON c.id=f.company_id"):
        if r["fc"] != r["cc"]:
            add("Units/currency", "ERROR", f"{name[r['company_id']]} {r['period_end']}: financials currency {r['fc']} != company currency {r['cc']}")
        if r["revenue"] and r["revenue_usd"] and r["fx_rate"] and abs(r["revenue"] * r["fx_rate"] / r["revenue_usd"] - 1) > 1e-6:
            add("Units/currency", "ERROR", f"{name[r['company_id']]} {r['period_end']}: revenue_usd != revenue*fx_rate")
        if r["fc"] != "USD" and not r["fx_rate"]:
            add("Units/currency", "ERROR", f"{name[r['company_id']]} {r['period_end']}: non-USD without fx_rate")
        if r["revenue"] and r["revenue"] < 1e5:
            add("Units/currency", "WARN", f"{name[r['company_id']]} {r['period_end']}: revenue {r['revenue']} looks like millions, not full units")
    for r in con.execute("SELECT v.company_id,v.currency vc,c.currency cc FROM valuations v JOIN companies c ON c.id=v.company_id WHERE v.currency!=c.currency"):
        add("Units/currency", "WARN", f"{name[r['company_id']]}: valuation currency {r['vc']} != reporting currency {r['cc']}")
    for r in con.execute("""SELECT v.company_id,v.ev_ebitda,v.enterprise_value,f.ebitda FROM valuations v JOIN financials f ON f.company_id=v.company_id
                            AND f.period_end=(SELECT MAX(period_end) FROM financials WHERE company_id=v.company_id)
                            WHERE v.ev_ebitda IS NOT NULL AND f.ebitda>0 AND v.enterprise_value>0"""):
        calc = r["enterprise_value"] / r["ebitda"]
        if abs(calc / r["ev_ebitda"] - 1) > 0.35:
            add("Units/currency", "WARN", f"{name[r['company_id']]}: reported EV/EBITDA {r['ev_ebitda']:.1f}x vs EV/latest-FY EBITDA {calc:.1f}x "
                "(TTM vs FY difference, or perimeter change) - treat multiple with caution")

    # 4. Duplicates ---------------------------------------------------------
    for t, keys in (("financials", "company_id,period_end,period_type"), ("valuations", "company_id,as_of_date"),
                    ("companies", "sector_id,ticker"), ("signals", "company_id,signal_type,value_text,signal_date")):
        for r in con.execute(f"SELECT {keys}, COUNT(*) n FROM {t} GROUP BY {keys} HAVING n>1"):
            add("Duplicates", "ERROR", f"{t}: duplicate rows on ({keys}) -> {tuple(r)[:-1]} x{r['n']}")
    dup_names = con.execute("SELECT name,COUNT(*) FROM companies GROUP BY name HAVING COUNT(*)>1").fetchall()
    for n, c in dup_names:
        add("Duplicates", "WARN", f"company name '{n}' appears {c} times")

    # 5. Orphans ------------------------------------------------------------
    for t in ("companies", "financials", "valuations", "signals", "sector_metrics_defense"):
        n = con.execute(f"SELECT COUNT(*) FROM {t} WHERE source_id IS NOT NULL AND source_id NOT IN (SELECT id FROM sources)").fetchone()[0]
        if n:
            add("Orphans", "ERROR", f"{t}: {n} rows with source_id not in sources")
        nn = con.execute(f"SELECT COUNT(*) FROM {t} WHERE source_id IS NULL").fetchone()[0]
        if nn:
            add("Orphans", "ERROR", f"{t}: {nn} rows with NULL source_id")
    for t, col in (("financials", "company_id"), ("valuations", "company_id"), ("signals", "company_id"),
                   ("sector_metrics_defense", "company_id"), ("data_gaps", "company_id")):
        n = con.execute(f"SELECT COUNT(*) FROM {t} WHERE {col} NOT IN (SELECT id FROM companies)").fetchone()[0]
        if n:
            add("Orphans", "ERROR", f"{t}: {n} rows referencing missing companies")
    unused = con.execute("""SELECT COUNT(*) FROM sources WHERE id NOT IN (SELECT source_id FROM companies WHERE source_id IS NOT NULL)
        AND id NOT IN (SELECT source_id FROM financials) AND id NOT IN (SELECT source_id FROM valuations)
        AND id NOT IN (SELECT source_id FROM signals) AND id NOT IN (SELECT source_id FROM sector_metrics_defense WHERE source_id IS NOT NULL)""").fetchone()[0]
    if unused:
        add("Orphans", "INFO", f"{unused} sources rows are not referenced by any data row")

    # 6. Fiscal-year alignment ---------------------------------------------
    latest = {}
    for r in con.execute("SELECT c.id,c.fiscal_year_end fye,MAX(f.period_end) pe,s.name sector FROM companies c JOIN financials f ON f.company_id=c.id "
                         "JOIN sectors s ON s.id=c.sector_id GROUP BY c.id"):
        latest[r["id"]] = (r["sector"], r["pe"])
        if r["fye"] and abs(int(r["pe"][5:7]) - int(r["fye"][:2])) > 1 and abs(int(r["pe"][5:7]) - int(r["fye"][:2])) != 11:
            add("Fiscal-year alignment", "WARN", f"{name[r['id']]}: latest period {r['pe']} vs company fiscal_year_end {r['fye']}")
    by_sector = defaultdict(list)
    for cid, (s, pe) in latest.items():
        by_sector[s].append((pe, cid))
    for s, lst in by_sector.items():
        pes = sorted(lst)
        span = (date.fromisoformat(pes[-1][0]) - date.fromisoformat(pes[0][0])).days
        if span > 180:
            add("Fiscal-year alignment", "WARN", f"{s}: latest fiscal periods span {span} days ({pes[0][0]} {name[pes[0][1]]} .. {pes[-1][0]} "
                f"{name[pes[-1][1]]}); cross-company comparisons are not calendar-aligned")
        months = {pe[5:7] for pe, _ in lst}
        if len(months) > 1:
            add("Fiscal-year alignment", "INFO", f"{s}: fiscal year-end months differ: {sorted(months)}")
    for r in con.execute("SELECT m.company_id,MAX(m.period_end) mp,(SELECT MAX(period_end) FROM financials WHERE company_id=m.company_id) fp "
                         "FROM sector_metrics_defense m GROUP BY m.company_id"):
        gap = abs((date.fromisoformat(r["mp"]) - date.fromisoformat(r["fp"])).days)
        if gap > 7:
            add("Fiscal-year alignment", "INFO", f"{name[r['company_id']]}: latest defense-metric period {r['mp']} vs latest financials {r['fp']}")
    for r in con.execute("SELECT m.company_id,m.period_end,f.period_end fp FROM sector_metrics_defense m JOIN financials f ON f.company_id=m.company_id "
                         "AND ABS(julianday(f.period_end)-julianday(m.period_end)) BETWEEN 1 AND 7"):
        add("Fiscal-year alignment", "INFO", f"{name[r['company_id']]}: backlog period {r['period_end']} vs yfinance period {r['fp']} differ by a few days (52/53-week year)")

    # 7. Verification consistency ----------------------------------------------
    for r in con.execute("SELECT id,company_id,period_end,verify_note FROM financials WHERE verified=1"):
        fv = {x["field"]: x["status"] for x in con.execute("SELECT field,status FROM field_verification WHERE company_id=? AND period_end=?",
                                                          (r["company_id"], r["period_end"]))}
        note = r["verify_note"] or ""
        if "mismatch" in note.lower() or "mismatch" in fv.values():
            add("Verification", "ERROR", f"{name[r['company_id']]} {r['period_end']}: verified=1 but a checked field mismatches the filing ({note[:120]})")
        if fv and not (fv.get("revenue") == "match" and fv.get("net_income") == "match"):
            add("Verification", "ERROR", f"{name[r['company_id']]} {r['period_end']}: verified=1 without revenue AND net_income matching a filing ({fv})")
    n = con.execute("SELECT COUNT(*) FROM field_verification WHERE source_id NOT IN (SELECT id FROM sources)").fetchone()[0]
    if n:
        add("Verification", "ERROR", f"field_verification: {n} rows with source_id not in sources")
    for st, c in con.execute("SELECT status,COUNT(*) FROM field_verification GROUP BY status"):
        add("Verification", "INFO", f"field checks vs filings: {c} x {st}")

    # report ----------------------------------------------------------------
    errors = 0
    order = ["NULLs", "Outliers", "Units/currency", "Duplicates", "Orphans", "Fiscal-year alignment", "Verification"]
    print(f"DB validation report - {DB.name} - {date.today()}\n" + "=" * 60)
    for sec in order:
        items = findings.get(sec, [])
        print(f"\n## {sec}: {'OK' if not items else f'{len(items)} finding(s)'}")
        for lvl, msg in sorted(items, key=lambda x: ["ERROR", "WARN", "INFO"].index(x[0])):
            print(f"  [{lvl}] {msg}")
            errors += lvl == "ERROR"
    print(f"\nSummary: {errors} ERROR, {sum(l == 'WARN' for v in findings.values() for l, _ in v)} WARN, "
          f"{sum(l == 'INFO' for v in findings.values() for l, _ in v)} INFO")
    if "--strict" in sys.argv and errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
