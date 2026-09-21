"""Write data/verification_sheet.csv: a random, sector-stratified sample of numeric DB values that are still UNVERIFIED (SEC-verified rows are excluded) with source URLs,
so a human can spot-check them against filings and mark them verified.

Usage: python scripts/make_verification_sheet.py [--n 30] [--seed 7]
Workflow: fill `filing_value` + `verified_ok` (1/0) -> python scripts/apply_verification.py -> python scripts/build_db.py
(verified=1 is applied per ROW; only mark a row OK if its sampled value AND the rest of the row look right.)
"""
import argparse
import csv
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIN = ["revenue", "ebitda", "net_income", "net_debt", "fcf", "capex", "total_debt", "cash"]
VAL = ["market_cap", "enterprise_value", "ev_ebitda", "pe"]


def hint(table, metric, ticker, country):
    if table == "financials":
        where = ("SEC 10-K income statement/balance sheet/cash-flow statement" if country in ("US", "Israel")
                 else "annual report (IFRS consolidated statements) on the company's IR site")
        return f"Find '{metric}' in the {where} for the period; EBITDA/net_debt/fcf are Yahoo-derived, so reconstruct from statements"
    if table == "valuations":
        return "Market data snapshot: compare with the exchange/Yahoo quote page on/after the retrieval date"
    if table == "sector_metrics_defense":
        return "Search the 10-K/20-F for 'remaining performance obligation'; compare with the value (USD)"
    return "Check the company's latest annual report / profile for employee count"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    con = sqlite3.connect(ROOT / "data" / "finance.db")
    con.row_factory = sqlite3.Row
    pool = []
    for m in FIN:
        for r in con.execute(f"SELECT f.id,s.name sector,c.ticker,c.country,f.period_end,f.currency,f.{m} v,f.verified,so.url,so.retrieved_at "
                             "FROM financials f JOIN companies c ON c.id=f.company_id JOIN sectors s ON s.id=c.sector_id "
                             f"JOIN sources so ON so.id=f.source_id WHERE f.{m} IS NOT NULL AND f.verified=0"):
            pool.append(("financials", r["id"], m, r))
    for m in VAL:
        for r in con.execute(f"SELECT v.id,s.name sector,c.ticker,c.country,v.as_of_date period_end,v.currency,v.{m} v,v.verified,so.url,so.retrieved_at "
                             "FROM valuations v JOIN companies c ON c.id=v.company_id JOIN sectors s ON s.id=c.sector_id "
                             f"JOIN sources so ON so.id=v.source_id WHERE v.{m} IS NOT NULL AND v.verified=0"):
            pool.append(("valuations", r["id"], m, r))
    for r in con.execute("SELECT m.rowid id,s.name sector,c.ticker,c.country,m.period_end,'USD' currency,m.order_backlog v,m.verified,so.url,so.retrieved_at "
                         "FROM sector_metrics_defense m JOIN companies c ON c.id=m.company_id JOIN sectors s ON s.id=c.sector_id "
                         "JOIN sources so ON so.id=m.source_id WHERE m.order_backlog IS NOT NULL"):
        pool.append(("sector_metrics_defense", r["id"], "order_backlog", r))
    for r in con.execute("SELECT g.id,s.name sector,c.ticker,c.country,g.signal_date period_end,'' currency,g.value_num v,g.verified,so.url,so.retrieved_at "
                         "FROM signals g JOIN companies c ON c.id=g.company_id JOIN sectors s ON s.id=c.sector_id "
                         "JOIN sources so ON so.id=g.source_id WHERE g.signal_type='headcount' AND g.value_num IS NOT NULL"):
        pool.append(("signals", r["id"], "headcount", r))

    rnd = random.Random(a.seed)
    sectors = sorted({p[3]["sector"] for p in pool})
    per = a.n // len(sectors)
    picked = []
    for s in sectors:
        cand = [p for p in pool if p[3]["sector"] == s]
        rnd.shuffle(cand)
        seen, take = set(), []
        for p in cand:                       # spread across companies first
            if p[3]["ticker"] not in seen or len(seen) >= 8:
                take.append(p)
                seen.add(p[3]["ticker"])
            if len(take) == per:
                break
        picked += take
    out = ROOT / "data" / "verification_sheet.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sample_id", "sector", "ticker", "table", "row_id", "metric", "period_or_date", "db_value", "currency",
                    "source_url", "retrieved_at", "how_to_check", "filing_value", "verified_ok(1/0)", "notes"])
        for i, (t, rid, m, r) in enumerate(picked, 1):
            w.writerow([i, r["sector"], r["ticker"], t, rid, m, r["period_end"], r["v"], r["currency"], r["url"], r["retrieved_at"],
                        hint(t, m, r["ticker"], r["country"]), "", "", ""])
    print(f"wrote {out} ({len(picked)} rows from {len(sectors)} sectors)")


if __name__ == "__main__":
    main()
