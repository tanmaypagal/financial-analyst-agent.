"""Turn your filled-in data/verification_sheet.csv into data/curated/verified_rows.csv (kept across rebuilds).
Rows with verified_ok=1 are recorded by natural key (not row id), then build_db.py sets verified=1 on them.
Usage: python scripts/apply_verification.py && python scripts/build_db.py
"""
import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
con = sqlite3.connect(ROOT / "data" / "finance.db")
existing = {}
out = ROOT / "data" / "curated" / "verified_rows.csv"
if out.exists():
    for r in csv.DictReader(out.open(encoding="utf-8")):
        existing[(r["table"], r["ticker"], r["period_or_date"], r["metric"])] = r
n = 0
for r in csv.DictReader((ROOT / "data" / "verification_sheet.csv").open(encoding="utf-8")):
    if r["verified_ok(1/0)"].strip() == "1":
        existing[(r["table"], r["ticker"], r["period_or_date"], r["metric"])] = {
            "table": r["table"], "ticker": r["ticker"], "period_or_date": r["period_or_date"], "metric": r["metric"],
            "filing_value": r["filing_value"], "notes": r["notes"]}
        n += 1
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, lineterminator="\n", fieldnames=["table", "ticker", "period_or_date", "metric", "filing_value", "notes"])
    w.writeheader()
    w.writerows(existing.values())
print(f"{n} rows marked OK this run; {len(existing)} total in {out.name}. Now run: python scripts/build_db.py")
