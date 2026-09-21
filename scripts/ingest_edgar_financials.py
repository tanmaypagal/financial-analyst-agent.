"""Pull annual (10-K / 20-F) XBRL facts from SEC EDGAR for every SEC-registered company in sectors.yaml.

Used by build_db.py to CROSS-CHECK the yfinance numbers against the filings (and mark matching rows verified).
Usage: python scripts/ingest_edgar_financials.py [sector ...]
Writes data/raw/edgar/<sector>/<ticker>_facts.json  (annual facts only, last ~6 fiscal years).
CIKs come from sectors.yaml `cik`, else from SEC's official ticker->CIK map. Non-SEC companies are skipped (no filing feed).
"""
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "financial-analyst-agent-takehome gunjit.999@gmail.com"}
TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
}


def annual_facts(facts: dict, tags: list[str]) -> list[dict]:
    out = []
    for tag in tags:
        node = facts.get("us-gaap", {}).get(tag)
        if not node:
            continue
        for it in node["units"].get("USD", []):
            if it.get("form") not in ("10-K", "10-K/A", "20-F", "20-F/A") or "start" not in it:
                continue
            days = (date.fromisoformat(it["end"]) - date.fromisoformat(it["start"])).days
            if 340 <= days <= 380:
                out.append({"tag": tag, "start": it["start"], "end": it["end"], "val": it["val"], "form": it["form"],
                            "filed": it["filed"], "accn": it["accn"]})
    cutoff = f"{date.today().year - 6}-01-01"
    return [o for o in out if o["end"] >= cutoff]


def main():
    cfg = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]
    wanted = sys.argv[1:] or [s for s, v in cfg.items() if v.get("companies")]
    tmap = {v["ticker"].upper(): v["cik_str"] for v in requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=60).json().values()}
    for s in wanted:
        for c in cfg[s]["companies"]:
            cik = c.get("cik") or tmap.get(c["ticker"].upper())
            if not cik:
                print(f"skip {c['ticker']}: not an SEC registrant (no filing feed)")
                continue
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
            r = requests.get(url, headers=HEADERS, timeout=90)
            time.sleep(0.2)                                    # SEC fair-access limit is 10 req/s
            if r.status_code != 200:
                print(f"EDGAR {c['ticker']}: HTTP {r.status_code}")
                continue
            facts = r.json().get("facts", {})
            data = {k: annual_facts(facts, t) for k, t in TAGS.items()}
            out = ROOT / "data" / "raw" / "edgar" / s
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{c['ticker']}_facts.json").write_text(json.dumps({
                "ticker": c["ticker"], "cik": int(cik), "retrieved_at": date.today().isoformat(), "source_url": url,
                "facts": data}, indent=1), encoding="utf-8", newline="\n")
            print(f"ok  {s}/{c['ticker']}: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))


if __name__ == "__main__":
    main()
