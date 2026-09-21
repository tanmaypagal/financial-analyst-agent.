"""Pull remaining-performance-obligation (RPO) facts from SEC EDGAR XBRL for companies with a `cik`.

RPO is the closest machine-readable proxy for order backlog. It is NOT identical to the
company-defined "backlog" some primes report; build_db.py stores that caveat in the notes column.

Usage: python scripts/ingest_edgar.py [sector ...]
Writes data/raw/edgar/<sector>/<ticker>_rpo.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "financial-analyst-agent-takehome gunjit.999@gmail.com"}


def main():
    cfg = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]
    wanted = sys.argv[1:] or [s for s, v in cfg.items() if v.get("companies")]
    for s in wanted:
        for c in cfg[s]["companies"]:
            if not c.get("cik"):
                continue
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c['cik']):010d}.json"
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code != 200:
                print(f"EDGAR {c['ticker']}: HTTP {r.status_code} (will be a data gap)")
                continue
            tag = r.json().get("facts", {}).get("us-gaap", {}).get("RevenueRemainingPerformanceObligation")
            rows = []
            if tag:
                for unit, items in tag["units"].items():
                    for it in items:
                        rows.append({"end": it["end"], "val": it["val"], "unit": unit, "form": it.get("form"),
                                     "fy": it.get("fy"), "fp": it.get("fp"), "filed": it.get("filed"),
                                     "accn": it.get("accn")})
            out = ROOT / "data" / "raw" / "edgar" / s
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{c['ticker']}_rpo.json").write_text(json.dumps({
                "ticker": c["ticker"], "cik": c["cik"], "retrieved_at": date.today().isoformat(),
                "source_url": url, "rpo": rows}, indent=1), encoding="utf-8", newline="\n")
            print(f"ok  {c['ticker']}: {len(rows)} RPO facts")


if __name__ == "__main__":
    main()
