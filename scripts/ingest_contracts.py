"""Pull recent US federal contract awards (USAspending.gov, official API) for companies with `contract_recipients`.

Awards SIGNED in the last 180 days, largest first, de-duplicated. Recipient names are searched by text, so a parent-name
search can return subsidiaries - the recipient entity is always kept in the stored text so the reader can judge.
Usage: python scripts/ingest_contracts.py [sector]     -> data/raw/contracts/<sector>/<ticker>.json
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.usaspending.gov/api/v2/search/spending_by_award/"


def main():
    cfg = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]
    end, start = date.today(), date.today() - timedelta(days=180)
    for s in sys.argv[1:] or [k for k, v in cfg.items() if v.get("companies")]:
        for c in cfg[s]["companies"]:
            names = c.get("contract_recipients")
            if not names:
                continue
            body = {"filters": {"time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat(), "date_type": "date_signed"}],
                                "award_type_codes": ["A", "B", "C", "D"], "recipient_search_text": names},
                    "fields": ["Award ID", "Recipient Name", "Award Amount", "Description", "Start Date", "Awarding Agency", "generated_internal_id"],
                    "sort": "Award Amount", "order": "desc", "limit": 12, "page": 1}
            r = requests.post(API, json=body, timeout=90)
            if r.status_code != 200:
                print(f"{c['ticker']}: HTTP {r.status_code}")
                continue
            seen, awards = set(), []
            for x in r.json().get("results", []):
                key = (x["Recipient Name"], x["Award Amount"], (x["Description"] or "")[:40])
                if key in seen or not x.get("generated_internal_id"):
                    continue
                seen.add(key)
                awards.append(x)
            out = ROOT / "data" / "raw" / "contracts" / s
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{c['ticker']}.json").write_text(json.dumps({
                "ticker": c["ticker"], "retrieved_at": end.isoformat(), "window": [start.isoformat(), end.isoformat()],
                "search_terms": names, "source": "USAspending.gov spending_by_award (date_signed)", "awards": awards[:5]}, indent=1), encoding="utf-8", newline="\n")
            print(f"{c['ticker']}: {len(awards[:5])} awards, top: {awards[0]['Recipient Name'] if awards else None}")


if __name__ == "__main__":
    main()
