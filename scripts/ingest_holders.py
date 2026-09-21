"""Pull shareholder data from yfinance into data/raw/holders/<sector>/<ticker>.json.

Gives insider % / institution % and the top institutional holders with their % of shares outstanding.
This is holder DATA, not a control analysis: build_db.py turns it into a factual ownership note and
never asserts "no controlling shareholder" beyond what the numbers show.
Usage: python scripts/ingest_holders.py [sector ...]
"""
import json
import math
import sys
from datetime import date
from pathlib import Path

import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]


def num(x):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    except (TypeError, ValueError):
        return None


def main():
    cfg = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]
    for s in sys.argv[1:] or [k for k, v in cfg.items() if v.get("companies")]:
        for c in cfg[s]["companies"]:
            t = yf.Ticker(c["ticker"])
            major, top = {}, []
            try:
                mh = t.major_holders
                if mh is not None and not mh.empty:
                    major = {str(k): num(v) for k, v in mh["Value"].items()}
            except Exception as e:                                                      # noqa: BLE001
                major = {"error": str(e)[:100]}
            try:
                ih = t.institutional_holders
                if ih is not None and not ih.empty:
                    for _, r in ih.head(5).iterrows():
                        top.append({"holder": str(r.get("Holder")), "pct": num(r.get("pctHeld")),
                                    "reported": str(r.get("Date Reported"))[:10]})
            except Exception:                                                           # noqa: BLE001
                pass
            out = ROOT / "data" / "raw" / "holders" / s
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{c['ticker']}.json").write_text(json.dumps({
                "ticker": c["ticker"], "retrieved_at": date.today().isoformat(),
                "source_url": f"https://finance.yahoo.com/quote/{c['ticker']}/holders",
                "major_holders": major, "top_institutional": top}, indent=1), encoding="utf-8", newline="\n")
            print(f"{s}/{c['ticker']}: major={list(major)[:4]} top={[(h['holder'][:18], h['pct']) for h in top[:2]]}")


if __name__ == "__main__":
    main()
