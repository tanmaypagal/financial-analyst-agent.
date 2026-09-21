"""Pull raw market/financial data from yfinance into data/raw/ (no interpretation, no estimates).

Usage: python scripts/ingest_yfinance.py [sector ...]     (default: all sectors with companies)

Writes:
  data/raw/yfinance/<sector>/<ticker>.json   info subset, annual income/balance/cashflow, news
  data/raw/fx/<CUR>USD.json                  daily FX closes (native -> USD)
Every file carries `retrieved_at`. Anything yfinance does not return is simply absent; build_db.py
turns absences into NULL + data_gaps.
"""
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

INFO_KEYS = [
    "longName", "currency", "financialCurrency", "exchange", "fullExchangeName", "country",
    "longBusinessSummary", "fullTimeEmployees", "marketCap", "enterpriseValue", "trailingPE",
    "enterpriseToEbitda", "enterpriseToRevenue", "dividendYield", "trailingAnnualDividendYield",
    "dividendRate", "currentPrice", "regularMarketPrice", "lastFiscalYearEnd",
    "mostRecentQuarter", "sector", "industry", "heldPercentInsiders", "heldPercentInstitutions",
]
INC_ROWS = ["Total Revenue", "EBITDA", "EBIT", "Net Income", "Gross Profit", "Diluted EPS"]
BS_ROWS = ["Total Debt", "Net Debt", "Cash And Cash Equivalents",
           "Cash Cash Equivalents And Short Term Investments"]
CF_ROWS = ["Capital Expenditure", "Free Cash Flow", "Operating Cash Flow"]


def clean(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return None


def frame_to_dict(df, rows):
    out = {}
    if df is None or df.empty:
        return out
    for col in df.columns:
        d = str(col.date())
        out[d] = {r: clean(df.loc[r, col]) for r in rows if r in df.index}
    return out


def pull_company(sector, c, retrieved_at):
    t = yf.Ticker(c["ticker"])
    info = t.info or {}
    news = []
    try:
        for n in (t.news or [])[:10]:
            cont = n.get("content", n)
            url = (cont.get("canonicalUrl") or cont.get("clickThroughUrl") or {}).get("url") or cont.get("link")
            news.append({
                "title": cont.get("title"),
                "publisher": (cont.get("provider") or {}).get("displayName") or cont.get("publisher"),
                "published": cont.get("pubDate") or cont.get("providerPublishTime"),
                "url": url,
            })
    except Exception as e:  # news is best effort
        news = [{"error": str(e)}]
    doc = {
        "ticker": c["ticker"], "sector": sector, "retrieved_at": retrieved_at,
        "source_url": f"https://finance.yahoo.com/quote/{c['ticker']}",
        "info": {k: clean(info.get(k)) for k in INFO_KEYS},
        "income_stmt": frame_to_dict(t.income_stmt, INC_ROWS),
        "balance_sheet": frame_to_dict(t.balance_sheet, BS_ROWS),
        "cashflow": frame_to_dict(t.cashflow, CF_ROWS),
        "news": news,
    }
    out = RAW / "yfinance" / sector
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{c['ticker']}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8", newline="\n")
    return doc


def pull_fx(cur, retrieved_at):
    if cur in ("USD", None):
        return
    pair = f"{cur}USD=X"
    hist = yf.Ticker(pair).history(start="2021-01-01", end=date.today().isoformat(), auto_adjust=False)
    series = {str(i.date()): round(float(v), 6) for i, v in hist["Close"].items() if v == v}
    out = RAW / "fx"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cur}USD.json").write_text(json.dumps({
        "pair": pair, "retrieved_at": retrieved_at,
        "source_url": f"https://finance.yahoo.com/quote/{pair}", "closes": series}), encoding="utf-8", newline="\n")


def main():
    cfg = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]
    wanted = sys.argv[1:] or [s for s, v in cfg.items() if v.get("companies")]
    retrieved_at = date.today().isoformat()
    currencies = set()
    for s in wanted:
        for c in cfg[s]["companies"]:
            for attempt in range(3):
                try:
                    d = pull_company(s, c, retrieved_at)
                    fc = d["info"].get("financialCurrency")
                    currencies.add("GBP" if fc == "GBp" else fc)
                    cur = d["info"].get("currency")
                    currencies.add("GBP" if cur == "GBp" else cur)
                    print(f"ok  {s}/{c['ticker']}  years={list(d['income_stmt'])[:4]}")
                    break
                except Exception as e:
                    print(f"retry {c['ticker']}: {e}")
                    time.sleep(2)
            else:
                print(f"FAILED {c['ticker']} - will be recorded as a data gap by build_db")
    for cur in sorted(x for x in currencies if x):
        pull_fx(cur, retrieved_at)
        print("fx", cur)


if __name__ == "__main__":
    main()
