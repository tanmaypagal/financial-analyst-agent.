"""Rebuild data/finance.db from data/raw/* (ingest scripts) and data/curated/*.csv.

Rules enforced here:
  * nothing is estimated: a value missing from the source is NULL and produces a data_gaps row
  * derived fields (margins, net_debt when the source omits it, USD conversion) are arithmetic on
    sourced numbers only
  * every row carries source_id + as_of_date; verified is always 0 unless a curated CSV says 1

Usage: python scripts/build_db.py
"""
import csv
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "finance.db"
RAW = ROOT / "data" / "raw"
CUR = ROOT / "data" / "curated"


TAG_ORDER = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements"],
}


def norm_cur(c):
    return "GBP" if c == "GBp" else c


def num(x):
    return None if x in (None, "") else float(x)


class Builder:
    def __init__(self):
        if DB.exists():
            DB.unlink()
        self.db = sqlite3.connect(DB)
        self.db.executescript((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))
        self._src = {}
        self.fx = {p.stem.replace("USD", ""): json.loads(p.read_text(encoding="utf-8"))
                   for p in (RAW / "fx").glob("*.json")} if (RAW / "fx").exists() else {}

    def source(self, url, publisher, doc_type, retrieved_at):
        key = (url, publisher, doc_type, retrieved_at)
        if key not in self._src:
            cur = self.db.execute("INSERT INTO sources(url,publisher,doc_type,retrieved_at) VALUES (?,?,?,?)", key)
            self._src[key] = cur.lastrowid
        return self._src[key]

    def gap(self, cid, field, reason):
        self.db.execute("INSERT INTO data_gaps(company_id,field,reason) VALUES (?,?,?)", (cid, field, reason))

    def fx_rate(self, cur, on_date):
        """Latest FX close on/before on_date. Returns (rate, actual_date) or (None, None)."""
        if cur == "USD":
            return 1.0, on_date
        closes = self.fx.get(cur, {}).get("closes", {})
        ds = [d for d in closes if d <= on_date]
        if not ds:
            return None, None
        d = max(ds)
        return closes[d], d

    # ------------------------------------------------------------------
    def build(self):
        cfg = yaml.safe_load((ROOT / "config" / "sectors.yaml").read_text(encoding="utf-8"))["sectors"]
        meta = {}
        p = CUR / "company_metadata.csv"
        if p.exists():
            meta = {r["ticker"]: r for r in csv.DictReader(p.open(encoding="utf-8"))}
        ids = {}
        for sname, sc in cfg.items():
            sid = self.db.execute("INSERT INTO sectors(name) VALUES (?)", (sname,)).lastrowid
            for c in sc.get("companies") or []:
                raw_path = RAW / "yfinance" / sname / f"{c['ticker']}.json"
                if not raw_path.exists():
                    print(f"skip {c['ticker']}: no raw pull")
                    continue
                cid = self.load_company(sid, sname, c, json.loads(raw_path.read_text(encoding="utf-8")),
                                        meta.get(c["ticker"]))
                ids[(sname, c["ticker"])] = cid
                if sname == "defense":
                    self.load_defense_edgar(cid, sname, c)
        self.load_curated_defense(ids)
        self.load_contracts(ids)
        self.load_curated_signals(ids)
        self.load_caveats(ids)
        self.verify_edgar(ids)
        self.apply_verified()
        self.db.commit()
        n = {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in ("companies", "financials", "valuations", "signals", "sector_metrics_defense", "data_gaps", "sources")}
        print("built", DB, n)
        try:                                                    # keep the README verification caveat in sync with the data
            import update_readme_caveats
            update_readme_caveats.main()
        except Exception as e:                                  # noqa: BLE001 - a docs problem must not break a data build
            print("README caveat not updated:", e)

    # ------------------------------------------------------------------
    def load_company(self, sid, sname, c, raw, meta):
        info, rt = raw["info"], raw["retrieved_at"]
        src = self.source(raw["source_url"], "Yahoo Finance (via yfinance)", "aggregator_api", rt)
        cur = norm_cur(info.get("financialCurrency"))
        fye = None
        if info.get("lastFiscalYearEnd"):
            fye = datetime.fromtimestamp(info["lastFiscalYearEnd"], tz=timezone.utc).strftime("%m-%d")
        acct = own = None
        if meta:
            acct = meta["accounting_standard"] or None
            own = meta["ownership_notes"] or None
            if own and meta.get("ownership_source_url"):
                own += f" [source: {meta['ownership_source_url']}; retrieved {meta['retrieved_at']}; unverified]"
        own = self.holder_note(sname, c, own)
        cid = self.db.execute(
            "INSERT INTO companies(sector_id,ticker,name,country,exchange,currency,fiscal_year_end,"
            "accounting_standard,ownership_notes,description,source_id,as_of_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, c["ticker"], c["name"], c.get("country"), c.get("exchange"), cur, fye, acct, own,
             info.get("longBusinessSummary"), src, rt)).lastrowid
        if not fye:
            self.gap(cid, "companies.fiscal_year_end", "not returned by yfinance")
        if not acct:
            self.gap(cid, "companies.accounting_standard", "no sourced value in curated CSV")
        if not own:
            self.gap(cid, "companies.ownership_notes", "no sourced ownership info in curated CSV (unknown, not 'none')")
        if not info.get("longBusinessSummary"):
            self.gap(cid, "companies.description", "not returned by yfinance")

        # ---- financials ------------------------------------------------
        inc, bs, cf = raw["income_stmt"], raw["balance_sheet"], raw["cashflow"]
        for pe in sorted(inc, reverse=True):
            i, b, f = inc.get(pe, {}), bs.get(pe, {}), cf.get(pe, {})
            rev, ebitda, ebit, ni = (i.get("Total Revenue"), i.get("EBITDA"), i.get("EBIT"), i.get("Net Income"))
            if rev is None and ebitda is None and ni is None:
                continue                    # yfinance pads old years with all-NaN columns; not a real period
            gp = i.get("Gross Profit")
            debt = b.get("Total Debt")
            cash = b.get("Cash And Cash Equivalents")
            nd = b.get("Net Debt")
            if nd is None and debt is not None and cash is not None:
                nd = debt - cash            # derived; documented in README
            capex = f.get("Capital Expenditure")
            capex = abs(capex) if capex is not None else None
            fcf = f.get("Free Cash Flow")
            gm = gp / rev if gp is not None and rev else None
            em = ebitda / rev if ebitda is not None and rev else None
            rate, fxd = self.fx_rate(cur, pe)
            rev_usd = rev * rate if rev is not None and rate else None
            self.db.execute(
                "INSERT INTO financials(company_id,period_end,period_type,revenue,ebitda,ebit,net_income,gross_margin,"
                "ebitda_margin,net_debt,total_debt,cash,capex,fcf,currency,revenue_usd,fx_rate,fx_date,source_id,as_of_date,verified)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (cid, pe, "FY", rev, ebitda, ebit, ni, gm, em, nd, debt, cash, capex, fcf, cur, rev_usd, rate, fxd, src, rt))
            for col, v in dict(revenue=rev, ebitda=ebitda, ebit=ebit, net_income=ni, gross_margin=gm, net_debt=nd,
                               total_debt=debt, cash=cash, capex=capex, fcf=fcf, revenue_usd=rev_usd).items():
                if v is None:
                    self.gap(cid, f"financials.{col}@{pe}", "not provided by yfinance for this period (NULL, not estimated)")
        if not inc:
            self.gap(cid, "financials.*", "yfinance returned no annual statements")

        # ---- valuation (live snapshot as of retrieval date) -----------------
        dy = info.get("dividendYield")
        dy = dy / 100 if dy is not None else None      # yfinance reports percent
        if dy is None and info.get("trailingAnnualDividendYield") == 0:
            dy = 0.0                                    # Yahoo's trailing annual yield is 0.0: a sourced "pays no dividend", not a gap
        vcur = norm_cur(info.get("financialCurrency") or info.get("currency"))
        rate, fxd = self.fx_rate(vcur, rt)
        mc = info.get("marketCap")
        self.db.execute(
            "INSERT INTO valuations(company_id,as_of_date,market_cap,enterprise_value,ev_ebitda,pe,ev_sales,dividend_yield,"
            "currency,market_cap_usd,fx_rate,fx_date,source_id,verified) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
            (cid, rt, mc, info.get("enterpriseValue"), info.get("enterpriseToEbitda"), info.get("trailingPE"),
             info.get("enterpriseToRevenue"), dy, vcur, mc * rate if mc and rate else None, rate, fxd, src))
        for col, v in dict(market_cap=mc, enterprise_value=info.get("enterpriseValue"), ev_ebitda=info.get("enterpriseToEbitda"),
                           pe=info.get("trailingPE"), ev_sales=info.get("enterpriseToRevenue"), dividend_yield=dy).items():
            if v is None:
                self.gap(cid, f"valuations.{col}", "not returned by yfinance (e.g. no dividend, negative earnings, or no data)")

        # ---- signals: headcount + news ------------------------------------
        emp = info.get("fullTimeEmployees")
        if emp:
            fye_date = datetime.fromtimestamp(info["lastFiscalYearEnd"], tz=timezone.utc).strftime("%Y-%m-%d") \
                if info.get("lastFiscalYearEnd") else rt
            self.db.execute(
                "INSERT INTO signals(company_id,signal_type,value_num,value_text,signal_date,source_id,as_of_date,verified)"
                " VALUES (?,?,?,?,?,?,?,0)",
                (cid, "headcount", emp,
                 "fullTimeEmployees from Yahoo Finance profile; Yahoo does not date it, so signal_date is set to the last "
                 "fiscal year end (conservative for staleness)", fye_date, src, rt))
        else:
            self.gap(cid, "signals.headcount", "fullTimeEmployees not returned by yfinance")
        n_news = 0
        for n in raw.get("news", []):
            if not n.get("title") or not n.get("url") or not n.get("published"):
                continue
            pub = str(n["published"])[:10]
            nsrc = self.source(n["url"], n.get("publisher") or "unknown", "press", rt)
            self.db.execute(
                "INSERT INTO signals(company_id,signal_type,value_num,value_text,signal_date,source_id,as_of_date,verified)"
                " VALUES (?,?,?,?,?,?,?,0)", (cid, "news", None, n["title"], pub, nsrc, rt))
            n_news += 1
        if not n_news:
            self.gap(cid, "signals.news", "no headlines returned by yfinance")
        self.gap(cid, "signals.hiring", "no free structured hiring source; fill data/curated/signals_curated.csv")
        self.gap(cid, "signals.contract_award", "no free structured source; fill data/curated/signals_curated.csv")
        return cid

    def holder_note(self, sname, c, own):
        """Append a factual holder-data note (US-domiciled companies only: Yahoo's holder data only sees US 13F filers,
        so for non-US names it is misleading and is not used)."""
        p = RAW / "holders" / sname / f"{c['ticker']}.json"
        if c.get("country") != "US" or not p.exists():
            return own
        d = json.loads(p.read_text(encoding="utf-8"))
        ins = d["major_holders"].get("insidersPercentHeld")
        top = [h for h in d["top_institutional"] if h.get("pct")]
        if ins is None or not top:
            return own
        txt = (f"HOLDER DATA (Yahoo Finance, {d['retrieved_at']}): insiders hold {ins:.1%} of shares; top reported institutional holders: "
               + ", ".join(f"{h['holder']} {h['pct']:.1%}" for h in top[:3]) + ".")
        if ins < 0.20 and max(h["pct"] for h in top) < 0.20:
            txt += " No reported holder exceeds 20% (indicative only: no controlling shareholder visible in this data; not a proxy-statement check)."
        else:
            txt += " A holder above 20% or high insider ownership is present - check control before assuming a take-private is feasible."
        txt += f" [source: {d['source_url']}; unverified]"
        return f"{own} | {txt}" if own else txt

    # ------------------------------------------------------------------
    def load_defense_edgar(self, cid, sname, c):
        p = RAW / "edgar" / sname / f"{c['ticker']}_rpo.json"
        if not p.exists():
            self.gap(cid, "sector_metrics_defense.order_backlog",
                     "no SEC XBRL RPO source (non-US filer); fill data/curated/defense_metrics_curated.csv from annual report")
            return
        d = json.loads(p.read_text(encoding="utf-8"))
        best = {}
        for r in d["rpo"]:
            if r["unit"] != "USD":
                continue
            if r["end"] not in best or r["filed"] > best[r["end"]]["filed"]:
                best[r["end"]] = r
        fy = sorted((r for r in best.values() if r["form"] in ("10-K", "20-F")), key=lambda r: r["end"], reverse=True)[:3]
        latest_q = [r for r in best.values() if r["form"] == "10-Q" and (not fy or r["end"] > fy[0]["end"])]
        rows = fy + sorted(latest_q, key=lambda r: r["end"], reverse=True)[:1]
        if not rows:
            self.gap(cid, "sector_metrics_defense.order_backlog", "EDGAR returned no RPO facts")
            return
        for r in rows:
            url = f"https://www.sec.gov/Archives/edgar/data/{int(d['cik'])}/{r['accn'].replace('-', '')}/"
            src = self.source(url, "SEC EDGAR (XBRL companyfacts)", "sec_filing", d["retrieved_at"])
            self.db.execute(
                "INSERT INTO sector_metrics_defense(company_id,period_end,order_backlog,book_to_bill,govt_revenue_share_pct,"
                "export_revenue_share_pct,notes,source_id,as_of_date,verified) VALUES (?,?,?,?,?,?,?,?,?,1)",
                (cid, r["end"], r["val"], None, None, None,
                 f"order_backlog = us-gaap RevenueRemainingPerformanceObligation ({r['form']}, USD); a proxy, NOT the "
                 f"company-defined backlog. Value read directly from the SEC filing's XBRL (primary source).", src, d["retrieved_at"]))
        for col in ("book_to_bill", "govt_revenue_share_pct", "export_revenue_share_pct"):
            self.gap(cid, f"sector_metrics_defense.{col}", "not disclosed in machine-readable form; NULL, not estimated")

    def load_contracts(self, ids):
        """Recent US federal contract awards (USAspending.gov) -> signals(contract_award)."""
        for (sname, tk), cid in ids.items():
            p = RAW / "contracts" / sname / f"{tk}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            for a in d["awards"]:
                src = self.source(f"https://www.usaspending.gov/award/{a['generated_internal_id']}", "USAspending.gov (US Treasury)",
                                  "gov_database", d["retrieved_at"])
                text = (f"{a['Recipient Name']} - {a['Awarding Agency']} - {(a['Description'] or 'no description').strip()[:160]} "
                        f"(award {a['Award ID']}; amount is the award's total obligated value in USD)")
                self.db.execute(
                    "INSERT INTO signals(company_id,signal_type,value_num,value_text,signal_date,source_id,as_of_date,verified)"
                    " VALUES (?,?,?,?,?,?,?,0)", (cid, "contract_award", a["Award Amount"], text, a["Start Date"], src, d["retrieved_at"]))
            if d["awards"]:
                self.db.execute("DELETE FROM data_gaps WHERE company_id=? AND field='signals.contract_award'", (cid,))
                self.gap(cid, "signals.contract_award",
                         f"only US federal awards signed in the last 180 days, top 5 by value, found by recipient-name text search "
                         f"({d['search_terms']}); subsidiaries may appear and non-US government contracts are not covered")

    def load_curated_defense(self, ids):
        p = CUR / "defense_metrics_curated.csv"
        for r in csv.DictReader(p.open(encoding="utf-8")) if p.exists() else []:
            cid = ids.get(("defense", r["ticker"]))
            if not cid or not r["source_url"]:
                print("curated defense row skipped (unknown ticker or no source_url):", r["ticker"])
                continue
            src = self.source(r["source_url"], "curated", "curated_csv", r["retrieved_at"])
            self.db.execute(
                "INSERT OR REPLACE INTO sector_metrics_defense VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, r["period_end"], num(r["order_backlog"]), num(r["book_to_bill"]), num(r["govt_revenue_share_pct"]),
                 num(r["export_revenue_share_pct"]), r["notes"] or None, src, r["retrieved_at"], int(r["verified"] or 0)))

    def load_caveats(self, ids):
        """Analyst-written data-quality caveats (with a source URL) surfaced via data_gaps/get_data_quality."""
        p = CUR / "data_caveats.csv"
        for r in csv.DictReader(p.open(encoding="utf-8")) if p.exists() else []:
            cid = next((v for (s, t), v in ids.items() if t == r["ticker"]), None)
            if cid:
                self.gap(cid, r["field"], f"CAVEAT: {r['reason']} (source: {r['source_url']})")

    def verify_edgar(self, ids, rel=0.005):
        """Cross-check yfinance financials against SEC XBRL annual facts, field by field (table field_verification).

        revenue / net_income: compared; a difference is a 'mismatch' (Yahoo's value is kept and flagged).
        capex / fcf: where the filing has them, the FILING's value is stored (fcf = SEC operating cash flow - SEC PP&E capex),
        Yahoo's value is kept in other_value/verify_note. 'sec_replaced' = Yahoo differed, 'sec_filled' = Yahoo was empty.
        financials.verified = 1 iff revenue and net_income match and no checked field is a mismatch.
        Writes data/raw/edgar_crosscheck.csv (every comparison, including what was replaced)."""
        rows_out, n_ok, n_bad = [], 0, 0
        for (sname, tk), cid in ids.items():
            p = RAW / "edgar" / sname / f"{tk}_facts.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            facts = d["facts"]

            def pick(metric, pe):
                cands = [f for f in facts.get(metric, []) if abs((date.fromisoformat(f["end"]) - date.fromisoformat(pe)).days) <= 7]
                for tag in TAG_ORDER[metric]:                       # first tag in priority order that has a fact
                    same = sorted((f for f in cands if f["tag"] == tag), key=lambda f: f["filed"], reverse=True)
                    if same:
                        return same[0]
                return None

            def filing_source(fact):
                url = f"https://www.sec.gov/Archives/edgar/data/{d['cik']}/{fact['accn'].replace('-', '')}/"
                return self.source(url, "SEC EDGAR (XBRL companyfacts)", "sec_filing", d["retrieved_at"])

            for r in self.db.execute("SELECT id,period_end,revenue,net_income,fcf,capex FROM financials WHERE company_id=?", (cid,)).fetchall():
                fid, pe, yf_rev, yf_ni, yf_fcf, yf_capex = r
                if pe < f"{date.today().year - 5}-01-01":
                    continue
                rv, ni, oc, cx = pick("revenue", pe), pick("net_income", pe), pick("op_cash_flow", pe), pick("capex", pe)
                checks = {}                                          # field -> (status, db_value, other_value, fact, note)

                def compare(field, yf, fact):
                    if fact is None:
                        return
                    sec = fact["val"]
                    if yf is None:
                        checks[field] = ("sec_filled", sec, None, fact, "Yahoo had no value")
                    elif abs(yf - sec) <= rel * max(abs(sec), 1):
                        checks[field] = ("match", yf, sec, fact, None)
                    else:
                        checks[field] = ("mismatch", yf, sec, fact, "Yahoo value kept; filing value in other_value")
                compare("revenue", yf_rev, rv)
                compare("net_income", yf_ni, ni)
                new_capex, new_fcf = yf_capex, yf_fcf
                if cx:
                    if yf_capex is None or abs(yf_capex - cx["val"]) > rel * max(abs(cx["val"]), 1):
                        st = "sec_filled" if yf_capex is None else "sec_replaced"
                        checks["capex"] = (st, cx["val"], yf_capex, cx, f"filing value = {cx['tag']} (PP&E purchases only); "
                                                                        "Yahoo's capex is a broader definition (e.g. includes capitalised software)")
                        new_capex = cx["val"]
                    else:
                        checks["capex"] = ("match", yf_capex, cx["val"], cx, None)
                if oc and cx:
                    fcf_sec = oc["val"] - cx["val"]
                    if yf_fcf is None or abs(yf_fcf - fcf_sec) > rel * max(abs(fcf_sec), 1):
                        st = "sec_filled" if yf_fcf is None else "sec_replaced"
                        checks["fcf"] = (st, fcf_sec, yf_fcf, oc, "filing operating cash flow minus filing PP&E capex; may differ from the company's own FCF definition")
                        new_fcf = fcf_sec
                    else:
                        checks["fcf"] = ("match", yf_fcf, fcf_sec, oc, None)
                for field, (st, dbv, other, fact, note) in checks.items():
                    self.db.execute("INSERT INTO field_verification VALUES (?,?,?,?,?,?,?,?)",
                                    (cid, pe, field, st, dbv, other, filing_source(fact), note))
                if new_capex != yf_capex or new_fcf != yf_fcf:
                    self.db.execute("UPDATE financials SET capex=?, fcf=? WHERE id=?", (new_capex, new_fcf, fid))
                    for col in ("capex", "fcf"):
                        self.db.execute("DELETE FROM data_gaps WHERE company_id=? AND field=?", (cid, f"financials.{col}@{pe}"))
                verified = int(checks.get("revenue", ("",))[0] == "match" and checks.get("net_income", ("",))[0] == "match"
                               and not any(v[0] == "mismatch" for v in checks.values()))
                if checks:
                    acc = (rv or ni or oc or cx)["accn"]
                    parts = [f"{f} {v[0]}" for f, v in checks.items()]
                    note = f"SEC XBRL {(rv or ni or oc or cx)['form']} {acc}: " + ", ".join(parts)
                    repl = [f"{f}: Yahoo {v[2]:,.0f} -> filing {v[1]:,.0f}" for f, v in checks.items() if v[0] == "sec_replaced" and v[2] is not None]
                    if repl:
                        note += " (" + "; ".join(repl) + ")"
                    note += "; ebitda, net_debt and other fields are still Yahoo-derived"
                else:
                    note = "no SEC fact found for this period"
                self.db.execute("UPDATE financials SET verified=?, verify_note=? WHERE id=?", (verified, note, fid))
                n_ok += verified
                n_bad += not verified
                g = lambda f, i: checks[f][i] if f in checks else None    # noqa: E731
                rows_out.append([sname, tk, pe, yf_rev, rv["val"] if rv else None, g("revenue", 0), yf_ni, ni["val"] if ni else None, g("net_income", 0),
                                 yf_capex, cx["val"] if cx else None, g("capex", 0), yf_fcf, (oc["val"] - cx["val"]) if oc and cx else None,
                                 g("fcf", 0), verified, (rv or ni or oc or cx or {}).get("accn")])
        out = RAW / "edgar_crosscheck.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["sector", "ticker", "period_end", "yf_revenue", "sec_revenue", "revenue_status", "yf_net_income", "sec_net_income",
                        "net_income_status", "yf_capex", "sec_capex", "capex_status", "yf_fcf", "sec_fcf(ocf-capex)", "fcf_status",
                        "verified", "sec_accession"])
            w.writerows(rows_out)
        print(f"SEC cross-check: {n_ok} rows verified, {n_bad} not verified (see {out.name})")

    def apply_verified(self):
        """Rows a human confirmed against filings (data/curated/verified_rows.csv, made by apply_verification.py)."""
        p = CUR / "verified_rows.csv"
        n = 0
        for r in csv.DictReader(p.open(encoding="utf-8")) if p.exists() else []:
            t, tk, pd_ = r["table"], r["ticker"], r["period_or_date"]
            key = {"financials": "period_end", "valuations": "as_of_date", "sector_metrics_defense": "period_end", "signals": "signal_date"}[t]
            extra = " AND signal_type='headcount'" if t == "signals" else ""
            cur = self.db.execute(f"UPDATE {t} SET verified=1 WHERE {key}=? AND company_id=(SELECT id FROM companies WHERE ticker=?){extra}", (pd_, tk))
            n += cur.rowcount
        if n:
            print(f"marked {n} rows verified from verified_rows.csv")

    def load_curated_signals(self, ids):
        p = CUR / "signals_curated.csv"
        for r in csv.DictReader(p.open(encoding="utf-8")) if p.exists() else []:
            cid = next((v for (s, t), v in ids.items() if t == r["ticker"]), None)
            if not cid or not r["source_url"]:
                print("curated signal skipped (unknown ticker or no source_url):", r["ticker"])
                continue
            src = self.source(r["source_url"], r["publisher"] or "curated", "curated_csv", r["retrieved_at"])
            self.db.execute(
                "INSERT INTO signals(company_id,signal_type,value_num,value_text,signal_date,source_id,as_of_date,verified)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (cid, r["signal_type"], num(r["value_num"]), r["value_text"] or None, r["signal_date"], src,
                 r["retrieved_at"], int(r["verified"] or 0)))
            self.db.execute("DELETE FROM data_gaps WHERE company_id=? AND field=?", (cid, f"signals.{r['signal_type']}"))


if __name__ == "__main__":
    Builder().build()
