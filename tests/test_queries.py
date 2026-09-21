import asyncio

import pytest

from mcp_server import queries as q


def test_sector_enforced_and_explicit_not_found():
    r = q.get_company_profile("tech", "Lockheed Martin")
    assert r["found"] is False and "defense" in r["reason"]
    r = q.get_financials("defense", "Tesla")
    assert r["found"] is False and "No data" in r["reason"]
    assert q.get_company_profile("space", "x")["found"] is False


def test_lookup_by_ticker_name_case_insensitive_and_unique_partial():
    for key in ("LMT", "lmt", "Lockheed Martin", "lockheed"):
        assert q.get_company_profile("defense", key)["profile"]["ticker"] == "LMT"
    assert q.get_company_profile("defense", "SAAB-B.ST")["found"]


def test_ambiguous_partial_is_not_guessed():
    r = q.get_company_profile("defense", "systems")     # Elbit Systems / BAE Systems
    assert r["found"] is False and "ambiguous" in r["reason"]
    assert q.get_company_profile("defense", "ab")["found"] is False   # too short to fuzzy match


def test_every_result_has_provenance_and_stale():
    for r in (q.get_company_profile("defense", "RTX"), q.get_financials("defense", "RTX"), q.get_valuation("defense", "RTX"),
              q.get_signals("defense", "RTX", "headcount"), q.get_sector_stats("defense", "ev_ebitda"), q.get_defense_metrics("RTX")):
        assert r["found"] and {"source_id", "as_of_date", "stale"} <= set(r)
        assert r["source_id"] is not None


def test_derived_ratios_are_server_side():
    f = q.get_financials("defense", "LMT", 3)["financials"]
    assert len(f) == 3 and set(f[0]["derived"]) >= {"revenue_growth_yoy", "net_debt_to_ebitda", "fcf_conversion"}
    assert abs(f[0]["derived"]["net_debt_to_ebitda"] - f[0]["net_debt"] / f[0]["ebitda"]) < 1e-9


def test_stale_flag_follows_policy(monkeypatch):
    monkeypatch.setenv("AGENT_TODAY", "2030-01-01")
    assert q.get_valuation("defense", "RTX")["stale"] is True
    assert q.get_financials("logistics", "HUBG")["stale"] is True
    monkeypatch.setenv("AGENT_TODAY", "2026-09-21")
    assert q.get_valuation("defense", "RTX")["stale"] is False


def test_null_values_stay_null_and_are_gaps():
    v = q.get_valuation("logistics", "RXO")["valuation"]
    assert v["pe"] is None
    dq = q.get_data_quality("logistics", "RXO")["companies"]["RXO"]
    assert any(g["field"] == "valuations.pe" for g in dq["gaps"])
    assert q.get_signals("defense", "LMT", "hiring")["signals"] == []


def test_sector_stats_and_defense_only_metrics():
    s = q.get_sector_stats("defense", "ev_ebitda")
    assert s["min"] <= s["q1"] <= s["median"] <= s["q3"] <= s["max"]
    assert q.get_defense_metrics("Microsoft")["found"] is False
    assert q.get_sector_stats("tech", "order_backlog_to_revenue")["n"] == 0


def test_sec_filers_verified_and_others_not():
    lmt = q.get_financials("defense", "LMT", 3)["financials"]
    assert all(f["verified"] == 1 and "SEC XBRL" in f["verify_note"] for f in lmt)
    saab = q.get_financials("defense", "Saab", 3)["financials"]
    assert all(f["verified"] == 0 for f in saab)
    assert q.get_valuation("defense", "LMT")["valuation"]["verified"] == 0          # market data is never filing-verified


def test_implied_book_to_bill_is_derived_and_labelled():
    r = q.get_defense_metrics("LMT")
    assert r["implied_book_to_bill"] and "DERIVED" in r["implied_book_to_bill_method"]
    assert all(m["book_to_bill"] is None for m in r["metrics"])                     # disclosed column stays NULL


def test_contract_awards_present_for_us_primes():
    s = q.get_signals("defense", "LMT", "contract_award")["signals"]
    assert s and all(x["value_text"] and x["source_id"] for x in s)


def test_no_free_form_sql_surface():
    import mcp_server.server as s
    names = {t.name for t in asyncio.run(s.mcp.list_tools())}
    assert not any("sql" in n or "query" in n for n in names) and "get_schema" in names


def test_sector_stats_staleness_is_per_contributing_company(monkeypatch):
    monkeypatch.setenv("AGENT_TODAY", "2026-09-21")
    s = q.get_sector_stats("logistics", "ebitda_margin")
    assert "Hub Group" in s["stale_companies"] and s["n_stale"] == len(s["stale_companies"]) >= 1     # HUBG's latest FY is 2024
    assert s["stale"] is True and s["data_date_per_company"]["Hub Group"] == "2024-12-31"
    assert s["as_of_date"] > "2026-01-01"                       # the retrieval date alone would have said "fresh"
    v = q.get_sector_stats("logistics", "ev_ebitda")             # market data was retrieved today: nothing stale
    assert v["stale"] is False and v["stale_companies"] == [] and v["n_stale"] == 0
    d = q.get_sector_stats("defense", "ebitda_margin")
    assert d["stale_companies"] == [] and d["stale"] is False    # every defense company has FY2025 data
    monkeypatch.setenv("AGENT_TODAY", "2030-01-01")
    assert q.get_sector_stats("defense", "ebitda_margin")["n_stale"] == 13


def test_enterprise_value_usd_is_server_side_and_null_safe(tmp_path, monkeypatch):
    v = q.get_valuation("logistics", "DSV")["valuation"]
    assert v["enterprise_value_usd"] == pytest.approx(v["enterprise_value"] * v["fx_rate"])
    assert q.get_valuation("logistics", "Hub Group")["valuation"]["enterprise_value_usd"] == q.get_valuation("logistics", "Hub Group")["valuation"]["enterprise_value"]  # USD: rate 1
    s = q.get_sector_stats("defense", "enterprise_value_usd")
    assert s["per_company"]["Lockheed Martin"] > 100e9 and s["per_company"]["Kratos Defense & Security Solutions"] < 25e9
    import shutil, sqlite3
    db = tmp_path / "t.db"
    shutil.copy(q.DB_PATH, db)
    c = sqlite3.connect(db)
    c.execute("UPDATE valuations SET enterprise_value=NULL WHERE company_id=(SELECT id FROM companies WHERE ticker='SAIA')")
    c.commit()
    c.close()
    monkeypatch.setattr(q, "DB_PATH", db)
    assert q.get_valuation("logistics", "Saia")["valuation"]["enterprise_value_usd"] is None


def test_pe_persona_carries_the_owner_to_confirm_cap_and_other_thresholds_are_untouched():
    import yaml
    t = yaml.safe_load(open("config/personas/pe_analyst.yaml", encoding="utf-8"))["thresholds"]
    assert t["largest_practical_deal_ev_usd"] == 25000000000
    assert (t["total_leverage_ceiling_x"], t["min_fcf_conversion"], t["max_capex_to_revenue"]) == (4.5, 0.40, 0.08)
    assert "OWNER TO CONFIRM" in open("config/personas/pe_analyst.yaml", encoding="utf-8").read()


# ----------------------------------------------------------------------------- Stage 8: company lookup
@pytest.mark.parametrize("sector, text, ticker", [
    ("tech", "Oracle Corp", "ORCL"), ("tech", "Oracle Corporation", "ORCL"), ("tech", "oracle corp.", "ORCL"),
    ("logistics", "Hub Group Inc", "HUBG"), ("logistics", "Knight-Swift Transportation Holdings", "KNX"),
    ("defense", "Kongsberg Gruppen ASA", "KOG.OL"), ("defense", "Dassault Aviation SA", "AM.PA"), ("defense", "BAE Systems plc", "BA.L"),
    ("defense", "Elbit Systems Ltd", "ESLT"), ("defense", "Kratos Defense & Security Solutions Inc", "KTOS"),
    ("defense", "What do you think about Saab?", "SAAB-B.ST"),            # a DB name appears inside the query string
    ("defense", "Give me Rheinmetall's outlook", "RHM.DE"),
])
def test_lookup_strips_suffixes_and_finds_names_inside_query_strings(sector, text, ticker):
    r = q.get_company_profile(sector, text)
    assert r["found"] is True and r["profile"]["ticker"] == ticker, r


@pytest.mark.parametrize("name", ["Tesla", "Boeing", "Nvidia", "UPS", "Airbus", "Airbus SE", "Tesla Inc", "United Parcel Service",
                                  "What do you think about Nvidia and Boeing?"])
def test_lookup_never_resolves_companies_that_are_not_in_the_database(name):
    for sector in ("defense", "tech", "logistics"):
        r = q.get_company_profile(sector, name)
        assert r["found"] is False and r["reason_code"] == "NOT_IN_DB", (sector, name, r)


def test_lookup_ambiguity_and_wrong_sector_and_short_tickers():
    r = q.get_company_profile("defense", "General Dynamics versus Northrop Grumman")
    assert r["found"] is False and r["reason_code"] == "AMBIGUOUS" and len(r["reason"]) > 0     # two names in one query: refuse, do not guess
    assert q.get_company_profile("defense", "Microsoft Corp")["reason_code"] == "WRONG_SECTOR"
    for text in ("what should I do right now", "open the box", "the dt price", "is now a good time"):   # NOW / BOX / DT are words, not names
        assert q.get_company_profile("tech", text)["found"] is False
    assert q.get_company_profile("tech", "NOW")["found"] is True and q.get_company_profile("tech", "dt")["found"] is True   # exact tickers still work
