"""Stage 3: per-field verification, filing-sourced FCF/capex, validator, generated README caveat."""
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp_server import queries as q

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "finance.db"


@pytest.fixture()
def con():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def test_arcbest_fcf_comes_from_the_filing_not_yahoo():
    f = next(x for x in q.get_financials("logistics", "ArcBest", 3)["financials"] if x["period_end"] == "2024-12-31")
    assert f["fcf"] == 62743000.0                                              # filing OCF 285,846,000 - filing capex 223,103,000 (Yahoo said 45,846,000)
    fcf = next(v for v in f["verified_fields"] if v["field"] == "fcf")
    assert fcf["status"] == "sec_replaced" and fcf["other_value"] == 45846000.0
    assert f["capex"] == 223103000.0 and "Yahoo 45,846,000 -> filing 62,743,000" in f["verify_note"]
    assert f["verified"] == 1                                                  # stored values now equal the filing
    srcs = q.get_financials("logistics", "ArcBest", 3)["sources"]
    assert srcs[fcf["source_id"]]["url"].startswith("https://www.sec.gov/Archives/edgar/data/")   # per-field SEC source


def test_fcf_conversion_and_capex_intensity_use_the_stored_filing_values():
    f = q.get_financials("logistics", "ArcBest", 3)["financials"][1]           # FY2024
    assert f["derived"]["fcf_conversion"] == pytest.approx(f["fcf"] / f["ebitda"])
    assert f["derived"]["capex_to_revenue"] == pytest.approx(f["capex"] / f["revenue"])


def test_get_financials_exposes_verified_and_unverified_fields():
    lmt = q.get_financials("defense", "LMT", 1)["financials"][0]
    assert {v["field"] for v in lmt["verified_fields"]} >= {"revenue", "net_income"}
    assert "ebitda" in lmt["unverified_fields"] and "net_debt" in lmt["unverified_fields"]       # never filing-checked
    saab = q.get_financials("defense", "Saab", 1)["financials"][0]
    assert saab["verified_fields"] == [] and set(saab["unverified_fields"]) == set(q.KEY_FIELDS) and saab["verified"] == 0


def test_verified_rows_are_consistent_with_their_field_checks(con):
    assert con.execute("SELECT COUNT(*) FROM field_verification WHERE status='mismatch'").fetchone()[0] == 0
    bad = con.execute("""SELECT f.id FROM financials f WHERE f.verified=1 AND EXISTS (SELECT 1 FROM field_verification v WHERE v.company_id=f.company_id
                         AND v.period_end=f.period_end AND v.status='mismatch')""").fetchall()
    assert not bad
    for r in con.execute("SELECT f.fcf, v.db_value FROM financials f JOIN field_verification v ON v.company_id=f.company_id "
                         "AND v.period_end=f.period_end AND v.field='fcf'"):
        assert r["fcf"] == r["db_value"]                                       # what is stored is what was verified


def test_mismatch_is_surfaced_in_data_gaps_even_if_the_model_ignores_it():
    from agent.core import _finalize
    from agent.models import LLMAnswer, ToolCall
    res = [{"found": True, "company": "X Corp", "financials": [{"period_end": "2025-12-31", "verified_fields": [
        {"field": "revenue", "status": "mismatch", "db_value": 100.0, "other_value": 90.0}]}]}]
    ans = LLMAnswer(answer="ok", persona_output={"stance": "NEUTRAL", "verdicts": [], "portfolio_fit_notes": "n/a"}, confidence="high")
    r = _finalize(ans, "mf_analyst", "defense", "m", [ToolCall(name="get_financials", args={}, latency_ms=1.0, found=True)], res, time.perf_counter())
    assert any("FIELD MISMATCH" in g and "X Corp" in g and "revenue" in g for g in r.data_gaps)


def _validate(db: Path):
    return subprocess.run([sys.executable, "scripts/validate_db.py", "--strict"], cwd=ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, "FINANCE_DB": str(db), "PYTHONIOENCODING": "utf-8"})


def test_validator_fails_when_a_verified_row_has_a_mismatch(tmp_path):
    ok = _validate(DB)
    assert ok.returncode == 0, ok.stdout[-400:]
    bad = tmp_path / "bad.db"
    shutil.copy(DB, bad)
    c = sqlite3.connect(bad)
    c.execute("UPDATE field_verification SET status='mismatch' WHERE rowid=(SELECT MIN(rowid) FROM field_verification WHERE field='revenue')")
    c.commit()
    c.close()
    r = _validate(bad)
    assert r.returncode == 1 and "verified=1 but a checked field mismatches" in r.stdout


def test_readme_caveat_is_generated_from_the_data(con):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- AUTO:verification -->")[1].split("<!-- /AUTO:verification -->")[0]
    n_ver = con.execute("SELECT SUM(verified) FROM financials").fetchone()[0]
    assert f"**{n_ver} of " in block
    for (name,) in con.execute("SELECT DISTINCT c.name FROM field_verification v JOIN companies c ON c.id=v.company_id WHERE v.status='sec_replaced'"):
        assert name in block                                                   # every affected company is named, none by hand


def test_every_script_compiles_and_the_build_still_regenerates_the_readme(tmp_path):
    """Scripts are not imported by the other tests, so a syntax error in one would go unnoticed until someone ran it."""
    import py_compile
    for f in sorted((ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(f), cfile=str(tmp_path / (f.name + "c")), doraise=True)
    r = subprocess.run([sys.executable, "scripts/update_readme_caveats.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "regenerated" in r.stdout, r.stdout + r.stderr
    assert b"\r\n" not in (ROOT / "README.md").read_bytes()                       # LF preserved by the generator
