"""Regenerate the generated blocks of README.md (never edit them by hand):
  <!-- AUTO:verification --> ... <!-- /AUTO:verification -->   verification status, from data/finance.db
  <!-- AUTO:counts -->       ... <!-- /AUTO:counts -->         test / eval-case / tool counts, from the repo itself
build_db.py calls this at the end of every build, so the README cannot drift from the data or the tests.
Usage: python scripts/update_readme_caveats.py
"""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- AUTO:verification -->", "<!-- /AUTO:verification -->"
CSTART, CEND = "<!-- AUTO:counts -->", "<!-- /AUTO:counts -->"


def build_block(db_path: Path) -> str:
    con = sqlite3.connect(db_path)
    one = lambda q, *a: con.execute(q, a).fetchone()[0]                          # noqa: E731
    n_rows, n_ver = one("SELECT COUNT(*) FROM financials"), one("SELECT COALESCE(SUM(verified),0) FROM financials")
    unver = con.execute("SELECT c.name, COUNT(*) FROM financials f JOIN companies c ON c.id=f.company_id WHERE f.verified=0 GROUP BY c.name ORDER BY c.name").fetchall()
    sec_cos = one("SELECT COUNT(DISTINCT company_id) FROM field_verification")
    status = dict(con.execute("SELECT field||'/'||status, COUNT(*) FROM field_verification GROUP BY 1").fetchall())
    repl = con.execute("""SELECT c.name, v.field, COUNT(*), AVG(ABS(v.db_value-v.other_value)/NULLIF(ABS(v.db_value),0))
                          FROM field_verification v JOIN companies c ON c.id=v.company_id
                          WHERE v.status='sec_replaced' AND v.other_value IS NOT NULL GROUP BY c.name, v.field ORDER BY c.name, v.field""").fetchall()
    by_co: dict[str, list[str]] = {}
    for name, field, n, avg in repl:
        by_co.setdefault(name, []).append(f"{field} x{n}, Yahoo off by {avg:.0%} on average")
    mism = one("SELECT COUNT(*) FROM field_verification WHERE status='mismatch'")
    fcf_repl = one("SELECT COUNT(*) FROM field_verification WHERE field='fcf' AND status='sec_replaced'")
    capex_repl = one("SELECT COUNT(*) FROM field_verification WHERE field='capex' AND status='sec_replaced'")
    n_cos_repl = one("SELECT COUNT(DISTINCT company_id) FROM field_verification WHERE status='sec_replaced'")
    lines = [
        f"* **Verification status (generated from the database).** Every yfinance row starts unverified. For the {sec_cos} SEC-registered "
        f"companies each annual row is checked field by field against the 10-K/20-F XBRL (table `field_verification`; `get_financials` returns "
        f"`verified_fields` and `unverified_fields`). `verified=1` means revenue **and** net income match the filing and no checked field "
        f"mismatches: **{n_ver} of {n_rows} financial rows**. Checked: revenue ({status.get('revenue/match', 0)} match), net income "
        f"({status.get('net_income/match', 0)} match), plus capex and FCF. **EBITDA, net debt, cash, debt and margins are never checked against a "
        f"filing** - they remain Yahoo-derived on every row, verified or not. Field mismatches left uncorrected: {mism}.",
        f"* **Where Yahoo and the filing disagree, the filing's value is stored** ({fcf_repl} FCF and {capex_repl} capex values across {n_cos_repl} companies; "
        f"Yahoo's value is kept in `field_verification.other_value` and `verify_note`). FCF = filing operating cash flow - filing PP&E purchases. "
        f"Yahoo's capex is a broader definition (it includes items such as capitalised software), so for software-heavy or intangible-investing companies "
        f"the filing-based FCF can be higher than the company's own or Yahoo's figure. Affected: "
        + "; ".join(f"{k} ({', '.join(v)})" for k, v in by_co.items()) + ".",
        f"* **Not filing-checked at all:** {', '.join(f'{n} ({c} rows)' for n, c in unver)} (not SEC registrants - no filing feed here), and every "
        f"valuation / market-data row. `data/verification_sheet.csv` samples only values that are still unverified. Confidence is capped at *medium* "
        f"whenever most rows used are unverified.",
    ]
    return "\n".join(lines)


def build_counts() -> str:
    """Exact counts: tests via pytest --collect-only (parametrised cases included), eval cases from cases.yaml, MCP tools from server.py."""
    cases = len(yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))["cases"])
    tools = len(re.findall(r"^@mcp\.tool\(\)", (ROOT / "mcp_server" / "server.py").read_text(encoding="utf-8"), flags=re.M))
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    tests = m.group(1) if m else "?"
    return f"**{tests} automated tests** (`pytest -q`, no API key needed) | **{cases} eval cases** (`python -m eval.run_eval`) | **{tools} MCP tools**"


def replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        return text
    a, b = text.index(start) + len(start), text.index(end)
    return text[:a] + "\n" + body + "\n" + text[b:]


def main():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README has no AUTO:verification markers; skipped")
        return
    text = replace_block(text, START, END, build_block(ROOT / "data" / "finance.db"))
    text = replace_block(text, CSTART, CEND, build_counts())
    readme.write_text(text, encoding="utf-8", newline="\n")
    print("README generated blocks (verification, counts) regenerated")


if __name__ == "__main__":
    main()
