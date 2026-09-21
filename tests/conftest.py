import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _db():
    if not (ROOT / "data" / "finance.db").exists():
        pytest.skip("run python scripts/build_db.py first", allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def _pin_today(_db):
    """Staleness is computed from 'today'. Pin it to the database's own retrieval date so the tests do not depend on the real clock
    (or on when the DB was last rebuilt): market data is fresh, FY2024 data (e.g. Hub Group) is stale, whatever day the tests run.
    The MCP server subprocess inherits AGENT_TODAY (see agent/mcp_client.SERVER_ENV_VARS)."""
    con = sqlite3.connect(ROOT / "data" / "finance.db")
    pinned = con.execute("SELECT MAX(as_of_date) FROM valuations").fetchone()[0]
    con.close()
    old = os.environ.get("AGENT_TODAY")
    os.environ["AGENT_TODAY"] = pinned
    yield pinned
    if old is None:
        os.environ.pop("AGENT_TODAY", None)
    else:
        os.environ["AGENT_TODAY"] = old
