-- Financial-analyst agent database. Rebuilt by scripts/build_db.py; never hand-edited.
-- Monetary values are stored in the company's NATIVE reporting currency in FULL units
-- (not millions), with a USD-converted value plus the FX rate/date used.
PRAGMA foreign_keys = ON;

CREATE TABLE sectors (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE                    -- e.g. 'defense'
);

CREATE TABLE sources (
    id           INTEGER PRIMARY KEY,
    url          TEXT NOT NULL,
    publisher    TEXT NOT NULL,
    doc_type     TEXT NOT NULL,                  -- aggregator_api | sec_filing | press | curated_csv | ...
    retrieved_at TEXT NOT NULL                   -- ISO date the value was retrieved
);

CREATE TABLE companies (
    id                  INTEGER PRIMARY KEY,
    sector_id           INTEGER NOT NULL REFERENCES sectors(id),
    ticker              TEXT NOT NULL,
    name                TEXT NOT NULL,
    country             TEXT,
    exchange            TEXT,
    currency            TEXT,                    -- native reporting currency (ISO 4217)
    fiscal_year_end     TEXT,                    -- 'MM-DD'
    accounting_standard TEXT,                    -- 'US GAAP' | 'IFRS' | NULL if not sourced
    ownership_notes     TEXT,                    -- state/family stakes; NULL if not sourced
    description         TEXT,
    source_id           INTEGER REFERENCES sources(id),
    as_of_date          TEXT,
    UNIQUE (sector_id, ticker)
);

CREATE TABLE financials (
    id            INTEGER PRIMARY KEY,
    company_id    INTEGER NOT NULL REFERENCES companies(id),
    period_end    TEXT NOT NULL,
    period_type   TEXT NOT NULL,                 -- FY | Q
    revenue       REAL, ebitda REAL, ebit REAL, net_income REAL,
    gross_margin  REAL,                          -- fraction, derived = gross profit / revenue
    ebitda_margin REAL,                          -- fraction, derived = ebitda / revenue
    net_debt      REAL, total_debt REAL, cash REAL,
    capex         REAL,                          -- positive number = cash outflow
    fcf           REAL,
    currency      TEXT NOT NULL,
    revenue_usd   REAL, fx_rate REAL, fx_date TEXT,   -- native->USD rate applied at period end
    source_id     INTEGER NOT NULL REFERENCES sources(id),
    as_of_date    TEXT NOT NULL,
    verified      INTEGER NOT NULL DEFAULT 0,   -- 1 only if core values matched a filing / a human check (see verify_note)
    verify_note   TEXT,                         -- how/what was verified, or the discrepancy found
    UNIQUE (company_id, period_end, period_type)
);

-- Per-field verification of the financials against a filing. A field appears here only if it was actually checked.
--   match         stored value equals the filing's value (within 0.5%)
--   mismatch      stored value differs from the filing and was NOT corrected (flag it; the row is not verified)
--   sec_replaced  Yahoo's value differed; the FILING's value is stored (Yahoo's is in other_value)
--   sec_filled    Yahoo had no value; the FILING's value is stored
-- source_id is the filing the value/check came from (differs per field). financials.verified = 1 only if revenue and
-- net_income are 'match' and no checked field is 'mismatch'.
CREATE TABLE field_verification (
    company_id  INTEGER NOT NULL REFERENCES companies(id),
    period_end  TEXT NOT NULL,
    field       TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('match','mismatch','sec_replaced','sec_filled')),
    db_value    REAL,
    other_value REAL,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    note        TEXT,
    PRIMARY KEY (company_id, period_end, field)
);

CREATE TABLE valuations (
    id             INTEGER PRIMARY KEY,
    company_id     INTEGER NOT NULL REFERENCES companies(id),
    as_of_date     TEXT NOT NULL,
    market_cap REAL, enterprise_value REAL,
    ev_ebitda REAL, pe REAL, ev_sales REAL,
    dividend_yield REAL,                         -- fraction (0.025 = 2.5%)
    currency       TEXT NOT NULL,
    market_cap_usd REAL, fx_rate REAL, fx_date TEXT,   -- added vs brief: cross-country size comparison
    source_id      INTEGER NOT NULL REFERENCES sources(id),
    verified       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (company_id, as_of_date)
);

CREATE TABLE signals (
    id          INTEGER PRIMARY KEY,
    company_id  INTEGER NOT NULL REFERENCES companies(id),
    signal_type TEXT NOT NULL CHECK (signal_type IN ('headcount','hiring','news','contract_award')),
    value_num   REAL,
    value_text  TEXT,
    signal_date TEXT NOT NULL,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    as_of_date  TEXT NOT NULL,
    verified    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE sector_metrics_defense (
    company_id             INTEGER NOT NULL REFERENCES companies(id),
    period_end             TEXT NOT NULL,
    order_backlog          REAL,                 -- native currency; see notes for definition
    book_to_bill           REAL,
    govt_revenue_share_pct REAL,
    export_revenue_share_pct REAL,
    notes                  TEXT,                 -- e.g. 'EDGAR RemainingPerformanceObligation, not company-defined backlog'
    source_id              INTEGER REFERENCES sources(id),
    as_of_date             TEXT NOT NULL,
    verified               INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (company_id, period_end)
);

CREATE TABLE data_gaps (
    id         INTEGER PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id),
    field      TEXT NOT NULL,                    -- 'table.column[@period]'
    reason     TEXT NOT NULL
);

CREATE INDEX idx_companies_sector   ON companies(sector_id);
CREATE INDEX idx_financials_company ON financials(company_id);
CREATE INDEX idx_valuations_company ON valuations(company_id);
CREATE INDEX idx_signals_company    ON signals(company_id);
CREATE INDEX idx_defmetrics_company ON sector_metrics_defense(company_id);
CREATE INDEX idx_gaps_company       ON data_gaps(company_id);
