# Configurable Financial-Analyst Agent (MCP + SQLite)

One agent, three personas (Mutual Fund / Equity / PE analyst), three sectors (Defense / Tech / Logistics) = 9 valid combinations.
Every fact comes from a SQLite database queried **live through MCP tools**; nothing about a company is hardcoded in a prompt or in code.
The same `agent.core.run_agent()` serves a REST API and a Streamlit chat UI.

> **My notes:** <!-- YOUR NOTES HERE (reviewer / author) -->

## TL;DR

* **What it is.** `run_agent(query, persona, sector)` builds a system prompt from YAML (persona lens, priority metrics -> tools, thresholds, required sections, forbidden reasoning, output JSON schema), discovers tools from an MCP server (`list_tools`), and runs an OpenAI tool-use loop (cap 8 iterations). The answer is validated JSON (Pydantic + the persona's JSON Schema, one retry).
* **Where facts come from.** 37 companies in SQLite, populated from yfinance, SEC EDGAR XBRL, USAspending.gov and a few sourced notes. Only the MCP server process opens the DB; the agent talks to it over the MCP protocol. Every result carries `source_id`, `as_of_date`, `stale`, `verified`.
* **What stops the model from bluffing (all in code, all tested).**
  1. *Sector isolation*: the agent refuses any tool call whose `sector` differs from the user's selection; the server refuses a company outside the sector it is given.
  2. *Numeric grounding*: every number in `data_points`, `answer` and `persona_output` is checked against numeric tool output (per company, unit-aware). Ungrounded data points are dropped; ungrounded prose numbers are listed in `data_gaps` and cap confidence at *medium*. The check itself is tested against random fabricated numbers (see "Numeric grounding").
  3. *Confidence policy*: rules named in `config/data_policy.yaml` are implemented one-to-one in `agent/confidence.py` (stale data, unverified share, missing priority metrics, company not found...). Confidence = min(model's, policy's).
  4. *Verification*: for the SEC-registered companies each annual row is checked field by field against the filing; where Yahoo and the filing disagree the filing value is stored.
* **What is NOT verified.** EBITDA, net debt and market data are still Yahoo-derived everywhere; the 8 non-US companies have no filing check at all; ownership notes are partly secondary sources. See "Known data caveats".
* **Owner decisions still open.** The PE hard deal-size cap (`largest_practical_deal_ev_usd`, placeholder USD 25bn, marked `OWNER TO CONFIRM` in `config/personas/pe_analyst.yaml`) and every threshold in the persona YAMLs are defaults, not conclusions.
* **Run it.** `pip install -r requirements.txt`, copy `.env.example` to `.env` and set the key and model, then `streamlit run ui/app.py` or `uvicorn api.main:app`. `pytest -q` needs no key.
* **Size.** <!-- AUTO:counts -->
**132 automated tests** (`pytest -q`, no API key needed) | **31 eval cases** (`python -m eval.run_eval`) | **11 MCP tools**
<!-- /AUTO:counts -->
* **Deviations from the brief / spec** (details below): `mcp<2` pinned; the numeric-grounding tolerance is the exact rounding bound instead of `min(rounding, 0.5%)`; an **11th MCP tool** `get_company_snapshot` (the spec listed ten; it is additive and the ten are unchanged); extra fields (`market_cap_usd`, `enterprise_value_usd`, `sources` map, `history`, `verify_note`, `field_verification` table).

## Model used

* **Model:** `<OWNER: model name used for the reported results>`  (set with `OPENAI_MODEL` in `.env`)
* **Reasoning effort:** `<OWNER: minimal | low | medium | high>`  (set with `OPENAI_REASONING_EFFORT`; unset = provider default)
* **Judge model:** `<OWNER: JUDGE_MODEL>`  (must differ from the agent model)

The model is swappable with no code change. Any OpenAI chat model that supports function calling works; `reasoning_effort` is only sent when the variable is set.

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then set OPENAI_API_KEY, OPENAI_MODEL (and JUDGE_MODEL for evals). Never commit .env.

# 1. Data (a built DB is committed; rebuild any time from raw pulls + curated CSVs)
python scripts/build_db.py                             # offline: data/raw/* + data/curated/*.csv -> data/finance.db (also regenerates the README blocks)
#   refresh raw pulls (needs internet), in this order:
#   python scripts/ingest_yfinance.py && python scripts/ingest_edgar.py defense && python scripts/ingest_edgar_financials.py \
#     && python scripts/ingest_holders.py && python scripts/ingest_contracts.py defense

# 2. MCP server (the agent spawns it itself over stdio; run manually only to inspect it)
python scripts/mcp_cli.py tools                        # list tools via a real MCP client session
python scripts/mcp_cli.py smoke                        # call every tool once + negative tests
python -m mcp_server.server --transport http --port 8765   # optional: streamable HTTP (then set MCP_SERVER_URL)

# 3. Interfaces
uvicorn api.main:app --port 8000                       # REST
streamlit run ui/app.py                                # chat UI
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"Which companies look like buyout targets?","persona":"pe_analyst","sector":"logistics"}'

# 4. Quality
pytest -q                                              # no API key needed (scripted LLM stands in for OpenAI)
python scripts/validate_db.py [--strict]               # data-quality report; --strict exits 1 on any ERROR
python scripts/make_verification_sheet.py              # -> data/verification_sheet.csv (30 still-unverified values to spot-check)
python scripts/apply_verification.py && python scripts/build_db.py   # after you fill the sheet: keep your confirmations across rebuilds
python -m eval.run_eval --dry-run                      # validate eval cases
python -m eval.run_eval                                # run all cases -> eval/results/<ts>/summary.md  (needs API key)
```

**REST.** `POST /query {query, persona, sector, history?}`, `GET /health[?deep=true]`, `GET /personas`, `GET /sectors`. `history` is an optional list of `{query, answer}` (the last 3 are used, only to resolve references like "it"; every number is re-fetched). Streamlit sends the last 3 completed exchanges the same way.
Response: `answer, persona, sector, companies_referenced[], data_points[{company,metric,value,unit,source_id,as_of,stale}], persona_output, confidence, confidence_reasons[], data_gaps[], tools_called[{name,args,latency_ms,found}], model, latency_ms` plus an additive `sources{id: url}` map.

| Status | `detail.error` | Cause |
|---|---|---|
| 422 | (validation) | bad persona/sector; the body lists the valid options |
| 429 | `llm_rate_limited` | provider rate limit (after the SDK's own retries, `OPENAI_MAX_RETRIES`) |
| 502 | `llm_auth_failed`, `llm_provider_error`, `invalid_agent_output` | provider rejected the key / provider error / model returned invalid JSON twice |
| 503 | `not_configured` | `OPENAI_API_KEY` or `OPENAI_MODEL` missing |
| 504 | `agent_timeout`, `llm_timeout` | run exceeded `AGENT_TIMEOUT_S`, or one LLM call exceeded `OPENAI_TIMEOUT` |

## Architecture

```
 Streamlit UI ─┐                                   ┌─ persona YAML (lens, metrics->tools, thresholds, sections, schema, forbidden reasoning)
               ├─► agent.core.run_agent() ◄────────┤─ sectors.yaml (universe, analyst notes)   ── system prompt built at runtime
 FastAPI ──────┘        │  OpenAI function-calling  └─ data_policy.yaml (staleness, confidence rules -> agent/confidence.py)
                        │  loop (max 8 iterations), sector guard, grounding, confidence policy
                        ▼
              agent.mcp_client  ── MCP ClientSession (stdio | streamable HTTP) ── list_tools / call_tool
                        │            tool schemas auto-converted -> OpenAI tool defs (no hand-written copy)
   ═════════════ MCP protocol boundary ═════════════
                        ▼
              mcp_server.server (FastMCP)  ──►  mcp_server.queries (typed, parameterized, read-only)  ──►  data/finance.db
                                                                        ▲
        scripts/ingest_*.py -> data/raw/  +  data/curated/*.csv  ─► scripts/build_db.py
```
Only the MCP server process opens the SQLite file (read-only URI). The agent imports no DB code. The server subprocess receives only `PATH`, `PYTHONPATH`, `FINANCE_DB` and `AGENT_TODAY` (plus the MCP SDK's own safe defaults): **`OPENAI_API_KEY` is not passed to it.**

## MCP design
Typed tools (the count is in the TL;DR): `list_sectors, list_companies, get_company_profile, get_financials, get_valuation, get_company_snapshot, get_signals, get_sector_stats, get_defense_metrics, get_data_quality, get_schema`.
* **No free-form SQL** is exposed; enums (`signal_type`, `metric`) are real JSON-Schema enums the model sees.
* **Sector isolation is enforced in two places.** *Server:* a company must belong to the `sector` argument it is given, otherwise the tool returns `{"found": false, "reason_code": "WRONG_SECTOR", ...}` (never an empty guess; e.g. Microsoft asked under `defense`). *Agent:* the model chooses tool arguments, so `agent/core.py` refuses any call whose `sector` differs from the user's selection (`WRONG_SELECTED_SECTOR`) before it reaches the MCP server. The server alone cannot stop a model from querying another (valid) sector.
* **Company lookup** (`mcp_server/queries.py::_candidates`): exact ticker/name (also after dropping corporate suffixes: "Oracle Corp" -> Oracle), else the query is a substring of a name/ticker (>= 3 chars), else a company *name* (>= 4 chars) appears as whole words inside the query ("What do you think about Saab?"). Short tickers such as NOW/BOX/DT only match exactly, never inside a sentence. Ambiguity (e.g. two names in one query) is refused with candidates, never guessed. Tesla, Boeing, Nvidia, UPS and Airbus resolve to `NOT_IN_DB`.
* **Every `found:false` has a machine-readable `reason_code`:** `NOT_IN_DB, WRONG_SECTOR, AMBIGUOUS, UNKNOWN_SECTOR, BAD_ARGS, NO_ROWS` (the agent adds `WRONG_SELECTED_SECTOR`).
* Every result carries `source_id`, `as_of_date`, `stale` (from `data_policy.yaml`); rows also carry `verified`. `get_financials` adds per period `verified_fields` (checked vs a filing, with status and source) and `unverified_fields`. `get_sector_stats` reports `stale_companies` / `n_stale` computed **per contributing company** from its own period, not from the retrieval date.
* **`get_company_snapshot(sector, companies[])`** (typed, read-only): latest annual financials + derived ratios + per-field verification, latest valuation (+ premium to sector median, `enterprise_value_usd`) and data-quality flags for up to 12 companies in ONE call, each company with its own `source_id`, `as_of_date`, `stale`, `verified`. It uses the same lookup and sector rules as every other tool (missing companies come back under `not_found` with a `reason_code`), and the old tools are unchanged. Purpose: sector-wide questions previously took dozens of per-company calls and many model turns; for 12 companies the snapshot is 1 call and about 40% of the bytes of the equivalent separate calls (`tests/test_snapshot.py`). The prompt tells the model to use it for company detail and `get_financials` only for multi-year history.
* **Derived ratios are computed server-side** (growth, margin change, net debt/EBITDA, FCF conversion, capex intensity, premium to sector median, gap to best, `enterprise_value_usd`). The model rarely does arithmetic, so its numbers stay checkable against tool output.
* `get_defense_metrics` is exposed to the model only when the sector is Defense (`sector_tools` in `sectors.yaml`) and is refused otherwise.
* stdio by default (the agent owns the child process: no port, no auth surface, no lifecycle); streamable HTTP is available for a shared service.

## Guardrails around the model

**Numeric grounding** (`agent/grounding.py`). The pool holds only *numeric leaf values* of tool output (no numbers parsed from strings, no ids/years/dates), scoped per company where the result names one. A displayed number may be a rescaled tool number only as its own unit allows (`x` identity; `%` x100 or identity; `B`/`M`/`K` divide by 1e9/1e6/1e3; no unit identity or divide) and only where that makes sense (dividing needs a big amount, x100 needs a fraction). Tolerance is half a unit of the last displayed decimal. Data points are checked against *their company's* numbers; prose is checked line by line against the one company the line names (otherwise against everything, which is weaker). Numeric strings such as `"$41.7B"` are parsed and checked; unparseable values are dropped.
*Why it matters and how it was tested:* the previous check accepted **98.4%** of random fabricated numbers on a realistic PE-screen pool, so "99-100% grounded" said nothing. `tests/test_grounding.py` rebuilds that pool from the real query layer and measures both directions (run `pytest -s tests/test_grounding.py` for current values): at the time of writing about **4.7%** of fabricated numbers are accepted when tied to a company (per-seed range 4.0-5.2%, so **borderline against the 5% target**), about **11-13%** with no company context, and **100%** of genuine numbers formatted as an LLM would (14.4x, 12.3%, $75.0B, 75,048 million). The eval summary repeats this control on every run's own pool.
*Deviation:* the brief asked for `min(half a unit of the last decimal, 0.5% relative)`. Measured, the 0.5% cap rejects correctly rounded small numbers ("3.2%" from 3.244%) and accepted only ~85% of genuine numbers, so the tolerance is the exact rounding bound.
*Known limits:* signs are not flipped (a "47.5% discount" for a tool value of -47.5 is flagged); model-computed arithmetic (e.g. headroom = 4.5x - 0.247x) and counts ("11 of 12 companies") are flagged as unverified by design; a correct number used for a wrong conclusion passes.

**Confidence** (`agent/confidence.py`). `config/data_policy.yaml` lists rule names; each is a registered function; a test fails if the YAML and the code disagree. `downgrade_to_low_if`: `company_not_found`, `no_tool_returned_data`, `more_than_half_of_cited_companies_have_gaps_in_priority_metrics`. `downgrade_one_level_if`: `any_data_point_stale` (including a stale contributor to a sector median), `any_data_point_unverified_share_above: 0.5`, `key_metric_missing`. A not-found failure counts only if no *later* successful call returned a matching company (typo then recovery is not penalised); model mistakes (`BAD_ARGS`, `WRONG_SELECTED_SECTOR`) say nothing about the data and are not penalised. Ungrounded prose numbers cap confidence at *medium*.

**Verification rule.** The prompt tells the model to state mismatched or unverified key fields in the answer and `data_gaps`; any `mismatch` in `verified_fields` is also added to `data_gaps` deterministically.

**Untrusted text.** News headlines, descriptions, ownership notes and contract text are treated as data, never instructions (prompt rule 12).

## Personas and owner decisions
Persona YAMLs (`config/personas/*.yaml`) hold everything that differs by persona: `system_prompt`, `lens_description`, `priority_metrics` (drive the tool plan), `thresholds`, `required_sections`, `output_schema_extension` (JSON Schema for `persona_output`), `forbidden_reasoning`. **All thresholds are defaults for the owner to review.** In particular the PE persona has a **hard deal-size cap** `largest_practical_deal_ev_usd` (placeholder USD 25bn, `OWNER TO CONFIRM`): above it `lbo_score` is at most 4, `key_blockers` must contain "exceeds practical sponsor deal size", and the company can never be `top_pick`. The size comes from the server (`enterprise_value_usd` in `get_valuation` and as a `get_sector_stats` metric), and `eval/checks.py::check_deal_size_cap` verifies the rule on every PE answer.

## Schema and sourcing
Tables: `sectors, companies, financials, valuations, signals, sector_metrics_defense, field_verification, sources, data_gaps` (`db/schema.sql`; indexes on `company_id`/`sector_id`). Values are in **native currency, full units**, with `revenue_usd` + `fx_rate` + `fx_date` (Yahoo FX close on/before period end); `valuations` additionally has `market_cap_usd`. Every data row has `source_id` and `as_of_date`.

| Sector | Companies (12-13) | Sources |
|---|---|---|
| Defense | Lockheed Martin, Northrop Grumman, General Dynamics, RTX, Kratos, Thales, Dassault Aviation, Rheinmetall, Hensoldt, Elbit, BAE Systems, Saab, Kongsberg | yfinance (financials, market data, headcount, news); SEC EDGAR XBRL (backlog = `RemainingPerformanceObligation`, plus annual facts used to cross-check yfinance); USAspending.gov contract awards (US primes); sourced ownership notes |
| Tech | Microsoft, Oracle, Adobe, Salesforce, ServiceNow, Intuit, Dynatrace, DocuSign, Pegasystems, Commvault, Box, Twilio | yfinance, cross-checked against SEC XBRL |
| Logistics | XPO, GXO, RXO, Saia, Landstar, Hub Group, Forward Air, ArcBest, Werner, Knight-Swift, C.H. Robinson, DSV | yfinance, cross-checked against SEC XBRL (all but DSV) |

Raw pulls are kept in `data/raw/`. All 13 proposed Defense names had usable data, so none were swapped.

### Known data caveats (read before trusting a number)
<!-- AUTO:verification -->
* **Verification status (generated from the database).** Every yfinance row starts unverified. For the 29 SEC-registered companies each annual row is checked field by field against the 10-K/20-F XBRL (table `field_verification`; `get_financials` returns `verified_fields` and `unverified_fields`). `verified=1` means revenue **and** net income match the filing and no checked field mismatches: **116 of 148 financial rows**. Checked: revenue (116 match), net income (116 match), plus capex and FCF. **EBITDA, net debt, cash, debt and margins are never checked against a filing** - they remain Yahoo-derived on every row, verified or not. Field mismatches left uncorrected: 0.
* **Where Yahoo and the filing disagree, the filing's value is stored** (24 FCF and 25 capex values across 7 companies; Yahoo's value is kept in `field_verification.other_value` and `verify_note`). FCF = filing operating cash flow - filing PP&E purchases. Yahoo's capex is a broader definition (it includes items such as capitalised software), so for software-heavy or intangible-investing companies the filing-based FCF can be higher than the company's own or Yahoo's figure. Affected: ArcBest (capex x4, Yahoo off by 9% on average, fcf x4, Yahoo off by 14% on average); Box (capex x4, Yahoo off by 561% on average, fcf x4, Yahoo off by 7% on average); C.H. Robinson (capex x4, Yahoo off by 194% on average, fcf x4, Yahoo off by 7% on average); Dynatrace (capex x3, Yahoo off by 10% on average, fcf x2, Yahoo off by 1% on average); Intuit (capex x4, Yahoo off by 32% on average, fcf x4, Yahoo off by 1% on average); RTX (capex x4, Yahoo off by 24% on average, fcf x4, Yahoo off by 11% on average); ServiceNow (capex x2, Yahoo off by 5% on average, fcf x2, Yahoo off by 1% on average).
* **Not filing-checked at all:** BAE Systems (4 rows), DSV (4 rows), Dassault Aviation (4 rows), Hensoldt (4 rows), Kongsberg Gruppen (4 rows), Rheinmetall (4 rows), Saab (4 rows), Thales (4 rows) (not SEC registrants - no filing feed here), and every valuation / market-data row. `data/verification_sheet.csv` samples only values that are still unverified. Confidence is capped at *medium* whenever most rows used are unverified.
<!-- /AUTO:verification -->
* **Backlog = SEC "remaining performance obligation"**, read straight from the filing (verified) but a proxy, not each company's own backlog definition; only the 6 SEC filers have it. **Book-to-bill as disclosed and government/export revenue share remain NULL for everyone.** `get_defense_metrics` additionally returns a clearly labelled **derived `implied_book_to_bill`** ((change in RPO + revenue) / revenue) for the SEC filers; it is never written to the disclosed `book_to_bill` column.
* **Contract awards** exist only for the US-linked names (Lockheed, Northrop, General Dynamics, RTX, Kratos, Elbit America): the top 5 US federal contracts signed in the last 180 days from USAspending.gov, found by recipient-name search (subsidiaries can appear, so the recipient is shown in every record). No European government contracts.
* **Hiring signals are empty** (no free structured source that respects site terms). "Headcount" is Yahoo's `fullTimeEmployees`, which is undated, so it is dated at the last fiscal year end (conservative) and is therefore flagged *stale* under the 180-day signal rule even though annual headcount is normally that old. Consider a longer threshold for headcount.
* **News signals** are yfinance headlines (title + URL + date), not analysed for sentiment.
* **Currency:** native values are never compared across currencies by the agent; use ratios or `*_usd`. FX is a spot close at period end; USD market cap and enterprise value use the retrieval-date rate. Yahoo quotes BAE in pence (GBp); it is normalised to GBP and the dividend yield is taken from Yahoo's percent field (its trailing-yield field is off by 100x for BAE).
* **IFRS vs US GAAP:** margins/EBITDA are not fully comparable across the two (leases, capitalised development). `accounting_standard` is rule-based (10-K filers = US GAAP; EU/UK/Norway listed groups = IFRS; Elbit = US GAAP because its XBRL is in the `us-gaap` namespace), **not read from each filing**.
* **Fiscal-year mismatch:** Tech year-ends span Jan-Jul (latest periods span 243 days); Hub Group's latest annual data is FY2024 (stale by policy) - a deliberate test case. `validate_db.py` reports these.
* **yfinance gaps:** `net_debt` is derived as total debt - cash when Yahoo omits it; several EV/EBITDA and P/E values are TTM and differ from FY figures (flagged by the validator); P/E is NULL for loss-makers. A missing Yahoo dividend yield is stored as **0.0 where Yahoo's trailing annual yield is 0.0** (a sourced "pays no dividend"), otherwise NULL.
* **Ownership notes** exist for all 37 companies but differ in strength. US-domiciled companies get a factual *holder-data* note (insiders %, top institutional holders, from Yahoo) saying whether any holder exceeds 20% - indicative only, **not a proxy-statement control check**. Non-US names use sourced control statements with the URL embedded in the note: Thales, Hensoldt, Saab, Kongsberg, Elbit, Dassault Aviation (family group ~67% of capital / ~80% of votes), Rheinmetall, BAE (15% foreign-voting cap; major-holder percentages not established) and DSV. These come from company pages and secondary summaries, some possibly outdated (Elbit's figure is from the 2022 20-F), and all are `unverified`. Yahoo holder data is deliberately *not* used for non-US names or Elbit because it only sees US-registered funds and would mislead. Kongsberg carries a caveat about a 2026 Maritime spin-off that may affect multiple comparability.

## Design decisions (and what I rejected)
1. **Typed tools vs. a `run_sql` tool.** Rejected free SQL: unbounded queries, sector isolation would need SQL parsing, the model could join its way to numbers with no provenance, and evals become unreviewable. Typed tools make company-in-sector checks, staleness and `found:false` semantics unbypassable on the server, let the agent inspect every `sector` argument, and give the eval harness a clean surface.
2. **Server-side derived metrics.** Rejected letting the LLM compute ratios: it makes numbers ungroundable and error-prone. Cost: adding a metric means a server change.
3. **stdio vs HTTP.** stdio default (simplest, no exposed surface); HTTP supported. Rejected HTTP-only because it adds a service to start for a single-user demo.
4. **One agent, three personas vs. three agents.** One `run_agent` + YAML configs: persona differences live in the *prompt-assembled reasoning plan* and are checked by code. Rejected three code paths (drift, triple maintenance) and tone-only prompts (the brief explicitly wants reasoning differences).
5. **Deterministic guardrails around the LLM** rather than trusting the prompt: schema validation with one retry, strict numeric grounding, a YAML-driven confidence policy, agent-side sector guard. Rejected "prompt it harder": rules the model can silently break are not rules.
6. **Filing value wins over aggregator value.** When SEC XBRL and Yahoo disagree on FCF/capex the filing is stored and Yahoo's value kept beside it. Rejected keeping Yahoo's number and merely flagging it (a flagged wrong number still feeds every ratio).
7. **mcp<2 pinned.** mcp 2.x renamed FastMCP; 1.x (`mcp==1.30`) is what is tested.

## Evaluation
`eval/cases.yaml` holds the brief's samples across persona x sector, cross-persona groups (Tech, Defense, Logistics-buyout), out-of-scope tests (real companies: Boeing, Nvidia, UPS, Tesla; plus a company that exists but in a *different* sector), grounding tests (headcount, backlog, hiring gap), missing-field and stale-data tests (counts in the TL;DR).
Programmatic checks (`eval/checks.py`, no LLM): response-schema validity; **numeric grounding using the same strict pool as the agent**, with a **negative control** (each run's own pool is fed random fabricated numbers; the summary reports how many the check would wrongly accept); out-of-scope refusal; tool expectations; gap/staleness acknowledgement; confidence caps; forbidden reasoning; the PE deal-size cap; and **persona divergence**.
*Persona divergence.* Tools, sector-stat metrics and schema fields come straight from each persona's YAML, so they differ by construction and are reported for information only. A same-question group **fails only if all personas give the same top pick AND their mean per-company Spearman rank correlation exceeds 0.9** (favourability: MF CORE_HOLDING=2/HOLD_WATCH=1/AVOID=0, Equity BUY/HOLD/SELL=2/1/0, PE `lbo_score`); the distinct top picks and the correlations are printed. For sector-level questions whose answers name fewer than 3 shared companies the correlation is undefined and the group cannot fail for lack of evidence - that means divergence is *unmeasured* there, not proven.
*LLM judge.* `eval/judge.py` (`JUDGE_MODEL` must differ from `OPENAI_MODEL`); write your rubric from `eval/rubrics/template.yaml` (the judge refuses to run with placeholders left in). The judge receives an **ordered, per-company digest** of the run's tool output, not a raw dump cut at 40,000 characters (which silently dropped the last tool results); the summary reports how often the old cut would have bitten and whether the digest needed compaction (`JUDGE_MAX_CHARS`).

### Eval results
<!-- OWNER: paste your real run here: model, reasoning effort, date, pass count, the grounding numbers with the negative-control line, the divergence table, judge scores, latency. Nothing in this README is a measured eval result. -->

**Reading the results honestly.** The programmatic checks test *structure and honesty* (valid JSON, grounded numbers, refusals, acknowledged gaps), not whether a rating is a good judgement; that is what the judge rubric is for. A run that passes every check can still contain a poor recommendation.

<!-- YOUR EVAL NOTES HERE -->

## One thing I'd improve with more time
Replace the remaining aggregator-derived fields (EBITDA, net debt, market data) with filing-sourced values (SEC XBRL for US names; ESEF/annual-report tables for European names) so the 8 non-SEC companies can be verified too and confidence can reach *high*. This also fixes IFRS/GAAP normalisation and gives dated headcount.

## What I'd do next
1. Run the eval with real keys, write the judge rubric, iterate on prompts using the failure analysis (and decide the PE deal-size cap and the persona thresholds).
2. Curate the empty inputs: disclosed book-to-bill, government/export share, hiring signals; proxy-statement control checks for all names.
3. Measure latency and token use on your own run (sector-wide questions used to take many model turns; `get_company_snapshot` is the first fix), then shorten tool output further if needed.
4. Longer/typed staleness rules (annual headcount vs daily prices), and calendarised (LTM) financials to remove fiscal-year mismatch.
5. Streaming responses in the UI, per-request caching of MCP sessions, auth + rate limits on the API (measure usage first, then set soft and hard limits), and a persistent MCP HTTP service.
