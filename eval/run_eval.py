"""Eval runner.

  python -m eval.run_eval                       # all cases
  python -m eval.run_eval --ids buyout_logistics_pe,oos_tesla_defense
  python -m eval.run_eval --tags out_of_scope
  python -m eval.run_eval --judge persona_quality_v1     # also run the LLM judge with eval/rubrics/<name>.yaml
  python -m eval.run_eval --dry-run             # validate cases.yaml against config + DB, no LLM calls

Writes eval/results/<timestamp>/{<case>.json, results.json, summary.md}.
"""
import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import valid_personas, valid_sectors  # noqa: E402
from agent.core import run_agent  # noqa: E402
from eval import checks as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dry_run(spec) -> int:
    db = sqlite3.connect(ROOT / "data" / "finance.db")
    in_sector = {(r[0].lower(), r[1]) for r in db.execute("SELECT c.name, s.name FROM companies c JOIN sectors s ON s.id=c.sector_id")}
    anywhere = {n for n, _ in in_sector}
    bad = 0
    for c in spec["cases"]:
        if c["persona"] not in valid_personas() or c["sector"] not in valid_sectors():
            print(f"BAD persona/sector in {c['id']}")
            bad += 1
        for n in c.get("expect", {}).get("no_data_for", []):
            if (n.lower(), c["sector"]) in in_sector:
                print(f"{c['id']}: '{n}' IS in sector {c['sector']}, so it is not out-of-scope")
                bad += 1
            if "cross_sector" in c.get("tags", []) and n.lower() not in anywhere:
                print(f"{c['id']}: tagged cross_sector but '{n}' is in no sector")
                bad += 1
    print(f"{len(spec['cases'])} cases, {len({c.get('group') for c in spec['cases']} - {None})} groups, {bad} problems")
    return bad


async def run_one(case, model, semaphore):
    async with semaphore:
        trace: list = []
        try:
            resp = await run_agent(case["query"], case["persona"], case["sector"], model=model, trace=trace)
            return {"response": resp.model_dump(), "trace": trace, "error": None}
        except Exception as e:                                                  # noqa: BLE001 - record and continue
            return {"response": None, "trace": trace, "error": f"{type(e).__name__}: {e}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids"); ap.add_argument("--tags"); ap.add_argument("--judge"); ap.add_argument("--model")
    ap.add_argument("--concurrency", type=int, default=2); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text(encoding="utf-8"))
    if a.dry_run:
        sys.exit(1 if dry_run(spec) else 0)
    cases = spec["cases"]
    if a.ids:
        cases = [c for c in cases if c["id"] in a.ids.split(",")]
    if a.tags:
        cases = [c for c in cases if set(a.tags.split(",")) & set(c.get("tags", []))]
    cfg, forbidden = spec["config"], spec.get("persona_forbidden_terms", {})
    rubric = None
    if a.judge:
        from eval.judge import judge, load_rubric
        rubric = load_rubric(a.judge)

    out_dir = Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    runs = await asyncio.gather(*(run_one(c, a.model, sem) for c in cases))

    results, by_group = [], {}
    for case, rec in zip(cases, runs):
        if rec["error"]:
            item = {"id": case["id"], "case": case, "error": rec["error"], "checks": [], "passed": False}
        else:
            chks = C.run_case_checks(case, rec, cfg, forbidden)
            item = {"id": case["id"], "case": case, "record": rec, "checks": [c.to_dict() for c in chks],
                    "passed": all(c.passed for c in chks)}
            if case.get("group"):
                by_group.setdefault(case["group"], {})[case["persona"]] = rec
            if rubric:
                from eval.judge import judge
                try:
                    item["judge"] = (await judge(case["query"], case["persona"], rec["response"]["answer"],
                                                 rec["trace"], rubric)).model_dump()
                except Exception as e:                                          # noqa: BLE001
                    item["judge"] = {"error": str(e)}
        results.append(item)
        (out_dir / f"{case['id']}.json").write_text(json.dumps(item, indent=1, default=str), encoding="utf-8")

    divergence = {g: C.check_divergence(recs, cfg["divergence_max_rank_corr"]) for g, recs in by_group.items() if len(recs) >= 2}
    (out_dir / "results.json").write_text(json.dumps({"cases": results, "divergence": divergence}, indent=1, default=str), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_md(results, divergence, rubric is not None), encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nresults written to {out_dir}")


def summary_md(results, divergence, judged) -> str:
    n_pass = sum(r["passed"] for r in results)
    L = [f"# Eval summary ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"**{n_pass}/{len(results)} cases passed all programmatic checks.**", "",
         "| case | persona | sector | result | failed checks | confidence | tools | ms |" + (" judge |" if judged else ""),
         "|---|---|---|---|---|---|---|---|" + ("---|" if judged else "")]
    for r in results:
        c = r["case"]
        if r.get("error"):
            L.append(f"| {r['id']} | {c['persona']} | {c['sector']} | ERROR | {r['error'][:80]} | | | |" + (" |" if judged else ""))
            continue
        resp = r["record"]["response"]
        failed = "; ".join(f"{k['name']}" for k in r["checks"] if not k["passed"]) or "-"
        j = ""
        if judged:
            jr = r.get("judge", {})
            j = f" {jr.get('overall', jr.get('error', ''))} |"
        L.append(f"| {r['id']} | {c['persona']} | {c['sector']} | {'PASS' if r['passed'] else 'FAIL'} | {failed} | {resp['confidence']} | "
                 f"{len(resp['tools_called'])} | {resp['latency_ms']:.0f} |" + j)
    from eval.judge import judge_input_report
    jr = judge_input_report([r["record"]["trace"] for r in results if r.get("record")])
    L += ["", "## Judge input", "",
          f"Tool output per run reaches the judge as an ordered per-company digest. Over {jr['runs']} runs: the OLD raw 40,000-char cut would have "
          f"dropped data in **{jr['old_cut_would_have_dropped_data']}** (largest raw output {jr['max_raw_chars']:,} chars); the digest needed compaction "
          f"in {jr['digest_compacted_level_1_or_2']} and per-company text truncation in **{jr['digest_truncated_level_3']}** "
          f"(largest digest {jr['max_digest_chars']:,} chars)."]
    ctl = [C.fabrication_control(r["record"]) for r in results if r.get("record")]
    if ctl:
        L += ["", "## Negative control of the numeric-grounding check", "",
              f"Feeding each run's own tool-output pool {300} random FABRICATED numbers, the check would wrongly accept "
              f"**{sum(a for a, _ in ctl) / len(ctl):.1%}** with no company context (weak path) and "
              f"**{sum(b for _, b in ctl) / len(ctl):.1%}** when the number is tied to one company (the data-point path). "
              "If these were near 100% the grounding pass-rate above would mean nothing."]
    L += ["", "## Persona divergence (same question, different personas)", "",
          "Fails only if all personas share the top pick AND their mean per-company rank correlation is above the limit "
          "(favourability: MF CORE_HOLDING=2/HOLD_WATCH=1/AVOID=0, Equity BUY/HOLD/SELL=2/1/0, PE lbo_score).", "",
          "| group | distinct top picks | top pick per persona (tied at top) | mean Spearman (pairs with >=3 shared cos) | verdict |",
          "|---|---|---|---|---|"]
    for g, d in divergence.items():
        tops = ", ".join(f"{p}: {v['company']} ({v['tied_at_top']})" for p, v in d["top_picks"].items())
        rho = d["mean_rank_correlation"]
        pairs = "; ".join(f"{k.replace('_analyst', '')}: {v['spearman']:.2f} (n={v['n_shared']})" if v["spearman"] is not None
                          else f"{k.replace('_analyst', '')}: n/a (n={v['n_shared']})" for k, v in d["pairwise_rank_correlation"].items())
        L.append(f"| {g} | {d['distinct_top_picks']} | {tops} | {('%.2f' % rho) if rho is not None else 'n/a'} - {pairs} | {'PASS' if d['passed'] else 'FAIL'} |")
    L += ["", "Structural dimensions (informational - they come from the persona YAML, so they differ by construction): "
          "mean Jaccard similarity of tools / sector-stat metrics / metrics cited / schema fields per group:", "",
          "| group | tools | sector-stat metrics | metrics cited | schema fields |", "|---|---|---|---|---|"]
    for g, d in divergence.items():
        dm = d["dimensions"]
        L.append(f"| {g} | {dm['tools']['mean_jaccard']} | {dm['sector_stat_metrics']['mean_jaccard']} | {dm['metrics_cited']['mean_jaccard']} | {dm['schema_fields']['mean_jaccard']} |")
    L += [""]
    L += [          "## Failures", ""]
    any_fail = False
    for r in results:
        for k in r.get("checks", []):
            if not k["passed"]:
                any_fail = True
                L.append(f"- **{r['id']}** / {k['name']}: {k['detail']}")
        if r.get("error"):
            any_fail = True
            L.append(f"- **{r['id']}** / run error: {r['error']}")
    if not any_fail:
        L.append("None.")
    L += ["", "## Failure analysis", "", "<!-- YOUR NOTES: for each failure above, is it a model error, a bad check, or a data problem? -->"]
    return "\n".join(L)


if __name__ == "__main__":
    asyncio.run(main())
