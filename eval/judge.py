"""LLM-judge hook. Sends (query, persona, answer, tool outputs, rubric) to a judge model and returns scored JSON.

The judge model is configured by JUDGE_MODEL (.env) and MUST differ from the agent's OPENAI_MODEL.
Rubrics live in eval/rubrics/*.yaml (see template.yaml - you write them).

Tool output reaches the judge as an ORDERED, PER-COMPANY DIGEST (digest_tool_outputs), not a raw dump cut at N characters: a raw cut
silently drops the LAST tool results, which is exactly where a run's final checks (data quality, verification) sit.
"""
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from common.naming import normalize_company

load_dotenv()
RUBRICS = Path(__file__).parent / "rubrics"
OLD_CUT = 40000                       # the previous raw truncation length, kept only to report how often it would have bitten
DEFAULT_MAX_CHARS = 120000            # digest budget (JUDGE_MAX_CHARS overrides)


class CriterionScore(BaseModel):
    score: float
    rationale: str


class JudgeResult(BaseModel):
    scores: dict[str, CriterionScore]
    overall: float
    flags: list[str] = []
    passed: bool | None = None
    raw_chars: int = 0                # size of the tool outputs as JSON
    digest_chars: int = 0             # size of what the judge actually saw
    digest_level: int = 0             # 0 = lossless-ish compact digest; 1-2 = progressively compacted; 3 = per-entity truncation
    truncated: bool = False           # True only at level 3 (some entity text was cut; every entity is still present)


def load_rubric(name: str) -> dict:
    path = RUBRICS / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"rubric '{name}' not found in {RUBRICS}")
    text = path.read_text(encoding="utf-8")
    if "<<" in text:
        raise ValueError(f"{path.name} still contains <<PLACEHOLDER>> text - fill in the rubric before judging")
    return yaml.safe_load(text)


def _check_models(judge_model: str, agent_model: str | None):
    if not judge_model:
        raise RuntimeError("JUDGE_MODEL is not set (put it in .env)")
    if agent_model and judge_model == agent_model:
        raise ValueError(f"JUDGE_MODEL ({judge_model}) must differ from the agent model OPENAI_MODEL ({agent_model})")


# ------------------------------------------------------------------------------------------------ digest
def _entries(tool_outputs: list) -> list[tuple[str, dict, dict]]:
    """Accept raw results or trace items {"tool","args","result"} -> [(tool, args, result)] in call order."""
    out = []
    for i, t in enumerate(tool_outputs):
        if isinstance(t, dict) and "result" in t and "tool" in t:
            out.append((t["tool"], t.get("args") or {}, t["result"]))
        else:
            out.append((f"call_{i + 1}", {}, t))
    return out


def _round(o, sig: int):
    if isinstance(o, float):
        return float(f"{o:.{sig}g}")
    if isinstance(o, dict):
        return {k: _round(v, sig) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, sig) for v in o]
    return o


def _compact(o, level: int):
    """Level 0: drop the sources map, shorten long strings. Level 1: 4 significant digits, no verification notes, short strings, <=5 signals.
    Level 2: also drop per-item gap lists and every field-verification note."""
    limit = {0: 300, 1: 120, 2: 80}[level]
    if isinstance(o, dict):
        drop = {"sources"}
        if level >= 1:
            drop |= {"note", "notes", "verify_note", "units", "description"}
        if level >= 2:
            drop |= {"gaps", "data_gaps", "source_ids", "data_date_per_company", "reason"}
        out = {k: _compact(v, level) for k, v in o.items() if k not in drop}
        if level >= 1 and isinstance(out.get("signals"), list):
            out["signals"] = out["signals"][:5]
        return out
    if isinstance(o, list):
        return [_compact(v, level) for v in o]
    if isinstance(o, str) and len(o) > limit:
        return o[:limit] + "..."
    return o


def _entity(result: dict, tool: str) -> str:
    """Which company a result is about ('_sector' for sector-level results)."""
    prof = result.get("profile") or {}
    name = result.get("company") or prof.get("name")
    return name if isinstance(name, str) else "_sector"


def digest_tool_outputs(tool_outputs: list, max_chars: int | None = None) -> tuple[str, dict]:
    """Ordered per-company digest of every tool result of a run: {entity: {tool: result, ...}} in order of first appearance, with
    sector-level results under '_sector'. Compaction escalates only if the digest exceeds max_chars; at the last level each entity's text is
    cut to its share of the budget, so no tool result / company disappears. Returns (json_text, stats)."""
    max_chars = max_chars or int(os.environ.get("JUDGE_MAX_CHARS", DEFAULT_MAX_CHARS))
    entries = _entries(tool_outputs)
    raw = len(json.dumps([r for _, _, r in entries], default=str))

    def build(level: int, sig: int | None) -> dict:
        groups: dict[str, dict] = {}
        for tool, args, res in entries:
            if isinstance(res.get("companies"), list) and res["companies"] and isinstance(res["companies"][0], dict):
                for e in res["companies"]:                                       # get_company_snapshot: one entity per company
                    groups.setdefault(e.get("company", "_sector"), {})[tool] = _compact(e, level) if not sig else _round(_compact(e, level), sig)
                rest = {k: v for k, v in res.items() if k not in ("companies", "sources")}
                groups.setdefault("_sector", {})[f"{tool}(meta)"] = _compact(rest, level)
                continue
            r = _compact(res, level)
            if sig:
                r = _round(r, sig)
            key = tool if tool not in groups.get(_entity(res, tool), {}) else f"{tool}#{len(groups[_entity(res, tool)]) + 1}"
            if tool.startswith("get_sector_stats") or tool.startswith("list_") or tool == "get_schema":
                key = f"{tool}({args.get('metric') or args.get('sector') or ''})"
            groups.setdefault(_entity(res, tool), {})[key] = r
        return groups

    for level, sig in ((0, 6), (1, 4), (2, 4)):
        text = json.dumps(build(level, sig), default=str, separators=(",", ":"))
        if len(text) <= max_chars:
            return text, {"raw_chars": raw, "digest_chars": len(text), "level": level, "truncated": False}
    groups = build(2, 4)                                                       # level 3: cut each entity to its share, keep all of them
    share = max(200, max_chars // max(1, len(groups)) - 40)
    cut = {e: (s if len(s) <= share else s[:share] + "...[cut]") for e, s in ((e, json.dumps(v, default=str, separators=(",", ":"))) for e, v in groups.items())}
    text = json.dumps(cut, default=str, separators=(",", ":"))
    return text, {"raw_chars": raw, "digest_chars": len(text), "level": 3, "truncated": True}


def judge_input_report(traces: list[list]) -> dict:
    """How often the judge's input needed help: the OLD raw cut (40,000 chars) vs the new digest, over a set of runs."""
    stats = [digest_tool_outputs(t)[1] for t in traces]
    return {"runs": len(stats), "old_cut_would_have_dropped_data": sum(s["raw_chars"] > OLD_CUT for s in stats),
            "max_raw_chars": max((s["raw_chars"] for s in stats), default=0),
            "digest_compacted_level_1_or_2": sum(s["level"] in (1, 2) for s in stats),
            "digest_truncated_level_3": sum(s["truncated"] for s in stats),
            "max_digest_chars": max((s["digest_chars"] for s in stats), default=0)}


# ------------------------------------------------------------------------------------------------ judging
def build_messages(query, persona, answer, tool_outputs, rubric, max_chars: int | None = None) -> tuple[list[dict], dict]:
    digest, stats = digest_tool_outputs(tool_outputs, max_chars)
    system = (rubric["judge_instructions"].strip() + "\n\nScore each criterion on the scale "
              f"{rubric['scale']['min']}-{rubric['scale']['max']}. Return ONLY JSON: "
              '{"scores": {"<criterion_id>": {"score": <number>, "rationale": "<why>"}}, "overall": <mean>, "flags": [<strings>]}')
    user = (f"RUBRIC:\n{yaml.safe_dump(rubric['criteria'], sort_keys=False)}\n\nPERSONA: {persona}\n\nQUERY: {query}\n\n"
            f"ANSWER UNDER REVIEW:\n{answer}\n\nTOOL OUTPUTS - ground truth for this run, digested per company in call order "
            f"(digest level {stats['level']}{', TEXT CUT' if stats['truncated'] else ''}):\n{digest}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], stats


async def judge(query: str, persona: str, answer: str, tool_outputs: list, rubric: dict, *, judge_model: str | None = None,
                client=None) -> JudgeResult:
    judge_model = judge_model or os.environ.get("JUDGE_MODEL", "")
    _check_models(judge_model, os.environ.get("OPENAI_MODEL"))
    if client is None:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
    messages, stats = build_messages(query, persona, answer, tool_outputs, rubric)
    resp = await client.chat.completions.create(model=judge_model, messages=messages, response_format={"type": "json_object"})
    res = JudgeResult.model_validate_json(resp.choices[0].message.content)
    res.raw_chars, res.digest_chars, res.digest_level, res.truncated = stats["raw_chars"], stats["digest_chars"], stats["level"], stats["truncated"]
    thr = rubric.get("pass_threshold")
    if thr is not None:
        res.passed = res.overall >= float(thr)
    return res
