"""The one agent implementation. Both the REST API and the Streamlit UI call run_agent()."""
import asyncio
import json
import os
import re
import time
from typing import Any

import jsonschema
from dotenv import load_dotenv
from pydantic import ValidationError

from agent.confidence import LEVELS, policy_confidence
from agent.config import build_system_prompt, load_personas, load_sectors, validate_selection
from agent.grounding import GroundingPool, decimals_of, parse_display, persona_numbers, unit_of
from agent.mcp_client import MCPToolClient
from agent.models import AgentResponse, DataPoint, LLMAnswer, ToolCall

load_dotenv()
MAX_TOOL_CHARS = 24000


class AgentOutputError(RuntimeError):
    """The model failed to produce a valid structured answer twice."""


class AgentTimeoutError(RuntimeError):
    """The whole run exceeded AGENT_TIMEOUT_S."""


def _model_name(model: str | None) -> str:
    m = model or os.environ.get("OPENAI_MODEL")
    if not m:
        raise RuntimeError("OPENAI_MODEL is not set (put it in .env; see .env.example)")
    return m


def _make_client():
    from openai import AsyncOpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env; see .env.example)")
    return AsyncOpenAI(max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", 6)),        # SDK backs off on 429 / 5xx
                       timeout=float(os.environ.get("OPENAI_TIMEOUT", 120)))            # per LLM request, seconds


def _llm_kwargs() -> dict:
    """Optional reasoning-effort knob for reasoning models (OPENAI_REASONING_EFFORT=minimal|low|medium|high)."""
    eff = os.environ.get("OPENAI_REASONING_EFFORT")
    return {"reasoning_effort": eff} if eff else {}


def _hidden_tools(sector: str) -> set[str]:
    """Sector-specific tools (config/sectors.yaml `sector_tools`) are only exposed for their own sector."""
    hidden = set()
    for name, cfg in load_sectors().items():
        if name != sector:
            hidden |= set(cfg.get("sector_tools") or [])
    return hidden


def _llm_view(result: dict) -> str:
    """What the model sees: same JSON, minus the (bulky) sources map - source_id is enough to cite."""
    slim = {k: v for k, v in result.items() if k != "sources"}
    s = json.dumps(slim, default=str)
    return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + '..."truncated"'


def _parse_final(text: str, persona_key: str) -> LLMAnswer:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    ans = LLMAnswer.model_validate_json(text)
    jsonschema.validate(ans.persona_output, load_personas()[persona_key]["output_schema_extension"])
    return ans


def _dp_number(dp: DataPoint):
    """(value, decimals, unit) of a data point, parsing numeric strings like '$41.7B', '14.4x', '12%'. None if non-numeric."""
    v = dp.value
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v), decimals_of(v), unit_of(dp.unit)
    if isinstance(v, str):
        nums = parse_display(v)
        if nums:
            val, dec, unit = nums[0]
            return val, dec, unit or unit_of(dp.unit)
    return None


def _prose_texts(ans: LLMAnswer) -> list[str]:
    """Text to number-check: `answer` plus the string / numeric leaves of `persona_output` (lbo_score is a judgement, not data)."""
    texts = [ans.answer]

    def walk(o, key=None):
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, (int, float)) and not isinstance(o, bool) and key != "lbo_score":
            texts.append(str(o))
    walk(ans.persona_output)
    return texts


def _finalize(ans: LLMAnswer, persona: str, sector: str, model: str, calls: list[ToolCall], results: list[dict],
              t0: float) -> AgentResponse:
    pool = GroundingPool(extra=persona_numbers(load_personas()[persona]["thresholds"]))
    for r in results:
        pool.add_result(r)
    kept, dropped, non_numeric = [], [], []
    for dp in ans.data_points:
        if dp.value is None:                                   # a null metric reported as null: nothing to ground
            kept.append(dp)
            continue
        parsed = _dp_number(dp)
        if parsed is None:
            non_numeric.append(f"{dp.company}/{dp.metric}={dp.value!r}")
        elif not pool.check(*parsed, company=dp.company, scoped=True):
            dropped.append(f"{dp.company}/{dp.metric}={dp.value}")
        else:
            kept.append(dp)
    gaps = list(ans.data_gaps)
    if dropped:
        gaps.append("Removed ungrounded data points (value not found in that company's tool results): " + "; ".join(dropped))
    if non_numeric:
        gaps.append("Removed non-numeric data points: " + "; ".join(non_numeric))
    for r in results:                                          # field-level mismatches vs filings are always surfaced, model or not
        for row in r.get("financials") or []:
            for vf in row.get("verified_fields") or []:
                if vf.get("status") == "mismatch":
                    msg = (f"FIELD MISMATCH vs filing: {r.get('company')} {row.get('period_end')} {vf['field']} stored {vf['db_value']:,.0f} "
                           f"but the filing says {vf['other_value']:,.0f}")
                    if msg not in gaps:
                        gaps.append(msg)
    # numbers in the prose: never rewritten, but reported and they cap confidence
    bad = []
    for t in _prose_texts(ans):
        bad += [f"{v:g}{unit or ''}" for v, _, unit in pool.ungrounded_in(t) if f"{v:g}{unit or ''}" not in bad]
    cap, reasons = policy_confidence(results, calls, persona, ans.companies_referenced)
    if bad:
        gaps.append("Numbers in the answer/persona_output not found in any tool result (unverified): " + ", ".join(bad[:15]))
        reasons.append(f"policy: {len(bad)} number(s) in the answer could not be traced to a tool result")
        cap = LEVELS[min(LEVELS.index(cap), LEVELS.index("medium"))]
    conf = LEVELS[min(LEVELS.index(ans.confidence), LEVELS.index(cap))]
    sources: dict[int, str] = {}
    for r in results:
        for sid, s in (r.get("sources") or {}).items():
            sources[int(sid)] = s["url"]
    return AgentResponse(
        answer=ans.answer, persona=persona, sector=sector, companies_referenced=ans.companies_referenced,
        data_points=kept, persona_output=ans.persona_output, confidence=conf,
        confidence_reasons=ans.confidence_reasons + reasons, data_gaps=gaps, tools_called=calls, model=model,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1), sources=sources)


async def run_agent(query: str, persona: str, sector: str, *, history: list | None = None, timeout_s: float | None = None,
                    **kw) -> AgentResponse:
    """Answer `query` as `persona` about `sector`, grounding every fact in MCP tool results.

    history: up to the last 3 earlier turns [{"query","answer"}] - used only to resolve references ("it", "that company").
    timeout_s / AGENT_TIMEOUT_S: overall limit for the whole run (LLM calls + tools); raises AgentTimeoutError.
    `client`/`model`/`mcp_url`/`max_iterations` are injectable for tests; `trace` (optional list) receives raw tool results.
    """
    limit = timeout_s if timeout_s is not None else float(os.environ.get("AGENT_TIMEOUT_S", 300))
    try:
        return await asyncio.wait_for(_run_agent(query, persona, sector, history, **kw), timeout=limit or None)
    except asyncio.TimeoutError as e:
        raise AgentTimeoutError(f"agent run exceeded the {limit:g}s limit (AGENT_TIMEOUT_S)") from e


async def _run_agent(query: str, persona: str, sector: str, history: list | None, *, client: Any = None, model: str | None = None,
                     mcp_url: str | None = None, max_iterations: int | None = None, trace: list | None = None) -> AgentResponse:
    t0 = time.perf_counter()
    validate_selection(persona, sector)
    model = _model_name(model)
    client = client or _make_client()
    max_iter = max_iterations or int(os.environ.get("MAX_TOOL_ITERATIONS", 8))
    hidden = _hidden_tools(sector)
    calls: list[ToolCall] = []
    results: list[dict] = []

    async with MCPToolClient(mcp_url) as mcp:
        tools = [t for t in mcp.openai_tools() if t["function"]["name"] not in hidden]
        messages: list[dict] = [{"role": "system", "content": build_system_prompt(persona, sector)}]
        for turn in (history or [])[-3:]:                       # earlier turns: context only; every number must be re-fetched
            messages += [{"role": "user", "content": str(turn["query"])}, {"role": "assistant", "content": str(turn["answer"])}]
        messages.append({"role": "user", "content": query})
        final_text = None
        for i in range(max_iter + 1):
            last = i == max_iter                      # cap reached: force an answer with no tools
            resp = await client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="none" if last else "auto", **_llm_kwargs())
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None) or []
            if not tcs or last:
                final_text = msg.content
                messages.append({"role": "assistant", "content": msg.content})
                break
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tcs]})
            allowed = {t["function"]["name"] for t in tools}
            takes_sector = {t["function"]["name"] for t in tools if "sector" in t["function"]["parameters"].get("properties", {})}

            async def exec_one(tc):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name in hidden or name not in allowed:
                    return tc, name, args, {"found": False, "reason_code": "BAD_ARGS",
                                            "reason": f"tool '{name}' is not available for sector '{sector}'"}, 0.0
                # Sector isolation is enforced HERE (the model picks tool arguments): a call for any sector other than the
                # user's selection never reaches the MCP server.
                if name in takes_sector and str(args.get("sector", sector)).strip().lower() != sector.lower():
                    return tc, name, args, {"found": False, "reason_code": "WRONG_SELECTED_SECTOR",
                                            "reason": f"The selected sector is '{sector}'; the call asked for '{args.get('sector')}'. "
                                                      f"No data returned. Only query the selected sector."}, 0.0
                try:
                    out, ms = await mcp.call(name, args)
                except Exception as e:                          # bad args etc. surface to the model, not the user
                    out, ms = {"found": False, "reason_code": "BAD_ARGS", "reason": f"tool call failed: {e}"}, 0.0
                return tc, name, args, out, ms

            for tc, name, args, out, ms in await asyncio.gather(*(exec_one(tc) for tc in tcs)):   # order preserved
                calls.append(ToolCall(name=name, args=args, latency_ms=round(ms, 1), found=out.get("found")))
                results.append(out)
                if trace is not None:
                    trace.append({"tool": name, "args": args, "result": out})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _llm_view(out)})

        try:
            ans = _parse_final(final_text, persona)
        except (ValidationError, jsonschema.ValidationError, ValueError) as e:
            messages.append({"role": "user", "content": "Your reply was not valid. Return ONLY the JSON object described in the "
                             f"system prompt. Validation error: {str(e)[:800]}"})
            resp = await client.chat.completions.create(model=model, messages=messages, tool_choice="none",
                                                        tools=tools, response_format={"type": "json_object"}, **_llm_kwargs())
            try:
                ans = _parse_final(resp.choices[0].message.content, persona)
            except (ValidationError, jsonschema.ValidationError, ValueError) as e2:
                raise AgentOutputError(f"model returned invalid structured output twice: {str(e2)[:500]}") from e2

    return _finalize(ans, persona, sector, model, calls, results, t0)


def run_agent_sync(query: str, persona: str, sector: str, **kw) -> AgentResponse:
    """Blocking wrapper for callers without an event loop (Streamlit, scripts)."""
    return asyncio.run(run_agent(query, persona, sector, **kw))
