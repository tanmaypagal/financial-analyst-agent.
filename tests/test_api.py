import inspect

from fastapi.testclient import TestClient

import agent.core as core
import api.main as m
from agent.models import AgentResponse

client = TestClient(m.app)


def test_meta_endpoints():
    assert client.get("/health").json()["db_present"] is True
    assert {p["key"] for p in client.get("/personas").json()} == {"mf_analyst", "equity_analyst", "pe_analyst"}
    assert {s["key"] for s in client.get("/sectors").json()} == {"defense", "tech", "logistics"}


def test_422_lists_valid_options():
    r = client.post("/query", json={"query": "x", "persona": "nope", "sector": "space"})
    assert r.status_code == 422
    e = r.json()["detail"]["errors"]
    assert "equity_analyst" in e["persona"]["valid"] and "logistics" in e["sector"]["valid"]
    assert client.post("/query", json={"persona": "mf_analyst"}).status_code == 422


def test_query_returns_structured_response(monkeypatch):
    async def fake(q, p, s, **kw):
        return AgentResponse(answer="a", persona=p, sector=s, companies_referenced=["XPO"], data_points=[], persona_output={},
                             confidence="low", confidence_reasons=[], data_gaps=[], tools_called=[], model="m", latency_ms=1.0)
    monkeypatch.setattr(m, "run_agent", fake)
    r = client.post("/query", json={"query": "q", "persona": "equity_analyst", "sector": "logistics"})
    assert r.status_code == 200 and r.json()["companies_referenced"] == ["XPO"] and r.json()["persona"] == "equity_analyst"


def test_api_and_ui_share_one_agent_function():
    assert m.run_agent is core.run_agent
    assert "run_agent_sync" in open("ui/app.py", encoding="utf-8").read()
    assert "asyncio.run(run_agent" in inspect.getsource(core.run_agent_sync)


# ----------------------------------------------------------------------------- Stage 8: error mapping, history
import httpx
import openai
import pytest

from agent.core import AgentOutputError, AgentTimeoutError

_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _resp(code):
    return httpx.Response(code, request=_REQ)


@pytest.mark.parametrize("exc, status, code", [
    (AgentTimeoutError("too slow"), 504, "agent_timeout"),
    (AgentOutputError("bad json"), 502, "invalid_agent_output"),
    (openai.RateLimitError("slow down", response=_resp(429), body=None), 429, "llm_rate_limited"),
    (openai.APITimeoutError(request=_REQ), 504, "llm_timeout"),
    (openai.AuthenticationError("bad key", response=_resp(401), body=None), 502, "llm_auth_failed"),
    (openai.APIError("boom", request=_REQ, body=None), 502, "llm_provider_error"),
    (openai.APIConnectionError(request=_REQ), 502, "llm_provider_error"),
    (RuntimeError("OPENAI_MODEL is not set"), 503, "not_configured"),
])
def test_agent_and_provider_errors_map_to_status_codes_with_json_detail(monkeypatch, exc, status, code):
    async def boom(*a, **kw):
        raise exc
    monkeypatch.setattr(m, "run_agent", boom)
    r = client.post("/query", json={"query": "q", "persona": "mf_analyst", "sector": "defense"})
    assert r.status_code == status
    assert r.json()["detail"]["error"] == code and r.json()["detail"]["message"]
    assert "sk-" not in r.text                                                     # provider messages never echo credentials


def test_history_is_accepted_and_forwarded(monkeypatch):
    seen = {}

    async def fake(q, p, s, **kw):
        seen.update(kw)
        return AgentResponse(answer="a", persona=p, sector=s, companies_referenced=[], data_points=[], persona_output={}, confidence="low",
                             confidence_reasons=[], data_gaps=[], tools_called=[], model="m", latency_ms=1.0)
    monkeypatch.setattr(m, "run_agent", fake)
    body = {"query": "and its margins?", "persona": "equity_analyst", "sector": "logistics",
            "history": [{"query": "Tell me about XPO", "answer": "XPO is ..."}]}
    assert client.post("/query", json=body).status_code == 200
    assert seen["history"] == [{"query": "Tell me about XPO", "answer": "XPO is ..."}]


def test_streamlit_sends_the_last_three_completed_turns_to_run_agent(monkeypatch):
    from streamlit.testing.v1 import AppTest
    calls = []

    def fake(q, persona, sector, **kw):
        calls.append((q, persona, sector, kw.get("history")))
        return AgentResponse(answer=f"answer {q}", persona=persona, sector=sector, companies_referenced=[], data_points=[], persona_output={},
                             confidence="low", confidence_reasons=[], data_gaps=[], tools_called=[], model="m", latency_ms=1.0)
    monkeypatch.setattr(core, "run_agent_sync", fake)
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "ui" / "app.py"), default_timeout=30).run()
    for i in range(5):
        at.chat_input[0].set_value(f"q{i}").run()
    assert not at.exception
    assert calls[0][3] == [] and calls[1][3] == [{"query": "q0", "answer": "answer q0"}]
    assert calls[4][3] == [{"query": f"q{i}", "answer": f"answer q{i}"} for i in (1, 2, 3)]          # exactly the last three
