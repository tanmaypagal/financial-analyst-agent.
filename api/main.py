"""REST API. Thin wrapper: everything goes through agent.core.run_agent (same function the UI uses).

Run:  uvicorn api.main:app --port 8000
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.config import load_personas, load_sectors, valid_personas, valid_sectors
import openai

from agent.core import AgentOutputError, AgentTimeoutError, run_agent
from agent.models import AgentResponse, QueryRequest

app = FastAPI(title="Financial Analyst Agent", version="1.0")


@app.post("/query", response_model=AgentResponse)
async def query(req: QueryRequest):
    bad = {}
    if req.persona not in valid_personas():
        bad["persona"] = {"got": req.persona, "valid": valid_personas()}
    if req.sector not in valid_sectors():
        bad["sector"] = {"got": req.sector, "valid": valid_sectors()}
    if bad:
        raise HTTPException(status_code=422, detail={"message": "invalid persona and/or sector", "errors": bad})
    try:
        return await run_agent(req.query, req.persona, req.sector, history=[t.model_dump() for t in req.history])
    except AgentTimeoutError as e:
        raise HTTPException(status_code=504, detail={"error": "agent_timeout", "message": str(e)})
    except AgentOutputError as e:
        raise HTTPException(status_code=502, detail={"error": "invalid_agent_output",
                                                     "message": f"agent could not produce a valid structured answer: {e}"})
    except openai.RateLimitError as e:
        raise HTTPException(status_code=429, detail={"error": "llm_rate_limited", "message": "the LLM provider rate-limited the request; retry later"})
    except openai.APITimeoutError as e:
        raise HTTPException(status_code=504, detail={"error": "llm_timeout", "message": "the LLM provider timed out"})
    except openai.AuthenticationError as e:
        raise HTTPException(status_code=502, detail={"error": "llm_auth_failed", "message": "the LLM provider rejected the server's API key"})
    except openai.APIError as e:                     # connection errors, 5xx, bad requests from the provider
        raise HTTPException(status_code=502, detail={"error": "llm_provider_error", "message": f"{type(e).__name__}: {str(e)[:200]}"})
    except RuntimeError as e:                       # e.g. missing OPENAI_API_KEY / OPENAI_MODEL
        raise HTTPException(status_code=503, detail={"error": "not_configured", "message": str(e)})


@app.get("/health")
async def health(deep: bool = False):
    from mcp_server.queries import DB_PATH
    out = {"status": "ok", "db_present": Path(DB_PATH).exists(), "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
           "model": os.environ.get("OPENAI_MODEL")}
    if deep:                                        # spawn the MCP server and list its tools
        from agent.mcp_client import MCPToolClient
        try:
            async with MCPToolClient() as c:
                out["mcp_tools"] = [t.name for t in c.tools]
        except Exception as e:
            out["status"], out["mcp_error"] = "degraded", str(e)
    if not out["db_present"]:
        out["status"] = "degraded"
    return out


@app.get("/personas")
async def personas():
    return [{"key": k, "display_name": v["display_name"], "lens": v["lens_description"].strip()}
            for k, v in load_personas().items()]


@app.get("/sectors")
async def sectors():
    cfg = load_sectors()
    return [{"key": k, "display_name": cfg[k].get("display_name", k), "n_companies": len(cfg[k]["companies"])}
            for k in valid_sectors()]
