"""Pydantic models for the agent's structured output and the REST response contract."""
from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]


class DataPoint(BaseModel):
    company: str
    metric: str
    value: float | str | None = None
    unit: str | None = None
    source_id: int | None = None
    as_of: str | None = None
    stale: bool | None = None


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    latency_ms: float
    found: bool | None = None          # False when the tool returned {"found": false}


class LLMAnswer(BaseModel):
    """What the model must return as its final message (validated; one retry on failure)."""
    answer: str
    companies_referenced: list[str] = Field(default_factory=list)
    data_points: list[DataPoint] = Field(default_factory=list)
    persona_output: dict[str, Any]
    confidence: Confidence
    confidence_reasons: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str
    persona: str
    sector: str
    companies_referenced: list[str]
    data_points: list[DataPoint]
    persona_output: dict[str, Any]
    confidence: Confidence
    confidence_reasons: list[str]
    data_gaps: list[str]
    tools_called: list[ToolCall]
    model: str
    latency_ms: float
    sources: dict[int, str] = Field(default_factory=dict, description="source_id -> URL (additive convenience field)")


class Turn(BaseModel):
    query: str
    answer: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    persona: str
    sector: str
    history: list[Turn] = Field(default_factory=list, description="optional earlier turns (last 3 are used) to resolve references")
