"""Scripted stand-in for AsyncOpenAI so the agent loop can be tested without an API key.
Everything except the LLM (MCP subprocess, tool conversion, validation, grounding, confidence) is real."""
import json
from types import SimpleNamespace as NS


def tool_msg(calls):
    return NS(content=None, tool_calls=[NS(id=f"call_{i}", function=NS(name=n, arguments=json.dumps(a)))
                                        for i, (n, a) in enumerate(calls)])


def final_msg(obj):
    return NS(content=obj if isinstance(obj, str) else json.dumps(obj), tool_calls=None)


class ScriptedClient:
    """steps: list of callables(messages) -> message. Records every request in .requests."""
    def __init__(self, steps):
        self.steps, self.requests = list(steps), []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kw):
        self.requests.append(kw)
        step = self.steps.pop(0)
        msg = step(kw["messages"]) if callable(step) else step
        return NS(choices=[NS(message=msg)])


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
