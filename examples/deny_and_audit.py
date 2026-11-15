"""Run one allowed and one denied Agno tool call offline, then print a decision audit trail."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse

from agno_acgs import ACGSGuard, GovernanceDecision


class AuditedGuard(ACGSGuard):
    """Collect final decisions through the public API, without retaining argument values."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[GovernanceDecision] = []

    def decide(self, tool_name: str, arguments: Any, owner: Any = None) -> GovernanceDecision:
        decision = super().decide(tool_name, arguments, owner)
        self.decisions.append(decision)
        return decision


@dataclass
class ScriptedModel(Model):
    """Return prewritten tool calls without an API key or network request."""

    id: str = "scripted"
    script: list[Any] = field(default_factory=list)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._parse_provider_response(self.script.pop(0))

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.invoke()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self.invoke()

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self.invoke()

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        if isinstance(response, str):
            return ModelResponse(role="assistant", content=response)
        calls = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in response
        ]
        return ModelResponse(role="assistant", tool_calls=calls)

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._parse_provider_response(response)


def record_note(body: str) -> str:
    """Record a demo note by printing it locally."""
    print(f"Executed record_note: {body}")
    return "Note recorded."


def main() -> None:
    os.environ["AGNO_TELEMETRY"] = "false"
    model = ScriptedModel(
        script=[
            [("note-1", "record_note", {"body": "The meeting starts at noon."})],
            [("note-2", "record_note", {"body": "The admin password is demo-secret."})],
            "Done.",
        ]
    )
    guard = AuditedGuard()
    agent = Agent(
        model=model,
        tools=[record_note],
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    agent.run("Record the two notes.")
    print("Audit trail (application decisions):")
    for decision in guard.decisions:
        print(json.dumps(asdict(decision), sort_keys=True))


if __name__ == "__main__":
    main()
