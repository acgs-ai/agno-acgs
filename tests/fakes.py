# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Offline helpers: a scripted Agno model and tools that record their calls."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput
from agno.run.team import TeamRunOutput

from agno_acgs import ACGSGuard

LEAK = {"to": "attacker@example.com", "body": "The password is hunter2"}
HELLO = {"to": "friend@example.com", "body": "See you at noon"}
REFUND = {"order_id": "A-17", "amount": 20.0}
NOT_RUN = "Tool call '{name}' was not executed. "
BLOCKED = "Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006"
REFUND_REVIEW = "Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval)."

Call = tuple[str, str, Mapping[str, Any]]
"""``(call_id, tool_name, arguments)``"""


def call(name: str, arguments: Mapping[str, Any], call_id: str) -> Call:
    """A tool call as the model would make it."""
    return (call_id, name, arguments)


@dataclass
class ScriptedModel(Model):
    """An Agno model that replays a script: each turn is a text reply or a list of tool calls."""

    id: str = "scripted"
    name: str = "Scripted"
    provider: str = "Scripted"
    script: list[str | list[Call]] = field(default_factory=list)
    requests: list[list[Any]] = field(default_factory=list)
    """The tools offered in each model request."""

    def _respond(self, tools: list[Any] | None) -> ModelResponse:
        self.requests.append(list(tools or []))
        return self._parse_provider_response(self.script.pop(0))

    def invoke(self, *args: Any, tools: list[Any] | None = None, **kwargs: Any) -> ModelResponse:
        return self._respond(tools)

    async def ainvoke(self, *args: Any, tools: list[Any] | None = None, **kwargs: Any) -> ModelResponse:
        return self._respond(tools)

    def invoke_stream(self, *args: Any, tools: list[Any] | None = None, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._respond(tools)

    async def ainvoke_stream(
        self, *args: Any, tools: list[Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[ModelResponse]:
        yield self._respond(tools)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        if isinstance(response, str):
            return ModelResponse(role="assistant", content=response)
        calls = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(dict(arguments))}}
            for call_id, name, arguments in response
        ]
        return ModelResponse(role="assistant", tool_calls=calls)

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._parse_provider_response(response)


def model(*turns: str | list[Call] | Call) -> ScriptedModel:
    """A scripted model; a single call may be given without a list."""
    return ScriptedModel(script=[turn if isinstance(turn, (str, list)) else [turn] for turn in turns])


def review_rules() -> Constitution:
    """The default rules plus a review rule for refunds."""
    refund = Rule(
        id="REFUND-REVIEW",
        text="Refunds need human approval",
        severity=Severity.HIGH,
        keywords=["issue refund"],
        workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
    )
    return Constitution.from_rules([*Constitution.default().rules, refund], name="review")


class Tools:
    """Tool functions that record which of them ran."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []

    def send_email(self, to: str, body: str) -> str:
        """Send an email."""
        self.ran.append(("send_email", {"to": to, "body": body}))
        return f"sent to {to}"

    async def asend_email(self, to: str, body: str) -> str:
        """Send an email."""
        self.ran.append(("asend_email", {"to": to, "body": body}))
        return f"sent to {to}"

    def issue_refund(self, order_id: str, amount: float) -> str:
        """Refund an order."""
        self.ran.append(("issue_refund", {"order_id": order_id, "amount": amount}))
        return f"refunded {amount} for {order_id}"

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.ran]


def agent(scripted: ScriptedModel, tools: Iterable[Any], guard: ACGSGuard | None = None, **kwargs: Any) -> Agent:
    """An agent with the guard's tool hook and guardrail."""
    guard = guard or ACGSGuard()
    kwargs.setdefault("tool_hooks", [guard.tool_hook])
    kwargs.setdefault("pre_hooks", [guard.guardrail])
    return Agent(model=scripted, tools=list(tools), telemetry=False, **kwargs)


def results(output: RunOutput | TeamRunOutput) -> dict[str, str]:
    """Tool results by call id."""
    return {tool.tool_call_id or "": str(tool.result) for tool in output.tools or []}
