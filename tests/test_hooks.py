# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The tool hook runs allowed calls and returns the reason for the others, in every way Agno runs tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from agno.run.agent import ToolCallCompletedEvent
from agno.tools import Toolkit

from agno_acgs import ACGSGuard, GovernanceDecision, Verdict
from tests.fakes import BLOCKED, HELLO, LEAK, Tools, agent, call, model, results


def test_blocked_call_is_not_run_and_the_model_reads_why() -> None:
    tools = Tools()
    scripted = model([call("send_email", LEAK, "e1"), call("send_email", HELLO, "e2")], "Done.")
    output = agent(scripted, [tools.send_email]).run("Email them")
    assert tools.ran == [("send_email", HELLO)]
    found = results(output)
    assert found["e1"].startswith(BLOCKED)
    assert found["e2"] == "sent to friend@example.com"
    assert output.content == "Done."


async def test_async_run_with_an_async_tool() -> None:
    tools = Tools()
    scripted = model([call("asend_email", LEAK, "e1"), call("asend_email", HELLO, "e2")], "Done.")
    output = await agent(scripted, [tools.asend_email]).arun("Email them")
    assert tools.names == ["asend_email"]
    assert results(output)["e1"].startswith("Tool call 'asend_email' was not executed. Blocked by ACGS rule ACGS-006")


async def test_async_run_with_a_sync_tool() -> None:
    tools = Tools()
    output = await agent(model(call("send_email", LEAK, "e1"), "Done."), [tools.send_email]).arun("x")
    assert tools.ran == []
    assert results(output)["e1"].startswith(BLOCKED)


def test_streaming_run() -> None:
    tools = Tools()
    governed = agent(
        model([call("send_email", LEAK, "e1"), call("send_email", HELLO, "e2")], "Done."), [tools.send_email]
    )
    events = list(governed.run("x", stream=True, stream_events=True))
    assert tools.ran == [("send_email", HELLO)]
    completed = [event.tool for event in events if isinstance(event, ToolCallCompletedEvent) and event.tool]
    assert {tool.tool_call_id: str(tool.result) for tool in completed}["e1"].startswith(BLOCKED)


async def test_async_streaming_run() -> None:
    tools = Tools()
    governed = agent(model(call("send_email", LEAK, "e1"), "Done."), [tools.send_email])
    events = [event async for event in governed.arun("x", stream=True, stream_events=True)]
    assert tools.ran == []
    assert any(type(event).__name__ == "ToolCallCompletedEvent" for event in events)


def _async_hook(log: list[str]) -> Callable[..., Any]:
    async def audit(function_name: str, function_call: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        log.append(function_name)
        result = function_call(**arguments)
        return await result if hasattr(result, "__await__") else result

    return audit


def test_sync_run_skips_other_async_hooks_but_never_the_guard() -> None:
    # Agno's run() drops async tool hooks with a warning. The guard's hook is a plain function, so it stays.
    tools, log = Tools(), list[str]()
    guard = ACGSGuard()
    governed = agent(
        model(call("send_email", LEAK, "e1"), "Done."),
        [tools.send_email],
        guard,
        tool_hooks=[_async_hook(log), guard.tool_hook],
    )
    output = governed.run("x")
    assert log == []
    assert tools.ran == []
    assert results(output)["e1"].startswith(BLOCKED)


async def test_async_run_with_another_async_hook_still_checks_the_call() -> None:
    tools, log = Tools(), list[str]()
    guard = ACGSGuard()
    scripted = model([call("send_email", LEAK, "e1"), call("send_email", HELLO, "e2")], "Done.")
    governed = agent(scripted, [tools.send_email], guard, tool_hooks=[_async_hook(log), guard.tool_hook])
    output = await governed.arun("x")
    assert sorted(log) == ["send_email", "send_email"]
    assert tools.ran == [("send_email", HELLO)]
    assert results(output)["e1"].startswith(BLOCKED)


def test_the_reason_never_repeats_argument_values() -> None:
    tools = Tools()
    output = agent(model(call("send_email", LEAK, "e1"), "Done."), [tools.send_email]).run("x")
    assert "hunter2" not in results(output)["e1"]
    assert "attacker" not in results(output)["e1"]


def test_an_error_while_deciding_denies_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = ACGSGuard()

    def broken(tool_name: str, arguments: Any) -> GovernanceDecision:
        raise RuntimeError("boom")

    monkeypatch.setattr(guard, "evaluate", broken)
    tools = Tools()
    output = agent(model(call("send_email", HELLO, "e1"), "Done."), [tools.send_email], guard).run("x")
    assert tools.ran == []
    assert results(output)["e1"] == (
        "Tool call 'send_email' was not executed. ACGS governance could not evaluate this tool call, so it was denied."
    )
    decision = guard.decide("send_email", HELLO)
    assert decision.verdict is Verdict.DENY
    assert decision.error == "RuntimeError: boom"


def test_toolkit_tools_are_checked() -> None:
    tools = Tools()
    toolkit = Toolkit(name="mail", tools=[tools.send_email])
    output = agent(model([call("send_email", LEAK, "e1"), call("send_email", HELLO, "e2")], "Done."), [toolkit]).run(
        "x"
    )
    assert tools.ran == [("send_email", HELLO)]
    assert results(output)["e1"].startswith(BLOCKED)


def test_agno_built_in_tools_are_checked() -> None:
    # Agno's own tools (here the agentic session state tool) get the agent's tool hooks too.
    scripted = model(call("update_session_state", {"updates": {"api_key": "sk-test"}}, "s1"), "Done.")
    output = agent(scripted, [], enable_agentic_state=True, session_state={}).run("x")
    assert results(output)["s1"].startswith("Tool call 'update_session_state' was not executed. Blocked by ACGS rule")
    assert "api_key" not in (output.session_state or {})


def test_a_tool_with_its_own_hooks_still_gets_the_guard() -> None:
    # Agent(tool_hooks=...) replaces a tool's own hooks, so a tool cannot opt out of the guard.
    from agno.tools import tool

    ran: list[str] = []

    def passthrough(function_name: str, function_call: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        return function_call(**arguments)

    @tool(tool_hooks=[passthrough])
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        ran.append(to)
        return "sent"

    output = agent(model(call("send_email", LEAK, "e1"), "Done."), [send_email]).run("x")
    assert ran == []
    assert results(output)["e1"].startswith(BLOCKED)


def test_options_are_validated() -> None:
    with pytest.raises(ValueError, match="on_review must be 'deny' or 'confirmed'"):
        ACGSGuard(on_review="ask")  # type: ignore[arg-type]
    assert repr(ACGSGuard()) == "ACGSGuard(agent_id='agno', on_review='deny', allow_ungoverned=False)"
