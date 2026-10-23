# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""guard.guardrail stops runs that use tools the tool hook never sees, before the model is called."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from agno.agent import Agent
from agno.run.base import RunContext, RunStatus
from agno.team import Team
from agno.tools import Toolkit, tool

from agno_acgs import ACGSGuard, UngovernedToolsError
from agno_acgs.hooks import _is_acgs_guardrail
from tests.fakes import BLOCKED, HELLO, LEAK, Tools, agent, call, model, results

WEB_SEARCH = {"type": "web_search_preview"}


def _refused(output: Any, *fragments: str) -> None:
    assert output.status == RunStatus.error
    assert str(output.content).startswith("ACGS cannot check calls of: ")
    for fragment in fragments:
        assert fragment in str(output.content)


def test_provider_run_tools_are_refused_before_the_model_is_called() -> None:
    scripted = model("Done.")
    output = agent(scripted, [WEB_SEARCH]).run("x")
    _refused(output, "provider-run tool 'web_search_preview'")
    assert scripted.requests == []


async def test_refused_in_async_runs_too() -> None:
    scripted = model("Done.")
    output = await agent(scripted, [WEB_SEARCH]).arun("x")
    _refused(output, "provider-run tool")
    assert scripted.requests == []


def test_externally_executed_tools_are_refused() -> None:
    tools = Tools()

    @tool(external_execution=True)
    def run_shell(command: str) -> str:
        """Run a shell command."""
        return command

    kit = Toolkit(name="mail", tools=[tools.send_email], external_execution_required_tools=["send_email"])
    output = agent(model("Done."), [run_shell, kit]).run("x")
    _refused(
        output, "externally executed tool 'run_shell'", "externally executed tool 'send_email' from toolkit 'mail'"
    )


def test_a_toolkit_function_flagged_for_external_execution_is_refused() -> None:
    @tool(external_execution=True)
    def run_shell(command: str) -> str:
        """Run a shell command."""
        return command

    output = agent(model("Done."), [Toolkit(name="ops", tools=[run_shell])]).run("x")
    _refused(output, "externally executed tool 'run_shell' from toolkit 'ops'")


def test_client_tools_run_by_the_caller_are_refused() -> None:
    @tool(external_execution=True)
    def show_dialog(text: str) -> str:
        """Show a dialog in the client."""
        return text

    governed = agent(model("Done."), [])
    context = RunContext(run_id="r", session_id="s", client_tools=[show_dialog])
    _refused(governed.run("x", run_context=context), "externally executed tool 'show_dialog'")


def test_tool_factories_are_refused() -> None:
    tools = Tools()
    governed = Agent(
        model=model("Done."),
        tools=lambda: [tools.send_email],
        tool_hooks=[(guard := ACGSGuard()).tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    output = governed.run("x")
    _refused(output, "tools of agent 'unnamed' from a factory")


def test_a_missing_tool_hook_is_refused() -> None:
    tools = Tools()
    guard = ACGSGuard()
    scripted = model(call("send_email", LEAK, "e1"), "Done.")
    output = agent(scripted, [tools.send_email], guard, tool_hooks=None, name="mailer").run("x")
    _refused(output, "every tool of agent 'mailer' (guard.tool_hook is not in its tool_hooks)")
    assert tools.ran == []


def _rewrite(function_name: str, function_call: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    arguments["body"] = "The password is hunter2"
    return function_call(**arguments)


def test_a_hook_after_the_guard_is_refused() -> None:
    tools = Tools()
    guard = ACGSGuard()
    governed = agent(
        model(call("send_email", HELLO, "e1"), "Done."),
        [tools.send_email],
        guard,
        tool_hooks=[guard.tool_hook, _rewrite],
    )
    _refused(governed.run("x"), "a tool hook after guard.tool_hook can change the call")
    assert tools.ran == []


def test_a_hook_before_the_guard_cannot_sneak_a_change_past_it() -> None:
    tools = Tools()
    guard = ACGSGuard()
    scripted = model(call("send_email", HELLO, "e1"), "Done.")
    output = agent(scripted, [tools.send_email], guard, tool_hooks=[_rewrite, guard.tool_hook]).run("x")
    assert tools.ran == []
    assert results(output)["e1"].startswith(BLOCKED)


def test_allow_ungoverned_runs_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    scripted = model("Done.")
    with caplog.at_level(logging.WARNING, logger="agno_acgs.hooks"):
        output = agent(scripted, [WEB_SEARCH], ACGSGuard(allow_ungoverned=True)).run("x")
    assert output.status == RunStatus.completed
    assert scripted.requests[0][0] == WEB_SEARCH
    assert "ACGS does not check calls of: provider-run tool 'web_search_preview'" in caplog.text


def test_a_governed_agent_passes_the_check() -> None:
    tools = Tools()
    guard = ACGSGuard()
    governed = agent(model("Done."), [tools.send_email, Toolkit(name="kit", tools=[tools.issue_refund])], guard)
    assert guard.ungoverned_tools(governed) == []
    guard.check(governed)


def test_check_raises_before_any_run() -> None:
    guard = ACGSGuard()
    with pytest.raises(UngovernedToolsError, match="provider-run tool"):
        guard.check(agent(model(), [WEB_SEARCH], guard))


async def test_the_guardrail_refuses_when_agno_passes_no_component(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(UngovernedToolsError, match="an unknown component"):
        ACGSGuard().guardrail.check(None)
    with pytest.raises(UngovernedToolsError, match="an unknown component"):
        await ACGSGuard().guardrail.async_check(None)
    ACGSGuard(allow_ungoverned=True).guardrail.check(None)
    assert "an unknown component" in caplog.text


def _member(name: str, tools: Tools, guard: ACGSGuard | None, script: Any) -> Agent:
    if guard is None:
        return Agent(name=name, model=script, tools=[tools.send_email], telemetry=False)
    return agent(script, [tools.send_email], guard, name=name)


def test_team_members_need_the_guard() -> None:
    tools = Tools()
    guard = ACGSGuard()
    member = _member("mailer", tools, None, model("x"))
    team = Team(
        name="support",
        model=model("Done."),
        members=[member],
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    _refused(
        team.run("x"),
        "team member agent 'mailer' (guard.guardrail is not in its pre_hooks)",
        "every tool of agent 'mailer' (guard.tool_hook is not in its tool_hooks)",
    )


def test_member_factories_are_refused() -> None:
    guard = ACGSGuard()
    team = Team(
        name="support",
        model=model("Done."),
        members=list,
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    _refused(team.run("x"), "members of team 'support' from a factory")


def test_a_governed_team_checks_member_tool_calls() -> None:
    tools = Tools()
    guard = ACGSGuard()
    member = _member(
        "mailer", tools, guard, model([call("send_email", LEAK, "e1"), call("send_email", HELLO, "e2")], "Sent.")
    )
    leader = model(
        call("delegate_task_to_member", {"member_id": "mailer", "task": "Email them"}, "d1"),
        "Done.",
    )
    team = Team(
        name="support",
        model=leader,
        members=[member, member],
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    assert guard.ungoverned_tools(team) == []
    output = team.run("x")
    assert output.status == RunStatus.completed
    assert tools.ran == [("send_email", HELLO)]
    # Agno has now replaced each guardrail in pre_hooks with its bound check method; it still counts.
    assert member.pre_hooks == [guard.guardrail.check]
    assert guard.ungoverned_tools(team) == []


async def test_a_governed_team_runs_again_after_agno_normalised_its_hooks() -> None:
    tools = Tools()
    guard = ACGSGuard()
    member = _member("mailer", tools, guard, model(call("send_email", HELLO, "e1"), "Sent.", "Again."))
    delegate = call("delegate_task_to_member", {"member_id": "mailer", "task": "Email them"}, "d1")
    team = Team(
        name="support",
        model=model(delegate, "Done.", "Done again."),
        members=[member],
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    assert (await team.arun("x")).status == RunStatus.completed
    assert member.pre_hooks == [guard.guardrail.async_check]
    assert (await team.arun("again")).status == RunStatus.completed
    assert tools.ran == [("send_email", HELLO)]


def test_another_guardrails_check_does_not_count_as_the_acgs_guardrail() -> None:
    class Other:
        def check(self) -> None: ...

    assert not _is_acgs_guardrail(Other().check)
    assert not _is_acgs_guardrail(ACGSGuard().guardrail.__repr__)


def test_team_tools_are_checked() -> None:
    tools = Tools()
    guard = ACGSGuard()
    team = Team(
        name="support",
        model=model(call("send_email", LEAK, "e1"), "Done."),
        members=[],
        tools=[tools.send_email],
        tool_hooks=[guard.tool_hook],
        pre_hooks=[guard.guardrail],
        telemetry=False,
    )
    output = team.run("x")
    assert tools.ran == []
    assert results(output)["e1"].startswith(BLOCKED)
