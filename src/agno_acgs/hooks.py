# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance for Agno agents and teams: a tool hook that checks each tool call before the tool runs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from acgs_lite import Constitution
from agno.exceptions import CheckTrigger, InputCheckError
from agno.guardrails.base import BaseGuardrail
from agno.tools import Toolkit
from agno.tools.function import Function

from agno_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

logger = logging.getLogger(__name__)

_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."
_NO_REVIEWER = "No reviewer is configured for this agent, so it was denied."
_NOT_CONFIRMED = "The tool does not require confirmation, so no person approved this call and it was denied."
_CONFIRMED = "A person confirmed this call through Agno's confirmation flow."


class UngovernedToolsError(InputCheckError):
    """Raised by ``guard.guardrail`` before a run that uses tools ACGS cannot check.

    It is an ``agno.exceptions.InputCheckError``, so Agno stops the run before the model is called and
    returns a run output with ``status=RunStatus.error`` and this message as its content.
    """


def _is_acgs_hook(hook: object) -> bool:
    return isinstance(getattr(hook, "__self__", None), ACGSGuard) and getattr(hook, "__func__", None) is (
        ACGSGuard.tool_hook
    )


def _is_acgs_guardrail(hook: object) -> bool:
    # Agno replaces a guardrail in pre_hooks with its bound check method on the first run.
    return isinstance(hook, ACGSGuardrail) or (
        isinstance(getattr(hook, "__self__", None), ACGSGuardrail)
        and getattr(hook, "__func__", None) in (ACGSGuardrail.check, ACGSGuardrail.async_check)
    )


def _describe_tool(tool: object) -> list[str]:
    if isinstance(tool, dict):
        return [f"provider-run tool {tool.get('type', tool)!r} (the model provider runs it, not Agno)"]
    if isinstance(tool, Toolkit):
        names = list(tool.external_execution_required_tools)
        functions = [*tool.functions.values(), *tool.async_functions.values()]
        names += [function.name for function in functions if function.external_execution]
        return [
            f"externally executed tool '{name}' from toolkit '{tool.name}' (the caller runs it, not Agno)"
            for name in dict.fromkeys(names)
        ]
    if isinstance(tool, Function) and tool.external_execution:
        return [f"externally executed tool '{tool.name}' (the caller runs it, not Agno)"]
    return []


def _requires_confirmation(tools: object, name: str) -> bool:
    """Whether the tool Agno runs under ``name`` pauses for a person's confirmation before each call.

    Agno keeps the first tool registered under a name, so the first match decides.
    """
    if not isinstance(tools, (list, tuple)):
        return False
    for tool in tools:
        if isinstance(tool, Toolkit):
            function = tool.async_functions.get(name) or tool.functions.get(name)
            if function is not None:
                return bool(function.requires_confirmation) or name in tool.requires_confirmation_tools
        elif isinstance(tool, Function):
            if tool.name == name:
                return bool(tool.requires_confirmation)
        elif callable(tool) and getattr(tool, "__name__", None) == name:
            return getattr(tool, "_agno_approval_type", None) == "required"
    return False


def _with(decision: GovernanceDecision, verdict: Verdict, reason: str) -> GovernanceDecision:
    return GovernanceDecision(
        verdict=verdict,
        tool_name=decision.tool_name,
        reason=reason,
        rule_ids=decision.rule_ids,
        warning_rule_ids=decision.warning_rule_ids,
        error=decision.error,
    )


class ACGSGuardrail(BaseGuardrail):
    """Agno pre-hook that stops a run before the model is called if it uses tools ACGS cannot check.

    Use ``guard.guardrail``; do not create it directly. It is a guardrail (not a plain pre-hook) so that
    Agno runs it, and lets its error stop the run, even when hooks run in the background.
    """

    def __init__(self, guard: ACGSGuard) -> None:
        """Create the guardrail.

        Args:
            guard: The guard whose tool hook and options apply.
        """
        self.guard = guard

    def check(
        self,
        run_input: Any = None,  # noqa: ARG002  # Agno passes hook arguments by name
        agent: Any = None,
        team: Any = None,
        run_context: Any = None,
    ) -> None:
        """Check the agent or team that is about to run.

        Args:
            run_input: The run input (unused).
            agent: The running ``agno.agent.Agent``, if an agent runs.
            team: The running ``agno.team.Team``, if a team runs.
            run_context: The run context; its client tools are checked too.

        Raises:
            UngovernedToolsError: If the run uses tools ACGS cannot check and ``allow_ungoverned`` is off.
        """
        component = agent if agent is not None else team
        if component is None:
            self.guard.report_ungoverned(["an unknown component (Agno passed neither an agent nor a team)"])
            return
        found = self.guard.ungoverned_tools(component)
        client_tools = getattr(run_context, "client_tools", None) or []
        found += [description for tool in client_tools for description in _describe_tool(tool)]
        if found:
            self.guard.report_ungoverned(found)

    async def async_check(
        self, run_input: Any = None, agent: Any = None, team: Any = None, run_context: Any = None
    ) -> None:
        """Async variant of ``check``, used by ``arun``.

        Args:
            run_input: The run input (unused).
            agent: The running ``agno.agent.Agent``, if an agent runs.
            team: The running ``agno.team.Team``, if a team runs.
            run_context: The run context; its client tools are checked too.
        """
        self.check(run_input, agent=agent, team=team, run_context=run_context)


class ACGSGuard:
    """Check Agno tool calls against an ACGS constitution before the tool runs.

    Give an agent or team both parts::

        Agent(..., tool_hooks=[guard.tool_hook], pre_hooks=[guard.guardrail])

    ``guard.tool_hook`` runs around every tool call Agno executes (function tools, toolkits, MCP tools,
    the agent's built-in memory and knowledge tools, and confirmed or user-input tools when the run
    continues). It:

    - runs allowed calls unchanged;
    - does not run blocked calls: ``"Tool call '<name>' was not executed. <reason>"`` becomes the call's
      result, which the model reads;
    - denies calls that need review, unless ``on_review="confirmed"`` and the tool is one Agno pauses for a
      person's confirmation (``requires_confirmation``, ``@approval``), in which case the person's
      confirmation counts as the review.

    A blocking rule always wins over review rules, and any error while evaluating a call denies it.
    The hook is a plain function: Agno skips ``async def`` tool hooks in ``run()`` with only a warning,
    and runs plain ones in both ``run()`` and ``arun()``.

    ``guard.guardrail`` stops a run before the model is called when the agent or team uses tools the hook
    never sees (provider-run tools, externally executed tools, team members without the guard) or does not
    have ``guard.tool_hook`` as its last tool hook.

    Example:
        >>> from agno_acgs import ACGSGuard
        >>> guard = ACGSGuard()
        >>> guard.evaluate("send_email", {"to": "a@example.com", "body": "The password is hunter2"}).verdict
        <Verdict.DENY: 'deny'>
    """

    def __init__(
        self,
        *,
        constitution: Constitution | None = None,
        agent_id: str = "agno",
        on_review: Literal["deny", "confirmed"] = "deny",
        allow_ungoverned: bool = False,
        tool_call_formatter: ToolCallFormatter | None = None,
    ) -> None:
        """Create the guard.

        Args:
            constitution: The ACGS constitution to enforce. Defaults to ``acgs_lite.Constitution.default()``.
            agent_id: Agent identifier recorded by ACGS for each validation.
            on_review: What to do when only review rules match. ``"deny"`` denies the call.
                ``"confirmed"`` runs it only if the tool requires confirmation in Agno, so a person confirmed
                this call before it ran; other calls are denied.
            allow_ungoverned: Log a warning instead of stopping the run when it uses tools ACGS cannot check.
            tool_call_formatter: Optional function ``(tool_name, args) -> str`` that replaces ``format_tool_call``.

        Raises:
            ValueError: If ``on_review`` is invalid.
        """
        if on_review not in ("deny", "confirmed"):
            msg = f"on_review must be 'deny' or 'confirmed', got {on_review!r}"
            raise ValueError(msg)
        self._governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)
        self.constitution: Constitution = self._governor.constitution
        self.agent_id = agent_id
        self.on_review = on_review
        self.allow_ungoverned = allow_ungoverned
        self.guardrail = ACGSGuardrail(self)
        """The guardrail to pass in ``pre_hooks=[...]``."""

    def __repr__(self) -> str:
        """Show the options, not the constitution."""
        return (
            f"ACGSGuard(agent_id={self.agent_id!r}, on_review={self.on_review!r}, "
            f"allow_ungoverned={self.allow_ungoverned!r})"
        )

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        """Evaluate a tool call without running an agent. Never raises.

        Args:
            tool_name: Name of the tool.
            arguments: The tool arguments.

        Returns:
            The governance decision.
        """
        return self._governor.evaluate(tool_name, arguments)

    def decide(self, tool_name: str, arguments: Any, owner: Any = None) -> GovernanceDecision:
        """Decide one call the way ``tool_hook`` does, including review handling. Never raises.

        Args:
            tool_name: Name of the tool.
            arguments: The tool arguments.
            owner: The agent or team that runs the tool; its tools decide whether a person confirmed the call.

        Returns:
            The decision; ``ALLOW`` means the tool runs.
        """
        try:
            decision = self.evaluate(tool_name, arguments)
            if decision.verdict is not Verdict.REVIEW:
                return decision
            if self.on_review == "deny":
                return _with(decision, Verdict.DENY, f"{decision.reason} {_NO_REVIEWER}")
            if _requires_confirmation(getattr(owner, "tools", None), tool_name):
                return _with(decision, Verdict.ALLOW, f"{decision.reason} {_CONFIRMED}")
            return _with(decision, Verdict.DENY, f"{decision.reason} {_NOT_CONFIRMED}")
        except Exception as error:
            logger.warning("ACGS could not decide tool call '%s'; denying it (fail-closed).", tool_name, exc_info=True)
            return GovernanceDecision(
                verdict=Verdict.DENY,
                tool_name=tool_name,
                reason=_EVALUATION_FAILED,
                error=f"{type(error).__name__}: {error}",
            )

    def tool_hook(
        self,
        function_name: str,
        function_call: Callable[..., Any],
        arguments: dict[str, Any],
        agent: Any = None,
        team: Any = None,
    ) -> Any:
        """Agno tool hook: run the tool only if ACGS allows the call. Pass it as ``tool_hooks=[guard.tool_hook]``.

        It never raises for a denied call: the reason becomes the call's result, which the model reads.

        Args:
            function_name: Name of the tool (filled in by Agno).
            function_call: Runs the rest of the hook chain and the tool (filled in by Agno).
            arguments: The tool arguments sent by the model (filled in by Agno).
            agent: The agent that runs the tool (filled in by Agno).
            team: The team that runs the tool (filled in by Agno).

        Returns:
            The tool's result, or a message explaining why it was not run.
        """
        decision = self.decide(function_name, arguments, agent if agent is not None else team)
        if decision.verdict is Verdict.ALLOW:
            return function_call(**arguments)
        return f"Tool call '{function_name}' was not executed. {decision.reason}"

    def ungoverned_tools(self, component: Any) -> list[str]:
        """Describe what an agent or team can run that ``guard.tool_hook`` never sees.

        Reported: ``guard.tool_hook`` missing from ``tool_hooks`` or not the last tool hook (a later hook
        can change the call after the check); provider-run tools given as dicts; tools marked for external
        execution; tools given as a factory (``tools=callable``), which Agno resolves after the guardrail
        ran; and, for teams, members without ``guard.guardrail`` and ``guard.tool_hook``.

        Args:
            component: An ``agno.agent.Agent`` or ``agno.team.Team``.

        Returns:
            One description per ungoverned item, empty if everything is governed.
        """
        return self._ungoverned(component, set())

    def _ungoverned(self, component: Any, seen: set[int]) -> list[str]:
        if id(component) in seen:
            return []
        seen.add(id(component))
        label = f"{type(component).__name__.lower()} '{getattr(component, 'name', None) or 'unnamed'}'"
        found: list[str] = []
        hooks = list(getattr(component, "tool_hooks", None) or [])
        if not any(_is_acgs_hook(hook) for hook in hooks):
            found.append(f"every tool of {label} (guard.tool_hook is not in its tool_hooks)")
        elif not _is_acgs_hook(hooks[-1]):
            found.append(f"every tool of {label} (a tool hook after guard.tool_hook can change the call)")
        tools: Any = getattr(component, "tools", None)
        if isinstance(tools, (list, tuple)):
            found += [description for tool in tools for description in _describe_tool(tool)]
        elif callable(tools):
            found.append(f"tools of {label} from a factory (resolved after the guardrail ran)")
        members = getattr(component, "members", None)
        if members is not None:
            found += self._members(members, label, seen)
        return found

    def _members(self, members: Any, label: str, seen: set[int]) -> list[str]:
        if callable(members):
            return [f"members of {label} from a factory (resolved after the guardrail ran)"]
        found: list[str] = []
        for member in members:
            member_label = f"{type(member).__name__.lower()} '{getattr(member, 'name', None) or 'unnamed'}'"
            if not any(_is_acgs_guardrail(hook) for hook in getattr(member, "pre_hooks", None) or []):
                found.append(f"team member {member_label} (guard.guardrail is not in its pre_hooks)")
            found += self._ungoverned(member, seen)
        return found

    def check(self, component: Any) -> None:
        """Report what an agent or team can run without ACGS checks (see ``ungoverned_tools``).

        Call it after building the agent or team to fail early; ``guard.guardrail`` does the same before
        every run.

        Args:
            component: An ``agno.agent.Agent`` or ``agno.team.Team``.
        """
        found = self.ungoverned_tools(component)
        if found:
            self.report_ungoverned(found)

    def report_ungoverned(self, descriptions: Sequence[str]) -> None:
        """Raise ``UngovernedToolsError``, or log a warning when ``allow_ungoverned`` is set.

        Args:
            descriptions: What ACGS cannot check.

        Raises:
            UngovernedToolsError: Unless ``allow_ungoverned`` is set.
        """
        listed = "; ".join(descriptions)
        if self.allow_ungoverned:
            logger.warning("ACGS does not check calls of: %s (allow_ungoverned is set).", listed)
            return
        msg = (
            f"ACGS cannot check calls of: {listed}. Remove them, or pass allow_ungoverned=True to run them "
            "without ACGS checks."
        )
        raise UngovernedToolsError(msg, check_trigger=CheckTrigger.VALIDATION_FAILED)
