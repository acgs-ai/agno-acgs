# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Calls that need review are denied, unless a person confirmed them through Agno's confirmation flow."""

from __future__ import annotations

from typing import Any

from agno.run.base import RunStatus
from agno.tools import Toolkit, tool
from agno.tools.function import Function

from agno_acgs import ACGSGuard, Verdict
from agno_acgs.hooks import _requires_confirmation
from tests.fakes import LEAK, REFUND, REFUND_REVIEW, Tools, agent, call, model, results, review_rules

NOT_RUN = "Tool call 'issue_refund' was not executed. "


def _confirmed_refund(ran: list[Any]) -> Function:
    @tool(requires_confirmation=True)
    def issue_refund(order_id: str, amount: float) -> str:
        """Refund an order."""
        ran.append((order_id, amount))
        return f"refunded {amount} for {order_id}"

    return issue_refund


def test_review_is_denied_by_default() -> None:
    tools = Tools()
    governed = agent(
        model(call("issue_refund", REFUND, "r1"), "Done."), [tools.issue_refund], ACGSGuard(constitution=review_rules())
    )
    output = governed.run("Refund A-17")
    assert tools.ran == []
    assert (
        results(output)["r1"] == f"{NOT_RUN}{REFUND_REVIEW} No reviewer is configured for this agent, so it was denied."
    )


def test_confirmed_mode_runs_a_call_a_person_confirmed() -> None:
    ran: list[Any] = []
    guard = ACGSGuard(constitution=review_rules(), on_review="confirmed")
    governed = agent(model(call("issue_refund", REFUND, "r1"), "Done."), [_confirmed_refund(ran)], guard)
    paused = governed.run("Refund A-17")
    assert paused.status == RunStatus.paused
    assert ran == []
    for requirement in paused.tools or []:
        requirement.confirmed = True
    output = governed.continue_run(run_response=paused)
    assert ran == [("A-17", 20.0)]
    assert results(output)["r1"] == "refunded 20.0 for A-17"


async def test_confirmed_mode_async() -> None:
    ran: list[Any] = []
    guard = ACGSGuard(constitution=review_rules(), on_review="confirmed")
    governed = agent(model(call("issue_refund", REFUND, "r1"), "Done."), [_confirmed_refund(ran)], guard)
    paused = await governed.arun("Refund A-17")
    for requirement in paused.tools or []:
        requirement.confirmed = True
    await governed.acontinue_run(run_response=paused)
    assert ran == [("A-17", 20.0)]


def test_a_rejected_call_is_not_run() -> None:
    ran: list[Any] = []
    guard = ACGSGuard(constitution=review_rules(), on_review="confirmed")
    governed = agent(model(call("issue_refund", REFUND, "r1"), "Done."), [_confirmed_refund(ran)], guard)
    paused = governed.run("Refund A-17")
    for requirement in paused.tools or []:
        requirement.confirmed = False
    governed.continue_run(run_response=paused)
    assert ran == []


def test_confirmed_mode_denies_review_for_tools_without_confirmation() -> None:
    tools = Tools()
    guard = ACGSGuard(constitution=review_rules(), on_review="confirmed")
    output = agent(model(call("issue_refund", REFUND, "r1"), "Done."), [tools.issue_refund], guard).run("x")
    assert tools.ran == []
    assert results(output)["r1"] == (
        f"{NOT_RUN}{REFUND_REVIEW} The tool does not require confirmation, so no person approved this call and it "
        "was denied."
    )


def test_a_confirmation_never_overrides_a_blocking_rule() -> None:
    ran: list[Any] = []

    @tool(requires_confirmation=True)
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        ran.append(to)
        return "sent"

    guard = ACGSGuard(on_review="confirmed")
    governed = agent(model(call("send_email", LEAK, "e1"), "Done."), [send_email], guard)
    paused = governed.run("x")
    for requirement in paused.tools or []:
        requirement.confirmed = True
    output = governed.continue_run(run_response=paused)
    assert ran == []
    assert results(output)["e1"].startswith("Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006")


def test_values_a_person_types_in_are_checked_too() -> None:
    ran: list[str] = []

    @tool(requires_user_input=True, user_input_fields=["body"])
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        ran.append(body)
        return "sent"

    governed = agent(model(call("send_email", {"to": "a@example.com"}, "e1"), "Done."), [send_email])
    paused = governed.run("x")
    for requirement in paused.tools or []:
        for field in requirement.user_input_schema or []:
            field.value = "The password is hunter2"
    output = governed.continue_run(run_response=paused)
    assert ran == []
    assert results(output)["e1"].startswith("Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006")


def test_which_tools_count_as_confirmed() -> None:
    tools = Tools()
    confirmed = _confirmed_refund([])
    by_list = Toolkit(name="payments", tools=[tools.issue_refund], requires_confirmation_tools=["issue_refund"])
    plain_kit = Toolkit(name="payments", tools=[tools.issue_refund])
    listed_only = Toolkit(name="other", tools=[tools.send_email], requires_confirmation_tools=["issue_refund"])
    flagged = Toolkit(name="payments", tools=[confirmed])

    def approved(order_id: str) -> str:
        """Refund."""
        return order_id

    approved._agno_approval_type = "required"  # type: ignore[attr-defined]  # what @approval stamps
    approved.__name__ = "issue_refund"

    assert _requires_confirmation([confirmed], "issue_refund")
    assert _requires_confirmation([by_list], "issue_refund")
    assert _requires_confirmation([flagged], "issue_refund")
    assert _requires_confirmation([approved], "issue_refund")
    assert not _requires_confirmation([plain_kit, confirmed], "issue_refund")  # the first tool with the name wins
    assert not _requires_confirmation([tools.issue_refund], "issue_refund")
    assert not _requires_confirmation([Function(name="other"), {"type": "web_search"}], "issue_refund")
    # Listing a name the toolkit does not have does not make another tool with that name confirmed.
    assert not _requires_confirmation([listed_only, tools.issue_refund], "issue_refund")
    assert not _requires_confirmation(None, "issue_refund")
    assert not _requires_confirmation(lambda: [confirmed], "issue_refund")


def test_decide_without_an_owner_denies_review() -> None:
    guard = ACGSGuard(constitution=review_rules(), on_review="confirmed")
    assert guard.decide("issue_refund", REFUND).verdict is Verdict.DENY
    assert guard.decide("issue_refund", REFUND, owner=None).verdict is Verdict.DENY
