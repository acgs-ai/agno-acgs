# agno-acgs

[![CI](https://github.com/acgs-ai/agno-acgs/actions/workflows/ci.yml/badge.svg)](https://github.com/acgs-ai/agno-acgs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agno-acgs.svg)](https://pypi.org/project/agno-acgs/)
[![Python](https://img.shields.io/pypi/pyversions/agno-acgs.svg)](https://pypi.org/project/agno-acgs/)

`agno-acgs` checks [Agno](https://github.com/agno-agi/agno) tool calls against [ACGS](https://pypi.org/project/acgs-lite/)
constitution rules before the tool runs. It uses two documented Agno hooks:

- `guard.tool_hook`, an Agno tool hook (`tool_hooks=[...]`) that wraps every tool call Agno executes;
- `guard.guardrail`, an Agno guardrail (`pre_hooks=[...]`) that stops a run before the model is called if the
  agent or team can run tools the tool hook never sees.

What happens to a call:

- Calls that match no blocking rule run unchanged.
- Calls that match a blocking rule do not run. The model gets the reason as the tool result and can continue.
- Calls that only match review rules are denied by default. With `on_review="confirmed"`, they run only if the
  tool is one Agno pauses for a person's confirmation (`requires_confirmation=True`, `@approval`) and the person
  confirmed this call.
- A blocking rule always wins. A confirmation never runs a blocked call.
- Any error while evaluating a call denies it (fail-closed).
- A run of an agent or team with tools the hook cannot check (provider-run tools, externally executed tools,
  members without the guard, tool or member factories) stops with `UngovernedToolsError` before the model is
  called.

The package is a thin adapter over the public `acgs-lite` API.

> [!NOTE]
> This is a community package. It is not part of Agno and is not maintained, reviewed or endorsed by the Agno
> maintainers.

## Installation

```bash
pip install agno-acgs
```

Requires Python 3.10 or later, `agno>=3.1.0,<4` and `acgs-lite>=3.0.1,<4`. Install `agno[mcp]` to use `MCPTools`.

## Usage

```python
from agno.agent import Agent

from agno_acgs import ACGSGuard


def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


guard = ACGSGuard()
# model is any Agno model, for example OpenAIResponses(id=...) from agno.models.openai (pip install openai).
agent = Agent(
    model=model,
    tools=[send_email],
    tool_hooks=[guard.tool_hook],  # keep it last
    pre_hooks=[guard.guardrail],
)
guard.check(agent)  # optional: fail now instead of on the first run
```

Both parts are needed. The guardrail refuses to run an agent whose last tool hook is not `guard.tool_hook`.

## Quickstart

`examples/quickstart.py` runs offline. A small scripted model stands in for a real one:

```python
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
```

```text
confirm: issue_refund {'order_id': 'A-17', 'amount': 20.0}
email-1: Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses).
weather-1: Sunny in Paris
refund-1: refunded 20.0 for A-17
```

## How a call is evaluated

1. The tool name and the call's arguments are rendered as text by `format_tool_call`:

   ```python
   from agno_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

   The arguments are the ones Agno passes to the tool hook: what the model sent plus values a person typed in
   for `requires_user_input` tools.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | The tool runs |
   | Only `require_human_review` / `escalate_to_senior` rules matched | Review (see below) |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | Denied |
   | A rule ID not found in the constitution, or an invalid result without a rule | Denied |
   | Any exception while reading or evaluating the call | Denied (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.

A denied call is not executed. The tool hook returns `"Tool call '<name>' was not executed. <reason>"` as the
call's result, so the model reads the reason. The reason names the matched rules but never repeats argument
values. `ACGSGuard.evaluate(tool_name, args)` returns the `GovernanceDecision` without running anything, and
`ACGSGuard.decide(tool_name, args, owner)` applies the review handling as well.

The tool hook is a plain function on purpose: Agno's `run()` skips `async def` tool hooks with only a warning,
so an async guard would let every call through in sync runs. A plain hook runs in `run()`, `arun()` and both
streaming modes, and awaits async tools in `arun()`.

## Review

Agno has no way for a tool hook to ask a person and wait. The guard therefore offers two options:

- `on_review="deny"` (default): calls that need review are denied with "No reviewer is configured for this agent,
  so it was denied."
- `on_review="confirmed"`: Agno's own confirmation flow counts as the review. Mark the tools that may need
  review with `@tool(requires_confirmation=True)`, `Toolkit(requires_confirmation_tools=[...])` or `@approval`.
  Agno pauses the run before such a call; when the caller sets `requirement.confirmed = True` and continues the
  run, the call goes through the tool hook, which runs it. Calls of tools without that flag are still denied, and
  a rejected confirmation never reaches the hook.

```python
from agno_acgs import ACGSGuard

guard = ACGSGuard(constitution=constitution, on_review="confirmed")
```

Things to know:

- Whoever sets `confirmed = True` is the reviewer. The guard cannot tell who that was. Put it behind your own
  authentication.
- The confirmation applies to the tool, not to the rule: the person sees the tool name and arguments
  (`requirement.tool_args`), not which ACGS rule asked for review.
- The flag is looked up by tool name in the agent's (or team's) `tools`; the first tool with that name decides,
  as in Agno.

## What is covered

| Tool | Checked | How |
|---|---|---|
| Functions, `@tool`, `Function`, toolkits (`Toolkit`, `MCPTools`, Agno's built-in toolkits) | Yes | `tool_hook` |
| Agno's own tools (memory, knowledge, chat history, session state) | Yes | `tool_hook` |
| Tools with their own `tool_hooks` / `pre_hook` | Yes | Agent-level `tool_hooks` replace a tool's own hooks |
| Confirmed tools and tools with user input, after `continue_run` | Yes | `tool_hook`, with the final arguments |
| Team tools and member delegation | Yes | The team needs both parts; each member needs both parts as well |
| Provider-run tools given as dicts (for example `{"type": "web_search_preview"}`) | No | The provider runs them; refused |
| `external_execution=True` tools and `Toolkit(external_execution_required_tools=...)` | No | The caller runs them; refused |
| Client tools sent with a run (`RunContext.client_tools`) | Function tools yes; external or provider-run ones refused | `tool_hook` / guardrail |
| `tools=` or `members=` given as a factory | No | Resolved after the guardrail; refused |

Refused means `UngovernedToolsError` (an `InputCheckError`) before the model is called: the run ends with
`status=RunStatus.error` and the message as its content. Pass `allow_ungoverned=True` to run such tools with a
logged warning instead; their calls are not checked. `ACGSGuard.ungoverned_tools(agent_or_team)` lists what
would be refused.

```python
from agno.agent import Agent
from agno.team import Team

from agno_acgs import ACGSGuard

guard = ACGSGuard()
mailer = Agent(name="mailer", model=model, tools=[], tool_hooks=[guard.tool_hook], pre_hooks=[guard.guardrail])
team = Team(name="support", model=model, members=[mailer], tool_hooks=[guard.tool_hook], pre_hooks=[guard.guardrail])
assert guard.ungoverned_tools(team) == []
```

## Hook order

Agno runs tool hooks in list order, the first one outermost, and runs the tool with the arguments dict the
hooks share. A hook after the guard could change that dict after the check, so the guardrail refuses an agent
whose last tool hook is not `guard.tool_hook`. Hooks before the guard run first; any change they make is checked.

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from agno_acgs import ACGSGuard

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
guard = ACGSGuard(constitution=constitution, agent_id="payments-app")
```

`Constitution.from_yaml("rules.yaml")` works as well.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"agno"` | Agent identifier recorded by ACGS |
| `on_review` | `"deny"` | `"deny"`, or `"confirmed"` (Agno's confirmation flow counts as the review) |
| `allow_ungoverned` | `False` | Run tools the hook cannot check, with a logged warning, instead of stopping the run |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |

## Limitations

- Only tools listed as checked in [What is covered](#what-is-covered). User input, model output and tool results
  are not checked.
- In `arun()`, Agno runs the calls of one model turn concurrently. A denied call does not stop the others in the
  same turn.
- `continue_run()` does not run pre-hooks, so the guardrail does not run again then. Tool calls are still checked
  by the tool hook. If you change an agent's tools between `run()` and `continue_run()`, call `guard.check(agent)`.
- Tested with `run()`, `arun()`, both streaming modes, toolkits, a local stdio MCP server, Agno's session-state
  tool, confirmation and user-input pauses, client tools and teams, all with a scripted model, against agno 3.1.0
  and the latest release. Real providers, AgentOS, workflows and remote agents are not tested.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied
  under the default constitution whenever it is called.
- agno 3.1.0 is the minimum: with agno 3.0.x, allowed async tool calls did not run in `arun()` in our tests.

## Development

```bash
uv sync --group dev        # or use hatch: hatch run prepare
hatch run format           # ruff format + ruff check --fix
hatch run lint             # ruff check + ruff format --check
hatch run typecheck        # mypy --strict
hatch run cov              # pytest with a 100% branch-coverage gate, network disabled
hatch run min:cov          # the same tests against agno 3.1.0 and acgs-lite 3.0.1
pre-commit install         # format, lint, type check and tests on commit; commitizen on commit messages
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); `cz bump` updates the version
and the changelog.

## License

Apache-2.0. See [LICENSE](LICENSE).
