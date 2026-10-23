# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tools of a local MCP server (``MCPTools``) are Agno functions, so the tool hook checks their calls."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from agno.tools.mcp import MCPTools

from tests.fakes import BLOCKED, HELLO, LEAK, agent, call, model, results

SERVER = Path(__file__).with_name("mcp_server.py")


async def test_mcp_tool_calls_are_checked(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    command = shlex.join([sys.executable, str(SERVER), str(log)])
    scripted = model([call("send_email", LEAK, "m1"), call("send_email", HELLO, "m2")], "Done.")
    async with MCPTools(command=command, transport="stdio") as mcp:
        output = await agent(scripted, [mcp]).arun("Email them")
    found = results(output)
    assert found["m1"].startswith(BLOCKED)
    assert "sent to friend@example.com" in found["m2"]
    assert log.read_text(encoding="utf-8").splitlines() == ["send_email friend@example.com"]
