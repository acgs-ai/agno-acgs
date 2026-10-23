# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The README only shows code that runs, and the quickstart output it shows is the real output."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from agno.team import Team

from agno_acgs import ACGSGuard
from tests.fakes import model

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _blocks(language: str) -> list[str]:
    return [textwrap.dedent(block) for block in re.findall(rf"```{language}\n(.*?)```", README, re.DOTALL)]


def test_quickstart_output_matches_readme() -> None:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error", str(QUICKSTART)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        timeout=300,
        env={**os.environ, "AGNO_TELEMETRY": "false"},
    )
    [expected] = [block for block in _blocks("text") if block.startswith("confirm:")]
    assert completed.stdout == expected
    assert completed.stderr == ""


def test_readme_embeds_the_quickstart_verbatim() -> None:
    assert QUICKSTART.read_text(encoding="utf-8") in _blocks("python")


def test_readme_states_the_supported_versions() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("agno", "acgs-lite"):
        [requirement] = re.findall(rf'"({package}>=[^"]+)"', pyproject)
        assert f"`{requirement}`" in README
        [floor] = re.findall(rf'"{package}==([^"]+)"', pyproject)
        assert requirement.startswith(f"{package}>={floor},")
    assert README.count("agno 3.1.0") == 3


def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 6
    namespaces: list[dict[str, Any]] = []
    for block in blocks:
        namespace: dict[str, Any] = {"__name__": "readme", "model": model("Done."), "constitution": None}
        exec(compile(block, "<README>", "exec", dont_inherit=True), namespace)  # noqa: S102
        namespaces.append(namespace)
    assert "normalized: update config | dry run" in capsys.readouterr().out
    usage = namespaces[0]
    assert usage["agent"].tool_hooks == [usage["guard"].tool_hook]
    assert usage["agent"].pre_hooks == [usage["guard"].guardrail]
    assert namespaces[3]["guard"].on_review == "confirmed"
    assert isinstance(namespaces[4]["team"], Team)
    assert isinstance(namespaces[-1]["guard"], ACGSGuard)
    assert namespaces[-1]["guard"].agent_id == "payments-app"
