# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import agno_acgs


def test_public_api() -> None:
    assert sorted(agno_acgs.__all__) == sorted(
        [
            "ACGSGuard",
            "ACGSGuardrail",
            "GovernanceDecision",
            "ToolCallFormatter",
            "UngovernedToolsError",
            "Verdict",
            "format_tool_call",
        ]
    )
    for name in agno_acgs.__all__:
        assert getattr(agno_acgs, name) is not None
