# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import os

# Agno sends anonymous usage telemetry unless this is off; the tests run without network access.
os.environ["AGNO_TELEMETRY"] = "false"
