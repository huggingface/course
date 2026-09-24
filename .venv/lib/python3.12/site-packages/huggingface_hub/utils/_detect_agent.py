# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Detect whether the process is being invoked by an AI coding agent.

Detection is based on environment variables that AI agents set in their shell
sessions.  ``AI_AGENT`` and ``AGENT`` are treated as a universal standard (any
tool can set it); the remaining checks are tool-specific and ordered by
prevalence.

Inspired by ``@vercel/detect-agent`` (https://github.com/vercel/vercel/tree/main/packages/detect-agent).
"""

import os
from typing import Optional


# Standard env vars — value is used as the agent name directly.
_STANDARD_AGENT_VARS: tuple[str, ...] = ("AI_AGENT", "AGENT")


# NOTE: ``cowork`` must appear before ``claude-code`` so the more specific
# signal takes priority when both ``CLAUDE_CODE`` and ``CLAUDE_CODE_IS_COWORK``
# are set.
_TOOL_AGENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ANTIGRAVITY_AGENT",), "antigravity"),
    (("AUGMENT_AGENT",), "augment-cli"),
    (("CLINE_ACTIVE",), "cline"),
    (("CLAUDE_CODE_IS_COWORK",), "cowork"),
    (("CLAUDECODE", "CLAUDE_CODE"), "claude-code"),
    (("CODEX_SANDBOX", "CODEX_CI", "CODEX_THREAD_ID"), "codex"),
    (("CURSOR_TRACE_ID",), "cursor"),
    (("CURSOR_AGENT",), "cursor-cli"),
    (("GEMINI_CLI",), "gemini"),
    (("COPILOT_MODEL", "COPILOT_ALLOW_ALL", "COPILOT_GITHUB_TOKEN"), "github-copilot"),
    (("GOOSE_TERMINAL",), "goose"),
    (("OPENCLAW_SHELL",), "openclaw"),
    (("OPENCODE_CLIENT",), "opencode"),
    (("REPL_ID",), "replit"),
    (("ROO_ACTIVE",), "roo-code"),
    (("TRAE_AI_SHELL_ID",), "trae"),
)

_KNOWN_AGENTS = {"devin"} | {agent for _, agent in _TOOL_AGENTS}


def detect_agent() -> Optional[str]:
    """Return the name of the detected AI agent or ``None``.

    Checks environment variables in priority order and returns on the first
    match.  When ``AI_AGENT`` or ``AGENT`` is set, the value is checked against
    known agent names, unrecognized values are returned as ``"unknown"``.
    """
    for var in _STANDARD_AGENT_VARS:
        name = os.environ.get(var, "").strip().lower()
        if name:
            return name if name in _KNOWN_AGENTS else "unknown"

    for env_vars, agent_name in _TOOL_AGENTS:
        if any(os.environ.get(var) for var in env_vars):
            return agent_name

    return None


def is_agent() -> bool:
    """Return ``True`` if the process is being invoked by an AI coding agent."""
    return detect_agent() is not None
