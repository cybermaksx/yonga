"""Terminal presentation layer.

The UI renders; it never decides. Everything here consumes typed domain objects
and the :class:`~yonga.application.progress.ProgressEvent` stream, and nothing
here imports an adapter, OpenVINO or the Hugging Face Hub — which is what keeps
a future GUI able to reuse the same application layer.

Modules:

* :mod:`yonga.ui.console` — colour/JSON policy, error rendering, confirmations;
* :mod:`yonga.ui.progress` — a Rich ``ProgressSink`` implementation;
* :mod:`yonga.ui.tables` — pure render functions for every report;
* :mod:`yonga.ui.chat_tui` — the chat panels, prompt and command parser.
"""

from __future__ import annotations

from yonga.ui.chat_tui import (
    CHAT_COMMANDS,
    ChatCommand,
    ChatCommandKind,
    ChatMetrics,
    ChatSessionInfo,
    ChatView,
    help_text,
    parse_command,
    render_answer,
    render_metrics,
    session_facts,
    startup_panel,
)
from yonga.ui.console import (
    YongaConsole,
    build_console,
    diagnostic_lines,
    human_bytes,
    human_duration,
    json_ready,
    render_diagnostic,
    status_icon,
    status_style,
)
from yonga.ui.progress import RichProgressSink
from yonga.ui.tables import (
    SourceOption,
    artifact_list,
    benchmark_summary,
    doctor_report,
    inspect_report,
    preparation_plan,
    probe_report,
    registry_explanation,
)

__all__ = [
    "CHAT_COMMANDS",
    "ChatCommand",
    "ChatCommandKind",
    "ChatMetrics",
    "ChatSessionInfo",
    "ChatView",
    "RichProgressSink",
    "SourceOption",
    "YongaConsole",
    "artifact_list",
    "benchmark_summary",
    "build_console",
    "diagnostic_lines",
    "doctor_report",
    "help_text",
    "human_bytes",
    "human_duration",
    "inspect_report",
    "json_ready",
    "parse_command",
    "preparation_plan",
    "probe_report",
    "registry_explanation",
    "render_answer",
    "render_diagnostic",
    "render_metrics",
    "session_facts",
    "startup_panel",
    "status_icon",
    "status_style",
]
