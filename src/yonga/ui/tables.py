"""Pure render functions turning typed domain objects into Rich renderables.

Every function here is a *pure* projection: it takes domain values, returns a
renderable, and prints nothing. That is what lets the tests assert on the render
model and what keeps presentation out of the application layer. None of these
functions decides anything — the layouts follow ``docs/CLI_SPEC.md`` sections 3,
4, 6, 7, 10, 11 and 12.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rich.box import ROUNDED, SIMPLE
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yonga.domain.artifacts import ArtifactSummary, DeviceProbeReport, ProbeStatus
from yonga.domain.benchmarks import BenchmarkResult
from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceStatus,
    EvidenceLevel,
    ResolutionExplanation,
)
from yonga.domain.diagnostics import HealthSection, Severity
from yonga.domain.models import ModelMetadata
from yonga.domain.plans import PreparationPlan
from yonga.ui.console import human_bytes, human_duration, status_icon, status_style

if TYPE_CHECKING:  # Registry types are consumed, never imported at runtime.
    from yonga.registry.resolver import RegistryExplanation

__all__ = [
    "SourceOption",
    "artifact_list",
    "benchmark_summary",
    "doctor_report",
    "inspect_report",
    "preparation_plan",
    "probe_report",
    "registry_explanation",
]

_DASH = "—"

#: Severity shown for each device compatibility status.
_DEVICE_STATUS_SEVERITY: dict[DeviceStatus, Severity] = {
    DeviceStatus.SUPPORTED: Severity.OK,
    DeviceStatus.EXPERIMENTAL: Severity.WARNING,
    DeviceStatus.UNSUPPORTED: Severity.ERROR,
    DeviceStatus.UNKNOWN: Severity.UNKNOWN,
}

#: Severity shown for each probe status.
_PROBE_STATUS_SEVERITY: dict[ProbeStatus, Severity] = {
    ProbeStatus.OK: Severity.OK,
    ProbeStatus.FAILED: Severity.ERROR,
    ProbeStatus.SKIPPED: Severity.INFO,
    ProbeStatus.NOT_ATTEMPTED: Severity.UNKNOWN,
}


class SourceOption(BaseModel):
    """One preparation route offered to the user by ``yonga inspect``.

    The UI does not work out which routes exist — the application layer builds
    these from the resolved source and the registry's derivative candidates, and
    this type is only the view model that carries them here.
    """

    model_config = ConfigDict(extra="ignore")

    label: str
    available: bool = False
    repo_id: str | None = None
    detail: str | None = None
    requires_confirmation: bool = False


def _grid(*, padding: tuple[int, int] = (0, 2)) -> Table:
    """Return an unboxed key/value grid with consistent padding."""
    table = Table.grid(padding=padding)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    return table


def _text(value: object, *, style: str = "") -> Text:
    """Render any value as text, using an em dash for ``None`` and blanks."""
    if value is None:
        return Text(_DASH, style="dim")
    rendered = str(value).strip()
    if not rendered:
        return Text(_DASH, style="dim")
    return Text(rendered, style=style)


def _bool_text(value: bool, *, true_label: str = "yes", false_label: str = "no") -> Text:
    """Render a boolean as a coloured yes/no."""
    return Text(true_label if value else false_label, style="green" if value else "dim")


def _short_revision(revision: str | None, *, length: int = 12) -> str | None:
    """Abbreviate a 40-character commit sha, leaving branch/tag names alone."""
    if revision is None:
        return None
    cleaned = revision.strip()
    if len(cleaned) == 40 and all(char in "0123456789abcdef" for char in cleaned.lower()):
        return cleaned[:length]
    return cleaned


def _timestamp(moment: datetime | None) -> str:
    """Render a timestamp in local minutes precision."""
    if moment is None:
        return _DASH
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def _status_text(severity: Severity, label: str) -> Text:
    """Render ``icon label`` in the colour that matches ``severity``."""
    text = Text()
    text.append(f"{status_icon(severity)} ", style=status_style(severity))
    text.append(label, style=status_style(severity))
    return text


# ----------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------
def doctor_report(sections: Sequence[HealthSection]) -> RenderableType:
    """Render the ``yonga doctor`` report (``docs/CLI_SPEC.md`` section 3).

    Each section becomes a heading followed by one ``icon label value`` row per
    check. Remediation text is printed, indented, under the check that needs it,
    so a warning is never shown without telling the user what to do about it.
    """
    if not sections:
        return Text("No health checks were produced.", style="dim")

    blocks: list[RenderableType] = []
    for section in sections:
        blocks.append(Text(section.title, style="bold"))
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True, width=1)
        table.add_column(no_wrap=True)
        table.add_column(overflow="fold")
        if not section.checks:
            table.add_row("", Text("(no checks)", style="dim"), "")
        for check in section.checks:
            icon = Text(status_icon(check.status), style=status_style(check.status))
            value = _text(check.value)
            if check.detail:
                value = Text.assemble(value, ("  ", ""), (check.detail, "dim"))
            table.add_row(icon, Text(check.label), value)
            if check.status in (Severity.WARNING, Severity.ERROR, Severity.UNKNOWN):
                for step in check.remediation:
                    table.add_row("", "", Text(f"→ {step}", style="yellow"))
        blocks.extend((table, Text("")))
    return Group(*blocks[:-1])


# ----------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------
def inspect_report(
    metadata: ModelMetadata,
    compatibility: CompatibilityDecision | None = None,
    source_options: Sequence[SourceOption] = (),
    *,
    show_files: bool = False,
) -> RenderableType:
    """Render the ``yonga inspect`` report (``docs/CLI_SPEC.md`` section 4)."""
    facts = _grid()
    facts.add_row(Text("Model"), _text(metadata.identity.repo_id))
    facts.add_row(
        Text("Revision"),
        _text(_short_revision(metadata.resolved_revision or metadata.identity.revision)),
    )
    facts.add_row(Text("License"), _text(metadata.license))
    facts.add_row(Text("Task"), _text(metadata.pipeline_tag or metadata.task))
    facts.add_row(Text("Architecture"), _text(", ".join(metadata.architectures) or None))
    facts.add_row(Text("Model type"), _text(metadata.model_type))
    facts.add_row(Text("Pipeline"), _text(metadata.pipeline_kind.value))
    formats = ", ".join(
        observation.format.value for observation in metadata.formats if observation.files
    )
    facts.add_row(Text("Formats"), _text(formats or None))
    facts.add_row(
        Text("Size"),
        _text(f"~{human_bytes(metadata.weight_bytes, precision=1)} total weights")
        if metadata.weight_bytes
        else _text(None),
    )
    if metadata.parameter_count:
        facts.add_row(Text("Parameters"), _text(f"{metadata.parameter_count:,}"))
    if metadata.gated or metadata.private:
        flags = ", ".join(
            label for label, on in (("gated", metadata.gated), ("private", metadata.private)) if on
        )
        facts.add_row(Text("Access"), Text(flags, style="yellow"))
    if metadata.requires_remote_code:
        facts.add_row(
            Text("Remote code"),
            Text("required (needs --trust-remote-code)", style="yellow"),
        )

    blocks: list[RenderableType] = [facts]

    if compatibility is not None:
        blocks.extend((Text(""), Text("Compatibility on this host", style="bold")))
        blocks.append(_compatibility_table(compatibility))

    if source_options:
        blocks.extend((Text(""), Text("Preparation", style="bold")))
        options = _grid()
        for option in source_options:
            value = _bool_text(option.available)
            parts: list[str] = []
            if option.repo_id:
                parts.append(option.repo_id)
            if option.detail:
                parts.append(option.detail)
            if option.requires_confirmation:
                parts.append("approval required")
            if parts:
                value = Text.assemble(value, ("  ", ""), (" — ".join(parts), "dim"))
            options.add_row(Text(option.label), value)
        blocks.append(options)

    if show_files and metadata.files:
        blocks.extend((Text(""), Text("Files", style="bold")))
        files = Table(box=SIMPLE, show_edge=False, pad_edge=False)
        files.add_column("Path", overflow="fold")
        files.add_column("Size", justify="right", no_wrap=True)
        files.add_column("LFS", justify="center", no_wrap=True)
        for repo_file in metadata.files:
            files.add_row(
                repo_file.path,
                human_bytes(repo_file.size) if repo_file.size is not None else _DASH,
                "yes" if repo_file.lfs else "",
            )
        blocks.append(files)

    return Group(*blocks)


def _compatibility_table(decision: CompatibilityDecision) -> RenderableType:
    """Render the per-device compatibility rows of a decision."""
    if not decision.per_device:
        return Text("  No registry claim for this host.", style="dim")
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True, width=1)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for entry in decision.per_device:
        severity = _DEVICE_STATUS_SEVERITY[entry.status]
        detail = entry.reason or ""
        label = entry.status.value
        if entry.evidence is not EvidenceLevel.UNKNOWN:
            label = f"{label} ({entry.evidence.value})"
        value = Text(label, style=status_style(severity))
        if detail:
            value = Text.assemble(value, ("  ", ""), (detail, "dim"))
        table.add_row(
            Text(status_icon(severity), style=status_style(severity)),
            Text(entry.device, style="bold"),
            value,
        )
    return table


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------
def preparation_plan(plan: PreparationPlan) -> RenderableType:
    """Render the plan shown *before* any download or conversion happens.

    This is the screen that makes an expensive, irreversible decision reviewable:
    what will be fetched, where it will be written, how much temporary space the
    conversion needs, whether remote code is involved, and every reason the user
    is being asked to confirm.
    """
    source = plan.resolved_source
    body = _grid()
    body.add_row(Text("Requested model"), _text(str(plan.requested)))
    body.add_row(
        Text("Resolved source"),
        _text(source.source.repo_id),
    )
    body.add_row(
        Text("Revision"),
        _text(
            _short_revision(
                (source.metadata.resolved_revision if source.metadata is not None else None)
                or source.source.revision
                or plan.requested_revision
            )
        ),
    )
    body.add_row(Text("Relationship"), _text(source.relationship.value))
    body.add_row(Text("Source format"), _text(source.format.value))
    body.add_row(Text("Architecture"), _text(plan.architecture))
    body.add_row(Text("Task"), _text(plan.task or plan.pipeline.value))

    device_line = Text(plan.target_device, style="bold")
    decision = plan.compatibility
    if decision.selected_device and decision.status is not DeviceStatus.UNKNOWN:
        severity = _DEVICE_STATUS_SEVERITY[decision.status]
        device_line = Text.assemble(
            device_line,
            ("  ", ""),
            (f"{decision.status.value} ({decision.evidence_level.value})", status_style(severity)),
        )
    if decision.reasons:
        device_line = Text.assemble(device_line, ("  ", ""), (decision.reasons[0], "dim"))
    body.add_row(Text("Device decision"), device_line)

    body.add_row(Text("Profile"), _text(plan.profile.id))
    if plan.profile.summary:
        body.add_row(Text(""), Text(plan.profile.summary, style="dim"))
    quantization = plan.profile.quantization
    quant_parts: list[str] = [quantization.weight_format]
    if quantization.group_size is not None:
        quant_parts.append(f"group size {quantization.group_size}")
    if quantization.ratio is not None:
        quant_parts.append(f"ratio {quantization.ratio:g}")
    if quantization.symmetric is not None:
        quant_parts.append("symmetric" if quantization.symmetric else "asymmetric")
    if quantization.awq:
        quant_parts.append("AWQ")
    if quantization.scale_estimation:
        quant_parts.append("scale estimation")
    body.add_row(Text("Quantization"), _text(", ".join(quant_parts)))

    disk = plan.disk
    body.add_row(Text("Expected download"), _text(human_bytes(disk.expected_download_bytes)))
    temp = disk.temp
    temp_value = Text(human_bytes(temp.total_required_bytes), style="bold")
    temp_value = Text.assemble(
        temp_value,
        ("  ", ""),
        (f"free {human_bytes(temp.usable_free_bytes)} in {temp.job_temp_dir}", "dim"),
    )
    if not temp.sufficient:
        temp_value = Text.assemble(temp_value, ("  ", ""), ("insufficient", "red"))
    body.add_row(Text("Estimated temp requirement"), temp_value)
    if temp.system_temp_would_fail:
        body.add_row(
            Text(""),
            Text("System /tmp is too small; the managed temp directory is used.", style="yellow"),
        )
    if disk.estimated_artifact_bytes is not None:
        body.add_row(Text("Estimated artifact"), _text(human_bytes(disk.estimated_artifact_bytes)))
    body.add_row(Text("Artifact directory"), _text(plan.artifact_dir))
    if plan.reuse_existing_artifact is not None:
        body.add_row(Text("Reusing artifact"), _text(plan.reuse_existing_artifact))

    if plan.trust_remote_code_required:
        remote = Text(
            "enabled by --trust-remote-code" if plan.trust_remote_code_enabled else "NOT enabled",
            style="yellow" if plan.trust_remote_code_enabled else "red",
        )
        body.add_row(Text("Remote code"), Text.assemble(Text("required  ", style="bold"), remote))
    else:
        body.add_row(Text("Remote code"), Text("not required", style="dim"))

    blocks: list[RenderableType] = [body]
    blocks.extend(_bullet_block("Expected repairs", plan.expected_repairs, style="cyan"))
    blocks.extend(_bullet_block("Known limitations", plan.known_limitations, style="yellow"))

    if plan.confirmations:
        blocks.extend((Text(""), Text("Confirmation required", style="bold")))
        reasons = Table.grid(padding=(0, 2))
        reasons.add_column(no_wrap=True, width=1)
        reasons.add_column(overflow="fold")
        for reason in plan.confirmations:
            marker = "!" if reason.security_sensitive else "·"
            style = "red" if reason.security_sensitive else "yellow"
            suffix = " (never auto-approved by --yes)" if reason.security_sensitive else ""
            reasons.add_row(
                Text(marker, style=style),
                Text.assemble((reason.message, ""), (suffix, "dim")),
            )
        blocks.append(reasons)

    if plan.exporter.problems:
        blocks.extend(_bullet_block("Exporter problems", plan.exporter.problems, style="red"))

    return Panel(
        Group(*blocks),
        title="Preparation plan",
        subtitle=f"job {plan.job_id}",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


def _bullet_block(title: str, items: Sequence[str], *, style: str) -> list[RenderableType]:
    """Return a titled bullet list, or nothing at all when ``items`` is empty."""
    if not items:
        return []
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True, width=1)
    table.add_column(overflow="fold")
    for item in items:
        table.add_row(Text("•", style=style), Text(item))
    return [Text(""), Text(title, style="bold"), table]


# ----------------------------------------------------------------------
# list
# ----------------------------------------------------------------------
def artifact_list(summaries: Sequence[ArtifactSummary]) -> RenderableType:
    """Render ``yonga list`` (``docs/CLI_SPEC.md`` section 7)."""
    if not summaries:
        return Text(
            "No prepared artifacts yet. Run 'yonga prepare MODEL' to make one.", style="dim"
        )

    table = Table(box=SIMPLE, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Artifact", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Revision", no_wrap=True)
    table.add_column("Profile", no_wrap=True)
    table.add_column("Device", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Benchmark", no_wrap=True)
    table.add_column("State", no_wrap=True)

    for summary in summaries:
        severity = Severity.OK if summary.ready else Severity.WARNING
        table.add_row(
            summary.artifact_id,
            summary.repo_id,
            _short_revision(summary.resolved_revision) or _DASH,
            summary.profile or _DASH,
            summary.recommended_device or _DASH,
            human_bytes(summary.size_bytes),
            _timestamp(summary.created_at),
            _timestamp(summary.last_benchmark),
            _status_text(severity, "ready" if summary.ready else "incomplete"),
        )
    return table


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------
def probe_report(reports: Sequence[DeviceProbeReport]) -> RenderableType:
    """Render ``yonga probe`` (``docs/CLI_SPEC.md`` section 11).

    One block per device: the overall verdict, then a per-submodel read/compile
    row, then the smoke test, then the short explanation when it failed.
    """
    if not reports:
        return Text("No device probes were performed.", style="dim")

    blocks: list[RenderableType] = []
    for report in reports:
        severity = _PROBE_STATUS_SEVERITY[report.status]
        heading = Text.assemble(
            (f"{report.device}  ", "bold"),
            (f"{status_icon(severity)} {report.status.value}", status_style(severity)),
        )
        blocks.append(heading)

        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True, width=1)
        table.add_column(no_wrap=True)
        table.add_column(overflow="fold")
        rows = [*report.submodels]
        if report.smoke_test is not None:
            rows.append(report.smoke_test)
        if not rows:
            table.add_row("", Text("(nothing probed)", style="dim"), "")
        for result in rows:
            result_severity = _PROBE_STATUS_SEVERITY[result.status]
            label = result.role.value if result.role is not None else "smoke test"
            detail = Text(result.status.value, style=status_style(result_severity))
            if result.duration_seconds is not None:
                detail = Text.assemble(
                    detail, ("  ", ""), (human_duration(result.duration_seconds), "dim")
                )
            if result.message:
                detail = Text.assemble(detail, ("  ", ""), (result.message, "dim"))
            table.add_row(
                Text(status_icon(result_severity), style=status_style(result_severity)),
                Text(label),
                detail,
            )
        blocks.append(table)

        if report.diagnostic is not None:
            blocks.append(
                Text(f"  {report.diagnostic.title}: {report.diagnostic.message}", style="dim")
            )
        blocks.append(Text(""))
    return Group(*blocks[:-1])


# ----------------------------------------------------------------------
# bench
# ----------------------------------------------------------------------
def benchmark_summary(result: BenchmarkResult) -> RenderableType:
    """Render ``yonga bench`` (``docs/CLI_SPEC.md`` section 10).

    Columns are exactly Prompt / In tok / Out tok / TTFT / tok/s / TPOT / Wall.
    Input token counts are not aggregated by the domain layer, so they are taken
    from the measured (non-warmup) runs of the same prompt.
    """
    header = _grid()
    header.add_row(Text("Model"), _text(result.repo_id))
    header.add_row(Text("Artifact"), _text(result.artifact_id))
    header.add_row(Text("Device"), _text(result.device))
    header.add_row(Text("Profile"), _text(result.profile))
    suite = f"{result.suite}  ({result.runs} runs, {result.warmups} warmups)"
    header.add_row(Text("Suite"), _text(suite))
    header.add_row(Text("Max new tokens"), _text(result.settings.max_new_tokens))
    if result.compile_seconds is not None:
        header.add_row(Text("Compile"), _text(human_duration(result.compile_seconds)))

    table = Table(box=SIMPLE, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Prompt", no_wrap=True)
    table.add_column("In tok", justify="right", no_wrap=True)
    table.add_column("Out tok", justify="right", no_wrap=True)
    table.add_column("TTFT", justify="right", no_wrap=True)
    table.add_column("tok/s", justify="right", no_wrap=True)
    table.add_column("TPOT", justify="right", no_wrap=True)
    table.add_column("Wall", justify="right", no_wrap=True)

    if not result.aggregates:
        table.add_row(Text("(no measured runs)", style="dim"), "", "", "", "", "", "")
    for aggregate in result.aggregates:
        table.add_row(
            aggregate.prompt_id,
            _input_tokens_for(result, aggregate.prompt_id),
            _round(aggregate.output_tokens_mean),
            _millis(aggregate.ttft_ms_mean),
            _decimal(aggregate.throughput_tps_mean),
            _millis(aggregate.tpot_ms_mean),
            human_duration(aggregate.wall_seconds_mean)
            if aggregate.wall_seconds_mean is not None
            else _DASH,
        )

    blocks: list[RenderableType] = [header, Text(""), table]
    if result.errors:
        blocks.extend(_bullet_block("Errors", result.errors, style="red"))
    return Group(*blocks)


def _input_tokens_for(result: BenchmarkResult, prompt_id: str) -> str:
    """Return the input token count observed for ``prompt_id``, if any run reported one."""
    for run in result.run_metrics:
        if run.prompt_id == prompt_id and not run.warmup and run.input_tokens is not None:
            return str(run.input_tokens)
    return _DASH


def _round(value: float | None) -> str:
    """Render a mean count as a whole number."""
    if value is None:
        return _DASH
    return f"{value:.0f}"


def _decimal(value: float | None) -> str:
    """Render a rate with one decimal."""
    if value is None:
        return _DASH
    return f"{value:.1f}"


def _millis(value: float | None) -> str:
    """Render a millisecond figure with its unit."""
    if value is None:
        return _DASH
    return f"{value:.1f} ms"


# ----------------------------------------------------------------------
# registry explain
# ----------------------------------------------------------------------
def registry_explanation(explanation: RegistryExplanation) -> RenderableType:
    """Render ``yonga registry explain`` (``docs/CLI_SPEC.md`` section 12).

    Every candidate rule is listed with its specificity and priority, every
    rejected rule is listed *with the reasons it was rejected*, and the winner is
    marked. That auditability is the point of the registry.
    """
    header = _grid()
    header.add_row(Text("Model"), _text(explanation.repo_id))
    header.add_row(Text("Device"), _text(explanation.device))
    header.add_row(Text("Host tags"), _text(", ".join(explanation.host_tags) or None))
    if explanation.versions:
        versions = ", ".join(
            f"{name} {value}" for name, value in sorted(explanation.versions.items())
        )
        header.add_row(Text("Versions"), _text(versions))
    header.add_row(Text("Recommended device"), _text(explanation.recommended_device))

    blocks: list[RenderableType] = [header]
    resolutions: list[tuple[str, ResolutionExplanation | None, str | None]] = [
        ("Architecture", explanation.architecture, explanation.architecture_rule_id),
        ("Model rule", explanation.model, explanation.model_rule_id),
        ("Profile", explanation.profile, explanation.profile_id),
        ("Chat profile", explanation.chat, explanation.chat_profile_id),
        ("Derivative sources", explanation.derivative_resolution, None),
    ]
    for title, resolution, winner_id in resolutions:
        if resolution is None:
            continue
        blocks.extend((Text(""), Text(title, style="bold")))
        blocks.append(_resolution_table(resolution, winner_id))

    if explanation.derivatives:
        blocks.extend((Text(""), Text("Derivative candidates", style="bold")))
        table = Table(box=SIMPLE, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("Repository", overflow="fold")
        table.add_column("Relationship", no_wrap=True)
        table.add_column("Trust", no_wrap=True)
        table.add_column("Note", overflow="fold")
        for candidate in explanation.derivatives:
            table.add_row(
                candidate.repo_id,
                candidate.relationship.value,
                candidate.trust.value,
                candidate.note or "",
            )
        blocks.append(table)

    if explanation.device_compatibility:
        blocks.extend((Text(""), Text("Device compatibility", style="bold")))
        compatibility = Table.grid(padding=(0, 2))
        compatibility.add_column(no_wrap=True, width=1)
        compatibility.add_column(no_wrap=True)
        compatibility.add_column(overflow="fold")
        for entry in explanation.device_compatibility:
            severity = _DEVICE_STATUS_SEVERITY[entry.status]
            detail = Text(
                f"{entry.status.value} ({entry.evidence.value})", style=status_style(severity)
            )
            if entry.reason:
                detail = Text.assemble(detail, ("  ", ""), (entry.reason, "dim"))
            if entry.rule_id:
                detail = Text.assemble(detail, ("  ", ""), (entry.rule_id, "dim"))
            compatibility.add_row(
                Text(status_icon(severity), style=status_style(severity)),
                Text(entry.device, style="bold"),
                detail,
            )
        blocks.append(compatibility)

    if explanation.notes:
        blocks.extend(_bullet_block("Notes", explanation.notes, style="yellow"))
    return Group(*blocks)


def _resolution_table(resolution: ResolutionExplanation, winner_id: str | None) -> RenderableType:
    """Render one resolution's candidates, winner and rejection reasons."""
    winner = winner_id or resolution.winner_id
    if not resolution.candidates:
        note = Text("  no candidate rules", style="dim")
        if resolution.notes:
            return Group(note, *(Text(f"  {line}", style="dim") for line in resolution.notes))
        return note

    table = Table(box=SIMPLE, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("", no_wrap=True, width=1)
    table.add_column("Rule", overflow="fold")
    table.add_column("Kind", no_wrap=True)
    table.add_column("Spec", justify="right", no_wrap=True)
    table.add_column("Prio", justify="right", no_wrap=True)
    table.add_column("Outcome", overflow="fold")

    for candidate in resolution.candidates:
        is_winner = winner is not None and candidate.rule_id == winner
        if is_winner:
            marker, style, outcome = "✓", "green", "winner"
        elif candidate.matched:
            marker, style, outcome = "·", "dim", "matched"
        else:
            marker, style, outcome = "✗", "red", "rejected"
        reasons = "; ".join(candidate.reasons)
        verdict = Text(outcome, style=style)
        if reasons:
            verdict = Text.assemble(verdict, ("  ", ""), (reasons, "dim"))
        table.add_row(
            Text(marker, style=style),
            Text(candidate.rule_id, style="bold" if is_winner else ""),
            candidate.kind,
            str(candidate.specificity),
            str(candidate.priority),
            verdict,
        )
    if resolution.notes:
        return Group(table, *(Text(f"  {line}", style="dim") for line in resolution.notes))
    return table
