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
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict
from rich.box import ROUNDED, SIMPLE
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.measure import Measurement
from rich.panel import Panel
from rich.rule import Rule
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

#: The flag that lets a user proceed past an unmet exporter requirement. Spelled
#: here rather than imported so this pure render module keeps its domain-only
#: dependencies; it must stay equal to
#: ``yonga.application.prepare.IGNORE_EXPORTER_REQUIREMENTS_FLAG``, which a unit
#: test asserts.
_IGNORE_EXPORTER_REQUIREMENTS_FLAG = "--ignore-exporter-requirements"

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
    # Not an error on its own: the pipeline smoke test carries the verdict.
    ProbeStatus.DEFERRED: Severity.INFO,
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


def _architecture_line(plan: PreparationPlan) -> Text:
    """Render the architecture as the identity, with the matched rule id behind it.

    ``plan.architecture`` is what the model *is* (``llama``). The registry rule
    that matched covers a whole family, so its id answers a different question
    and is shown as provenance rather than as the answer.
    """
    identity = _text(plan.architecture)
    rule_id = (plan.architecture_rule_id or "").strip()
    if not rule_id or rule_id == str(plan.architecture or "").strip():
        return identity
    return Text.assemble(identity, ("  ", ""), (f"rule {rule_id}", "dim"))


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


def _timestamp(moment: datetime | None, *, minutes: bool = True) -> str:
    """Render a timestamp in local time, to the minute or, when narrow, to the day."""
    if moment is None:
        return _DASH
    return moment.astimezone().strftime("%Y-%m-%d %H:%M" if minutes else "%Y-%m-%d")


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
    architecture: str | None = None,
    identified_from: str | None = None,
) -> RenderableType:
    """Render the ``yonga inspect`` report (``docs/CLI_SPEC.md`` section 4).

    ``architecture`` is the identity the registry resolved, which can be known
    even when the repository declares none — a GGUF-only repo has no
    ``config.json``, so the architecture comes from the GGUF header instead.
    Passing it keeps the human table and ``--json`` telling the same story.

    ``identified_from`` names that weaker source when one was used. It is
    disclosed beside the compatibility block at default verbosity rather than
    in a ``--verbose`` note, because every device claim in that block rests on
    the identification and ``SPEC.md`` section 3 does not allow support to be
    presented as better established than it is.
    """
    facts = _grid()
    facts.add_row(Text("Model"), _text(metadata.identity.repo_id))
    facts.add_row(
        Text("Revision"),
        _text(_short_revision(metadata.resolved_revision or metadata.identity.revision)),
    )
    facts.add_row(Text("License"), _text(metadata.license))
    facts.add_row(Text("Task"), _text(metadata.pipeline_tag or metadata.task))
    facts.add_row(
        Text("Architecture"),
        _text(", ".join(metadata.architectures) or architecture),
    )
    facts.add_row(Text("Model type"), _text(metadata.model_type or architecture))
    facts.add_row(Text("Pipeline"), _text(metadata.pipeline_kind.value))
    formats = ", ".join(
        observation.format.value for observation in metadata.formats if observation.files
    )
    facts.add_row(Text("Formats"), _text(formats or None))
    facts.add_row(Text("Size"), _size_row(metadata))
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
        if identified_from is not None:
            blocks.append(_identity_provenance(metadata, identified_from))
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


def _size_row(metadata: ModelMetadata) -> Text:
    """Render the inspect "Size" row: usable weights, plus the repository total.

    ``weight_bytes`` is the number that matters to the user — what one model
    costs to download and what preflight reserves working space for. For most
    repositories it is also the whole story, and the row stays the short form.

    A repository that publishes several interchangeable copies of one model (the
    GGUF one-file-per-quantization convention) makes the two figures diverge,
    and hiding either would mislead: the total alone overstates the download
    several-fold, the primary alone contradicts the file listing right below it.
    So both are shown, on one line, with the repository figure qualified by how
    many files it spans.
    """
    usable = metadata.weight_bytes
    if not usable:
        return _text(None)
    repository = metadata.repo_weight_bytes
    if repository <= usable:
        return _text(f"~{human_bytes(usable, precision=1)} total weights")
    observation = metadata.primary_observation
    file_count = len(observation.files) if observation is not None else 0
    return _text(
        f"~{human_bytes(usable, precision=1)} (one model; "
        f"{human_bytes(repository, precision=1)} across {file_count} files in the repository)"
    )


def _identity_provenance(metadata: ModelMetadata, identified_from: str) -> Text:
    """State that the claims below rest on a hint rather than on ``config.json``.

    One line, because ``docs/CLI_SPEC.md`` section 1 asks the default output to
    stay concise; the alias that actually matched is left to the ``--verbose``
    note the registry already writes.
    """
    if metadata.architectures or metadata.model_type:
        tail = "the repository's config.json declares a different identity"
    else:
        tail = "this repository has no config.json"
    return Text(f"Architecture identified from {identified_from}; {tail}.", style="yellow")


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
    body.add_row(Text("Architecture"), _architecture_line(plan))
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

    blocks.extend(_exporter_block(plan))

    return Panel(
        Group(*blocks),
        title="Preparation plan",
        subtitle=f"job {plan.job_id}",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


def _exporter_block(plan: PreparationPlan) -> list[RenderableType]:
    """Render unmet exporter requirements as a *blocking* section.

    This is not an advisory list. Everything above it in the panel describes
    work that is about to happen and, at most, asks for approval; this section
    says the work will not happen at all. It therefore gets its own rule, an
    explicit ``BLOCKED`` heading and a red ``x`` marker per problem rather than
    the ``!``/``·`` bullets of ``Confirmation required``, and it names the one
    flag that overrides it — ``--yes`` deliberately does not.

    Rendered only when the pin really will stop the job: the section is keyed on
    :attr:`PreparationPlan.exporter_requirements_block`, the same property the
    preflight gate blocks on, so the panel cannot announce a block that will not
    happen (an already-OpenVINO source is copied, never converted).
    """
    if not plan.exporter_requirements_block:
        return []
    problems = Table.grid(padding=(0, 2))
    problems.add_column(no_wrap=True, width=1)
    problems.add_column(overflow="fold")
    for problem in plan.exporter.problems:
        problems.add_row(Text("x", style="bold red"), Text(problem, style="red"))
    return [
        Text(""),
        Rule(style="red"),
        Text("BLOCKED — exporter requirements not satisfied", style="bold red"),
        problems,
        Text(
            "Preparation stops at preflight, before the download and the conversion. "
            f"Override with {_IGNORE_EXPORTER_REQUIREMENTS_FLAG} if you know the pin is stale; "
            "--yes does not.",
            style="dim",
        ),
    ]


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
#: Every column ``yonga list`` can show, in display order, with its header.
_LIST_COLUMNS: tuple[tuple[str, str], ...] = (
    ("artifact", "Artifact"),
    ("model", "Model"),
    ("revision", "Revision"),
    ("profile", "Profile"),
    ("device", "Dev"),
    ("size", "Size"),
    ("created", "Created"),
    ("benchmark", "Benchmark"),
    ("ready", "Ready"),
)

#: Order in which columns are given up when the terminal cannot fit them all.
#: ``docs/CLI_SPEC.md`` section 7 asks for the artifact, its source revision,
#: profile, recommended device, size and last benchmark, so the columns it does
#: not ask for go first and the ones it does go last.
_LIST_DROP_ORDER: tuple[str, ...] = (
    "created",
    "model",
    "revision",
    "benchmark",
    "size",
    "ready",
)

#: Columns whose values are right-aligned.
_LIST_NUMERIC: frozenset[str] = frozenset({"size"})

#: Widest repository id printed before it is elided; long ids must not be able
#: to push the columns beside them out of the table.
_LIST_MODEL_MAX_WIDTH = 32

#: Commit sha length in the list. The full sha is one ``--json`` away.
_LIST_REVISION_LENGTH = 8

#: A width no real table reaches, used to measure a layout's natural size.
_LIST_MEASURE_WIDTH = 10_000


class _ListVariant(NamedTuple):
    """One rung of the ``yonga list`` layout ladder."""

    dropped: frozenset[str]
    long_timestamps: bool


def _list_variants() -> tuple[_ListVariant, ...]:
    """Build the layout ladder, widest first, narrowest last."""
    variants = [_ListVariant(frozenset(), True), _ListVariant(frozenset(), False)]
    dropped: set[str] = set()
    for key in _LIST_DROP_ORDER:
        dropped.add(key)
        variants.append(_ListVariant(frozenset(dropped), False))
    return tuple(variants)


#: The layout ladder, from every column with to-the-minute timestamps down to
#: the artifact id, its profile and its device.
_LIST_VARIANTS: tuple[_ListVariant, ...] = _list_variants()


def _list_cells(summary: ArtifactSummary, *, long_timestamps: bool) -> dict[str, Text]:
    """Render one artifact as a cell per column key."""
    severity = Severity.OK if summary.ready else Severity.WARNING
    ready = _status_text(severity, "yes" if summary.ready else "no")
    revision = _short_revision(summary.resolved_revision, length=_LIST_REVISION_LENGTH)
    return {
        "artifact": _text(summary.artifact_id),
        "model": _text(summary.repo_id),
        "revision": _text(revision),
        "profile": _text(summary.profile),
        "device": _text(summary.recommended_device),
        "size": _text(human_bytes(summary.size_bytes)),
        "created": _text(_timestamp(summary.created_at, minutes=long_timestamps)),
        "benchmark": _text(_timestamp(summary.last_benchmark, minutes=long_timestamps)),
        "ready": ready,
    }


def _artifact_table(summaries: Sequence[ArtifactSummary], variant: _ListVariant) -> Table:
    """Build the ``yonga list`` table for one rung of the ladder."""
    table = Table(
        box=SIMPLE,
        show_edge=False,
        pad_edge=False,
        header_style="bold",
        # One space between columns instead of two: the extra column that the
        # box rule occupies already separates them, and nine columns cannot
        # afford a second space each on an 80-column terminal.
        padding=(0, 1, 0, 0),
    )
    keys = [key for key, _ in _LIST_COLUMNS if key not in variant.dropped]
    headers = dict(_LIST_COLUMNS)
    for key in keys:
        table.add_column(
            headers[key],
            no_wrap=True,
            overflow="ellipsis",
            justify="right" if key in _LIST_NUMERIC else "left",
            max_width=_LIST_MODEL_MAX_WIDTH if key == "model" else None,
        )
    for summary in summaries:
        cells = _list_cells(summary, long_timestamps=variant.long_timestamps)
        table.add_row(*(cells[key] for key in keys))
    return table


class _ArtifactList:
    """The ``yonga list`` catalogue, laid out for the terminal it is printed on.

    Nine columns do not fit an 80-column terminal, and Rich's answer to that is
    to shrink every column until each one is an ellipsis. This renderable
    answers differently: it drops whole columns, in a fixed order, until the
    rest fit at their natural width. The artifact id, the profile and the
    device are never truncated, and the detail ``docs/CLI_SPEC.md`` section 7
    asks for survives down to 80 columns.
    """

    def __init__(self, summaries: Sequence[ArtifactSummary]) -> None:
        """Keep the summaries; the layout is chosen at render time."""
        self._summaries = tuple(summaries)

    def table_for(self, console: Console, options: ConsoleOptions) -> Table:
        """Return the widest layout whose natural width fits ``options``."""
        wide = options.update_width(_LIST_MEASURE_WIDTH)
        narrowest = _artifact_table(self._summaries, _LIST_VARIANTS[-1])
        for variant in _LIST_VARIANTS[:-1]:
            candidate = _artifact_table(self._summaries, variant)
            if Measurement.get(console, wide, candidate).maximum <= options.max_width:
                return candidate
        return narrowest

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the chosen layout."""
        yield self.table_for(console, options)

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        """Measure the layout that would be chosen for ``options``."""
        return Measurement.get(console, options, self.table_for(console, options))


def artifact_list(summaries: Sequence[ArtifactSummary]) -> RenderableType:
    """Render ``yonga list`` (``docs/CLI_SPEC.md`` section 7)."""
    if not summaries:
        return Text(
            "No prepared artifacts yet. Run 'yonga prepare MODEL' to make one.", style="dim"
        )
    return _ArtifactList(summaries)


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
