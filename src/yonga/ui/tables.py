"""Pure render functions turning typed domain objects into Rich renderables.

Every function here is a *pure* projection: it takes domain values, returns a
renderable, and prints nothing. That is what lets the tests assert on the render
model and what keeps presentation out of the application layer. None of these
functions decides anything — the layouts follow ``docs/CLI_SPEC.md`` sections 3,
4, 6, 7, 10, 11 and 12.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict
from rich.box import ROUNDED, SIMPLE
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.measure import Measurement
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from yonga.domain.artifacts import ArtifactSummary, DeviceProbeReport, ProbeResult, ProbeStatus
from yonga.domain.benchmarks import BenchmarkResult
from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceStatus,
    EvidenceLevel,
    ResolutionExplanation,
)
from yonga.domain.compile_cache import CacheEntryStatus, CompileCacheEntry
from yonga.domain.diagnostics import FailureCategory, HealthSection, Severity
from yonga.domain.evaluation import EvaluationResult
from yonga.domain.models import ModelMetadata
from yonga.domain.plans import JobStatus, JobSummary, PreparationPlan
from yonga.domain.publication import PublishPlan
from yonga.ui.console import human_bytes, human_duration, status_icon, status_style

if TYPE_CHECKING:  # Registry types are consumed, never imported at runtime.
    from yonga.application.search import SearchReport
    from yonga.registry.resolver import RegistryExplanation

__all__ = [
    "SourceOption",
    "artifact_list",
    "benchmark_summary",
    "compile_cache_entries",
    "doctor_report",
    "inspect_report",
    "job_heading",
    "job_list",
    "local_clock",
    "preparation_plan",
    "probe_report",
    "registry_explanation",
    "search_results",
    "utc_offset",
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
    # Neither a pass nor a failure: the probe stopped before it had evidence.
    ProbeStatus.INCONCLUSIVE: Severity.WARNING,
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
    remote_code_required: bool | None = None,
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

    ``remote_code_required`` is the registry-aware decision ``prepare`` acts on.
    A repository can ship modelling code that is never run, because
    Transformers implements the architecture natively; the row then says so
    instead of asking for ``--trust-remote-code``. ``None`` falls back to what
    the repository declares.
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
        sized, basis = metadata.size_for_matching()
        parameters = Text(f"{metadata.parameter_count:,}")
        if sized is None:
            # Said here because nothing else will: a size band would quietly not apply.
            parameters.append(f"  not used to choose a profile: {basis}", style="dim")
        else:
            parameters.append(
                f"  {metadata.parameter_count_origin}; includes embeddings and any vision tower",
                style="dim",
            )
        facts.add_row(Text("Parameters"), parameters)
    if metadata.gated or metadata.private:
        flags = ", ".join(
            label for label, on in (("gated", metadata.gated), ("private", metadata.private)) if on
        )
        facts.add_row(Text("Access"), Text(flags, style="yellow"))
    required = (
        metadata.requires_remote_code if remote_code_required is None else remote_code_required
    )
    if required:
        facts.add_row(
            Text("Remote code"),
            Text("required (needs --trust-remote-code)", style="yellow"),
        )
    elif metadata.requires_remote_code:
        facts.add_row(
            Text("Remote code"),
            Text("not required (shipped, but Transformers implements this architecture)"),
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

    gguf = plan.resolved_source.gguf_file
    if gguf is not None:
        # Imported as-is: the file decides the weights, the profile only the runtime.
        body.add_row(
            Text("GGUF file"),
            Text.assemble((gguf, "bold"), ("  imported directly by OpenVINO GenAI", "dim")),
        )
        types = ", ".join(
            f"{name} x{count}" for name, count in plan.resolved_source.gguf_tensor_types.items()
        )
        body.add_row(Text("Quantization"), _text(f"the file's own: {types}"))
        body.add_row(
            Text("Runtime profile"),
            Text.assemble((plan.profile.id, ""), ("  runtime properties only", "dim")),
        )
    else:
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
        selection = plan.profile_selection
        if selection is not None and selection.applies and selection.sentence:
            body.add_row(Text("Chosen because"), Text(selection.sentence, style="dim"))

    if plan.reuses_artifact:
        # Nothing is fetched, converted or written: download, temp space and
        # export environment would all describe work that does not happen.
        body.add_row(
            Text("Reusing artifact"),
            Text.assemble(
                (str(plan.reuse_existing_artifact), "bold"),
                ("  ready at this revision and profile; --force to prepare again", "dim"),
            ),
        )
    else:
        _work_rows(body, plan)

    if plan.trust_remote_code_required:
        remote = Text(
            "enabled by --trust-remote-code" if plan.trust_remote_code_enabled else "NOT enabled",
            style="yellow" if plan.trust_remote_code_enabled else "red",
        )
        body.add_row(Text("Remote code"), Text.assemble(Text("required  ", style="bold"), remote))
    else:
        body.add_row(Text("Remote code"), Text("not required", style="dim"))
    if not plan.reuses_artifact:
        body.add_row(Text("Export environment"), _export_environment_line(plan))

    reusable = plan.reusable_export
    if reusable is not None:
        finished = reusable.completed_at.strftime("%Y-%m-%d %H:%M UTC")
        body.add_row(
            Text("Export"),
            Text.assemble(
                Text(f"reused from {reusable.job_label} {reusable.job_id[:8]}", style="cyan"),
                (f"  completed {finished}, identical inputs; --fresh-export to redo", "dim"),
            ),
        )

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

    blocks.extend(_refusal_block(plan))
    blocks.extend(_exporter_block(plan))

    return Panel(
        Group(*blocks),
        title="Preparation plan",
        subtitle=f"job {plan.job_id}",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


def _work_rows(body: Table, plan: PreparationPlan) -> None:
    """Add the download, temp-space and artifact rows of a plan that does the work."""
    disk = plan.disk
    download = Text(human_bytes(disk.expected_download_bytes), style="bold")
    if plan.download_ignore_patterns:
        download.append("  only the files the conversion reads", style="dim")
    body.add_row(Text("Expected download"), download)
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
        body.add_row(
            Text("Estimated artifact"), _text(f"about {human_bytes(disk.estimated_artifact_bytes)}")
        )
    body.add_row(Text("Artifact directory"), _text(plan.artifact_dir))


def _export_environment_line(plan: PreparationPlan) -> Text:
    """Say which Python the export runs in, and why that one.

    Always shown, so a plan that exports in yonga's own environment says so
    rather than leaving it to be inferred from a missing row.
    """
    source = plan.resolved_source
    if source.already_openvino:
        return Text("none needed  the published OpenVINO IR is copied, not exported", style="dim")
    if source.gguf_file is not None:
        return Text(
            "none needed  OpenVINO GenAI imports the GGUF file in this process", style="dim"
        )
    environment = plan.export_environment
    if environment is None:
        if plan.runtime_missing:
            why = "no managed environment is planned without the OpenVINO runtime"
        elif plan.exporter.problems:
            why = "managed environments are not in use, so the unmet requirements apply here"
        else:
            why = "the installed packages meet the exporter requirements"
        return Text.assemble(
            Text("this environment", style="cyan"),
            (f"  {sys.executable}  ", ""),
            (why, "dim"),
        )
    pins = ", ".join(
        f"{name}{specifier}"
        for name, specifier in environment.packages.items()
        if name == "transformers" and specifier
    )
    state = (
        "exists, reused"
        if plan.export_environment_ready
        else "built only if the reused export fails its checks"
        if plan.reusable_export is not None
        else "will be built"
    )
    return Text.assemble(
        Text(f"managed {environment.name}", style="cyan"),
        (f"  {pins}  " if pins else "  ", ""),
        (state, "dim"),
    )


def _refusal_block(plan: PreparationPlan) -> list[RenderableType]:
    """Render why a plan will stop before preflight, whatever is confirmed.

    Keyed on the same properties the service refuses on, so the panel never
    announces a stop that will not happen, nor hides one that will.
    """
    reasons: list[str] = []
    if plan.runtime_missing:
        reasons.append(
            "The OpenVINO runtime is not installed in this environment: "
            "pip install 'yonga[openvino]'"
        )
    if plan.device_known_unsupported:
        fallback = next(iter(plan.compatibility.fallback_candidates), None)
        reasons.append(
            f"The registry records {plan.target_device} as unsupported for this model "
            f"({plan.compatibility.evidence_level.value})."
            + (f" Recommended fallback: --device {fallback.lower()}" if fallback else "")
        )
    if plan.auth_missing:
        reasons.append(
            "The repository is gated or private and no Hugging Face token was found: hf auth login"
        )
    if not reasons:
        return []
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True, width=1)
    table.add_column(overflow="fold")
    for reason in reasons:
        table.add_row(Text("x", style="bold red"), Text(reason, style="red"))
    return [
        Text(""),
        Rule(style="red"),
        Text("BLOCKED — this preparation cannot run", style="bold red"),
        table,
        Text("Nothing is downloaded or converted.", style="dim"),
    ]


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
#: ``Rev`` and ``Dev`` are abbreviated because a header wider than its values
#: sets the column's width, and an 80-column terminal has no cell to spare.
_LIST_COLUMNS: tuple[tuple[str, str], ...] = (
    ("artifact", "Artifact"),
    ("model", "Model"),
    ("revision", "Rev"),
    ("profile", "Profile"),
    ("device", "Dev"),
    ("size", "Size"),
    ("created", "Created"),
    ("benchmark", "Benchmark"),
    ("ready", "Ready"),
)

#: Order in which columns are given up when the terminal cannot fit them all.
#: The repository id and its revision are what tell two artifacts apart, so they
#: outlast everything but the artifact id, the profile and the device; the
#: creation time goes first, then the last benchmark, a timestamp the wider
#: layouts and ``--json`` still carry. ``Ready`` is given up before the size,
#: but never silently: an artifact that is not ready is then named under the
#: table instead.
_LIST_DROP_ORDER: tuple[str, ...] = (
    "created",
    "benchmark",
    "ready",
    "size",
    "revision",
    "model",
)

#: Columns whose values are right-aligned.
_LIST_NUMERIC: frozenset[str] = frozenset({"size"})

#: Widest repository id printed before it is elided; long ids must not be able
#: to push the columns beside them out of the table.
_LIST_MODEL_MAX_WIDTH = 32

#: Narrowest the repository id is elided to before the layout gives up another
#: column instead; any shorter and the tail no longer names the model.
_LIST_MODEL_MIN_WIDTH = 12

#: Commit sha length in the list, the length ``git`` abbreviates to. The full
#: sha is one ``--json`` away.
_LIST_REVISION_LENGTH = 7

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


def _elide_repo_id(repo_id: str, width: int) -> str:
    """Fit ``repo_id`` into ``width`` cells, giving up the organisation first.

    The model name is what tells the artifacts in one list apart, and its start
    (family, version, size) more than its end; the organisation is the part
    they most often share. So a repository id that does not fit loses its
    namespace to ``…/`` first, and only a name that still does not fit is cut,
    from the right.
    """
    if Text(repo_id).cell_len <= width:
        return repo_id
    name = repo_id.rpartition("/")[2]
    if name != repo_id and Text(f"…/{name}").cell_len <= width:
        return f"…/{name}"
    return name[: max(width - 1, 0)] + "…"


def _list_cells(
    summary: ArtifactSummary, *, long_timestamps: bool, model_width: int
) -> dict[str, Text]:
    """Render one artifact as a cell per column key."""
    severity = Severity.OK if summary.ready else Severity.WARNING
    ready = _status_text(severity, "yes" if summary.ready else "no")
    revision = _short_revision(summary.resolved_revision, length=_LIST_REVISION_LENGTH)
    return {
        "artifact": _text(summary.artifact_id),
        "model": _text(_elide_repo_id(summary.repo_id, model_width)),
        "revision": _text(revision),
        "profile": _text(summary.profile),
        "device": _text(summary.recommended_device),
        "size": _text(human_bytes(summary.size_bytes)),
        "created": _text(_timestamp(summary.created_at, minutes=long_timestamps)),
        "benchmark": _text(_timestamp(summary.last_benchmark, minutes=long_timestamps)),
        "ready": ready,
    }


def _artifact_table(
    summaries: Sequence[ArtifactSummary],
    variant: _ListVariant,
    *,
    model_width: int = _LIST_MODEL_MAX_WIDTH,
) -> Table:
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
        )
    for summary in summaries:
        cells = _list_cells(
            summary, long_timestamps=variant.long_timestamps, model_width=model_width
        )
        table.add_row(*(cells[key] for key in keys))
    return table


class _ArtifactList:
    """The ``yonga list`` catalogue, laid out for the terminal it is printed on.

    Nine columns do not fit an 80-column terminal, and Rich's answer to that is
    to shrink every column until each one is an ellipsis. This renderable
    answers differently: it drops whole columns, in a fixed order, until the
    rest fit at their natural width. Once the creation time has gone, the
    repository id may also be elided from the left (down to
    ``_LIST_MODEL_MIN_WIDTH`` cells) before another column is given up, since
    without it two artifacts of different models with one profile look alike.

    The artifact id, the profile and the device are never truncated. On an
    80-column terminal the list keeps the model, its revision, the size and
    readiness (the last as a note under the table when its column has had to
    go), and leaves the creation and benchmark times to wider terminals and to
    ``--json``.
    """

    def __init__(self, summaries: Sequence[ArtifactSummary]) -> None:
        """Keep the summaries; the layout is chosen at render time."""
        self._summaries = tuple(summaries)

    def _natural_model_width(self) -> int:
        """Cells the model column needs before any elision."""
        widest = max((Text(item.repo_id).cell_len for item in self._summaries), default=0)
        return min(_LIST_MODEL_MAX_WIDTH, max(widest, len("Model")))

    def layout_for(self, console: Console, options: ConsoleOptions) -> tuple[_ListVariant, Table]:
        """Return the widest layout that fits ``options``, with the rung it came from."""
        wide = options.update_width(_LIST_MEASURE_WIDTH)
        natural = self._natural_model_width()
        for variant in _LIST_VARIANTS[:-1]:
            candidate = _artifact_table(self._summaries, variant, model_width=natural)
            width = Measurement.get(console, wide, candidate).maximum
            if width <= options.max_width:
                return variant, candidate
            elidable = "created" in variant.dropped and "model" not in variant.dropped
            narrower = natural - (width - options.max_width)
            if elidable and narrower >= _LIST_MODEL_MIN_WIDTH:
                return variant, _artifact_table(self._summaries, variant, model_width=narrower)
        last = _LIST_VARIANTS[-1]
        return last, _artifact_table(self._summaries, last)

    def table_for(self, console: Console, options: ConsoleOptions) -> Table:
        """Return the table of the layout chosen for ``options``."""
        return self.layout_for(console, options)[1]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the chosen layout, naming unready artifacts if ``Ready`` was dropped."""
        variant, table = self.layout_for(console, options)
        yield table
        if "ready" not in variant.dropped:
            return
        unready = [item.artifact_id for item in self._summaries if not item.ready]
        if unready:
            yield _status_text(Severity.WARNING, f"Not ready: {', '.join(unready)}")

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
# compile cache
# ----------------------------------------------------------------------
#: Severity shown for each compile cache entry status.
_CACHE_STATUS_SEVERITY: dict[CacheEntryStatus, Severity] = {
    CacheEntryStatus.IN_USE: Severity.OK,
    CacheEntryStatus.ORPHANED: Severity.WARNING,
    CacheEntryStatus.STALE_RUNTIME: Severity.WARNING,
    CacheEntryStatus.UNATTRIBUTED: Severity.UNKNOWN,
}


def _release(version: str | None) -> str | None:
    """Shorten ``2026.4.0-22959-99c81491cc3-releases/2026/4`` to ``2026.4.0``."""
    if version is None:
        return None
    return version.split("-", 1)[0].strip() or version


def _cache_owner(entry: CompileCacheEntry) -> Text:
    """The owning artifact; ``*`` marks one that lives outside the artifact store."""
    label = entry.artifact_id or (entry.artifact_path.name if entry.artifact_path else None)
    if label is None:
        return Text(_DASH, style="dim")
    owner = Text(label, style="" if entry.owner_exists else "dim")
    if entry.owner_exists and not entry.owner_in_store:
        owner.append("*", style="cyan")
    return owner


#: Optional ``yonga cache`` columns, in the order they are given up when the
#: terminal is too narrow. Key, size, owner and status always stay.
_CACHE_DROP_ORDER: tuple[str, ...] = ("Model", "Last used", "OpenVINO", "Dev")


class _CacheTable:
    """The compile cache entries, laid out for the terminal they are printed on.

    Like ``yonga list``, whole columns are dropped, in a fixed order, until the
    rest fit at their natural width, so a key or a status is never cut short.
    """

    def __init__(
        self, entries: Sequence[CompileCacheEntry], reasons: dict[str, str] | None
    ) -> None:
        """Build every cell once; the layout is chosen at render time."""
        self._why = reasons is not None and any(
            reasons.get(entry.key) != entry.status.value for entry in entries
        )
        self._rows: list[dict[str, Text]] = []
        for entry in entries:
            severity = _CACHE_STATUS_SEVERITY[entry.status]
            row = {
                "Key": Text(entry.key),
                "Size": Text(human_bytes(entry.size_bytes)),
                "Last used": Text(_timestamp(entry.last_used_at, minutes=False)),
                "Dev": _text(entry.device),
                "OpenVINO": _text(_release(entry.openvino_version)),
                "Owner": _cache_owner(entry),
                "Status": _status_text(severity, entry.status.value),
                "Model": _text(entry.repo_id),
            }
            if self._why and reasons is not None:
                row["Why"] = _text(reasons.get(entry.key))
            self._rows.append(row)

    def _table(self, dropped: frozenset[str]) -> Table:
        headers = ["Key", "Size", "Last used", "Dev", "OpenVINO", "Owner", "Status"]
        if self._why:
            headers.append("Why")
        headers.append("Model")
        table = Table(
            box=SIMPLE, show_edge=False, pad_edge=False, header_style="bold", padding=(0, 1, 0, 0)
        )
        shown = [header for header in headers if header not in dropped]
        for header in shown:
            table.add_column(
                header,
                no_wrap=True,
                overflow="ellipsis",
                justify="right" if header == "Size" else "left",
                max_width=_LIST_MODEL_MAX_WIDTH if header == "Model" else None,
            )
        for row in self._rows:
            table.add_row(*(row[header] for header in shown))
        return table

    def table_for(self, console: Console, options: ConsoleOptions) -> Table:
        """Return the widest layout whose natural width fits ``options``."""
        wide = options.update_width(_LIST_MEASURE_WIDTH)
        dropped: set[str] = set()
        candidate = self._table(frozenset())
        for header in _CACHE_DROP_ORDER:
            if Measurement.get(console, wide, candidate).maximum <= options.max_width:
                return candidate
            dropped.add(header)
            candidate = self._table(frozenset(dropped))
        return candidate

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the chosen layout."""
        yield self.table_for(console, options)

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        """Measure the layout that would be chosen for ``options``."""
        return Measurement.get(console, options, self.table_for(console, options))


def compile_cache_entries(
    entries: Sequence[CompileCacheEntry],
    *,
    reasons: dict[str, str] | None = None,
) -> RenderableType:
    """Render the entries of ``yonga cache list`` or a ``cache prune`` plan.

    ``reasons`` adds a column saying why each entry was selected for pruning,
    when that is not simply its status. An owner marked ``*`` exists, but
    outside this artifact store (another ``--data-dir``, or a running
    ``prepare``'s staging directory).
    """
    if not entries:
        return Text("The compile cache is empty.", style="dim")
    table = _CacheTable(entries, reasons)
    if not any(entry.owner_exists and not entry.owner_in_store for entry in entries):
        return table
    return Group(table, Text("* the artifact exists, outside this artifact store", style="dim"))


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------
def search_results(report: SearchReport, *, device: str | None = None) -> RenderableType:
    """Render ``yonga search`` (``docs/CLI_SPEC.md`` section 5).

    With ``device`` set only that device's column is shown. An architecture
    identified from a GGUF header is marked ``*`` and footnoted, the same
    disclosure ``inspect`` makes, because every device claim on that row rests
    on the identification. A declared derivative is marked ``+`` and named in a
    footnote, so long repository ids keep the width they need.
    """
    if not report.hits:
        return Text(f"No models on the Hub match {report.query!r}.", style="dim")
    devices = [device.upper()] if device and device.lower() != "auto" else ["CPU", "GPU", "NPU"]
    table = Table(box=SIMPLE, show_edge=False, pad_edge=False)
    table.add_column("Model", overflow="fold", min_width=24)
    table.add_column("Arch", no_wrap=True)
    for name in devices:
        table.add_column(name, no_wrap=True, justify="center")
    table.add_column("Format", no_wrap=True)
    table.add_column("DL", no_wrap=True, justify="right")
    hinted = False
    derivatives: list[str] = []
    for hit in report.hits:
        arch = Text(hit.architecture or _DASH, style="" if hit.architecture else "dim")
        if hit.identified_from:
            arch.append("*", style="yellow")
            hinted = True
        cells: list[RenderableType] = [Text(hit.repo_id), arch]
        for name in devices:
            entry = hit.compatibility_for(name)
            if entry is None or entry.status is DeviceStatus.UNKNOWN:
                cells.append(Text("?", style="dim"))
                continue
            severity = _DEVICE_STATUS_SEVERITY[entry.status]
            cells.append(Text(status_icon(severity), style=status_style(severity)))
        formats = Text(",".join(hit.formats) or _DASH, style="" if hit.formats else "dim")
        if hit.derivative:
            formats.append("+", style="cyan")
            derivatives.append(f"+ {hit.repo_id} converts from {hit.derivative} (needs approval)")
        cells.append(formats)
        cells.append(Text(_compact_count(hit.downloads)))
        table.add_row(*cells)
    footer: list[RenderableType] = []
    if hinted:
        footer.append(
            Text("* architecture identified from GGUF metadata; no config.json.", style="dim")
        )
    footer.extend(Text(line, style="dim") for line in derivatives)
    footer.append(
        Text("✓ supported   ✗ unsupported   ! experimental   ? no registry claim", style="dim")
    )
    footer.extend(Text(note, style="dim") for note in report.notes)
    return Group(table, *footer)


def _compact_count(value: int | None) -> str:
    """Render a count as ``9.8M`` / ``896k`` / ``512``."""
    if value is None:
        return _DASH
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------
def probe_report(reports: Sequence[DeviceProbeReport], *, verbose: bool = False) -> RenderableType:
    """Render ``yonga probe`` (``docs/CLI_SPEC.md`` section 11).

    One block per device: the overall verdict, then a per-submodel read/compile
    row, then the smoke test, then the short explanation when it failed.

    A row says in one line what happened; the raw runtime text behind it (a
    native compiler dump runs to dozens of lines) is shown only with
    ``verbose``, and is always in ``--json`` and the job log.
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
            message = _probe_row_message(result, verbose=verbose)
            if message:
                detail = Text.assemble(detail, ("  ", ""), (message, "dim"))
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


def _probe_row_message(result: ProbeResult, *, verbose: bool) -> str:
    """The detail text of one probe row: a summary, or the raw text when verbose.

    A deferred raw compile is expected on the NPU for every dynamically shaped
    LLM and needs nothing from the user, so it reads as one neutral line rather
    than as the compiler dump it carries. A failure the registry recognised is
    named by its diagnostic's title. Anything else keeps its own text, which is
    all there is to go on. Blank lines are dropped either way.
    """
    raw = _without_blank_lines(result.message or "")
    if verbose:
        return raw
    if result.status is ProbeStatus.DEFERRED:
        return "the raw compile is not conclusive here; the GenAI pipeline decides"
    found = result.diagnostic
    if (
        result.status is ProbeStatus.FAILED
        and found is not None
        and found.category is not FailureCategory.UNKNOWN_RUNTIME_ERROR
    ):
        return found.title.strip() or raw
    return raw


def _without_blank_lines(text: str) -> str:
    """``text`` with its empty and whitespace-only lines removed."""
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


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
    runs = _count(result.runs, "run")
    warmups = _count(result.warmups, "warmup")
    suite = f"{result.suite}  ({runs}, {warmups})"
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


def _count(value: int, noun: str) -> str:
    """``1 run``, ``3 runs``, ``0 warmups``."""
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


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
        if resolution.profile_selection is not None and resolution.profile_selection.sentence:
            blocks.append(
                Text.assemble(
                    ("Chosen because  ", "bold"), (resolution.profile_selection.sentence, "dim")
                )
            )
        if resolution is explanation.chat and explanation.chat_profile_id is not None:
            tools = explanation.chat_tool_calling or "not declared; serve refuses tools"
            blocks.append(Text.assemble(("Tool calling  ", "bold"), (tools, "dim")))

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


def registry_profiles(rows: Sequence[Mapping[str, object]]) -> RenderableType:
    """Render ``yonga registry profiles``: one line per profile ``--profile`` accepts.

    ``rows`` are the command's JSON entries. The size band and the priority
    decide the automatic choice, so both are shown; a profile that is not
    active has its status in front of its summary.
    """
    table = Table(box=SIMPLE, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Profile", overflow="fold")
    table.add_column("Device", no_wrap=True)
    table.add_column("Weights", overflow="fold")
    table.add_column("Size band", overflow="fold")
    table.add_column("Prio", justify="right", no_wrap=True)
    table.add_column("Summary", overflow="fold")
    for row in rows:
        status = str(row.get("status") or "")
        summary = _text(row.get("summary"), style="dim")
        if status and status != "active":
            summary = Text.assemble((status, "yellow"), ("  ", ""), summary)
        table.add_row(
            Text(str(row["id"]), style="bold"),
            str(row["device"]),
            str(row["weights"]),
            _text(row.get("band")),
            str(row["priority"]),
            summary,
        )
    return table


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


def publication_plan(plan: PublishPlan) -> RenderableType:
    """Render what ``yonga publish`` will upload, and on what evidence, before it does."""
    body = _grid()
    visibility = "private" if plan.private else "PUBLIC"
    state = "existing, files replaced" if plan.repo_exists else "new"
    body.add_row(
        Text("Destination"),
        Text.assemble((plan.url, "cyan"), (f"  {visibility}, {state}", "bold")),
    )
    body.add_row(Text("Artifact"), _text(plan.artifact_id))
    for assessment in plan.licenses:
        label = "License" if assessment.role == "source" else "Requested model license"
        body.add_row(
            Text(label),
            Text.assemble(
                (assessment.license or "not declared", "bold"),
                (f"  {assessment.classification.value}  {assessment.repo_id}", "dim"),
            ),
        )
    for evidence in plan.devices:
        style = {"ok": "green", "failed": "red", "inconclusive": "yellow"}.get(
            evidence.status, "dim"
        )
        word = {
            "ok": "works",
            "failed": "failed",
            "inconclusive": "inconclusive",
            "not_tested": "not tested",
        }.get(evidence.status, evidence.status)
        body.add_row(
            Text(f"Device {evidence.device}"),
            Text.assemble(
                (word, style), (f"  {evidence.detail}" if evidence.detail else "", "dim")
            ),
        )
    body.add_row(
        Text("Upload"),
        _text(f"{len(plan.files)} files, {human_bytes(plan.upload_bytes)}"),
    )
    blocks: list[RenderableType] = [body]
    blocks.extend(_bullet_block("Repairs disclosed", plan.repairs, style="cyan"))
    blocks.extend(_bullet_block("Warnings", plan.warnings, style="yellow"))
    return Panel(
        Group(*blocks),
        title="Publication plan",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


def evaluation_result(result: EvaluationResult) -> RenderableType:
    """Render one ``yonga eval`` comparison: the totals, then every passage."""
    summary = result.summary
    head = _grid()
    head.add_row(
        Text("Candidate"),
        Text.assemble(
            (result.candidate.artifact_id, "bold"), (f"  {result.candidate.profile}", "dim")
        ),
    )
    head.add_row(
        Text("Reference"),
        Text.assemble(
            (result.reference.artifact_id, "bold"), (f"  {result.reference.profile}", "dim")
        ),
    )
    precision = result.runtime_properties.get("INFERENCE_PRECISION_HINT", "device default")
    head.add_row(Text("Device"), _text(f"{result.device} (inference precision {precision})"))
    head.add_row(Text("Corpus"), _text(f"{result.corpus_id}, {summary.positions} scored tokens"))
    head.add_row(Text("Mean KL"), Text(f"{summary.kl_mean:.4f} nats", style="bold"))
    agreement = f"{100 * summary.top1_agreement:.2f} %"
    head.add_row(Text("Top-1 agreement"), Text(agreement, style="bold"))
    head.add_row(
        Text("Perplexity"),
        _text(
            f"{summary.candidate_perplexity:.3f} against {summary.reference_perplexity:.3f} "
            f"({summary.perplexity_change_pct:+.2f} %)"
        ),
    )
    table = Table(box=SIMPLE, show_edge=False, pad_edge=False)
    for column in ("Passage", "Tokens", "Mean KL", "Max KL", "Top-1", "PPL ref", "PPL cand"):
        table.add_column(column, justify="left" if column == "Passage" else "right")
    for passage in result.passages:
        table.add_row(
            passage.passage_id,
            str(passage.positions),
            f"{passage.kl_mean:.4f}",
            f"{passage.kl_max:.3f}",
            f"{100 * passage.top1_agreement:.1f} %",
            f"{passage.reference_perplexity:.2f}",
            f"{passage.candidate_perplexity:.2f}",
        )
    notes = [Text(note, style="dim") for note in result.notes]
    return Panel(
        Group(head, Text(""), table, *notes),
        title="Quality evaluation",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


# ----------------------------------------------------------------------
# logs
# ----------------------------------------------------------------------
#: Severity shown for each job status in ``yonga logs``.
_JOB_STATUS_SEVERITY: dict[JobStatus, Severity] = {
    JobStatus.OK: Severity.OK,
    JobStatus.FAILED: Severity.ERROR,
    JobStatus.CANCELLED: Severity.WARNING,
    JobStatus.RUNNING: Severity.INFO,
    JobStatus.UNFINISHED: Severity.WARNING,
    JobStatus.UNKNOWN: Severity.UNKNOWN,
}

#: Job id characters shown in the index; ``yonga logs`` accepts any unique prefix.
_JOB_ID_PREFIX = 8

#: Narrowest the model column is squeezed to before whole columns are dropped.
_JOB_MODEL_MIN_WIDTH = 16

#: The index's columns, in order, as ``(key, header)``.
_JOB_COLUMNS: tuple[tuple[str, str], ...] = (
    ("job", "job"),
    ("started", "started"),
    ("command", "command"),
    ("model", "model"),
    ("device", "device"),
    ("status", "status"),
    ("took", "took"),
    ("log", "log"),
)

#: Columns given up, in this order, when the index does not fit the terminal.
_JOB_DROP_ORDER: tuple[str, ...] = ("log", "device", "took", "started")

#: Columns whose values are right-aligned.
_JOB_NUMERIC: frozenset[str] = frozenset({"took", "log"})


def utc_offset(moment: datetime) -> str:
    """Name the local zone ``moment`` is shown in, as ``UTC+03:00``."""
    offset = moment.astimezone().strftime("%z") or "+0000"
    return f"UTC{offset[:3]}:{offset[3:]}"


def local_clock(stamp: object) -> str:
    """Turn a job log record's UTC ``timestamp`` into local ``HH:MM:SS.mmm``.

    A value that does not parse is shown as the log wrote it, cut to the clock.
    """
    text = str(stamp or "")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text[11:23]
    if moment.tzinfo is None:
        return text[11:23]
    local = moment.astimezone()
    return local.strftime("%H:%M:%S.") + f"{local.microsecond // 1000:03d}"


def _job_status_text(summary: JobSummary) -> Text:
    """Render a job's outcome in the colour of its severity."""
    return _status_text(_JOB_STATUS_SEVERITY[summary.status], summary.status.value)


def _job_model(summary: JobSummary) -> str | None:
    """The model reference as the index shows it: a path by its last component."""
    model = summary.model
    if model and model.startswith(("/", "~", ".")):
        return Path(model).name or model
    return model


def _job_cells(summary: JobSummary, *, zone: str | None) -> dict[str, Text]:
    """Render one job as a cell per column key."""
    started = _timestamp(summary.started_at)
    if summary.started_at is not None and zone is None:
        started = f"{started} {utc_offset(summary.started_at)}"
    duration = summary.duration_seconds
    return {
        "job": Text(summary.job_id[:_JOB_ID_PREFIX], style="bold"),
        "started": _text(started),
        "command": _text(summary.command),
        "model": _text(_job_model(summary)),
        "device": _text(summary.device),
        "status": _job_status_text(summary),
        "took": _text(human_duration(duration) if duration is not None else None),
        "log": _text(human_bytes(summary.size_bytes)),
    }


def _job_table(
    rows: Sequence[dict[str, Text]],
    *,
    dropped: frozenset[str],
    model_width: int | None,
    caption: str | None,
) -> Table:
    """Build the ``yonga logs`` index for one layout."""
    table = Table(
        box=SIMPLE,
        show_edge=False,
        pad_edge=False,
        header_style="bold",
        padding=(0, 1, 0, 0),
        caption=caption,
        caption_justify="left",
        caption_style="dim",
    )
    keys = [key for key, _ in _JOB_COLUMNS if key not in dropped]
    headers = dict(_JOB_COLUMNS)
    for key in keys:
        table.add_column(
            headers[key],
            no_wrap=True,
            overflow="ellipsis",
            justify="right" if key in _JOB_NUMERIC else "left",
            max_width=model_width if key == "model" else None,
        )
    for cells in rows:
        table.add_row(*(cells[key] for key in keys))
    return table


class _JobList:
    """The ``yonga logs`` index, laid out for the terminal it is printed on.

    The model column gives way first, down to :data:`_JOB_MODEL_MIN_WIDTH`;
    after that whole columns are dropped in :data:`_JOB_DROP_ORDER`, so the job
    prefix, the command and the outcome are never reduced to an ellipsis.
    """

    def __init__(self, summaries: Sequence[JobSummary]) -> None:
        """Render the cells once; the layout is chosen at render time."""
        offsets = {
            utc_offset(summary.started_at)
            for summary in summaries
            if summary.started_at is not None
        }
        zone = next(iter(offsets)) if len(offsets) == 1 else None
        self._caption = f"Start times are local, {zone}." if zone else None
        self._rows = [_job_cells(summary, zone=zone) for summary in summaries]
        self._model_width = max([len("model"), *(row["model"].cell_len for row in self._rows)])

    def table_for(self, console: Console, options: ConsoleOptions) -> Table:
        """Return the widest layout that fits ``options``."""
        wide = options.update_width(_LIST_MEASURE_WIDTH)
        dropped: set[str] = set()
        for key in (None, *_JOB_DROP_ORDER):
            if key is not None:
                dropped.add(key)
            candidate = self._layout(frozenset(dropped), None)
            excess = Measurement.get(console, wide, candidate).maximum - options.max_width
            if excess <= 0:
                return candidate
            squeezed = self._model_width - excess
            if squeezed >= _JOB_MODEL_MIN_WIDTH:
                return self._layout(frozenset(dropped), squeezed)
        return self._layout(frozenset(dropped), _JOB_MODEL_MIN_WIDTH)

    def _layout(self, dropped: frozenset[str], model_width: int | None) -> Table:
        """One layout of the index."""
        caption = None if "started" in dropped else self._caption
        return _job_table(self._rows, dropped=dropped, model_width=model_width, caption=caption)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the chosen layout."""
        yield self.table_for(console, options)

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        """Measure the layout that would be chosen for ``options``."""
        return Measurement.get(console, options, self.table_for(console, options))


def job_list(summaries: Sequence[JobSummary]) -> RenderableType:
    """Render the ``yonga logs`` index: which job is which, and how each ended.

    Start times are local. The zone is named once, under the table, when every
    row shares it, and on each row otherwise (a list spanning a daylight saving
    change).
    """
    return _JobList(summaries)


def job_heading(summary: JobSummary) -> RenderableType:
    """Render the lines above one job's records: what ran, how it ended, the zone."""
    lines: list[RenderableType] = []
    facts = [value for value in (summary.command, summary.model, summary.device) if value]
    status = _job_status_text(summary)
    if summary.code:
        status.append(f" ({summary.code})", style="dim")
    duration = summary.duration_seconds
    if duration is not None:
        status.append(f", took {human_duration(duration)}", style="dim")
    lines.append(Text.assemble(("  ", ""), (" ".join(facts) + "  " if facts else "", ""), status))
    if summary.started_at is not None:
        local = summary.started_at.astimezone()
        lines.append(
            Text(
                f"  Started {local.strftime('%Y-%m-%d %H:%M:%S')} {utc_offset(local)}; "
                "the times below are local.",
                style="dim",
            )
        )
    return Group(*lines)
