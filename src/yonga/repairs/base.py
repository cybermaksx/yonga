"""The repair framework: typed, narrowly scoped, registry-selected file fixes.

A *repair* is a small, exactly matched, reversible transformation applied to files
inside a tool-owned artifact or staging directory. Repairs exist because upstream
conversion tooling occasionally emits artifacts that the runtime of the very same
release cannot load; the alternative — telling the user to hand-edit generated XML —
is worse.

Two rules shape everything in this module.

**The registry is data, never code.** A registry repair rule names an
*implementation* (``repair.implementation: tokenizer_string_precision``) and the
files and substring that must be present for it to be considered. It never carries
a transformation. :data:`REPAIR_IMPLEMENTATIONS` is the only place a name becomes
behaviour, so a hostile or merely careless registry override can at worst select an
existing, reviewed, tested repair — it can never describe a new one. This is why
:func:`get_repair` raises on an unknown name instead of degrading to a no-op.

**Repairs only ever touch tool-owned directories.** ``SPEC.md`` section 18.4 forbids
patching the shared Hugging Face cache: it is shared state that other tools and other
copies of this tool depend on, and a repaired file there would silently poison every
future export. :meth:`RepairContext.assert_tool_owned` enforces this on every path a
repair is about to read or write, and additionally refuses any path that resolves
inside a Hugging Face cache root even when a caller has mistakenly declared it owned.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from yonga.application.progress import NullSink, ProgressEvent, ProgressKind, ProgressSink
from yonga.domain.artifacts import RepairRecord
from yonga.domain.diagnostics import (
    FailureCategory,
    RepairError,
    Severity,
    diagnostic,
)
from yonga.util.jsonlog import JobLogger, null_logger
from yonga.util.versions import installed_version, satisfies

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.application.jobs import JobContext

__all__ = [
    "BUILTIN_REPAIR_MODULES",
    "REPAIR_IMPLEMENTATIONS",
    "REPAIR_VERSION_DISTRIBUTIONS",
    "ConfirmCallback",
    "Repair",
    "RepairContext",
    "RepairMatch",
    "RepairResult",
    "RepairRuleView",
    "RepairSafety",
    "apply_repairs",
    "get_repair",
    "huggingface_cache_roots",
    "installed_repair_versions",
    "known_implementations",
    "register_repair",
]

#: Modules that define built-in repair implementations. They are imported lazily by
#: :func:`get_repair` / :func:`known_implementations` so that importing this module
#: alone never pulls in an implementation's dependencies.
BUILTIN_REPAIR_MODULES: tuple[str, ...] = ("yonga.repairs.tokenizer_string_precision",)

#: Distribution names behind the version constraints a repair rule may carry. These
#: are the same two packages ``yonga.registry.resolver.VERSION_DISTRIBUTIONS`` names
#: for ``openvino`` / ``openvino_tokenizers``, so both layers judge one rule's
#: ``constraints`` against identical facts.
REPAIR_VERSION_DISTRIBUTIONS: Mapping[str, str] = {
    "openvino": "openvino",
    "tokenizers": "openvino-tokenizers",
}


def installed_repair_versions() -> dict[str, str | None]:
    """Return the installed versions the repair version gate is evaluated against.

    Only distribution metadata is read, so this never imports OpenVINO and stays
    safe on a machine that has none.
    """
    return {key: installed_version(dist) for key, dist in REPAIR_VERSION_DISTRIBUTIONS.items()}


#: Environment variables that may relocate the Hugging Face cache.
_HF_CACHE_ENV_KEYS: tuple[str, ...] = (
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
)


def _text(value: object, default: str = "") -> str:
    """Render a registry value as plain text, unwrapping enum members.

    ``str(SomeStrEnum.MEMBER)`` yields ``"SomeStrEnum.MEMBER"``, not the value, so
    registry enums must be unwrapped explicitly or a status comparison silently fails.
    """
    if value is None:
        return default
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


class RepairSafety(str, Enum):
    """How much authority a repair has to run on its own.

    The registry rule carries this value; the orchestration in :func:`apply_repairs`
    is the only place it is interpreted, so every caller gets the same policy.
    """

    AUTOMATIC = "automatic"
    """Exactly matched and provably narrow: applied without asking."""

    CONFIRM = "confirm"
    """Applied only when the caller's confirmation callback returns ``True``."""

    MANUAL_ONLY = "manual_only"
    """Never applied automatically; reported so the user can act."""

    @classmethod
    def parse(cls, raw: object, default: RepairSafety | None = None) -> RepairSafety:
        """Coerce a registry value into a safety level, falling back to ``default``.

        An unrecognised value degrades to the *most* restrictive interpretation
        rather than raising: a typo in a user override must never be able to make a
        repair run more eagerly than intended.
        """
        if isinstance(raw, cls):
            return raw
        text = _text(raw).strip().lower()
        if text:
            with suppress(ValueError):
                return cls(text)
            alias = _SAFETY_ALIASES.get(text)
            if alias is not None:
                return alias
        return default if default is not None else cls.MANUAL_ONLY


#: Spellings other layers use for a safety level, mapped onto this module's vocabulary.
#: ``yonga.registry.schema.RepairSafety`` spells the strictest level ``manual``.
_SAFETY_ALIASES: Mapping[str, RepairSafety] = {
    "manual": RepairSafety.MANUAL_ONLY,
    "manual-only": RepairSafety.MANUAL_ONLY,
    "never": RepairSafety.MANUAL_ONLY,
    "auto": RepairSafety.AUTOMATIC,
    "confirmation": RepairSafety.CONFIRM,
    "ask": RepairSafety.CONFIRM,
}

#: Rule statuses that take a rule out of service. Everything else (``active``,
#: ``experimental``, an empty string) is usable, matching ``RuleStatus.is_usable``.
_INACTIVE_STATUSES: frozenset[str] = frozenset(
    {"deprecated", "withdrawn", "disabled", "inactive", "retired", "superseded"}
)


def huggingface_cache_roots(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Return every directory that may hold the shared Hugging Face cache.

    Repairs must never write inside these. The list is deliberately generous: a
    false positive costs a refused repair (recoverable), a false negative corrupts
    shared state for every other tool on the machine.
    """
    source = os.environ if env is None else env
    roots: list[Path] = []

    def _add(raw: str | None, *suffix: str) -> None:
        if not raw:
            return
        with suppress(OSError, ValueError):
            roots.append(Path(raw).expanduser().joinpath(*suffix).resolve())

    for key in _HF_CACHE_ENV_KEYS:
        _add(source.get(key))
    _add(source.get("HF_HOME"))
    _add(source.get("HF_HOME"), "hub")
    _add(source.get("XDG_CACHE_HOME"), "huggingface")
    with suppress(OSError, RuntimeError):
        roots.append((Path.home() / ".cache" / "huggingface").resolve())
    # Deduplicate while keeping a deterministic order.
    seen: dict[Path, None] = {}
    for root in roots:
        seen.setdefault(root, None)
    return tuple(seen)


@functools.lru_cache(maxsize=1)
def _shared_null_logger() -> JobLogger:
    """Return the process-wide no-op logger used by contexts built without one.

    Cached because a :class:`JobLogger` holds an open file handle; one per context
    would leak a descriptor for every repair considered.
    """
    return null_logger()


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is ``root`` or lives underneath it."""
    try:
        return path == root or path.is_relative_to(root)
    except (OSError, ValueError):  # pragma: no cover - defensive
        return False


# --- context ---------------------------------------------------------------------------


@dataclass
class RepairContext:
    """Everything a repair needs to decide and to justify what it did.

    ``owned_roots`` is the security boundary. When it is left empty the artifact
    directory itself is the only root, which is the correct default: a repair
    invoked on one artifact has no business reaching outside it.
    """

    artifact_dir: Path
    """The tool-owned artifact (or staging) directory being repaired."""

    owned_roots: tuple[Path, ...] = ()
    """Extra roots (staging root, artifacts root) that count as tool-owned."""

    openvino_version: str | None = None
    """Installed OpenVINO version, used to evaluate a rule's version constraint."""

    tokenizers_version: str | None = None
    """Installed ``openvino-tokenizers`` version, recorded in the repair detail."""

    rule: RepairRuleView | None = None
    """The registry rule that selected the repair, when there is one."""

    logger: JobLogger = field(default_factory=_shared_null_logger)
    """Structured job logger; repairs log every file they touch."""

    progress: ProgressSink = field(default_factory=NullSink)
    """Where :attr:`ProgressKind.REPAIR` events go."""

    dry_run: bool = False
    """When true, detect and compute hashes but write nothing."""

    confirm: ConfirmCallback | None = None
    """Callback consulted for :attr:`RepairSafety.CONFIRM` repairs."""

    env: Mapping[str, str] | None = None
    """Environment used to locate the Hugging Face cache; defaults to ``os.environ``."""

    @classmethod
    def for_job(
        cls,
        ctx: JobContext,
        artifact_dir: Path,
        *,
        openvino_version: str | None = None,
        tokenizers_version: str | None = None,
        rule: RepairRuleView | None = None,
        dry_run: bool = False,
        confirm: ConfirmCallback | None = None,
    ) -> RepairContext:
        """Build a context from a :class:`~yonga.application.jobs.JobContext`.

        The job's staging and artifacts roots are added as tool-owned roots so a
        repair may legitimately reach a sibling directory of the artifact inside the
        same job, and nothing else.

        A version left as ``None`` is filled in from installed distribution metadata
        rather than being treated as unknown. A rule that reached this point was
        already selected by :meth:`yonga.registry.RegistryResolver.repairs_for`,
        which judged the same ``constraints`` against the same installed versions;
        re-deriving them here keeps the two gates from disagreeing and silently
        dropping every repair when the orchestration forgets to pass them.
        """
        paths = ctx.paths
        detected = installed_repair_versions()
        return cls(
            artifact_dir=artifact_dir,
            owned_roots=(paths.staging_dir, paths.artifacts_dir, ctx.staging_dir),
            openvino_version=(
                openvino_version if openvino_version is not None else detected["openvino"]
            ),
            tokenizers_version=(
                tokenizers_version if tokenizers_version is not None else detected["tokenizers"]
            ),
            rule=rule,
            logger=ctx.logger,
            progress=ctx.progress,
            dry_run=dry_run,
            confirm=confirm,
        )

    def roots(self) -> tuple[Path, ...]:
        """Return the resolved set of directories this context may write inside."""
        declared = [self.artifact_dir, *self.owned_roots]
        resolved: dict[Path, None] = {}
        for root in declared:
            with suppress(OSError, ValueError):
                resolved.setdefault(root.expanduser().resolve(), None)
        return tuple(resolved)

    def assert_tool_owned(self, path: Path) -> Path:
        """Return ``path`` resolved, or raise :class:`RepairError` if it is not ours.

        Two independent checks must both pass: the path must be inside a declared
        tool-owned root, and it must not be inside any Hugging Face cache root. The
        second check is not redundant — it catches an artifact directory that a
        caller wrongly pointed at the cache.
        """
        try:
            resolved = path.expanduser().resolve()
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            raise _error(
                "repair.path_unresolvable",
                f"The repair target path could not be resolved: {path}",
                remediation=["Re-run preparation so the artifact directory is recreated."],
            ) from exc

        for cache_root in huggingface_cache_roots(self.env):
            if _is_within(resolved, cache_root):
                raise _error(
                    "repair.path_in_hf_cache",
                    (
                        f"Refusing to repair {resolved}: it is inside the shared Hugging Face "
                        f"cache ({cache_root}). Repairs only ever modify files this tool owns."
                    ),
                    remediation=[
                        "Repairs run on the converted artifact, never on the downloaded source.",
                        "Re-run preparation so the export lands in a tool-owned staging directory.",
                    ],
                    path=str(resolved),
                    cache_root=str(cache_root),
                )

        roots = self.roots()
        if not any(_is_within(resolved, root) for root in roots):
            raise _error(
                "repair.path_not_tool_owned",
                (
                    f"Refusing to repair {resolved}: it is outside every tool-owned "
                    f"directory ({', '.join(str(root) for root in roots) or 'none declared'})."
                ),
                remediation=[
                    "Point the repair at the artifact or staging directory this job created.",
                ],
                path=str(resolved),
            )
        return resolved

    def relative_name(self, path: Path) -> str:
        """Return ``path`` relative to the artifact directory, as a POSIX string."""
        with suppress(OSError, ValueError):
            resolved = path.expanduser().resolve()
            base = self.artifact_dir.expanduser().resolve()
            if _is_within(resolved, base):
                return resolved.relative_to(base).as_posix()
        return path.name

    def emit(self, message: str, *, status: str | None = None, **data: object) -> None:
        """Emit a repair progress event. Never raises."""
        event = ProgressEvent(
            kind=ProgressKind.REPAIR,
            message=message,
            status=status,
            severity=Severity.INFO,
            data=dict(data),
        )
        with suppress(Exception):
            self.progress.emit(event)


# --- match / result --------------------------------------------------------------------


class RepairMatch(BaseModel):
    """What a repair found, before anything has been changed.

    A match is evidence, not a decision: :func:`apply_repairs` still has to clear it
    through the safety policy. ``counts`` exists so the post-repair verification can
    assert the exact number of substitutions instead of trusting a regex twice.
    """

    model_config = ConfigDict(extra="forbid")

    repair_id: str
    files: tuple[Path, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)
    details: dict[str, object] = Field(default_factory=dict)
    rule_id: str | None = None

    @property
    def total_occurrences(self) -> int:
        """Total number of matched occurrences across every matched file."""
        return sum(self.counts.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Relative names of the matched files, in the order they were matched."""
        return tuple(self.counts)


class RepairResult(BaseModel):
    """The outcome of applying (or declining to apply) a repair.

    This is the in-process value; :meth:`to_record` narrows it to the manifest shape
    defined by :class:`~yonga.domain.artifacts.RepairRecord`.
    """

    model_config = ConfigDict(extra="forbid")

    repair_id: str
    description: str = ""
    safety: RepairSafety = RepairSafety.MANUAL_ONLY
    applied: bool = False
    files: tuple[str, ...] = ()
    before_hashes: dict[str, str] = Field(default_factory=dict)
    after_hashes: dict[str, str] = Field(default_factory=dict)
    backup_dir: Path | None = None
    detail: dict[str, object] = Field(default_factory=dict)
    rule_id: str | None = None
    applied_at: datetime | None = None
    artifact_dir: Path | None = None

    def with_detail(self, **extra: object) -> RepairResult:
        """Return a copy with ``extra`` merged into :attr:`detail`."""
        merged = dict(self.detail)
        merged.update(extra)
        return self.model_copy(update={"detail": merged})

    def to_record(self) -> RepairRecord:
        """Build the manifest record for this repair.

        ``backup_dir`` is stored relative to the artifact directory when it lives
        inside it, so a manifest stays valid if the artifact is moved.
        """
        backup: str | None = None
        if self.backup_dir is not None:
            backup = str(self.backup_dir)
            if self.artifact_dir is not None:
                with suppress(OSError, ValueError):
                    resolved = self.backup_dir.expanduser().resolve()
                    base = self.artifact_dir.expanduser().resolve()
                    if _is_within(resolved, base):
                        backup = resolved.relative_to(base).as_posix()
        return RepairRecord(
            repair_id=self.repair_id,
            description=self.description,
            applied=self.applied,
            safety=self.safety.value,
            files=list(self.files),
            before_hashes=dict(self.before_hashes),
            after_hashes=dict(self.after_hashes),
            backup_dir=backup,
            detail=dict(self.detail),
            rule_id=self.rule_id,
            applied_at=self.applied_at,
        )


# --- registry rule view ------------------------------------------------------------------


def _field(source: object, *names: str) -> object:
    """Read the first present field from a mapping or an object, else ``None``."""
    for name in names:
        if isinstance(source, Mapping):
            if name in source:
                return source[name]
            continue
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def _string_tuple(value: object) -> tuple[str, ...]:
    """Coerce a registry value into a tuple of non-empty strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


class RepairRuleView(BaseModel):
    """A registry repair rule reduced to the fields the repair framework needs.

    The registry module owns the full ``RepairRule`` schema. This view is built from
    it structurally (:meth:`from_rule` also accepts a plain mapping) so the repair
    package stays usable — and testable — without importing the registry, and so a
    later schema addition cannot break repairs.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    implementation: str
    safety: RepairSafety | None = None
    """The rule's explicit safety level; ``None`` means "use the implementation's"."""
    status: str = "active"
    summary: str = ""
    files: tuple[str, ...] = ()
    contains: str | None = None
    openvino: str | None = None
    tokenizers: str | None = None
    requires_tokenizer_read: bool = False
    postconditions: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        """Whether the rule may take part in repair resolution.

        Expressed as a deny-list so an unfamiliar-but-live status (``experimental``)
        keeps working rather than silently disabling a repair.
        """
        return self.status.strip().lower() not in _INACTIVE_STATUSES

    def effective_safety(self, default: RepairSafety) -> RepairSafety:
        """Return the rule's safety level, falling back to the implementation's.

        An explicit ``manual_only`` in registry data always wins over an
        implementation that would otherwise run automatically: the registry may only
        ever make a repair *less* eager, never more.
        """
        return self.safety if self.safety is not None else default

    def version_matches(self, *, openvino: str | None, tokenizers: str | None) -> bool:
        """Whether the installed versions satisfy the rule's constraints.

        A rule with no constraint matches everything. A constraint with no installed
        version does *not* match: a repair gated on ``openvino >=2026.4`` must not run
        when we cannot tell what is installed.
        """
        return satisfies(openvino, self.openvino) and satisfies(tokenizers, self.tokenizers)

    @classmethod
    def from_rule(cls, rule: object) -> RepairRuleView:
        """Build a view from a registry ``RepairRule``, a mapping, or another view.

        Accepts both spellings seen in the specifications: ``match.files`` from
        ``docs/REGISTRY_SPEC.md`` section 7 and ``match.file_names`` from
        ``registry/examples/qwen3_5.yaml``.
        """
        if isinstance(rule, cls):
            return rule

        match = _field(rule, "match") or {}
        spec = _field(rule, "repair") or {}
        validation = _field(rule, "validation") or {}
        constraints = _field(rule, "constraints") or {}

        implementation = _text(_field(spec, "implementation", "impl")).strip()
        if not implementation:
            raise _error(
                "repair.rule_missing_implementation",
                (
                    "A registry repair rule does not name an implementation. Registry data "
                    "selects a reviewed implementation; it cannot describe a transformation."
                ),
                remediation=[
                    "Add `repair.implementation: <name>` to the rule.",
                    f"Known implementations: {', '.join(sorted(known_implementations()))}.",
                ],
            )

        rule_id = _text(_field(rule, "id", "rule_id")).strip()
        raw_safety = _field(spec, "safety")
        contains = _field(match, "contains", "contains_text")
        # Version gates live on ``match`` in docs/REGISTRY_SPEC.md section 7 and in the
        # shared ``constraints`` mapping in yonga.registry.schema.BaseRule.
        openvino = _field(match, "openvino", "openvino_version") or _field(constraints, "openvino")
        tokenizers = _field(match, "tokenizers", "openvino_tokenizers") or _field(
            constraints, "openvino_tokenizers", "tokenizers"
        )
        requires = _field(validation, "requires_tokenizer_read")
        postconditions = _field(validation, "postconditions") or _field(rule, "postconditions")

        return cls(
            id=rule_id or implementation,
            implementation=implementation,
            safety=RepairSafety.parse(raw_safety) if raw_safety is not None else None,
            status=_text(_field(rule, "status"), "active"),
            summary=_text(_field(rule, "summary", "description")),
            files=_string_tuple(_field(match, "files", "file_names", "filenames")),
            contains=_text(contains) or None,
            openvino=_text(openvino) or None,
            tokenizers=_text(tokenizers) or None,
            requires_tokenizer_read=bool(requires),
            postconditions=_string_tuple(postconditions),
        )


ConfirmCallback = Callable[["Repair", RepairMatch], bool]
"""Asks the user whether a :attr:`RepairSafety.CONFIRM` repair may run."""


# --- the repair interface ------------------------------------------------------------------


class Repair(ABC):
    """One narrowly scoped, exactly matched transformation of artifact files.

    Implementations must be idempotent: :meth:`detect` returns ``None`` once the
    artifact is already correct, and :meth:`apply` is a no-op when handed a stale
    match. Every implementation declares its own default file scope rather than
    accepting arbitrary paths from registry data.
    """

    #: Stable, dotted-or-dashed repair identifier recorded in the manifest.
    id: ClassVar[str] = ""
    #: The name registry data uses in ``repair.implementation``.
    implementation_name: ClassVar[str] = ""
    #: One-line human description, copied into the manifest record.
    description: ClassVar[str] = ""
    #: Default authority level when no registry rule overrides it.
    safety: ClassVar[RepairSafety] = RepairSafety.MANUAL_ONLY

    @abstractmethod
    def detect(self, artifact: Path, context: RepairContext) -> RepairMatch | None:
        """Return a match when this repair applies to ``artifact``, else ``None``.

        Must not modify anything, and must return ``None`` for an already repaired
        artifact so that re-running preparation is a no-op.
        """

    @abstractmethod
    def apply(self, match: RepairMatch, artifact: Path, context: RepairContext) -> RepairResult:
        """Apply the transformation described by ``match``.

        Must back up the original bytes, write atomically, and record the pre- and
        post-repair SHA-256 of every file it touched.
        """

    @abstractmethod
    def validate(self, result: RepairResult, context: RepairContext) -> RepairResult:
        """Verify the post-conditions and return ``result`` enriched with evidence.

        Raises :class:`~yonga.domain.diagnostics.RepairError` when a post-condition
        that does not depend on optional third-party packages has failed.
        """


# --- implementation registry -----------------------------------------------------------


REPAIR_IMPLEMENTATIONS: dict[str, type[Repair]] = {}
"""The only mapping from a registry name to executable behaviour.

Keyed by both the repair id and the implementation name, so a rule may use either.
"""


def register_repair(cls: type[Repair]) -> type[Repair]:
    """Class decorator registering a repair under its id and implementation name."""
    keys = [key for key in (cls.id, cls.implementation_name) if key]
    if not keys:
        raise _error(
            "repair.implementation_unnamed",
            f"{cls.__name__} declares neither `id` nor `implementation_name`.",
        )
    for key in keys:
        existing = REPAIR_IMPLEMENTATIONS.get(key)
        if existing is not None and existing is not cls:
            raise _error(
                "repair.implementation_conflict",
                f"Two repair implementations claim the name {key!r}: "
                f"{existing.__name__} and {cls.__name__}.",
            )
        REPAIR_IMPLEMENTATIONS[key] = cls
    return cls


def _load_builtin_implementations() -> None:
    """Import the built-in repair modules so they self-register. Idempotent."""
    for module in BUILTIN_REPAIR_MODULES:
        if module not in sys.modules:
            importlib.import_module(module)


def known_implementations() -> frozenset[str]:
    """Return every name :func:`get_repair` accepts."""
    _load_builtin_implementations()
    return frozenset(REPAIR_IMPLEMENTATIONS)


def get_repair(name: str) -> Repair:
    """Instantiate the repair registered under ``name``.

    Raises :class:`~yonga.domain.diagnostics.RepairError` for an unknown name. A
    registry override naming a non-existent implementation is a configuration error
    the user must see, never something to silently skip.
    """
    _load_builtin_implementations()
    key = name.strip()
    cls = REPAIR_IMPLEMENTATIONS.get(key)
    if cls is None:
        known = ", ".join(sorted(REPAIR_IMPLEMENTATIONS)) or "none"
        raise _error(
            "repair.unknown_implementation",
            f"No repair implementation is registered under {name!r}.",
            remediation=[
                f"Known implementations: {known}.",
                "Registry data may only select a built-in repair implementation.",
            ],
            requested=key,
        )
    return cls()


# --- orchestration -----------------------------------------------------------------------


def apply_repairs(
    artifact: Path,
    rules: Sequence[object],
    context: RepairContext,
    *,
    confirm: ConfirmCallback | None = None,
) -> list[RepairRecord]:
    """Run ``detect -> confirm policy -> apply -> validate`` for each registry rule.

    ``rules`` are registry repair rules (or mappings, or :class:`RepairRuleView`
    instances). Rules that are inactive, version-gated out, or that detect nothing
    produce no record at all — a manifest should say what happened to this artifact,
    not enumerate the whole registry. A rule that matches but is not applied *does*
    produce a record with ``applied=False`` and a reason, because the user needs to
    know the artifact is knowingly left unrepaired.
    """
    decide = confirm if confirm is not None else context.confirm
    records: list[RepairRecord] = []

    for raw_rule in rules:
        view = RepairRuleView.from_rule(raw_rule)
        if not view.active:
            context.logger.debug("repair.rule_inactive", f"skipping rule {view.id}", rule=view.id)
            continue
        if not view.version_matches(
            openvino=context.openvino_version,
            tokenizers=context.tokenizers_version,
        ):
            context.logger.info(
                "repair.rule_version_mismatch",
                f"rule {view.id} does not apply to this OpenVINO build",
                rule=view.id,
                openvino=context.openvino_version or "unknown",
                constraint=view.openvino or "*",
            )
            continue

        repair = get_repair(view.implementation)
        rule_context = replace(context, rule=view)
        match = repair.detect(artifact, rule_context)
        if match is None:
            context.logger.debug(
                "repair.not_detected",
                f"{view.id}: nothing to repair",
                rule=view.id,
                repair=repair.id,
            )
            continue

        context.logger.info(
            "repair.detected",
            f"{repair.id}: {match.total_occurrences} occurrence(s) in {len(match.counts)} file(s)",
            rule=view.id,
            repair=repair.id,
            files=list(match.names),
        )
        rule_context.emit(
            f"Detected {repair.id} in {len(match.counts)} file(s)",
            status="detected",
            repair=repair.id,
            rule=view.id,
        )

        safety = view.effective_safety(repair.safety)
        blocked = _blocked_reason(safety, match, repair, decide)
        if blocked is not None:
            records.append(
                _declined(repair, match, view, safety, blocked, rule_context).to_record()
            )
            context.logger.warning(
                "repair.not_applied",
                f"{repair.id} was not applied: {blocked}",
                rule=view.id,
                repair=repair.id,
            )
            continue

        result = repair.apply(match, artifact, rule_context)
        result = repair.validate(result, rule_context)
        # Policy, not the implementation, owns the authority level that gets recorded.
        result = result.model_copy(update={"safety": safety})
        records.append(result.to_record())
        context.logger.info(
            "repair.applied" if result.applied else "repair.noop",
            f"{repair.id}: {'applied' if result.applied else 'no change needed'}",
            rule=view.id,
            repair=repair.id,
            files=list(result.files),
        )
        rule_context.emit(
            f"{repair.id}: {'applied' if result.applied else 'no change needed'}",
            status="applied" if result.applied else "noop",
            repair=repair.id,
        )

    return records


def _blocked_reason(
    safety: RepairSafety,
    match: RepairMatch,
    repair: Repair,
    decide: ConfirmCallback | None,
) -> str | None:
    """Return why this repair must not run now, or ``None`` when it may proceed.

    A dry run is deliberately *not* blocked here: the implementation honours
    ``context.dry_run`` by computing the would-be hashes without writing, which is
    what makes ``--dry-run`` a useful preview instead of a bare "would run".
    """
    if safety is RepairSafety.MANUAL_ONLY:
        return "repair is marked manual_only and must be applied by hand"
    if safety is RepairSafety.CONFIRM:
        if decide is None:
            return "repair requires confirmation and no confirmation callback was supplied"
        if not decide(repair, match):
            return "confirmation was declined"
    return None


def _declined(
    repair: Repair,
    match: RepairMatch,
    view: RepairRuleView,
    safety: RepairSafety,
    reason: str,
    context: RepairContext,
) -> RepairResult:
    """Build the ``applied=False`` result for a repair that was not run."""
    detail: dict[str, object] = dict(match.details)
    detail.update({"reason": reason, "occurrences": match.total_occurrences})
    return RepairResult(
        repair_id=repair.id,
        description=repair.description,
        safety=safety,
        applied=False,
        files=match.names,
        detail=detail,
        rule_id=view.id,
        artifact_dir=context.artifact_dir,
    )


def _error(
    code: str,
    message: str,
    *,
    category: FailureCategory = FailureCategory.CONFIGURATION_ERROR,
    remediation: Sequence[str] = (),
    **context: str,
) -> RepairError:
    """Build a :class:`RepairError` carrying a fully populated diagnostic."""
    return RepairError(
        diagnostic(
            code,
            category,
            "Repair could not be applied",
            message,
            remediation=list(remediation),
            context=dict(context),
        )
    )
