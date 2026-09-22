"""Loading, layering and validation of compatibility registry data.

Registry data is plain YAML read with ``yaml.safe_load`` only: it can describe
rules, never execute Python. Layers are merged in a fixed order — the rules
shipped inside the package first, then every user override directory — and an
override that reuses an id *replaces* that rule while a new id is added. A
duplicate id inside a single layer is an authoring mistake and raises
:class:`~yonga.domain.diagnostics.RegistryError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from yonga.domain.diagnostics import Diagnostic, FailureCategory, RegistryError, Severity
from yonga.registry.schema import (
    KNOWN_REPAIR_IMPLEMENTATIONS,
    SCHEMA_VERSION,
    AnyRule,
    ArchitectureRule,
    ChatProfileRule,
    DerivativeRule,
    IssueRule,
    ModelRule,
    ProfileRule,
    Registry,
    RegistryDocument,
    RepairRule,
    RuleStatus,
    constraint_problems,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.config.paths import Paths

#: Package holding the built-in registry YAML files.
BUILTIN_PACKAGE: str = "yonga.registry"
#: Directory inside :data:`BUILTIN_PACKAGE` holding the shipped YAML files.
BUILTIN_DIRNAME: str = "builtin"
#: Built-in files, loaded in this order as a single layer.
BUILTIN_FILENAMES: tuple[str, ...] = (
    "architectures.yaml",
    "profiles.yaml",
    "issues.yaml",
    "derivatives.yaml",
    "repairs.yaml",
    "chat.yaml",
)
#: File suffixes treated as registry documents inside an override directory.
REGISTRY_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

_BUILTIN_LAYER = "builtin"

__all__ = [
    "BUILTIN_DIRNAME",
    "BUILTIN_FILENAMES",
    "BUILTIN_PACKAGE",
    "KNOWN_REPAIR_IMPLEMENTATIONS",
    "REGISTRY_SUFFIXES",
    "builtin_registry_dir",
    "load_builtin_documents",
    "load_builtin_registry",
    "load_document",
    "load_registry",
    "load_registry_from_documents",
    "override_documents",
    "parse_document",
    "registry_override_dirs",
    "validate_registry",
]


def _registry_error(
    code: str,
    title: str,
    message: str,
    *,
    remediation: Sequence[str] = (),
    context: Mapping[str, str] | None = None,
    cause: BaseException | None = None,
) -> RegistryError:
    """Build a :class:`RegistryError` carrying a fully populated diagnostic."""
    return RegistryError(
        Diagnostic(
            code=code,
            category=FailureCategory.REGISTRY_ERROR,
            severity=Severity.ERROR,
            title=title,
            message=message,
            remediation=list(remediation),
            context=dict(context or {}),
        ),
        cause=cause,
    )


def parse_document(text: str, *, origin: str) -> RegistryDocument:
    """Parse one registry YAML document.

    ``yaml.safe_load`` is the only parser used, so registry data can never
    instantiate arbitrary Python objects.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise _registry_error(
            "registry.yaml_invalid",
            "Registry file could not be parsed",
            f"{origin} is not valid YAML: {error}",
            remediation=["Fix the YAML syntax, then run 'yonga registry validate'."],
            context={"origin": origin},
            cause=error,
        ) from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _registry_error(
            "registry.document_invalid",
            "Registry file has the wrong shape",
            f"{origin} must contain a YAML mapping at the top level.",
            remediation=["Wrap the rules in top-level sections such as 'architectures:'."],
            context={"origin": origin},
        )
    payload: dict[str, object] = {str(key): value for key, value in raw.items()}
    payload["origin"] = origin
    try:
        document = RegistryDocument.model_validate(payload)
    except Exception as error:  # pydantic.ValidationError and friends
        raise _registry_error(
            "registry.schema_violation",
            "Registry file does not match the schema",
            f"{origin} could not be loaded: {error}",
            remediation=["Compare the file with docs/REGISTRY_SPEC.md."],
            context={"origin": origin},
            cause=error,
        ) from error
    if document.schema_version > SCHEMA_VERSION:
        raise _registry_error(
            "registry.schema_version_unsupported",
            "Registry file is too new",
            (
                f"{origin} declares schema_version {document.schema_version}, but this build "
                f"understands version {SCHEMA_VERSION}."
            ),
            remediation=["Upgrade yonga, or pin the registry file to a supported version."],
            context={"origin": origin},
        )
    return document


def load_document(path: Path) -> RegistryDocument:
    """Read and parse one registry document from disk."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise _registry_error(
            "registry.file_unreadable",
            "Registry file could not be read",
            f"{path} could not be read: {error}",
            remediation=["Check the file permissions of the registry override directory."],
            context={"origin": str(path)},
            cause=error,
        ) from error
    return parse_document(text, origin=str(path))


def builtin_registry_dir() -> Path | None:
    """Return the packaged registry directory when it exists on a real filesystem."""
    root = files(BUILTIN_PACKAGE) / BUILTIN_DIRNAME
    try:
        candidate = Path(str(root))
    except (TypeError, ValueError):  # pragma: no cover - zip-imported distribution
        return None
    return candidate if candidate.is_dir() else None


def load_builtin_documents() -> list[RegistryDocument]:
    """Parse every built-in registry document, in load order."""
    root = files(BUILTIN_PACKAGE) / BUILTIN_DIRNAME
    documents: list[RegistryDocument] = []
    for name in BUILTIN_FILENAMES:
        resource = root / name
        try:
            text = resource.read_text(encoding="utf-8")
        except (OSError, FileNotFoundError) as error:
            raise _registry_error(
                "registry.builtin_missing",
                "Built-in registry data is missing",
                (f"The packaged registry file {BUILTIN_DIRNAME}/{name} could not be read: {error}"),
                remediation=["Reinstall yonga; the package data appears to be incomplete."],
                context={"origin": f"{_BUILTIN_LAYER}:{name}"},
                cause=error,
            ) from error
        documents.append(parse_document(text, origin=f"{_BUILTIN_LAYER}:{name}"))
    return documents


def override_documents(directory: Path) -> list[RegistryDocument]:
    """Parse every registry document in one override directory, sorted by name."""
    if not directory.is_dir():
        return []
    candidates = sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in REGISTRY_SUFFIXES
    )
    return [load_document(path) for path in candidates]


class _Layer:
    """One merge layer: a named group of documents with unique ids."""

    def __init__(self, name: str, documents: Sequence[RegistryDocument]) -> None:
        self.name = name
        self.documents = list(documents)

    def rules(self) -> list[tuple[AnyRule, str]]:
        """Return every rule in this layer paired with the origin that supplied it."""
        collected: list[tuple[AnyRule, str]] = []
        for document in self.documents:
            for rule in document.rules():
                collected.append((rule, document.origin))
        return collected


class _Merger:
    """Accumulates layers into a single indexed registry."""

    def __init__(self) -> None:
        self._rules: dict[str, AnyRule] = {}
        self._origins: dict[str, str] = {}
        self._order: list[str] = []
        self.layers: list[str] = []

    def add_layer(self, layer: _Layer) -> None:
        """Merge one layer, rejecting duplicate ids inside it."""
        self.layers.append(layer.name)
        seen: dict[str, str] = {}
        for rule, origin in layer.rules():
            previous = seen.get(rule.id)
            if previous is not None:
                raise _registry_error(
                    "registry.duplicate_id",
                    "Duplicate registry rule id",
                    (
                        f"Rule id {rule.id!r} is defined twice in the same registry layer "
                        f"({previous} and {origin}). Ids must be unique within a layer; use a "
                        f"separate override directory to replace a rule."
                    ),
                    remediation=[
                        "Rename one of the rules, or move the replacement into an override "
                        "directory."
                    ],
                    context={"rule_id": rule.id, "layer": layer.name, "origin": origin},
                )
            seen[rule.id] = origin
            existing = self._rules.get(rule.id)
            if existing is not None and existing.kind != rule.kind:
                raise _registry_error(
                    "registry.kind_conflict",
                    "Registry override changes a rule's kind",
                    (
                        f"Rule id {rule.id!r} is a {existing.kind} rule in an earlier layer but a "
                        f"{rule.kind} rule in {origin}."
                    ),
                    remediation=["Use a distinct id for the new rule."],
                    context={"rule_id": rule.id, "origin": origin},
                )
            if rule.id not in self._rules:
                self._order.append(rule.id)
            self._rules[rule.id] = rule
            self._origins[rule.id] = origin

    def build(self) -> Registry:
        """Produce the merged :class:`Registry`."""
        architectures: list[ArchitectureRule] = []
        profiles: list[ProfileRule] = []
        issues: list[IssueRule] = []
        repairs: list[RepairRule] = []
        derivatives: list[DerivativeRule] = []
        models: list[ModelRule] = []
        chat: list[ChatProfileRule] = []
        for rule_id in self._order:
            rule = self._rules[rule_id]
            if isinstance(rule, ArchitectureRule):
                architectures.append(rule)
            elif isinstance(rule, ProfileRule):
                profiles.append(rule)
            elif isinstance(rule, IssueRule):
                issues.append(rule)
            elif isinstance(rule, RepairRule):
                repairs.append(rule)
            elif isinstance(rule, DerivativeRule):
                derivatives.append(rule)
            elif isinstance(rule, ModelRule):
                models.append(rule)
            else:
                chat.append(rule)
        return Registry(
            schema_version=SCHEMA_VERSION,
            architectures=architectures,
            profiles=profiles,
            issues=issues,
            repairs=repairs,
            derivatives=derivatives,
            models=models,
            chat=chat,
            origins=dict(self._origins),
            layers=list(self.layers),
        )


def load_builtin_registry() -> Registry:
    """Load only the registry data shipped inside the package."""
    merger = _Merger()
    merger.add_layer(_Layer(_BUILTIN_LAYER, load_builtin_documents()))
    return merger.build()


def registry_override_dirs(
    paths: Paths | None = None,
    extra_dirs: Sequence[Path] = (),
) -> list[Path]:
    """Return the override directories that ``load_registry`` would read, in order."""
    directories: list[Path] = []
    if paths is not None:
        directories.append(paths.registry_overrides_dir)
    directories.extend(Path(entry) for entry in extra_dirs)
    ordered: list[Path] = []
    for directory in directories:
        if directory not in ordered:
            ordered.append(directory)
    return ordered


def load_registry(
    paths: Paths | None = None,
    extra_dirs: Sequence[Path] = (),
) -> Registry:
    """Load the built-in registry, then apply user override directories.

    ``paths`` supplies ``Paths.registry_overrides_dir``; ``extra_dirs`` adds
    further directories after it (later layers win). Each directory is one merge
    layer: within it ids must be unique, across layers an existing id is
    replaced and a new id is added. Missing directories are simply skipped.
    """
    merger = _Merger()
    merger.add_layer(_Layer(_BUILTIN_LAYER, load_builtin_documents()))
    for directory in registry_override_dirs(paths, extra_dirs):
        documents = override_documents(directory)
        if documents:
            merger.add_layer(_Layer(str(directory), documents))
    return merger.build()


def load_registry_from_documents(documents: Iterable[RegistryDocument]) -> Registry:
    """Merge already-parsed documents as a single layer (used by tests and tools)."""
    merger = _Merger()
    merger.add_layer(_Layer("<memory>", list(documents)))
    return merger.build()


def _describe_rule(rule: AnyRule) -> str:
    """Short ``kind 'id'`` label used in validation messages."""
    return f"{rule.kind} rule {rule.id!r}"


def _validate_constraints(rule: AnyRule, problems: list[str]) -> None:
    """Check the rule-level and device-level version constraints."""
    for problem in constraint_problems(rule.constraints):
        problems.append(f"{_describe_rule(rule)}: {problem}")
    devices = getattr(rule, "devices", None)
    if isinstance(devices, dict):
        for device, support in devices.items():
            for problem in constraint_problems(support.constraints):
                problems.append(f"{_describe_rule(rule)} device {device}: {problem}")


def _validate_references(registry: Registry, problems: list[str]) -> None:
    """Check every cross-rule reference resolves to an existing rule."""
    carriers: list[ArchitectureRule | ModelRule] = [*registry.architectures, *registry.models]
    for rule in carriers:
        for device, profile_id in rule.profiles.items():
            if registry.profile(profile_id) is None:
                problems.append(
                    f"{_describe_rule(rule)}: device {device} references unknown profile "
                    f"{profile_id!r}"
                )
                continue
            profile = registry.profile(profile_id)
            if profile is not None and profile.device != device:
                problems.append(
                    f"{_describe_rule(rule)}: device {device} references profile "
                    f"{profile_id!r}, which targets {profile.device}"
                )
        if rule.chat_profile is not None and registry.chat_profile(rule.chat_profile) is None:
            problems.append(
                f"{_describe_rule(rule)}: references unknown chat profile {rule.chat_profile!r}"
            )
        recommended = getattr(rule, "recommended_device", None)
        if isinstance(recommended, str) and rule.devices and recommended not in rule.devices:
            problems.append(
                f"{_describe_rule(rule)}: recommended_device {recommended} has no support entry"
            )
    for issue in registry.issues:
        repair_id = issue.recommendation.repair
        if repair_id is not None and registry.repair(repair_id) is None:
            problems.append(f"{_describe_rule(issue)}: references unknown repair {repair_id!r}")


def _validate_categories(registry: Registry, problems: list[str]) -> None:
    """Check every issue rule names a real failure category."""
    known = {item.value for item in FailureCategory}
    for issue in registry.issues:
        if issue.classification.category not in known:
            problems.append(
                f"{_describe_rule(issue)}: unknown failure category "
                f"{issue.classification.category!r}"
            )


def _validate_repairs(
    registry: Registry,
    known_implementations: Iterable[str],
    problems: list[str],
) -> None:
    """Check every repair rule selects a known built-in implementation."""
    known = set(known_implementations)
    for repair in registry.repairs:
        if repair.implementation not in known:
            problems.append(
                f"{_describe_rule(repair)}: unknown repair implementation "
                f"{repair.implementation!r} (known: {', '.join(sorted(known)) or 'none'})"
            )


def _signatures_overlap(left: IssueRule, right: IssueRule) -> bool:
    """True when two issue rules could fire on the same error text."""
    left_needles = left.match.exception_contains
    right_needles = right.match.exception_contains
    if not left_needles or not right_needles:
        return True
    for first in left_needles:
        for second in right_needles:
            a = first.casefold()
            b = second.casefold()
            if a in b or b in a:
                return True
    return False


def _same_scope(kind: str, first: AnyRule, second: AnyRule) -> bool:
    """True when two rules of the same kind compete for exactly the same subjects."""
    if kind == "profile" and isinstance(first, ProfileRule) and isinstance(second, ProfileRule):
        return first.device == second.device
    if kind == "issue" and isinstance(first, IssueRule) and isinstance(second, IssueRule):
        return _signatures_overlap(first, second)
    if (
        kind == "derivative"
        and isinstance(first, DerivativeRule)
        and isinstance(second, DerivativeRule)
    ):
        return first.requested_repo.casefold() == second.requested_repo.casefold()
    return True


def _validate_conflicts(registry: Registry, problems: list[str]) -> None:
    """Report rule pairs that no deterministic tiebreak could separate."""
    groups: list[tuple[str, Sequence[AnyRule]]] = [
        ("architecture", registry.architectures),
        ("profile", registry.profiles),
        ("issue", registry.issues),
        ("derivative", registry.derivatives),
        ("model", registry.models),
        ("chat", registry.chat),
    ]
    for kind, rules in groups:
        active = [rule for rule in rules if rule.status.is_usable]
        for index, first in enumerate(active):
            first_rungs = set(first.match_block.resolvable_specificities())
            for second in active[index + 1 :]:
                if first.priority != second.priority:
                    continue
                # Every rung a rule can *resolve* at, not just its best one: a
                # rule carrying aliases scores at the demoted alias rung
                # whenever a hint rescued the match, so two rules can tie there
                # while their static specificities differ. Comparing only the
                # static rung certifies a registry that then raises
                # ``registry.ambiguous_rules`` during resolution.
                shared = first_rungs & set(second.match_block.resolvable_specificities())
                if not shared:
                    continue
                if not first.match_block.overlaps(second.match_block):
                    continue
                if not _same_scope(kind, first, second):
                    continue
                problems.append(
                    f"{kind} rules {first.id!r} and {second.id!r} can both resolve at "
                    f"specificity {max(shared)} with the same priority ({first.priority}); "
                    f"resolution would be ambiguous"
                )


def validate_registry(
    registry: Registry,
    *,
    known_implementations: Iterable[str] | None = None,
) -> list[str]:
    """Return every human-readable problem found in ``registry``.

    This powers ``yonga registry validate``. It reports unknown repair
    implementation names, unparsable version specifiers, dangling profile,
    chat-profile and repair references, unknown failure categories, and rule
    pairs that share a priority and a specificity they can both resolve at
    (which would make resolution ambiguous). Specificity is compared over every
    rung a rule can reach, including the demoted alias rung, so a registry that
    passes validation never raises ``registry.ambiguous_rules`` while
    resolving. An empty list means the registry is sound.
    """
    implementations = (
        KNOWN_REPAIR_IMPLEMENTATIONS
        if known_implementations is None
        else set(known_implementations)
    )
    problems: list[str] = []
    for rule in registry.all_rules():
        _validate_constraints(rule, problems)
        if rule.schema_version > SCHEMA_VERSION:
            problems.append(
                f"{_describe_rule(rule)}: schema_version {rule.schema_version} is newer than the "
                f"supported version {SCHEMA_VERSION}"
            )
        if rule.status is RuleStatus.DEPRECATED and not rule.summary:
            problems.append(f"{_describe_rule(rule)}: deprecated rules should explain why")
    _validate_references(registry, problems)
    _validate_categories(registry, problems)
    _validate_repairs(registry, implementations, problems)
    _validate_conflicts(registry, problems)
    return problems
