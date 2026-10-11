"""Deterministic, explainable resolution of compatibility registry rules.

Every decision this module makes is reproducible and auditable:

* Version constraints are applied as a *filter before* scoring, and a rule they
  reject is still reported as a rejected candidate carrying its reason.
* Surviving rules are ranked on the specificity ladder (exact repository id 100,
  architecture plus host/device family 80, architecture 60, architecture reached
  through an identity alias 50, task 40, generic 20); a tie is broken by the
  rule's ``priority``. The alias rung sits below every declared-identity rung so
  a rule that recognises the model type a repository actually declares always
  beats one identified from a hint found outside ``config.json``. A hint is
  consulted only when the repository declares no identity at all: a declaration
  a rule does not list is a rejection, so a contradicting GGUF header resolves
  to "unknown" rather than to another architecture's rule.
* A tie that ``priority`` cannot break is a registry authoring bug and raises
  :class:`~yonga.domain.diagnostics.RegistryError` rather than silently picking
  by load order.

The one exception is :meth:`RegistryResolver.classify`, which turns raw error
text into a :class:`~yonga.domain.diagnostics.Diagnostic` and must never raise:
a failure while explaining a failure would hide the original problem.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from yonga.domain.compatibility import (
    DeviceCompatibility,
    DeviceStatus,
    EvidenceLevel,
    ProfileSelection,
    ResolutionExplanation,
    RuleCandidate,
)
from yonga.domain.diagnostics import (
    RAW_EXCERPT_MAX_CHARS,
    Diagnostic,
    FailureCategory,
    RegistryError,
    Severity,
    diagnostic,
    fill_placeholders,
)
from yonga.domain.hardware import normalise_device
from yonga.domain.models import ModelFormat
from yonga.domain.plans import ConversionProfile, QuantizationSettings
from yonga.registry.schema import (
    IDENTITY_HINT_SOURCE,
    AnyRule,
    ArchitectureRule,
    BaseRule,
    ChatProfileRule,
    DerivativeCandidate,
    DerivativeRule,
    GGUFRule,
    IssueRule,
    MatchSubject,
    ModelRule,
    ProfileRule,
    Registry,
    RepairRule,
    format_parameter_count,
    subject_from_metadata,
)
from yonga.util.redaction import redact_text
from yonga.util.versions import installed_version, satisfies

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.domain.hardware import HardwareInventory
    from yonga.domain.models import ModelIdentity, ModelMetadata

RuleT = TypeVar("RuleT", bound=BaseRule)

#: How a conversion profile was chosen; see :class:`ProfileSelection`.
ProfileBasis = Literal[
    "explicit",
    "named_by_model_rule",
    "named_by_architecture_rule",
    "size_band",
    "device_default",
]

#: An automatic profile choice: the profile, its basis and the rule that named it, if any.
_Automatic = tuple[ProfileRule, ProfileBasis, str | None]


def _size_phrase(subject: MatchSubject) -> str:
    """``has 8.19B parameters`` or ``has an unknown size (<basis>)``."""
    if subject.parameter_count is None:
        basis = subject.parameter_count_basis or "no parameter count is known"
        return f"has an unknown size ({basis})"
    return f"has {format_parameter_count(subject.parameter_count)} parameters"


#: ``config.json`` keys giving the input width of a family of linear layers, and
#: the lowest 4-bit ``ratio`` at which a group size that does not divide that
#: width is certain to be refused. NNCF quantizes each weight row in groups along
#: its input channels and raises ``InvalidGroupSizeError`` for any 4-bit layer
#: whose channel count the group size does not divide. ``hidden_size`` feeds the
#: q/k/v and gate/up projections, well over half of a decoder's weights, so no
#: ratio of one half or more can keep all of them at the 8-bit backup precision.
#: ``intermediate_size`` feeds only the down projections, which a mixed ratio
#: might happen to leave at 8 bits, so it is certain only at ratio 1.0.
_CHANNEL_WIDTHS: tuple[tuple[str, float], ...] = (
    ("hidden_size", 0.5),
    ("intermediate_size", 1.0),
)

#: The code of a group size that cannot quantize the model, whether NNCF said so
#: during the export or profile resolution predicted it before the download.
GROUP_SIZE_MISMATCH_CODE = "export.quantization.group_size_mismatch"

#: Weight formats quantized group-wise, where ``group_size`` takes effect.
_GROUPED_FORMATS: frozenset[str] = frozenset({"int4", "nf4", "mxfp4", "cb4"})


def channel_widths(meta: ModelMetadata | None) -> dict[str, int]:
    """The linear-layer input widths ``config.json`` declares, for the group-size check.

    A multimodal config keeps the language model's under ``text_config``; that
    is the part a text-generation export quantizes to 4 bits. Only positive
    integers count: anything else is not evidence of a width.
    """
    if meta is None:
        return {}
    sources: list[Mapping[str, object]] = [meta.config]
    text_config = meta.config.get("text_config")
    if isinstance(text_config, Mapping):
        sources.append(text_config)
    widths: dict[str, int] = {}
    for key, _ratio in _CHANNEL_WIDTHS:
        for source in sources:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                widths[key] = value
                break
    return widths


def group_size_conflict(
    quantization: QuantizationSettings, widths: Mapping[str, int]
) -> str | None:
    """Why ``quantization`` is certain to be refused for a model of ``widths``, or ``None``.

    The answer is ``None`` whenever the refusal is not certain: channel-wise
    or 8-bit quantization, an unknown width, or a mixed ratio low enough that
    the affected layers could all stay at the backup precision.
    """
    group = quantization.group_size
    if group is None or group <= 0 or quantization.weight_format not in _GROUPED_FORMATS:
        return None
    ratio = 1.0 if quantization.ratio is None else float(quantization.ratio)
    for key, certain_from in _CHANNEL_WIDTHS:
        width = widths.get(key)
        if width is None or width % group == 0 or ratio < certain_from:
            continue
        return (
            f"it quantizes in groups of {group}, which does not divide this model's "
            f"{key} of {width}"
        )
    return None


def _named_size_clause(
    subject: MatchSubject,
    automatic: _Automatic | None,
    device: str,
) -> str:
    """The size half of a named profile's sentence, and the choice the rule overrode.

    ``The model has 751.6M parameters (<basis>); without the rule it would be
    cpu-int4-r08, the CPU profile for models below 5B.`` A rule naming a profile
    bypasses the size bands, so the sentence says which band it bypassed.
    """
    if subject.parameter_count is not None:
        size = f"The model has {format_parameter_count(subject.parameter_count)} parameters"
        if subject.parameter_count_basis:
            size += f" ({subject.parameter_count_basis})"
    else:
        size = f"The model {_size_phrase(subject)}"
    if automatic is None:
        return size + "."
    rule, basis, named_by = automatic
    band = rule.applies_to.parameter_count
    if basis == "size_band" and band is not None:
        alternative = f"the {device} profile for models {band.describe()}"
    elif basis == "named_by_architecture_rule":
        alternative = f"named by architecture rule {named_by}"
    else:
        alternative = f"the highest-priority {device} profile"
    return f"{size}; without the rule it would be {rule.id}, {alternative}."


#: Constraint key -> distribution name used to look the installed version up.
VERSION_DISTRIBUTIONS: Mapping[str, str] = {
    "openvino": "openvino",
    "openvino_genai": "openvino-genai",
    "openvino_tokenizers": "openvino-tokenizers",
    "optimum": "optimum",
    "optimum_intel": "optimum-intel",
    "nncf": "nncf",
    "transformers": "transformers",
    "torch": "torch",
    "huggingface_hub": "huggingface-hub",
}

#: Characters of raw error text kept from the head when an excerpt is truncated.
_EXCERPT_HEAD_CHARS: int = 1200
_TRUNCATION_MARKER: str = "\n[... truncated ...]\n"

_UNKNOWN_CODE = "runtime.unknown_error"


def installed_registry_versions() -> dict[str, str | None]:
    """Return the installed versions of every package the registry constrains.

    Only distribution meta is read, so this stays safe to call on a machine
    where OpenVINO is not installed and never imports a heavy dependency.
    """
    versions: dict[str, str | None] = {
        key: installed_version(dist) for key, dist in VERSION_DISTRIBUTIONS.items()
    }
    versions["python"] = platform.python_version()
    return versions


def _excerpt(text: str) -> str:
    """Redact and bound raw error text for storage in a diagnostic."""
    cleaned = redact_text(text).strip()
    if len(cleaned) <= RAW_EXCERPT_MAX_CHARS:
        return cleaned
    tail = RAW_EXCERPT_MAX_CHARS - _EXCERPT_HEAD_CHARS - len(_TRUNCATION_MARKER)
    return cleaned[:_EXCERPT_HEAD_CHARS] + _TRUNCATION_MARKER + cleaned[-tail:]


def _error_text(raw_error: str | BaseException) -> str:
    """Render an exception or string as the text the classifier matches against.

    Rendering is defensive: some native wrappers raise from ``__str__``, and a
    failure there must not hide the failure being classified.
    """
    if not isinstance(raw_error, BaseException):
        return raw_error
    name = type(raw_error).__name__
    try:
        message = str(raw_error)
    except Exception:
        return name
    return f"{name}: {message}" if message else name


class _GenericSignature(BaseModel):
    """A last-resort, code-level inference rule used when no issue rule matches."""

    model_config = ConfigDict(frozen=True)

    code: str
    category: FailureCategory
    title: str
    message: str
    needles: tuple[str, ...]
    remediation: tuple[str, ...] = ()
    environment_broken: bool = False
    retry_same_device: bool = False

    def hit(self, text: str) -> str | None:
        """Return the longest needle present in ``text`` (case-insensitive)."""
        haystack = text.casefold()
        found = [needle for needle in self.needles if needle.casefold() in haystack]
        if not found:
            return None
        return max(found, key=len)


#: Category inference used only after every registry issue rule has been tried.
GENERIC_SIGNATURES: tuple[_GenericSignature, ...] = (
    _GenericSignature(
        code="device.permission_denied",
        category=FailureCategory.PERMISSION_DENIED,
        title="Accelerator device node is not accessible",
        message=(
            "The operating system refused access to the accelerator device node. This is a "
            "host permission problem, not a model compatibility problem."
        ),
        needles=("permission denied", "eacces", "operation not permitted"),
        remediation=(
            "Run 'yonga doctor' to see which device node is unreadable.",
            "Add your user to the group that owns the device node, then log out and back in.",
        ),
        environment_broken=True,
    ),
    _GenericSignature(
        code="disk.insufficient_space",
        category=FailureCategory.INSUFFICIENT_DISK,
        title="Not enough free disk space",
        message=(
            "A write failed because the filesystem ran out of space. Conversion needs several "
            "times the model size in scratch space."
        ),
        needles=(
            "no space left on device",
            "basic_ios::clear: iostream error",
            "disk quota exceeded",
            "enospc",
        ),
        remediation=(
            "Free space on the cache filesystem, or point YONGA_CACHE_DIR at a larger disk.",
        ),
    ),
    _GenericSignature(
        code="memory.out_of_memory",
        category=FailureCategory.OUT_OF_MEMORY,
        title="Out of memory",
        message="The host or the accelerator ran out of memory while running this step.",
        needles=(
            "std::bad_alloc",
            "bad_alloc",
            "out of memory",
            "cannot allocate memory",
            "cl_out_of_resources",
            "ze_result_error_out_of_device_memory",
            "ze_result_error_out_of_host_memory",
        ),
        remediation=(
            "Close other applications, or choose a profile with stronger weight compression.",
        ),
    ),
    _GenericSignature(
        code="hub.auth_required",
        category=FailureCategory.GATED_MODEL_AUTH_REQUIRED,
        title="Authentication required",
        message="The Hugging Face Hub refused the request because the repository is gated.",
        needles=(
            "401 client error",
            "403 client error",
            "gatedrepoerror",
            "access to model",
            "is restricted and you are not in the authorized list",
        ),
        remediation=("Run 'hf auth login' and accept the model licence on the model page.",),
    ),
    _GenericSignature(
        code="source.remote_code_required",
        category=FailureCategory.REMOTE_CODE_REQUIRED,
        title="Model requires remote code",
        message=(
            "Loading this repository executes Python shipped inside it, which yonga never "
            "enables on its own."
        ),
        needles=("trust_remote_code", "custom code", "requires you to execute"),
        remediation=("Re-run with --trust-remote-code after reviewing the repository's code.",),
    ),
    _GenericSignature(
        code="packages.abi_mismatch",
        category=FailureCategory.PACKAGE_ABI_MISMATCH,
        title="Installed packages are not binary compatible",
        message=(
            "A native library failed to load, which usually means the OpenVINO family of "
            "packages was mixed across releases."
        ),
        needles=("undefined symbol", "glibcxx_", "cxxabi", "symbol not found", "version `glibc"),
        remediation=(
            "Run 'yonga doctor' and reinstall openvino, openvino-genai and openvino-tokenizers "
            "from the same release.",
        ),
    ),
    _GenericSignature(
        code="device.driver_missing",
        category=FailureCategory.DRIVER_MISSING,
        title="Accelerator driver is not initialised",
        message=(
            "The accelerator runtime could not be initialised, so the device is not usable on "
            "this host."
        ),
        needles=("ze_result_error_uninitialized", "ze_result_error_device_lost", "no driver"),
        remediation=("Run 'yonga doctor' for the driver and kernel module status.",),
        environment_broken=True,
    ),
    _GenericSignature(
        code="device.unavailable",
        category=FailureCategory.DEVICE_UNAVAILABLE,
        title="Device is not available",
        message="OpenVINO does not expose the requested device on this host.",
        needles=(
            "is not registered in the openvino runtime",
            "was not found in the openvino runtime",
            "device is not available",
        ),
        remediation=("Run 'yonga doctor' to list the devices OpenVINO can see.",),
    ),
    _GenericSignature(
        code="ir.unsupported_element_type",
        category=FailureCategory.UNSUPPORTED_IR_TYPE,
        title="OpenVINO IR uses an unsupported element type",
        message="The IR declares an element type this OpenVINO build cannot read.",
        needles=("unsupported element type", "is_valid_type_idx"),
        # There is no `yonga validate`: validation runs inside `prepare`, and the
        # per-submodel view of an already-prepared artifact comes from `probe`.
        remediation=(
            "Run 'yonga probe ARTIFACT --all-submodels' to see which submodel is affected.",
            "Then re-convert it with 'yonga prepare MODEL --force'.",
        ),
    ),
)


def more_specific_chat_rule(
    named: ChatProfileRule | None,
    matched: ChatProfileRule | None,
) -> ChatProfileRule | None:
    """Return whichever of two chat rules constrains more, preferring ``named`` on a tie.

    ``named`` is the profile an architecture rule points at; ``matched`` is the
    profile whose own match block fits this model. A catch-all architecture rule
    naming a catch-all chat profile must not beat a rule written for the family.
    """
    if matched is None:
        return named
    if named is None:
        return matched
    if named.id == matched.id:
        return named
    if matched.match_block.static_specificity() > named.match_block.static_specificity():
        return matched
    return named


class RegistryExplanation(BaseModel):
    """Everything ``yonga registry explain`` needs to print for one model."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str | None = None
    device: str | None = None
    host_tags: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    architecture_rule_id: str | None = None
    architecture: ResolutionExplanation
    model_rule_id: str | None = None
    model: ResolutionExplanation
    profile_id: str | None = None
    profile: ResolutionExplanation | None = None
    chat_profile_id: str | None = None
    chat: ResolutionExplanation
    chat_tool_calling: str | None = None
    """The winning chat profile's tool-call format, described; ``None`` when it declares none."""
    derivatives: list[DerivativeCandidate] = Field(default_factory=list)
    derivative_resolution: ResolutionExplanation
    device_compatibility: list[DeviceCompatibility] = Field(default_factory=list)
    recommended_device: str | None = None
    notes: list[str] = Field(default_factory=list)


class RegistryResolver:
    """Resolves registry rules for one host, deterministically and explainably."""

    def __init__(
        self,
        registry: Registry,
        *,
        versions: Mapping[str, str | None] | None = None,
        host_tags: Sequence[str] = (),
        strict_versions: bool = False,
    ) -> None:
        """Bind a registry to the host facts its version constraints are checked against.

        ``versions`` maps a constraint key (``openvino``, ``transformers``, ...)
        to the installed version, and defaults to
        :func:`installed_registry_versions`. ``host_tags`` are the normalised
        device-family tags produced by the hardware adapter. When
        ``strict_versions`` is false (the default) a constraint on a package
        whose version is unknown is reported as unverified rather than treated
        as a rejection, so the tool keeps working when OpenVINO is absent.
        """
        self.registry = registry
        self.versions: dict[str, str | None] = (
            installed_registry_versions() if versions is None else dict(versions)
        )
        self.host_tags: tuple[str, ...] = tuple(host_tags)
        self.strict_versions = strict_versions

    # ------------------------------------------------------------------
    # shared machinery
    # ------------------------------------------------------------------
    def known_versions(self) -> dict[str, str]:
        """Installed versions that are actually known, for display."""
        return {key: value for key, value in self.versions.items() if value is not None}

    def _constraints_ok(self, constraints: Mapping[str, str]) -> tuple[bool, list[str]]:
        """Evaluate a ``constraints`` block against the bound host versions."""
        reasons: list[str] = []
        satisfied = True
        for key in sorted(constraints):
            specifier = constraints[key]
            installed = self.versions.get(key)
            if installed is None:
                if self.strict_versions:
                    satisfied = False
                    reasons.append(f"{key} is not installed, so {specifier} cannot be satisfied")
                else:
                    reasons.append(f"{key} version unknown, constraint {specifier} not verified")
                continue
            if satisfies(installed, specifier):
                reasons.append(f"{key} {installed} satisfies {specifier}")
            else:
                satisfied = False
                reasons.append(f"{key} {installed} does not satisfy {specifier}")
        return satisfied, reasons

    def _evaluate(self, rule: AnyRule, subject: MatchSubject) -> tuple[RuleCandidate, bool]:
        """Score one rule against ``subject``, returning its candidate record."""
        specificity = rule.match_block.static_specificity()
        if not rule.status.is_usable:
            return (
                RuleCandidate(
                    rule_id=rule.id,
                    kind=rule.kind,
                    specificity=specificity,
                    priority=rule.priority,
                    matched=False,
                    reasons=[f"rule status is {rule.status.value}"],
                ),
                False,
            )
        constraints_ok, constraint_reasons = self._constraints_ok(rule.constraints)
        if not constraints_ok:
            return (
                RuleCandidate(
                    rule_id=rule.id,
                    kind=rule.kind,
                    specificity=specificity,
                    priority=rule.priority,
                    matched=False,
                    reasons=constraint_reasons,
                ),
                False,
            )
        outcome = rule.match_block.evaluate(subject)
        return (
            RuleCandidate(
                rule_id=rule.id,
                kind=rule.kind,
                specificity=outcome.specificity,
                priority=rule.priority,
                matched=outcome.matched,
                reasons=[*constraint_reasons, *outcome.reasons],
            ),
            outcome.matched,
        )

    def _rank(
        self,
        scored: Sequence[tuple[RuleT, RuleCandidate]],
        *,
        subject: str,
    ) -> RuleT | None:
        """Return the winning rule, raising on a tie that priority cannot break."""
        matched = [(rule, candidate) for rule, candidate in scored if candidate.matched]
        if not matched:
            return None
        ordered = sorted(
            matched,
            key=lambda item: (-item[1].specificity, -item[1].priority, item[0].id),
        )
        best_rule, best_candidate = ordered[0]
        if len(ordered) > 1:
            runner_rule, runner_candidate = ordered[1]
            if (
                runner_candidate.specificity == best_candidate.specificity
                and runner_candidate.priority == best_candidate.priority
            ):
                raise self._conflict_error(subject, best_rule.id, runner_rule.id, best_candidate)
        return best_rule

    def _conflict_error(
        self,
        subject: str,
        first_id: str,
        second_id: str,
        candidate: RuleCandidate,
    ) -> RegistryError:
        """Build the error raised when two rules are indistinguishable."""
        return RegistryError(
            Diagnostic(
                code="registry.ambiguous_rules",
                category=FailureCategory.REGISTRY_ERROR,
                severity=Severity.ERROR,
                title="Registry rules conflict",
                message=(
                    f"Resolving {subject} matched {first_id!r} and {second_id!r} with the same "
                    f"specificity ({candidate.specificity}) and the same priority "
                    f"({candidate.priority}), so the outcome would depend on load order."
                ),
                remediation=[
                    "Run 'yonga registry validate' to list every ambiguous pair.",
                    "Give one of the rules a higher priority, or narrow its match block.",
                ],
                context={"subject": subject, "rules": f"{first_id}, {second_id}"},
            )
        )

    def _subject(
        self,
        meta: ModelMetadata | None,
        *,
        device: str | None = None,
        host_tags: Sequence[str] | None = None,
        size_applies: bool = True,
    ) -> MatchSubject:
        """Build a match subject from meta plus this resolver's host facts."""
        tags = self.host_tags if host_tags is None else tuple(host_tags)
        return subject_from_metadata(meta, device=device, host_tags=tags, size_applies=size_applies)

    # ------------------------------------------------------------------
    # architecture and exact-model rules
    # ------------------------------------------------------------------
    def resolve_architecture(
        self,
        meta: ModelMetadata | None,
        host_tags: Sequence[str] = (),
    ) -> tuple[ArchitectureRule | None, ResolutionExplanation]:
        """Resolve the architecture rule describing ``meta``."""
        tags = tuple(host_tags) or self.host_tags
        subject = self._subject(meta, host_tags=tags)
        explanation = ResolutionExplanation(subject="architecture rule")
        scored: list[tuple[ArchitectureRule, RuleCandidate]] = []
        for rule in self.registry.architectures:
            candidate, _matched = self._evaluate(rule, subject)
            explanation.candidates.append(candidate)
            scored.append((rule, candidate))
        winner = self._rank(scored, subject="architecture rule")
        if winner is None:
            explanation.notes.append("No architecture rule matched; runtime probing decides.")
        else:
            explanation.winner_id = winner.id
            _note_identity_hint(winner, subject, explanation)
        return winner, explanation

    def resolve_model_rule(
        self,
        meta: ModelMetadata | None,
    ) -> tuple[ModelRule | None, ResolutionExplanation]:
        """Resolve an exact-model rule, used for repositories that differ from their family."""
        subject = self._subject(meta)
        explanation = ResolutionExplanation(subject="model rule")
        scored: list[tuple[ModelRule, RuleCandidate]] = []
        for rule in self.registry.models:
            candidate, _matched = self._evaluate(rule, subject)
            explanation.candidates.append(candidate)
            scored.append((rule, candidate))
        winner = self._rank(scored, subject="model rule")
        if winner is not None:
            explanation.winner_id = winner.id
        return winner, explanation

    # ------------------------------------------------------------------
    # conversion profiles
    # ------------------------------------------------------------------
    def resolve_profile(
        self,
        *,
        device: str,
        arch_rule: ArchitectureRule | None = None,
        meta: ModelMetadata | None = None,
        model_rule: ModelRule | None = None,
        explicit_profile_id: str | None = None,
        size_applies: bool = True,
    ) -> tuple[ConversionProfile, ResolutionExplanation]:
        """Resolve the conversion profile to use for ``device``.

        An explicit profile id always wins. Otherwise a profile named by the
        exact-model rule is preferred, then one named by the architecture rule,
        then the best-scoring profile that targets the device. A named profile
        is a preference, not an override: it must itself be usable, satisfy its
        version constraints and match its own ``applies_to`` (a size band
        included), or resolution falls through to scoring.

        ``size_applies=False`` keeps the model's size out of the decision, for
        sources whose quantization no profile decides (a GGUF import, an
        OpenVINO copy). The explanation's ``profile_selection`` says why the
        winner won, in the sentence the plan and the manifest print.
        """
        target = normalise_device(device)
        subject = self._subject(meta, device=target, size_applies=size_applies)
        explanation = ResolutionExplanation(subject=f"conversion profile for {target}")
        # Like the size, the model's channel widths only matter where a profile's
        # quantization is applied.
        widths = channel_widths(meta) if size_applies else {}
        if explicit_profile_id is not None:
            profile = self._explicit_profile(explicit_profile_id, target, explanation)
            if not profile.status.is_usable:
                explanation.notes.append(
                    f"Profile {profile.id} is {profile.status.value}; it is used because it was "
                    "requested explicitly."
                )
            band = profile.applies_to.parameter_count
            if (
                band is not None
                and size_applies
                and band.verdict(subject.parameter_count) != "inside"
            ):
                explanation.notes.append(
                    f"Profile {profile.id} is meant for models {band.describe()}; this model "
                    f"{_size_phrase(subject)}. It is used because it was requested explicitly."
                )
            conflict = group_size_conflict(profile.quantization, widths)
            if conflict is not None:
                explanation.notes.append(
                    f"Profile {profile.id} cannot quantize this model: {conflict}, and NNCF "
                    f"refuses such a layer ({GROUP_SIZE_MISMATCH_CODE}). It is used "
                    "because it was requested explicitly."
                )
            automatic = self._quiet_automatic(target, arch_rule, model_rule, subject, widths=widths)
            explanation.profile_selection = self._profile_selection(
                "explicit",
                profile,
                subject,
                automatic=automatic
                if automatic is not None and automatic[0].id != profile.id
                else None,
            )
            return profile.to_conversion_profile(), explanation

        unfit: list[str] = []
        winner, basis, rule_id = self._automatic_profile(
            target, arch_rule, model_rule, subject, explanation, widths=widths, unfit=unfit
        )
        explanation.winner_id = winner.id
        # Only scoring can lose a profile to the model's size; a named
        # preference wins before any band is weighed.
        scored = basis in ("size_band", "device_default")
        # What the naming rule overrode: for a model rule, whatever the
        # architecture rule and scoring would choose; for an architecture rule,
        # scoring alone.
        bypassed = (
            None
            if scored
            else self._quiet_automatic(
                target,
                arch_rule if basis == "named_by_model_rule" else None,
                None,
                subject,
                widths=widths,
            )
        )
        explanation.profile_selection = self._profile_selection(
            basis,
            winner,
            subject,
            rule_id=rule_id,
            automatic=bypassed if bypassed is not None and bypassed[0].id != winner.id else None,
            skipped=[*(self._size_skipped(target, subject, winner) if scored else ()), *unfit],
            unfit=bool(unfit),
        )
        return winner.to_conversion_profile(), explanation

    def _quiet_automatic(
        self,
        target: str,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
        subject: MatchSubject,
        *,
        widths: Mapping[str, int] | None = None,
    ) -> _Automatic | None:
        """The automatic choice and its basis, worked out without touching the explanation.

        Like the real choice, it passes over a profile whose group size cannot
        quantize a model of ``widths``.
        """
        try:
            scratch = ResolutionExplanation(subject=f"conversion profile for {target}")
            choice = self._automatic_profile(
                target,
                arch_rule,
                model_rule,
                subject,
                scratch,
                widths=widths,
                suggest_devices=False,
            )
        except RegistryError:
            return None
        return choice

    def _automatic_profile(
        self,
        target: str,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
        subject: MatchSubject,
        explanation: ResolutionExplanation,
        *,
        widths: Mapping[str, int] | None = None,
        unfit: list[str] | None = None,
        suggest_devices: bool = True,
    ) -> tuple[ProfileRule, ProfileBasis, str | None]:
        """The profile chosen without ``--profile``: a named preference, else scoring.

        A profile whose group size cannot quantize a model of ``widths`` is
        passed over, named or scored: the export would only fail after the
        download. Each one passed over is described in ``unfit``, and when none is
        left the refusal names the devices whose own choice fits, unless
        ``suggest_devices`` is off (as it is while working those out).
        """
        passed_over: list[str] = [] if unfit is None else unfit

        def unusable(rule: ProfileRule, candidate: RuleCandidate) -> bool:
            conflict = group_size_conflict(rule.quantization, widths or {})
            if conflict is None:
                return False
            candidate.matched = False
            candidate.reasons.append(f"cannot quantize this model: {conflict}")
            passed_over.append(f"{rule.id} cannot quantize this model: {conflict}")
            return True

        for source_rule in (model_rule, arch_rule):
            if source_rule is None:
                continue
            named = source_rule.profile_id_for(target)
            if named is None:
                continue
            preferred = self.registry.profile(named)
            if preferred is None:
                explanation.notes.append(
                    f"{source_rule.kind} rule {source_rule.id} names unknown profile {named!r}"
                )
                continue
            candidate, matched = self._evaluate(preferred, subject)
            candidate.reasons.insert(0, f"named by {source_rule.kind} rule {source_rule.id}")
            explanation.candidates.append(candidate)
            if matched and unusable(preferred, candidate):
                continue
            if matched:
                basis: ProfileBasis = (
                    "named_by_model_rule"
                    if source_rule.kind == "model"
                    else "named_by_architecture_rule"
                )
                return preferred, basis, source_rule.id
            if not preferred.status.is_usable:
                explanation.notes.append(
                    f"{source_rule.kind} rule {source_rule.id} names {preferred.status.value} "
                    f"profile {named}; the {target} profiles were ranked instead."
                )
            # A band or a version constraint that rules the named profile out is
            # normal, not an anomaly: the candidate's reasons already say why.

        scored: list[tuple[ProfileRule, RuleCandidate]] = []
        for rule in self.registry.profiles:
            if rule.device != target:
                continue
            candidate, matched = self._evaluate(rule, subject)
            if matched:
                unusable(rule, candidate)
            explanation.candidates.append(candidate)
            scored.append((rule, candidate))
        winner = self._rank(scored, subject=f"conversion profile for {target}")
        if winner is None and passed_over:
            raise RegistryError(
                self._no_fitting_profile(
                    target,
                    passed_over,
                    arch_rule,
                    model_rule,
                    subject,
                    widths or {},
                    suggest_devices=suggest_devices,
                )
            )
        if winner is None:
            raise RegistryError(
                Diagnostic(
                    code="registry.profile_unresolved",
                    category=FailureCategory.REGISTRY_ERROR,
                    severity=Severity.ERROR,
                    title="No conversion profile is available",
                    message=(
                        f"The registry contains no usable conversion profile for {target}. "
                        f"Every candidate was rejected by its version constraints or match block."
                    ),
                    remediation=[
                        "Run 'yonga registry validate'.",
                        f"Add a profile targeting {target} in the registry override directory.",
                    ],
                    context={"device": target},
                )
            )
        basis = "size_band" if winner.applies_to.parameter_count is not None else "device_default"
        return winner, basis, None

    def _no_fitting_profile(
        self,
        target: str,
        unfit: Sequence[str],
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
        subject: MatchSubject,
        widths: Mapping[str, int],
        *,
        suggest_devices: bool,
    ) -> Diagnostic:
        """Refuse before the download when no profile for ``target`` can quantize the model.

        The way out is another device whose own automatic choice fits, worked out
        the same way. A profile made for one device is not offered for another:
        it carries that device's runtime properties (the NPU's ``MAX_PROMPT_LEN``),
        which the CPU and GPU plugins reject.
        """
        remediation: list[str] = []
        others = sorted({rule.device for rule in self.registry.profiles} - {target})
        for other in others if suggest_devices else ():
            scratch = ResolutionExplanation(subject=f"conversion profile for {other}")
            try:
                fitting, _basis, _rule_id = self._automatic_profile(
                    other,
                    arch_rule,
                    model_rule,
                    subject.model_copy(update={"device": other}),
                    scratch,
                    widths=widths,
                    suggest_devices=False,
                )
            except RegistryError:
                continue
            remediation.append(
                f"Prepare it for the {other} instead, where {fitting.id} fits: "
                f"yonga prepare MODEL --device {other.lower()}"
            )
        remediation.append(
            "Or add a profile whose group size divides the channel sizes above, in the "
            "registry override directory, and name it with --profile."
        )
        repo_id = subject.repo_id
        return diagnostic(
            GROUP_SIZE_MISMATCH_CODE,
            FailureCategory.CONFIGURATION_ERROR,
            f"No {target} profile can quantize this model",
            (
                f"Every {target} conversion profile quantizes in groups that do not fit this "
                f"model's channel sizes, so NNCF would refuse the export after the download: "
                f"{'; '.join(unfit)}. Nothing has been downloaded."
            ),
            remediation=remediation,
            context={"device": target, **({"repo_id": repo_id} if repo_id else {})},
        )

    def _size_skipped(self, target: str, subject: MatchSubject, winner: ProfileRule) -> list[str]:
        """One sentence per usable size-banded profile the model's size kept out.

        Only a profile that would have won without its band counts: one that
        loses to the winner anyway did not lose because of the size.
        """
        winner_rank = winner.applies_to.evaluate(subject).specificity
        skipped: list[str] = []
        for rule in self.registry.profiles:
            band = rule.applies_to.parameter_count
            if rule.device != target or band is None or rule.id == winner.id:
                continue
            if not rule.status.is_usable or not self._constraints_ok(rule.constraints)[0]:
                continue
            if band.verdict(subject.parameter_count) == "inside":
                continue
            outcome = rule.applies_to.model_copy(update={"parameter_count": None}).evaluate(subject)
            if not outcome.matched:
                continue  # it lost for another reason than size
            if (outcome.specificity, rule.priority) < (winner_rank, winner.priority):
                continue  # it would have lost to the winner at any size
            skipped.append(
                f"{rule.id} is for models {band.describe()}; this model {_size_phrase(subject)}"
            )
        return skipped

    def _profile_selection(
        self,
        basis: ProfileBasis,
        profile: ProfileRule,
        subject: MatchSubject,
        *,
        rule_id: str | None = None,
        automatic: _Automatic | None = None,
        skipped: Sequence[str] = (),
        unfit: bool = False,
    ) -> ProfileSelection:
        """Record why ``profile`` won, with the one sentence every surface prints.

        ``automatic`` is the profile that would have been chosen without
        ``--profile`` or without the naming rule, when it is a different one.
        """
        band = profile.applies_to.parameter_count
        device = profile.device
        automatic_profile_id = automatic[0].id if automatic is not None else None
        if basis == "explicit":
            sentence = "Requested with --profile."
            if automatic_profile_id is not None:
                sentence += f" The automatic choice would be {automatic_profile_id}."
        elif basis == "size_band":
            assert band is not None and subject.parameter_count is not None
            sentence = (
                f"{format_parameter_count(subject.parameter_count)} parameters "
                f"({subject.parameter_count_basis}) is {band.predicate()}: {profile.id} is the "
                f"{device} profile for that size (evidence: {profile.evidence.value})."
            )
        elif basis == "device_default":
            sentence = (
                f"Highest-priority {device} profile that can quantize this model."
                if unfit
                else f"Highest-priority {device} profile."
            )
        else:
            kind = "model" if basis == "named_by_model_rule" else "architecture"
            sentence = f"Named by {kind} rule {rule_id} for {device}."
            sentence += " " + _named_size_clause(subject, automatic, device)
        if skipped and basis != "explicit":
            sentence += " " + "; ".join(skipped) + "."
        return ProfileSelection(
            basis=basis,
            profile_id=profile.id,
            rule_id=rule_id,
            parameter_count=subject.parameter_count,
            parameter_count_basis=subject.parameter_count_basis,
            band=band.describe() if band is not None else None,
            band_evidence=profile.evidence.value if band is not None else None,
            automatic_profile_id=automatic_profile_id,
            skipped=list(skipped),
            sentence=sentence,
        )

    def _explicit_profile(
        self,
        profile_id: str,
        target: str,
        explanation: ResolutionExplanation,
    ) -> ProfileRule:
        """Return an explicitly requested profile, or raise a registry error."""
        profile = self.registry.profile(profile_id)
        if profile is None:
            on_device = [rule.id for rule in self.registry.profiles if rule.device == target]
            available = (
                f"Profiles for {target}: {', '.join(on_device)}."
                if on_device
                else f"No registry profile targets {target}."
            )
            raise RegistryError(
                Diagnostic(
                    code="registry.profile_unknown",
                    category=FailureCategory.REGISTRY_ERROR,
                    severity=Severity.ERROR,
                    title="Unknown conversion profile",
                    message=f"No registry profile is called {profile_id!r}.",
                    remediation=[
                        available,
                        "'yonga registry profiles' lists every profile with its device, "
                        "weights and size band.",
                    ],
                    context={"profile_id": profile_id},
                )
            )
        constraints_ok, reasons = self._constraints_ok(profile.constraints)
        explanation.candidates.append(
            RuleCandidate(
                rule_id=profile.id,
                kind=profile.kind,
                specificity=profile.match_block.static_specificity(),
                priority=profile.priority,
                matched=True,
                reasons=["requested explicitly by the user", *reasons],
            )
        )
        explanation.winner_id = profile.id
        if not constraints_ok:
            explanation.notes.append(
                f"Profile {profile.id} was requested explicitly although its version constraints "
                f"are not satisfied on this host."
            )
        if profile.device != target:
            explanation.notes.append(
                f"Profile {profile.id} targets {profile.device}, not the requested {target}."
            )
        return profile

    # ------------------------------------------------------------------
    # GGUF
    # ------------------------------------------------------------------
    def resolve_gguf(self) -> tuple[GGUFRule | None, list[str]]:
        """The GGUF support rule for the installed GenAI, and why others lost.

        Rules are ranked by priority among those whose version constraints the
        host satisfies. A rule whose constraints cannot be *verified* (GenAI
        not installed) does not count: direct GGUF import needs GenAI itself,
        so "unknown" is not support.
        """
        notes: list[str] = []
        best: GGUFRule | None = None
        for rule in self.registry.gguf:
            if not rule.status.is_usable:
                continue
            satisfied, reasons = self._constraints_ok(rule.constraints)
            unverified = any("not verified" in reason for reason in reasons)
            if not satisfied or unverified:
                notes.append(f"{rule.id}: " + "; ".join(reasons))
                continue
            if best is None or rule.priority > best.priority:
                best = rule
        return best, notes

    # derivatives
    # ------------------------------------------------------------------
    def resolve_derivatives(
        self,
        identity: ModelIdentity | str,
    ) -> tuple[list[DerivativeCandidate], ResolutionExplanation]:
        """Return the declared alternative sources for a requested repository."""
        repo_id = identity if isinstance(identity, str) else identity.repo_id
        explanation = ResolutionExplanation(subject=f"derivative sources for {repo_id}")
        subject = MatchSubject(repo_id=repo_id)
        scored: list[tuple[DerivativeRule, RuleCandidate]] = []
        for rule in self.registry.derivatives:
            candidate, _matched = self._evaluate(rule, subject)
            explanation.candidates.append(candidate)
            scored.append((rule, candidate))
        winner = self._rank(scored, subject=f"derivative sources for {repo_id}")
        if winner is None:
            explanation.notes.append("No declared derivative source for this repository.")
            return [], explanation
        explanation.winner_id = winner.id
        for candidate_source in winner.candidates:
            if candidate_source.requires_confirmation:
                explanation.notes.append(
                    f"{candidate_source.repo_id} needs explicit user confirmation "
                    f"(trust policy {candidate_source.trust.value})."
                )
        return list(winner.candidates), explanation

    # ------------------------------------------------------------------
    # failure classification
    # ------------------------------------------------------------------
    def classify(
        self,
        *,
        raw_error: str | BaseException,
        device: str | None = None,
        meta: ModelMetadata | None = None,
        stage: str | None = None,
        job_id: str | None = None,
        log_path: Path | None = None,
    ) -> Diagnostic:
        """Classify raw failure text into a diagnostic. Never raises."""
        result, _explanation = self.classify_explained(
            raw_error=raw_error,
            device=device,
            meta=meta,
            stage=stage,
            job_id=job_id,
            log_path=log_path,
        )
        return result

    def classify_explained(
        self,
        *,
        raw_error: str | BaseException,
        device: str | None = None,
        meta: ModelMetadata | None = None,
        stage: str | None = None,
        job_id: str | None = None,
        log_path: Path | None = None,
    ) -> tuple[Diagnostic, ResolutionExplanation]:
        """Classify raw failure text and return the candidate trail with it."""
        explanation = ResolutionExplanation(subject="failure classification")
        try:
            return self._classify(
                raw_error=raw_error,
                device=device,
                meta=meta,
                stage=stage,
                job_id=job_id,
                log_path=log_path,
                explanation=explanation,
            )
        except Exception as error:  # classification must never mask the real failure
            explanation.notes.append(f"Classification itself failed: {error}")
            return (
                self._unknown_diagnostic(
                    _error_text(raw_error),
                    device=device,
                    stage=stage,
                    meta=meta,
                    job_id=job_id,
                    log_path=log_path,
                ),
                explanation,
            )

    def _classify(
        self,
        *,
        raw_error: str | BaseException,
        device: str | None,
        meta: ModelMetadata | None,
        stage: str | None,
        job_id: str | None,
        log_path: Path | None,
        explanation: ResolutionExplanation,
    ) -> tuple[Diagnostic, ResolutionExplanation]:
        """Walk issue rules, then generic inference, then give up as unknown."""
        text = _error_text(raw_error)
        subject = self._subject(meta, device=device)
        scored: list[tuple[IssueRule, RuleCandidate, int]] = []
        for rule in self.registry.issues:
            candidate, matched = self._evaluate(rule, subject)
            hit_length = 0
            if matched and not rule.match.stage_matches(stage):
                candidate.matched = False
                candidate.reasons.append(f"rule applies to stages {', '.join(rule.match.stages)}")
                matched = False
            if matched:
                if not rule.match.exception_contains:
                    candidate.matched = False
                    candidate.reasons.append("rule carries no error signature")
                    matched = False
                else:
                    hit = rule.match.signature_hit(text)
                    if hit is None:
                        candidate.matched = False
                        candidate.reasons.append("no known signature found in the error text")
                        matched = False
                    else:
                        hit_length = len(hit)
                        candidate.reasons.append(f"error text contains {hit!r}")
            explanation.candidates.append(candidate)
            if matched:
                scored.append((rule, candidate, hit_length))
        winner = self._pick_issue(scored, explanation)
        if winner is not None:
            explanation.winner_id = winner.id
            return (
                self._diagnostic_from_issue(
                    winner,
                    text=text,
                    device=device,
                    stage=stage,
                    meta=meta,
                    job_id=job_id,
                    log_path=log_path,
                ),
                explanation,
            )
        for signature in GENERIC_SIGNATURES:
            hit = signature.hit(text)
            if hit is None:
                continue
            explanation.notes.append(f"Matched built-in category inference on {hit!r}.")
            return (
                diagnostic(
                    signature.code,
                    signature.category,
                    signature.title,
                    signature.message,
                    remediation=list(signature.remediation),
                    environment_broken=signature.environment_broken,
                    retry_same_device=signature.retry_same_device,
                    raw_excerpt=_excerpt(text),
                    job_id=job_id,
                    log_path=log_path,
                    context=_context(device=device, stage=stage, meta=meta),
                ),
                explanation,
            )
        explanation.notes.append("No registry rule and no built-in signature matched.")
        return (
            self._unknown_diagnostic(
                text,
                device=device,
                stage=stage,
                meta=meta,
                job_id=job_id,
                log_path=log_path,
            ),
            explanation,
        )

    def export_blocker(
        self,
        meta: ModelMetadata | None,
        *,
        versions: Mapping[str, str | None] | None = None,
    ) -> Diagnostic | None:
        """The known failure exporting ``meta`` is certain to hit, or ``None``.

        An issue rule's ``blocks_export`` (schema 4) applies when it lists the
        model's ``model_type`` and every constraint of the rule and of the block
        holds for ``versions``: the interpreter and packages that will run the
        export, by default the ones this resolver is bound to. Unlike matching
        elsewhere, a version that is not known does not pass. Refusing an export
        that would have worked costs more than a failure the rule still
        classifies once it happens.

        The diagnostic is the rule's own, with the block's ``message`` when it
        has one, and the versions that decided it in ``context["versions"]``.
        """
        model_type = (meta.model_type or "").strip().casefold() if meta is not None else ""
        if not model_type:
            return None
        bound = self.versions if versions is None else versions
        rules = sorted(
            (rule for rule in self.registry.issues if rule.status.is_usable),
            key=lambda rule: (-rule.priority, rule.id),
        )
        for rule in rules:
            block = rule.blocks_export
            if block is None or model_type not in block.model_types:
                continue
            if not (_verified(rule.constraints, bound) and _verified(block.constraints, bound)):
                continue
            keys = sorted({*rule.constraints, *block.constraints})
            found = self._diagnostic_from_issue(
                rule,
                text="",
                device=None,
                stage="export",
                meta=meta,
                job_id=None,
                log_path=None,
            )
            return found.model_copy(
                update={
                    "message": block.message or found.message,
                    "raw_excerpt": None,
                    "context": {
                        **found.context,
                        "versions": ", ".join(f"{key} {bound[key]}" for key in keys),
                    },
                }
            )
        return None

    @staticmethod
    def _pick_issue(
        scored: Sequence[tuple[IssueRule, RuleCandidate, int]],
        explanation: ResolutionExplanation,
    ) -> IssueRule | None:
        """Pick the most specific matching issue rule, deterministically."""
        if not scored:
            return None
        ordered = sorted(
            scored,
            key=lambda item: (
                -item[1].specificity,
                -item[1].priority,
                -item[2],
                item[0].id,
            ),
        )
        best = ordered[0]
        if len(ordered) > 1:
            runner = ordered[1]
            if (
                runner[1].specificity == best[1].specificity
                and runner[1].priority == best[1].priority
                and runner[2] == best[2]
            ):
                explanation.notes.append(
                    f"Issue rules {best[0].id!r} and {runner[0].id!r} are equally specific; "
                    f"the alphabetically first id was used. Fix this with 'yonga registry "
                    f"validate'."
                )
        return best[0]

    def _diagnostic_from_issue(
        self,
        rule: IssueRule,
        *,
        text: str,
        device: str | None,
        stage: str | None,
        meta: ModelMetadata | None,
        job_id: str | None,
        log_path: Path | None,
    ) -> Diagnostic:
        """Build the diagnostic described by a matching issue rule."""
        try:
            category = FailureCategory(rule.classification.category)
        except ValueError:
            category = FailureCategory.UNKNOWN_RUNTIME_ERROR
        recommendation = rule.recommendation
        title = recommendation.title or rule.summary or "Known failure"
        message = recommendation.message or rule.summary or title
        remediation = list(recommendation.remediation)
        context = _context(device=device, stage=stage, meta=meta)
        if recommendation.repair is not None:
            context["repair"] = recommendation.repair
            repair_rule = self.registry.repair(recommendation.repair)
            if repair_rule is not None:
                remediation.append(
                    f"Apply the '{repair_rule.id}' repair (implementation "
                    f"{repair_rule.implementation}) to the artifact."
                )
        return fill_placeholders(
            Diagnostic(
                code=rule.classification.code,
                category=category,
                severity=Severity.ERROR,
                title=title,
                message=message,
                remediation=remediation,
                fallback_devices=list(recommendation.fallback_devices),
                environment_broken=rule.classification.environment_broken,
                retry_same_device=rule.classification.retry_same_device,
                defer_to_pipeline=rule.classification.defer_to_pipeline,
                matched_rule_id=rule.id,
                raw_excerpt=_excerpt(text),
                log_path=log_path,
                job_id=job_id,
                context=context,
            )
        )

    @staticmethod
    def _unknown_diagnostic(
        text: str,
        *,
        device: str | None,
        stage: str | None,
        meta: ModelMetadata | None,
        job_id: str | None,
        log_path: Path | None,
    ) -> Diagnostic:
        """Build the diagnostic used when nothing recognised the failure."""
        return Diagnostic(
            code=_UNKNOWN_CODE,
            category=FailureCategory.UNKNOWN_RUNTIME_ERROR,
            severity=Severity.ERROR,
            title="Unrecognised failure",
            message=(
                "The failure does not match any known signature. The full error text is kept "
                "in the job log."
            ),
            remediation=[
                "Re-run with --debug and attach the job log to a bug report.",
            ],
            raw_excerpt=_excerpt(text),
            log_path=log_path,
            job_id=job_id,
            context=_context(device=device, stage=stage, meta=meta),
        )

    # ------------------------------------------------------------------
    # repairs
    # ------------------------------------------------------------------
    def repairs_for(
        self,
        *,
        files: Iterable[str | Path],
        contents_probe: Callable[[str, str], bool] | None = None,
    ) -> list[RepairRule]:
        """Return the repair rules that apply to an artifact.

        ``files`` are the file names present in the artifact directory.
        ``contents_probe(file_name, needle)`` answers whether a file contains the
        rule's ``contains`` marker; when it is ``None`` the content criterion is
        left unverified and the rule is returned as a candidate, because the
        repair implementation re-checks the file before touching it.
        """
        names = [Path(entry).name for entry in files]
        selected: list[RepairRule] = []
        for rule in self.registry.repairs:
            if not rule.status.is_usable:
                continue
            constraints_ok, _reasons = self._constraints_ok(rule.constraints)
            if not constraints_ok:
                continue
            candidates = rule.match.file_candidates(names)
            if not candidates:
                continue
            if not _contents_match(rule, candidates, contents_probe):
                continue
            selected.append(rule)
        selected.sort(key=lambda rule: (-rule.priority, rule.id))
        return selected

    # ------------------------------------------------------------------
    # chat profiles
    # ------------------------------------------------------------------
    def chat_profile(
        self,
        *,
        arch_rule: ArchitectureRule | None = None,
        repo_id: str | None = None,
        meta: ModelMetadata | None = None,
    ) -> ChatProfileRule | None:
        """Resolve the chat profile for a model family, if the registry has one."""
        rule, _explanation = self.resolve_chat_profile(
            arch_rule=arch_rule, repo_id=repo_id, meta=meta
        )
        return rule

    def resolve_chat_profile(
        self,
        *,
        arch_rule: ArchitectureRule | None = None,
        repo_id: str | None = None,
        meta: ModelMetadata | None = None,
    ) -> tuple[ChatProfileRule | None, ResolutionExplanation]:
        """Resolve the chat profile and return the candidate trail with it.

        Two answers are compared rather than one preferred outright. The
        architecture rule may *name* a chat profile, and a chat rule may match the
        model's own identity; the more specific of the two wins, with the named
        one breaking a tie because naming it was deliberate.

        Returning on the named profile alone is what used to make a catch-all
        architecture rule hide a family's own chat profile: every model covered by
        ``arch-generic-text-generation`` inherited ``chat-default`` even when the
        registry held a rule matching its exact ``model_type``.
        """
        explanation = ResolutionExplanation(subject="chat profile")
        named: ChatProfileRule | None = None
        if arch_rule is not None and arch_rule.chat_profile is not None:
            named = self.registry.chat_profile(arch_rule.chat_profile)
            if named is None:
                explanation.notes.append(
                    f"Architecture rule {arch_rule.id} names unknown chat profile "
                    f"{arch_rule.chat_profile!r}."
                )
        subject = self._chat_subject(arch_rule, repo_id, meta)
        scored: list[tuple[ChatProfileRule, RuleCandidate]] = []
        for rule in self.registry.chat:
            candidate, _matched = self._evaluate(rule, subject)
            if named is not None and rule.id == named.id and arch_rule is not None:
                candidate.reasons.append(f"named by architecture rule {arch_rule.id}")
            explanation.candidates.append(candidate)
            scored.append((rule, candidate))
        matched = self._rank(scored, subject="chat profile")
        winner = more_specific_chat_rule(named, matched)
        if winner is not None:
            explanation.winner_id = winner.id
        return winner, explanation

    def _chat_subject(
        self,
        arch_rule: ArchitectureRule | None,
        repo_id: str | None,
        meta: ModelMetadata | None,
    ) -> MatchSubject:
        """Build the match subject for chat resolution, falling back to the arch rule."""
        subject = self._subject(meta)
        if meta is not None and (subject.model_type or subject.architectures):
            if repo_id is not None and subject.repo_id is None:
                return subject.model_copy(update={"repo_id": repo_id})
            return subject
        model_type: str | None = subject.model_type
        architectures: tuple[str, ...] = subject.architectures
        if arch_rule is not None:
            if model_type is None and arch_rule.match.model_types:
                model_type = arch_rule.match.model_types[0]
            if not architectures:
                architectures = tuple(arch_rule.match.architectures)
        return MatchSubject(
            repo_id=repo_id or subject.repo_id,
            model_type=model_type,
            architectures=architectures,
            task=subject.task,
            host_tags=self.host_tags,
        )

    # ------------------------------------------------------------------
    # device compatibility
    # ------------------------------------------------------------------
    def device_compatibility(
        self,
        *,
        arch_rule: ArchitectureRule | None = None,
        inventory: HardwareInventory | None = None,
        model_rule: ModelRule | None = None,
    ) -> list[DeviceCompatibility]:
        """Return what the registry knows about every relevant device.

        Devices come from the host inventory and from the rules themselves, so a
        device the registry has an opinion about is reported even when OpenVINO
        does not expose it here.
        """
        available: list[str] = []
        if inventory is not None:
            available = list(inventory.available_device_names)
        devices: list[str] = list(available)
        for rule in (model_rule, arch_rule):
            if rule is None:
                continue
            for name in rule.devices:
                if name not in devices:
                    devices.append(name)
        results: list[DeviceCompatibility] = []
        for device in devices:
            results.append(
                self._device_entry(
                    device,
                    arch_rule=arch_rule,
                    model_rule=model_rule,
                    available=available,
                    inventory_present=inventory is not None,
                )
            )
        return results

    def _device_entry(
        self,
        device: str,
        *,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
        available: Sequence[str],
        inventory_present: bool,
    ) -> DeviceCompatibility:
        """Build the compatibility entry for a single device."""
        notes: list[str] = []
        if inventory_present and device not in available:
            notes.append("OpenVINO does not expose this device on this host.")
        for rule in (model_rule, arch_rule):
            if rule is None:
                continue
            support = rule.device_support(device)
            if support is None:
                continue
            constraints_ok, reasons = self._constraints_ok(support.constraints)
            if not constraints_ok:
                notes.extend(reasons)
                notes.append(
                    f"The registry claim for {device} is scoped to versions this host does not "
                    f"run, so it is treated as unknown."
                )
                return DeviceCompatibility(
                    device=device,
                    status=DeviceStatus.UNKNOWN,
                    evidence=EvidenceLevel.UNKNOWN,
                    reason=support.reason,
                    rule_id=rule.id,
                    notes=[*notes, *support.notes],
                )
            notes.extend(reasons)
            return DeviceCompatibility(
                device=device,
                status=support.status,
                evidence=support.evidence,
                reason=support.reason,
                rule_id=rule.id,
                notes=[*notes, *support.notes],
            )
        return DeviceCompatibility(
            device=device,
            status=DeviceStatus.UNKNOWN,
            evidence=EvidenceLevel.UNKNOWN,
            reason="no_registry_rule",
            rule_id=arch_rule.id if arch_rule is not None else None,
            notes=[*notes, "The registry has no claim about this device for this architecture."],
        )

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------
    def explain(
        self,
        *,
        meta: ModelMetadata | None = None,
        inventory: HardwareInventory | None = None,
        device: str | None = None,
    ) -> RegistryExplanation:
        """Produce the full audit trail printed by ``yonga registry explain``."""
        host_tags = list(self.host_tags)
        if inventory is not None and inventory.device_tags:
            host_tags = list(inventory.device_tags)
        arch_rule, arch_explanation = self.resolve_architecture(meta, host_tags)
        model_rule, model_explanation = self.resolve_model_rule(meta)
        chat_rule, chat_explanation = self.resolve_chat_profile(arch_rule=arch_rule, meta=meta)
        repo_id = meta.identity.repo_id if meta is not None else None
        derivatives: list[DerivativeCandidate] = []
        derivative_explanation = ResolutionExplanation(subject="derivative sources")
        if repo_id is not None:
            derivatives, derivative_explanation = self.resolve_derivatives(repo_id)
        compatibility = self.device_compatibility(
            arch_rule=arch_rule, inventory=inventory, model_rule=model_rule
        )
        recommended = arch_rule.recommended_device if arch_rule is not None else None
        notes: list[str] = []
        target = normalise_device(device) if device else recommended
        profile_explanation: ResolutionExplanation | None = None
        profile_id: str | None = None
        if target:
            # As prepare decides it: a GGUF or OpenVINO repository keeps its own
            # quantization, so its size must not choose a profile here either.
            primary = meta.primary_format if meta is not None else None
            size_applies = primary not in (ModelFormat.GGUF, ModelFormat.OPENVINO_IR)
            try:
                profile, profile_explanation = self.resolve_profile(
                    device=target,
                    arch_rule=arch_rule,
                    meta=meta,
                    model_rule=model_rule,
                    size_applies=size_applies,
                )
                profile_id = profile.id
            except RegistryError as error:
                notes.append(error.diagnostic.message)
        return RegistryExplanation(
            repo_id=repo_id,
            device=target or None,
            host_tags=host_tags,
            versions=self.known_versions(),
            architecture_rule_id=arch_rule.id if arch_rule is not None else None,
            architecture=arch_explanation,
            model_rule_id=model_rule.id if model_rule is not None else None,
            model=model_explanation,
            profile_id=profile_id,
            profile=profile_explanation,
            chat_profile_id=chat_rule.id if chat_rule is not None else None,
            chat=chat_explanation,
            chat_tool_calling=(
                chat_rule.tool_calling.describe()
                if chat_rule is not None and chat_rule.tool_calling is not None
                else None
            ),
            derivatives=derivatives,
            derivative_resolution=derivative_explanation,
            device_compatibility=compatibility,
            recommended_device=recommended,
            notes=notes,
        )


def _note_identity_hint(
    rule: ArchitectureRule,
    subject: MatchSubject,
    explanation: ResolutionExplanation,
) -> None:
    """Say so when the winning rule was reached through an identity hint.

    A hint only ever identifies a repository that declares nothing of its own,
    so the note can state that: it never reports a header overriding a
    ``config.json``, because that never happens.

    The rule's match block is re-evaluated rather than threaded back out of
    :meth:`RegistryResolver._evaluate`, which keeps that helper's signature
    shared by every rule kind. Evaluation is pure and deterministic, and only
    the winner is re-checked, so this cannot disagree with the ranking.
    """
    outcome = rule.match_block.evaluate(subject)
    if outcome.alias_hint is None:
        return
    explanation.notes.append(
        f"Architecture identified from {IDENTITY_HINT_SOURCE} "
        f"(alias {outcome.alias_hint!r}); this repository has no config.json."
    )


def _contents_match(
    rule: RepairRule,
    candidates: Sequence[str],
    contents_probe: Callable[[str, str], bool] | None,
) -> bool:
    """Check a repair rule's ``contains`` marker, if it has one and a probe exists."""
    needle = rule.match.contains
    if needle is None or contents_probe is None:
        return True
    return any(contents_probe(name, needle) for name in candidates)


def _verified(constraints: Mapping[str, str], versions: Mapping[str, str | None]) -> bool:
    """True when every constraint names a known version that satisfies it."""
    for key, specifier in constraints.items():
        version = versions.get(key)
        if version is None or not satisfies(version, specifier):
            return False
    return True


def _context(
    *,
    device: str | None,
    stage: str | None,
    meta: ModelMetadata | None,
) -> dict[str, str]:
    """Build the string-only diagnostic context block."""
    context: dict[str, str] = {}
    if device:
        context["device"] = normalise_device(device)
    if stage:
        context["stage"] = stage
    if meta is not None:
        context["repo_id"] = meta.identity.repo_id
        if meta.model_type:
            context["model_type"] = meta.model_type
        if meta.architectures:
            context["architecture"] = meta.architectures[0]
    return context
