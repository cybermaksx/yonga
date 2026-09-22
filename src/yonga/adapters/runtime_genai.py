"""OpenVINO GenAI chat/bench runtime and the domain-facing chat types.

The UI and the application services talk to :class:`ChatRuntime`; only this
module ever imports ``openvino_genai``. The conversation types
(:class:`ChatMessage`, :class:`ChatHistory`, :class:`GenerationResult`) are
*ours*, deliberately distinct from ``openvino_genai.ChatHistory`` — the GenAI
object is built fresh inside :meth:`GenAIChatRuntime.generate` and never leaves
this file. ``application/chat.py`` should re-export these names rather than
redefine them.

Two upstream details are hidden here on purpose:

* ``start_chat()``/``finish_chat()`` are deprecated and must not appear in the
  domain interface. The LLM path does not use them at all: multi-turn state
  travels as an ``openvino_genai.ChatHistory`` passed to ``generate``. The VLM
  path has no such overload in this build, so the adapter drives the deprecated
  lifecycle internally and feeds the pipeline only the newest user turn.
* Reasoning ("thinking") is switched off through the chat template variable via
  ``ChatHistory.set_extra_context``. The source repository's chat template is
  never rewritten. *Which* variable to send is decided from evidence: unless the
  registry profile states the answer outright, the artifact's own chat template
  is read and a candidate is applied only when that template actually references
  it. See :func:`resolve_thinking_variable`.

:class:`GenerationResult` carries the **raw** model text. ``display_text`` is set
equal to ``text`` and ``reasoning_text`` to ``None`` here; hiding reasoning
blocks is the chat service's ``sanitize_reasoning`` step, which owns the
show/hide policy and the unclosed-wrapper rule.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters.openvino_probe import (
    artifact_fingerprint,
    compile_cache_dir,
    compile_cache_key,
    openvino_runtime_version,
)
from yonga.config.settings import Settings
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import (
    CompileError,
    Diagnostic,
    FailureCategory,
    GenerationError,
    YongaError,
    diagnostic,
)
from yonga.domain.models import PipelineKind
from yonga.util.redaction import redact_text

__all__ = [
    "CHAT_TEMPLATE_FILES",
    "DEFAULT_REASONING_WRAPPERS",
    "DEFAULT_THINKING_VARIABLES",
    "ChatHistory",
    "ChatMessage",
    "ChatProfile",
    "ChatProfileLike",
    "ChatRuntime",
    "GenAIChatRuntime",
    "GenerationMetrics",
    "GenerationResult",
    "Role",
    "build_generation_config",
    "collect_metrics",
    "open_runtime",
    "read_chat_template",
    "resolve_thinking_variable",
    "template_references",
    "thinking_extra_context",
]

#: Reasoning wrappers understood by the default chat profile.
DEFAULT_REASONING_WRAPPERS: tuple[tuple[str, str], ...] = (("<think>", "</think>"),)

#: Thinking variables tried against an artifact's chat template when its chat
#: profile names none. Mirrors the registry's default so a profile-less runtime
#: behaves the same way as one driven by ``chat-default``.
DEFAULT_THINKING_VARIABLES: tuple[str, ...] = ("enable_thinking",)

#: Files an artifact may keep its chat template in, in the order they are read.
CHAT_TEMPLATE_FILES: tuple[str, ...] = (
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer_config.json",
)

_DEFAULT_DEVICE = "CPU"
_CACHE_DIR_PROPERTY = "CACHE_DIR"
_LENIENT = ConfigDict(extra="ignore")

#: The three conversation roles the chat template contract uses.
Role = Literal["system", "user", "assistant"]


# --- conversation types ---------------------------------------------------------------


class ChatMessage(BaseModel):
    """One turn in a conversation."""

    model_config = _LENIENT

    role: Role
    content: str

    def as_payload(self) -> dict[str, str]:
        """Return the ``{"role": ..., "content": ...}`` mapping GenAI expects."""
        return {"role": self.role, "content": self.content}


class ChatHistory(BaseModel):
    """An ordered conversation, owned by the domain rather than by GenAI.

    Keeping our own type means the UI, the chat service and the transcript writer
    never import ``openvino_genai``; the adapter converts this into the GenAI
    object immediately before each generation.
    """

    model_config = _LENIENT

    messages: list[ChatMessage] = Field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)

    def add_system(self, content: str) -> None:
        """Append a system message."""
        self.messages.append(ChatMessage(role="system", content=content))

    def add_user(self, content: str) -> None:
        """Append a user message."""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def reset(self, keep_system: bool = True) -> None:
        """Clear the conversation, optionally preserving the leading system turns."""
        if keep_system:
            self.messages = [message for message in self.messages if message.role == "system"]
        else:
            self.messages = []

    @property
    def system_message(self) -> str | None:
        """The first system message's content, when the conversation has one."""
        return next((m.content for m in self.messages if m.role == "system"), None)

    def to_payload(self) -> list[dict[str, str]]:
        """Return the conversation as a list of role/content mappings."""
        return [message.as_payload() for message in self.messages]


class GenerationMetrics(BaseModel):
    """Performance numbers extracted from one ``PerfMetrics`` object.

    Every field is optional: a missing or raising metric getter yields ``None``
    and must never break generation (CLI_SPEC section 9).
    """

    model_config = _LENIENT

    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tps: float | None = None
    load_ms: float | None = None


class GenerationResult(BaseModel):
    """The outcome of one generation turn.

    ``text`` is the raw model output. ``display_text``, ``reasoning_text`` and the
    two ``reasoning_*`` flags are filled in by the chat service's sanitiser; this
    adapter sets ``display_text = text`` and leaves the rest at their defaults so
    a caller that does not sanitise still shows something truthful.

    ``reasoning_hidden`` and ``reasoning_unterminated`` exist so the UI can
    explain an empty answer without looking at the text again. They describe the
    *shape* of what was hidden, never its content: a renderer may say that 200
    tokens of reasoning were suppressed and that the block never closed, and must
    not be able to reconstruct a single word of it.
    """

    model_config = _LENIENT

    text: str
    display_text: str
    reasoning_text: str | None = None
    reasoning_hidden: bool = False
    """True when a reasoning block was found and removed from ``display_text``."""

    reasoning_unterminated: bool = False
    """True when a reasoning block was opened and never closed before the text ended.

    In practice this means the model ran into its token limit mid-thought, which
    is the difference between "it had nothing to say" and "it never got to the
    answer".
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    throughput_tps: float | None = None
    tpot_ms: float | None = None
    wall_seconds: float = 0.0
    load_seconds: float | None = None
    stop_reason: str | None = None


# --- chat profile ---------------------------------------------------------------------


class ChatProfileLike(Protocol):
    """Structural type for a registry chat profile.

    The registry's ``ChatProfileRule`` satisfies this by declaring the same
    attribute names with the same types; the adapter reads nothing else.
    ``supports_thinking_flag`` is tri-state — ``None`` means "no opinion, decide
    from the artifact's chat template".
    """

    supports_thinking_flag: bool | None
    thinking_template_variable: str | None
    thinking_variable_candidates: list[str]
    default_thinking: bool


class ChatProfile(BaseModel):
    """Concrete chat profile, used when no registry rule matched.

    ``reasoning_wrappers`` is consumed by the chat service's sanitiser, not by
    this adapter.
    """

    model_config = _LENIENT

    supports_thinking_flag: bool | None = None
    thinking_template_variable: str | None = "enable_thinking"
    thinking_variable_candidates: list[str] = Field(
        default_factory=lambda: [*DEFAULT_THINKING_VARIABLES]
    )
    default_thinking: bool = False
    reasoning_wrappers: list[tuple[str, str]] = Field(
        default_factory=lambda: [*DEFAULT_REASONING_WRAPPERS]
    )


def read_chat_template(artifact_dir: Path) -> str | None:
    """Return the chat-template source an artifact ships, or ``None``.

    The template is read, never written: suppressing reasoning by editing the
    model's own template is forbidden (``SPEC.md`` section 18.4,
    ``MODEL_PIPELINE.md`` section 13), so this is evidence-gathering only.

    Exporters put the template in one of three places, checked in that order: a
    standalone ``chat_template.jinja``, a ``chat_template.json`` wrapper, or the
    ``chat_template`` key of ``tokenizer_config.json``. Newer Transformers
    releases may store a *list* of named templates there; every one of them is
    returned joined together, because any of them could be the one that runs.

    Never raises. An unreadable or malformed file is simply no evidence, and the
    caller then leaves the template exactly as the model author wrote it.
    """
    directory = Path(artifact_dir)
    for name in CHAT_TEMPLATE_FILES:
        path = directory / name
        try:
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        template = raw if path.suffix == ".jinja" else _template_from_json(raw)
        if template:
            return template
    return None


def _template_from_json(raw: str) -> str | None:
    """Pull the ``chat_template`` value out of a JSON config, whatever shape it has."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("chat_template")
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        parts = [
            entry["template"]
            for entry in value
            if isinstance(entry, dict) and isinstance(entry.get("template"), str)
        ]
        return "\n".join(parts) or None
    return None


def template_references(template: str | None, variable: str) -> bool:
    """True when ``template`` uses ``variable`` as a name of its own.

    The match is on whole identifiers, so ``enable_thinking`` is not found inside
    ``disable_enable_thinking2``. A name reached through an attribute, as in
    ``message.enable_thinking``, does not count either: that is a property of a
    message, not a variable the caller can set through extra context.
    """
    name = (variable or "").strip()
    if not template or not name:
        return False
    pattern = rf"(?<![\w.]){re.escape(name)}(?![\w])"
    return re.search(pattern, template) is not None


def resolve_thinking_variable(
    profile: ChatProfileLike | None,
    *,
    template: str | None = None,
) -> str | None:
    """Return the chat-template variable that drives thinking mode, or ``None``.

    Three sources, in order of authority:

    1. ``supports_thinking_flag: false`` is a veto. Nothing is sent, whatever the
       template mentions.
    2. ``supports_thinking_flag: true`` is a declaration. The profile's
       ``thinking_template_variable`` is used as-is, because the registry author
       has stated the family honours it.
    3. Otherwise the candidates are tried against ``template`` and the first one
       the template actually references wins. A profile that names a variable
       makes that name the only candidate.

    With no template (``None``) there is no evidence, so only an explicit
    declaration can produce a variable. That is the safe direction to fail: a
    template is left exactly as its author wrote it.
    """
    if profile is None:
        return None
    declared = getattr(profile, "supports_thinking_flag", None)
    named = str(getattr(profile, "thinking_template_variable", "") or "").strip()
    if declared is False:
        return None
    if declared is True and named:
        return named
    candidates = (named,) if named else _candidate_variables(profile)
    return next((name for name in candidates if template_references(template, name)), None)


def _candidate_variables(profile: ChatProfileLike) -> tuple[str, ...]:
    """Return a profile's candidate variable names, tolerating an odd value."""
    raw = getattr(profile, "thinking_variable_candidates", None)
    if raw is None:
        return DEFAULT_THINKING_VARIABLES
    try:
        items = list(raw)
    except TypeError:
        return DEFAULT_THINKING_VARIABLES
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def thinking_extra_context(
    profile: ChatProfileLike | None,
    *,
    thinking: bool | None = None,
    template: str | None = None,
) -> dict[str, bool]:
    """Return the chat-template variables to push through ``set_extra_context``.

    Returns an empty mapping when no thinking variable could be established,
    which is the safe default: an unknown template is left exactly as the model
    author wrote it. ``thinking`` overrides the profile's ``default_thinking``,
    and ``template`` is the artifact's own chat-template source, which is what
    turns a candidate name into evidence.
    """
    variable = resolve_thinking_variable(profile, template=template)
    if variable is None:
        return {}
    enabled = (
        bool(getattr(profile, "default_thinking", False)) if thinking is None else bool(thinking)
    )
    return {variable: enabled}


# --- runtime protocol -----------------------------------------------------------------


@runtime_checkable
class ChatRuntime(Protocol):
    """A loaded model that can answer turns. Deliberately free of GenAI types."""

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        """Generate one assistant turn for ``history``."""
        ...

    def close(self) -> None:
        """Release the pipeline. Must be idempotent."""
        ...


# --- lazy import ----------------------------------------------------------------------


def _import_genai() -> Any:
    """Import :mod:`openvino_genai` lazily. Raises ``ImportError`` when unavailable."""
    # Deliberately lazy: see the module docstring.
    import openvino_genai

    return openvino_genai


def _genai_missing_diagnostic(error: object) -> Diagnostic:
    """Build the diagnostic used when ``openvino_genai`` cannot be imported."""
    detail = redact_text(str(error))
    return diagnostic(
        "runtime.openvino_genai.missing",
        FailureCategory.PACKAGE_ABI_MISMATCH,
        "OpenVINO GenAI is not available",
        f"openvino_genai could not be imported, so no model can be run. {detail}",
        remediation=[
            "Install the runtime extras: pip install 'yonga[openvino]'",
            "openvino-genai must come from the same release family as openvino.",
        ],
        environment_broken=True,
        raw_excerpt=detail,
    )


def _require_genai() -> Any:
    """Return the ``openvino_genai`` module or raise a typed :class:`CompileError`."""
    try:
        return _import_genai()
    except Exception as exc:
        raise CompileError(_genai_missing_diagnostic(exc), cause=exc) from exc


# --- generation config ----------------------------------------------------------------


def build_generation_config(settings: GenerationSettings) -> Any:
    """Map :class:`GenerationSettings` onto an ``openvino_genai.GenerationConfig``.

    ``temperature``, ``top_p``, ``top_k`` and the RNG seed only take effect when
    ``do_sample`` is true, so they are applied only in that case — setting them
    for greedy decoding would silently suggest a sampling behaviour the runtime
    will not perform. ``repetition_penalty`` applies to both modes. The config is
    validated before it is returned.
    """
    genai = _require_genai()
    config = genai.GenerationConfig()
    config.max_new_tokens = max(1, int(settings.max_new_tokens))
    config.do_sample = bool(settings.do_sample)
    if settings.repetition_penalty is not None:
        config.repetition_penalty = float(settings.repetition_penalty)
    if settings.do_sample:
        if settings.temperature is not None:
            config.temperature = float(settings.temperature)
        if settings.top_p is not None:
            config.top_p = float(settings.top_p)
        if settings.top_k is not None:
            config.top_k = int(settings.top_k)
        if settings.seed is not None:
            config.rng_seed = int(settings.seed)
    try:
        config.validate()
    except Exception as exc:
        raise GenerationError(
            diagnostic(
                "runtime.generation.invalid_config",
                FailureCategory.CONFIGURATION_ERROR,
                "The generation settings were rejected by OpenVINO GenAI",
                redact_text(str(exc)),
                remediation=[
                    "Enable sampling (--temperature/--top-p/--top-k require do_sample).",
                    "Check that max_new_tokens is positive.",
                ],
                raw_excerpt=redact_text(str(exc)),
            ),
            cause=exc,
        ) from exc
    return config


# --- metrics --------------------------------------------------------------------------


def _finite(value: float) -> float | None:
    """Return ``value`` when it is a finite number, otherwise ``None``."""
    return value if math.isfinite(value) else None


def _metric_float(perf_metrics: object, getter: str) -> float | None:
    """Read one ``MeanStdPair`` (or plain float) metric defensively."""
    try:
        value = getattr(perf_metrics, getter)()
    except Exception:
        return None
    raw = getattr(value, "mean", value)
    try:
        return _finite(float(raw))
    except (TypeError, ValueError):
        return None


def _metric_int(perf_metrics: object, getter: str) -> int | None:
    """Read one integer metric defensively."""
    try:
        value = getattr(perf_metrics, getter)()
    except Exception:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_metrics(perf_metrics: object | None) -> GenerationMetrics:
    """Extract every metric we report from a GenAI ``PerfMetrics`` object.

    Each getter is called in isolation: one metric that is absent, raises, or
    returns something unexpected costs exactly that one field. ``get_load_time``
    returns a plain float while the others return ``MeanStdPair``; both shapes are
    handled by reading an optional ``mean`` attribute.
    """
    if perf_metrics is None:
        return GenerationMetrics()
    return GenerationMetrics(
        input_tokens=_metric_int(perf_metrics, "get_num_input_tokens"),
        output_tokens=_metric_int(perf_metrics, "get_num_generated_tokens"),
        ttft_ms=_metric_float(perf_metrics, "get_ttft"),
        tpot_ms=_metric_float(perf_metrics, "get_tpot"),
        throughput_tps=_metric_float(perf_metrics, "get_throughput"),
        load_ms=_metric_float(perf_metrics, "get_load_time"),
    )


def _first_text(raw: object) -> str:
    """Return the first decoded string of a GenAI result object."""
    with suppress(Exception):
        texts = list(raw.texts)  # type: ignore[attr-defined]
        if texts:
            return str(texts[0])
    with suppress(Exception):
        return str(raw)
    return ""


def _finish_reason(raw: object) -> str | None:
    """Return the first finish reason as a lower-case name, when reported."""
    with suppress(Exception):
        reasons = list(raw.finish_reasons)  # type: ignore[attr-defined]
        if reasons:
            first = reasons[0]
            name = getattr(first, "name", None)
            return str(name).lower() if name else str(first)
    return None


def _is_prefix(prefix: Sequence[ChatMessage], messages: Sequence[ChatMessage]) -> bool:
    """Return whether ``prefix`` is the leading run of ``messages``."""
    if len(prefix) > len(messages):
        return False
    return all(left == right for left, right in zip(prefix, messages, strict=False))


# --- the runtime ----------------------------------------------------------------------


class GenAIChatRuntime:
    """An OpenVINO GenAI pipeline kept alive across turns.

    The pipeline object is created once, in the constructor, and reused for every
    turn: reloading per prompt would pay the compile/load cost again and again
    (AGENT_PROMPT section 34). Call :meth:`close` (or use the runtime as a context
    manager) to release it.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        device: str = _DEFAULT_DEVICE,
        pipeline: PipelineKind = PipelineKind.LLM,
        properties: Mapping[str, str] | None = None,
        chat_profile: ChatProfileLike | None = None,
        settings: Settings | None = None,
        thinking: bool | None = None,
    ) -> None:
        """Load ``artifact_dir`` on ``device``.

        Raises :class:`~yonga.domain.diagnostics.CompileError` when the package is
        missing, the directory is not an artifact, or the pipeline fails to load;
        the raw runtime message is preserved for the registry classifier.
        """
        self._artifact_dir = Path(artifact_dir)
        self._device = (device or "").strip() or _DEFAULT_DEVICE
        self._pipeline_kind = pipeline
        self._chat_profile = chat_profile
        self._thinking = thinking
        self._settings = settings if settings is not None else Settings()
        self._properties = self._resolve_properties(properties)
        self._chat_template = read_chat_template(self._artifact_dir)
        self._thinking_variable = resolve_thinking_variable(
            chat_profile, template=self._chat_template
        )
        self._extra_context_error: str | None = None
        self._closed = False
        self._chat_started = False
        self._sent: list[ChatMessage] = []
        self._load_seconds = 0.0
        self._genai: Any = _require_genai()
        self._pipeline: Any = self._open_pipeline()

    # --- construction helpers --------------------------------------------

    def _resolve_properties(self, properties: Mapping[str, str] | None) -> dict[str, str]:
        """Merge the profile's runtime properties with the managed compile cache.

        The cache directory is keyed with the OpenVINO version so a runtime
        upgrade can never be handed blobs built by the previous release
        (ARCHITECTURE section 7). An explicit ``CACHE_DIR`` from the caller wins.
        """
        resolved = {str(name): str(value) for name, value in (properties or {}).items()}
        if not self._settings.compile_cache_enabled or _CACHE_DIR_PROPERTY in resolved:
            return resolved
        key = compile_cache_key(
            artifact_hash=artifact_fingerprint(self._artifact_dir),
            device=self._device,
            openvino_version=openvino_runtime_version(),
            properties=resolved,
        )
        resolved[_CACHE_DIR_PROPERTY] = str(compile_cache_dir(self._settings.paths, key))
        return resolved

    def _open_pipeline(self) -> Any:
        """Construct the GenAI pipeline for this artifact's pipeline kind."""
        if not self._artifact_dir.is_dir():
            raise CompileError(
                diagnostic(
                    "runtime.artifact.not_found",
                    FailureCategory.CONFIGURATION_ERROR,
                    "The prepared artifact directory does not exist",
                    f"No directory at {self._artifact_dir}.",
                    remediation=["Run `yonga prepare` first, or check the artifact path."],
                    context={"artifact_dir": str(self._artifact_dir)},
                )
            )
        started = time.perf_counter()
        try:
            if self._pipeline_kind is PipelineKind.VLM:
                pipeline = self._genai.VLMPipeline(
                    str(self._artifact_dir), self._device, **self._properties
                )
            else:
                pipeline = self._genai.LLMPipeline(
                    str(self._artifact_dir), self._device, **self._properties
                )
        except Exception as exc:
            raise CompileError(self._load_failed_diagnostic(exc), cause=exc) from exc
        self._load_seconds = time.perf_counter() - started
        return pipeline

    def _load_failed_diagnostic(self, error: BaseException) -> Diagnostic:
        """Build the load/compile failure diagnostic, keeping the raw runtime text."""
        raw = redact_text(str(error))
        return diagnostic(
            "runtime.genai.load_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"Could not load the model on {self._device}",
            raw,
            remediation=[
                "Run `yonga doctor` to check the device and driver state.",
                "Try another device, for example --device cpu.",
            ],
            raw_excerpt=raw,
            context={
                "device": self._device,
                "pipeline": self._pipeline_kind.value,
                "artifact_dir": str(self._artifact_dir),
            },
        )

    # --- introspection ---------------------------------------------------

    @property
    def artifact_dir(self) -> Path:
        """The artifact directory this runtime loaded."""
        return self._artifact_dir

    @property
    def device(self) -> str:
        """The OpenVINO device string the pipeline was created with."""
        return self._device

    @property
    def pipeline_kind(self) -> PipelineKind:
        """Whether this runtime drives an LLM or a VLM pipeline."""
        return self._pipeline_kind

    @property
    def properties(self) -> dict[str, str]:
        """The pipeline properties actually applied, including ``CACHE_DIR``."""
        return dict(self._properties)

    @property
    def load_seconds(self) -> float:
        """Wall-clock seconds spent constructing the pipeline (compile + load)."""
        return self._load_seconds

    @property
    def thinking_variable(self) -> str | None:
        """The chat-template variable this runtime drives, when one was established.

        ``None`` means the artifact's template names no thinking variable this
        profile knows about, so the template is left untouched.
        """
        return self._thinking_variable

    @property
    def extra_context_error(self) -> str | None:
        """Why the chat-template variables could not be applied, when that failed."""
        return self._extra_context_error

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    # --- generation ------------------------------------------------------

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        """Generate one assistant turn and collect its metrics.

        Raises :class:`~yonga.domain.diagnostics.GenerationError` on failure, with
        the raw runtime text preserved for the classifier.
        """
        if self._closed:
            raise GenerationError(
                diagnostic(
                    "runtime.genai.closed",
                    FailureCategory.CONFIGURATION_ERROR,
                    "The chat runtime has already been closed",
                    "Open a new runtime before generating again.",
                )
            )
        generation_config = build_generation_config(config)
        started = time.perf_counter()
        try:
            raw = self._generate_raw(history, generation_config)
        except YongaError:
            raise
        except Exception as exc:
            raise GenerationError(self._generation_failed_diagnostic(exc), cause=exc) from exc
        wall_seconds = time.perf_counter() - started
        text = _first_text(raw)
        metrics = collect_metrics(getattr(raw, "perf_metrics", None))
        load_seconds = (
            metrics.load_ms / 1000.0 if metrics.load_ms is not None else self._load_seconds
        )
        return GenerationResult(
            text=text,
            display_text=text,
            reasoning_text=None,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            ttft_ms=metrics.ttft_ms,
            throughput_tps=metrics.throughput_tps,
            tpot_ms=metrics.tpot_ms,
            wall_seconds=wall_seconds,
            load_seconds=load_seconds,
            stop_reason=_finish_reason(raw),
        )

    def _generate_raw(self, history: ChatHistory, generation_config: Any) -> Any:
        """Dispatch to the LLM or VLM generation path."""
        if self._pipeline_kind is PipelineKind.VLM:
            return self._generate_vlm(history, generation_config)
        return self._generate_llm(history, generation_config)

    def _generate_llm(self, history: ChatHistory, generation_config: Any) -> Any:
        """Generate with an ``LLMPipeline`` using a GenAI ``ChatHistory``.

        Passing the history object is what replaces the deprecated
        ``start_chat()``/``finish_chat()`` lifecycle: the whole conversation is
        re-supplied each turn and the pipeline applies the chat template itself.
        """
        genai_history = self._genai.ChatHistory()
        for message in history.messages:
            genai_history.append(message.as_payload())
        extra = thinking_extra_context(
            self._chat_profile, thinking=self._thinking, template=self._chat_template
        )
        if extra:
            try:
                genai_history.set_extra_context(dict(extra))
            except Exception as exc:
                # The template variable is a best-effort hint; the chat service's
                # sanitiser is the defence that actually hides reasoning text.
                self._extra_context_error = redact_text(f"{type(exc).__name__}: {exc}")
        return self._pipeline.generate(genai_history, generation_config)

    def _generate_vlm(self, history: ChatHistory, generation_config: Any) -> Any:
        """Generate with a ``VLMPipeline``.

        ``VLMPipeline.generate`` does not accept a ``ChatHistory`` in this build,
        so the deprecated chat lifecycle is driven here, entirely inside the
        adapter: the pipeline keeps its own conversation state and is handed only
        the turns it has not seen.
        """
        prompt = self._vlm_turn_prompt(history)
        return self._pipeline.generate(prompt, generation_config=generation_config)

    def _vlm_turn_prompt(self, history: ChatHistory) -> str:
        """Return the text to send to the VLM pipeline for this turn."""
        messages = list(history.messages)
        if not self._chat_started or not _is_prefix(self._sent, messages):
            self._restart_vlm_chat(messages)
        pending = messages[len(self._sent) :]
        prompt = "\n\n".join(
            message.content.strip()
            for message in pending
            if message.role == "user" and message.content.strip()
        )
        self._sent = messages
        if not prompt:
            raise GenerationError(
                diagnostic(
                    "runtime.generation.empty_prompt",
                    FailureCategory.CONFIGURATION_ERROR,
                    "There is no new user message to answer",
                    "Add a user message to the conversation before generating.",
                )
            )
        return prompt

    def _restart_vlm_chat(self, messages: Sequence[ChatMessage]) -> None:
        """(Re)start the hidden VLM chat session, seeding it with the system turn."""
        if self._chat_started:
            with suppress(Exception):
                self._pipeline.finish_chat()
            self._chat_started = False
        self._sent = []
        system = next((m.content for m in messages if m.role == "system"), "")
        try:
            if system:
                self._pipeline.start_chat(system)
            else:
                self._pipeline.start_chat()
        except Exception as exc:
            raise GenerationError(self._generation_failed_diagnostic(exc), cause=exc) from exc
        self._chat_started = True
        self._sent = [message for message in messages if message.role == "system"]

    def _generation_failed_diagnostic(self, error: BaseException) -> Diagnostic:
        """Build the generation failure diagnostic, keeping the raw runtime text."""
        raw = redact_text(str(error))
        return diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"Generation failed on {self._device}",
            raw,
            remediation=[
                "Retry with fewer tokens, or on another device.",
                "Run `yonga doctor` if the device stopped responding.",
            ],
            raw_excerpt=raw,
            context={
                "device": self._device,
                "pipeline": self._pipeline_kind.value,
                "artifact_dir": str(self._artifact_dir),
            },
        )

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Release the pipeline. Idempotent, and safe to call from a finally block."""
        if self._closed:
            return
        self._closed = True
        if self._chat_started:
            with suppress(Exception):
                self._pipeline.finish_chat()
            self._chat_started = False
        self._pipeline = None
        self._sent = []

    def __enter__(self) -> GenAIChatRuntime:
        """Return the runtime so it can be used as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the runtime when the ``with`` block ends."""
        self.close()


def open_runtime(
    artifact_dir: Path,
    *,
    device: str = _DEFAULT_DEVICE,
    pipeline: PipelineKind = PipelineKind.LLM,
    properties: Mapping[str, str] | None = None,
    chat_profile: ChatProfileLike | None = None,
    settings: Settings | None = None,
    thinking: bool | None = None,
) -> ChatRuntime:
    """Open a chat runtime for a prepared artifact.

    This is the single entry point the chat and bench services use. ``pipeline``
    comes from the artifact manifest's ``PipelineKind``;
    :attr:`~yonga.domain.models.PipelineKind.UNKNOWN` is driven as an LLM, which
    is what every text-only export is. ``thinking`` overrides the chat profile's
    ``default_thinking``.
    """
    return GenAIChatRuntime(
        artifact_dir,
        device=device,
        pipeline=pipeline,
        properties=properties,
        chat_profile=chat_profile,
        settings=settings,
        thinking=thinking,
    )
