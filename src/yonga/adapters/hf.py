"""Hugging Face Hub adapter: metadata, format detection and pinned downloads.

This module is the only place in yonga that talks to the Hugging Face Hub. It
implements the :class:`yonga.adapters.HubClient` Protocol on top of
``huggingface_hub`` — never hand-rolled HTTP — and it is deliberately read-only:
it reads repository metadata, fetches the handful of *small* JSON files needed to
identify an architecture, and materialises snapshots in the standard Hub cache.
It never mutates a source repository, never copies a snapshot and never enables
``trust_remote_code`` on the caller's behalf.

``huggingface_hub`` is a declared runtime dependency and is cheap to import, so
it is imported at module scope. ``transformers``, ``torch``, ``openvino`` and the
rest of the heavy stack are not imported here at all.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

import huggingface_hub
from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import (
    DisabledRepoError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.repocard import metadata_load
from huggingface_hub.utils import validate_repo_id  # type: ignore[attr-defined]
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm

from yonga.adapters.gguf_header import (
    GGUFFormatError,
    GGUFHeader,
    parse_gguf_header,
    read_local_header,
)
from yonga.application.progress import ProgressKind
from yonga.config.settings import Settings
from yonga.domain.diagnostics import (
    CancelledError,
    Diagnostic,
    FailureCategory,
    HubAuthError,
    Severity,
    SourceResolutionError,
    YongaError,
    diagnostic,
)
from yonga.domain.hardware import HuggingFaceAuthInfo
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    RepoFile,
)
from yonga.util.redaction import redact_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.application.jobs import JobContext

__all__ = [
    "CODE_AUTH_REQUIRED",
    "CODE_FILE_NOT_FOUND",
    "CODE_GATED",
    "CODE_HTTP_ERROR",
    "CODE_INVALID_REPO_ID",
    "CODE_NETWORK_ERROR",
    "CODE_OFFLINE_MISSING",
    "CODE_OFFLINE_SEARCH",
    "CODE_REPO_NOT_FOUND",
    "CODE_REVISION_NOT_FOUND",
    "CODE_REVISION_UNRESOLVED",
    "CODE_UNKNOWN",
    "CONFIG_FILENAME",
    "DEFAULT_REVISION",
    "HuggingFaceAdapter",
    "detect_formats",
    "detect_remote_code",
    "is_commit_sha",
    "translate_hub_error",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_REVISION: Final[str] = "main"
"""Revision used when the caller did not pin anything and we cannot resolve one."""

CONFIG_FILENAME: Final[str] = "config.json"
"""The one small file ``inspect`` downloads to learn the architecture."""

MAX_METADATA_FILE_BYTES: Final[int] = 8 * 1024 * 1024
"""Refuse to fetch a "small" metadata file bigger than this; note it instead."""

GGUF_METADATA_MAX_VALUE_CHARS: Final[int] = 512
"""GGUF metadata values longer than this (chat templates) are recorded as elided."""

GGUF_TOTAL_FILE_SIZE_KEY: Final[str] = "totalFileSize"
"""Hub GGUF metadata key holding the byte size of the repository's primary model file.

The Hub parses the repository's GGUF headers itself and reports the size of the
one model it considers the repository's subject, which is exactly the
``primary_bytes`` question. It is preferred over any local guess.
"""

GGUF_COMPANION_PREFIXES: Final[tuple[str, ...]] = ("mmproj",)
"""Filename prefixes marking a ``.gguf`` side-car that is not a runnable model.

HEURISTIC: this is a filename convention, not a fact carried in the file
listing. ``llama.cpp`` tooling names the multimodal projector that accompanies a
vision model ``mmproj-<model>.gguf``, and nothing in the Hub's file inventory
distinguishes it from a model file without downloading and parsing the header.
Getting this wrong only degrades a fallback estimate, never the figure the Hub
itself reported through :data:`GGUF_TOTAL_FILE_SIZE_KEY`.
"""

PROGRESS_MIN_INTERVAL_SECONDS: Final[float] = 0.2
"""Minimum wall-clock gap between two download progress events."""

OPENVINO_IR_PREFIX: Final[str] = "openvino"
SAFETENSORS_SUFFIX: Final[str] = ".safetensors"
GGUF_SUFFIX: Final[str] = ".gguf"
PYTORCH_BIN_SUFFIX: Final[str] = ".bin"
XML_SUFFIX: Final[str] = ".xml"
PYTHON_SUFFIX: Final[str] = ".py"

SAFETENSORS_INDEX_SUFFIX: Final[str] = ".safetensors.index.json"
PYTORCH_INDEX_SUFFIX: Final[str] = ".bin.index.json"

SAFETENSORS_SINGLE_FILENAME: Final[str] = "model.safetensors"
SAFETENSORS_INDEX_FILENAME: Final[str] = "model.safetensors.index.json"
"""The two layouts the Hub itself reads a parameter count from (``get_safetensors_metadata``)."""

MAX_SAFETENSORS_HEADER_BYTES: Final[int] = 100 * 1024 * 1024
"""The safetensors format's own ceiling on a header; a larger length is a damaged file."""

README_FILENAME: Final[str] = "README.md"
"""The model card, whose YAML front matter carries the licence, tags and pipeline tag."""

CUSTOM_CODE_PREFIXES: Final[tuple[str, ...]] = (
    "modeling_",
    "configuration_",
    "tokenization_",
    "processing_",
    "image_processing_",
    "video_processing_",
    "feature_extraction_",
    "audio_processing_",
)
"""Transformers' naming convention for repository-hosted custom model code."""

LOGIN_HINT: Final[str] = (
    "Authenticate with `hf auth login`, then re-run the command. "
    "A token exported as HF_TOKEN works too."
)
ACCEPT_HINT: Final[str] = (
    "Open the model page in a browser and accept its licence/access conditions first."
)

# Stable diagnostic codes. Application code and the registry match on these.
CODE_GATED: Final[str] = "hub.auth.gated"
CODE_AUTH_REQUIRED: Final[str] = "hub.auth.required"
CODE_REPO_NOT_FOUND: Final[str] = "hub.repo.not_found"
CODE_REVISION_NOT_FOUND: Final[str] = "hub.revision.not_found"
CODE_REVISION_UNRESOLVED: Final[str] = "hub.revision.unresolved"
CODE_OFFLINE_MISSING: Final[str] = "hub.offline.missing"
CODE_OFFLINE_SEARCH: Final[str] = "hub.offline.search_unavailable"
CODE_FILE_NOT_FOUND: Final[str] = "hub.file.not_found"
CODE_HTTP_ERROR: Final[str] = "hub.http.error"
CODE_RATE_LIMITED: Final[str] = "hub.rate_limited"
CODE_NETWORK_ERROR: Final[str] = "hub.network.error"
CODE_INVALID_REPO_ID: Final[str] = "hub.repo.invalid_id"
CODE_CANCELLED: Final[str] = "hub.download.cancelled"
CODE_UNKNOWN: Final[str] = "hub.unknown_error"

_COMMIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_SHARD_OF = re.compile(r"-of-(\d+)")
_NETWORK_MODULES: Final[frozenset[str]] = frozenset(
    {"httpx", "httpcore", "urllib3", "requests", "socket", "ssl"}
)


def is_commit_sha(value: str | None) -> bool:
    """Return ``True`` when *value* is a full 40-character git commit SHA.

    A branch or tag name is never enough for a reproducible preparation job
    (``SPEC.md`` section 8.3, ``docs/MODEL_PIPELINE.md`` section 4), so callers
    use this to decide whether a revision still has to be resolved.
    """
    return bool(value) and _COMMIT_SHA.match(value or "") is not None


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #


def _status_code(error: BaseException) -> int | None:
    """Best-effort HTTP status code carried by a ``huggingface_hub`` error."""
    response = getattr(error, "response", None)
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_network_error(error: BaseException) -> bool:
    """Return ``True`` for transport-level failures (DNS, TLS, timeouts, resets)."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    root = type(error).__module__.split(".", 1)[0]
    return root in _NETWORK_MODULES


def _hub_diagnostic(
    code: str,
    category: FailureCategory,
    title: str,
    message: str,
    *,
    error: BaseException | None,
    model: ModelIdentity | None,
    operation: str,
    remediation: Sequence[str] = (),
) -> Diagnostic:
    """Build a redacted :class:`Diagnostic` for a Hub failure.

    Both the message and the retained raw excerpt go through
    :func:`yonga.util.redaction.redact_text`, so a token that leaked into an
    upstream error string can never reach a log, a JSON document or the terminal.
    """
    context: dict[str, str] = {"operation": operation}
    if model is not None:
        context["repo_id"] = model.repo_id
        if model.revision:
            context["revision"] = model.revision
    status = _status_code(error) if error is not None else None
    if status is not None:
        context["http_status"] = str(status)
    return diagnostic(
        code,
        category,
        title,
        redact_text(message),
        severity=Severity.ERROR,
        remediation=[redact_text(item) for item in remediation],
        raw_excerpt=redact_text(str(error)) if error is not None else None,
        context=context,
    )


def _search_hit_metadata(info: Any) -> ModelMetadata:
    """Map one ``list_models`` result onto :class:`ModelMetadata`.

    Identity comes from the Hub's own ``config`` summary when the repository has
    a ``config.json``, and from the GGUF header otherwise -- the same two sources
    :meth:`HuggingFaceAdapter.inspect` uses, so search and inspect identify a
    repository the same way.
    """
    summary = getattr(info, "config", None)
    architectures: list[str] = []
    model_type: str | None = None
    if isinstance(summary, Mapping):
        architectures = _string_list(summary.get("architectures"))
        raw_type = summary.get("model_type")
        model_type = raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else None
    sha = getattr(info, "sha", None)
    return ModelMetadata(
        identity=ModelIdentity(repo_id=str(getattr(info, "id", "") or "")),
        resolved_revision=sha.strip() if isinstance(sha, str) and sha.strip() else None,
        gated=bool(getattr(info, "gated", False)),
        pipeline_tag=_optional_str(getattr(info, "pipeline_tag", None)),
        tags=_string_list(getattr(info, "tags", None)),
        library_name=_optional_str(getattr(info, "library_name", None)),
        architectures=architectures,
        model_type=model_type,
        parameter_count=_parameter_count(info),
        parameter_dtypes=_parameter_dtypes(info),
        gguf_metadata=_compact_gguf_metadata(getattr(info, "gguf", None)),
        downloads=_optional_int(getattr(info, "downloads", None)),
        likes=_optional_int(getattr(info, "likes", None)),
        last_modified=_isoformat(getattr(info, "last_modified", None)),
    )


def translate_hub_error(
    error: BaseException,
    *,
    model: ModelIdentity | None = None,
    operation: str = "hub request",
    offline: bool = False,
) -> YongaError:
    """Translate any ``huggingface_hub`` exception into a typed yonga error.

    The returned exception always carries a :class:`Diagnostic` with a stable
    code, the right :class:`FailureCategory` (and therefore the right process
    exit code) and English remediation text. Nothing in it can contain a
    credential: every string is passed through
    :func:`yonga.util.redaction.redact_text`.

    Categories deviate from the naive reading in one place, deliberately:
    ``RepositoryNotFoundError`` maps to :data:`FailureCategory.DOWNLOAD_FAILURE`
    rather than ``SOURCE_FORMAT_UNSUPPORTED``. A repository that does not exist
    says nothing about the *format* of any model; ``DOWNLOAD_FAILURE`` still
    yields exit code 20 (source resolution), which is the documented outcome.
    """
    if isinstance(error, YongaError):
        return error

    ref = str(model) if model is not None else "the requested repository"
    status = _status_code(error)

    if isinstance(error, HFValidationError):
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_INVALID_REPO_ID,
                FailureCategory.DOWNLOAD_FAILURE,
                "Invalid model reference",
                f"{ref} is not a valid Hugging Face repository id.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "Use the `namespace/name` form, for example `OpenVINO/Qwen3.5-9B-int4-ov`.",
                ],
            ),
            cause=error,
        )

    if isinstance(error, GatedRepoError):
        return HubAuthError(
            _hub_diagnostic(
                CODE_GATED,
                FailureCategory.GATED_MODEL_AUTH_REQUIRED,
                "Model access is gated",
                f"{ref} is gated: its owner requires you to accept conditions and "
                "authenticate before the files can be read.",
                error=error,
                model=model,
                operation=operation,
                remediation=[ACCEPT_HINT, LOGIN_HINT],
            ),
            cause=error,
        )

    if isinstance(error, (RepositoryNotFoundError, DisabledRepoError)):
        # VERIFIED on the live Hub: an anonymous request for a repository that
        # does not exist answers HTTP 401, not 404 — the Hub deliberately refuses
        # to confirm whether a private repository exists. So this branch must be
        # tried *before* the bare 401/403 check below, and its message has to
        # cover both readings at once.
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_REPO_NOT_FOUND,
                FailureCategory.DOWNLOAD_FAILURE,
                "Repository not found",
                f"{ref} does not exist on the Hugging Face Hub, is disabled, or is "
                "not visible to this account. The Hub answers the same way in all "
                "three cases, so the repository id and the login state both need "
                "checking.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "Check the spelling of the repository id, including its namespace.",
                    LOGIN_HINT,
                ],
            ),
            cause=error,
        )

    if isinstance(error, RevisionNotFoundError):
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_REVISION_NOT_FOUND,
                FailureCategory.DOWNLOAD_FAILURE,
                "Revision not found",
                f"The requested revision of {ref} does not exist. "
                "Branches, tags and commit SHAs are all accepted.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "List the repository's branches and tags on its Hub page, "
                    "or omit --revision to use the default branch.",
                ],
            ),
            cause=error,
        )

    if status in (401, 403):
        return HubAuthError(
            _hub_diagnostic(
                CODE_AUTH_REQUIRED,
                FailureCategory.GATED_MODEL_AUTH_REQUIRED,
                "Authentication required",
                f"The Hugging Face Hub refused access to {ref} "
                f"(HTTP {status}). The repository is private, gated, or the stored "
                "credential is not valid for it.",
                error=error,
                model=model,
                operation=operation,
                remediation=[LOGIN_HINT, ACCEPT_HINT],
            ),
            cause=error,
        )

    if isinstance(error, (LocalEntryNotFoundError, OfflineModeIsEnabled)) or (
        offline and isinstance(error, (EntryNotFoundError, FileNotFoundError))
    ):
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_OFFLINE_MISSING,
                FailureCategory.DOWNLOAD_FAILURE,
                "Not available offline",
                f"Offline mode is active and {ref} is not fully present in the "
                "local Hugging Face cache, so it cannot be used without network "
                "access.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "Re-run without offline mode (unset YONGA_OFFLINE / HF_HUB_OFFLINE) "
                    "to populate the cache.",
                    "Or point HF_HOME at a cache that already contains this revision.",
                ],
            ),
            cause=error,
        )

    if isinstance(error, EntryNotFoundError):
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_FILE_NOT_FOUND,
                FailureCategory.DOWNLOAD_FAILURE,
                "File not found in repository",
                f"A file expected in {ref} is not present at this revision.",
                error=error,
                model=model,
                operation=operation,
                remediation=["Inspect the repository's file list on its Hub page."],
            ),
            cause=error,
        )

    if isinstance(error, HfHubHTTPError) and status == 429:
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_RATE_LIMITED,
                FailureCategory.DOWNLOAD_FAILURE,
                "The Hugging Face Hub is rate-limiting these requests",
                f"The Hub answered HTTP 429 (too many requests) while performing {operation} "
                f"for {ref}. Anonymous requests have a much lower limit than logged-in ones.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "Log in so requests count against your account: hf auth login",
                    "Or wait a few minutes and run the same command again.",
                ],
            ),
            cause=error,
        )

    if isinstance(error, HfHubHTTPError):
        detail = f" (HTTP {status})" if status is not None else ""
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_HTTP_ERROR,
                FailureCategory.DOWNLOAD_FAILURE,
                "Hugging Face Hub request failed",
                f"The Hub returned an error{detail} while performing {operation} for {ref}.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "Retry in a moment; Hub errors in the 5xx range are usually transient.",
                    "Check https://status.huggingface.co/ if the failure persists.",
                ],
            ),
            cause=error,
        )

    if _is_network_error(error) or isinstance(error, OSError):
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_NETWORK_ERROR,
                FailureCategory.DOWNLOAD_FAILURE,
                "Cannot reach the Hugging Face Hub",
                f"The network request for {ref} failed before the Hub could answer.",
                error=error,
                model=model,
                operation=operation,
                remediation=[
                    "Check the machine's network connection and any proxy settings.",
                    "Use offline mode if the model is already in the local cache.",
                ],
            ),
            cause=error,
        )

    return YongaError(
        _hub_diagnostic(
            CODE_UNKNOWN,
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Unexpected Hugging Face Hub failure",
            f"{operation} for {ref} failed for an unrecognised reason.",
            error=error,
            model=model,
            operation=operation,
            remediation=["Re-run with --debug and attach the job log to a bug report."],
        ),
        cause=error,
    )


# --------------------------------------------------------------------------- #
# Pure detection helpers
# --------------------------------------------------------------------------- #


def _total_bytes(files: Sequence[RepoFile]) -> tuple[int, int]:
    """Return ``(sum_of_known_sizes, number_of_files_with_unknown_size)``."""
    total = 0
    unknown = 0
    for entry in files:
        if entry.size is None:
            unknown += 1
        else:
            total += entry.size
    return total, unknown


def _is_gguf_companion(entry: RepoFile) -> bool:
    """True when a ``.gguf`` file is a side-car rather than a runnable model.

    A filename-convention heuristic, never a format guarantee: see
    :data:`GGUF_COMPANION_PREFIXES`. Two details of the comparison are
    deliberate and are pinned by tests.

    * It case-folds and reads the *base name*, so neither ``MMPROJ-x.gguf`` nor
      ``sub/mmproj-x.gguf`` can slip past it.
    * Only a **prefix** counts. ``model-mmproj-ish.gguf`` is treated as a
      runnable model, because ``llama.cpp`` puts the marker first and a
      substring test would quietly swallow legitimately named models.
    """
    name = entry.name.casefold()
    return any(name.startswith(prefix.casefold()) for prefix in GGUF_COMPANION_PREFIXES)


def _gguf_primary_bytes(files: Sequence[RepoFile], declared_total: int | None) -> int:
    """Return the bytes of the single GGUF model a user would actually run.

    A GGUF repository conventionally publishes one file per quantization of the
    *same* model, so summing them answers the wrong question. Preference order:

    1. the size the Hub derived from the repository's GGUF headers
       (:data:`GGUF_TOTAL_FILE_SIZE_KEY`), when it is a positive integer;
    2. otherwise the largest single ``.gguf`` file that is not a companion — the
       unquantized copy is the largest, and picking the largest keeps the
       estimate conservative rather than optimistic.

    ``0`` means "unknown" and is returned whenever no non-companion file has a
    known size. That covers the repository holding nothing but companions: it
    publishes no runnable model, so answering with a companion's size would
    hand callers the size of the one file this module just classified as not
    runnable. Callers treat ``0`` as unknown and fall back to their own floor
    (the temp estimate reserves ``temp_min_bytes`` regardless).
    """
    if declared_total is not None and declared_total > 0:
        return declared_total
    models = [entry for entry in files if not _is_gguf_companion(entry)]
    return max((entry.size or 0 for entry in models), default=0)


def _size_note(unknown: int) -> list[str]:
    """Return a note when the Hub did not report every file size."""
    if unknown <= 0:
        return []
    return [
        f"Byte total is incomplete: the Hub reported no size for {unknown} file(s). "
        "Re-inspect with file metadata enabled for an exact figure."
    ]


def _expected_shards(names: Sequence[str]) -> int | None:
    """Read the shard count out of ``…-of-000NN…`` file names, when they agree."""
    counts = {
        int(match.group(1)) for name in names if (match := _SHARD_OF.search(name)) is not None
    }
    if len(counts) == 1:
        return counts.pop()
    return None


def _shard_completeness(
    weights: Sequence[RepoFile],
    index_paths: Sequence[str],
    shard_indexes: Mapping[str, Sequence[str]] | None,
    *,
    label: str,
) -> tuple[bool, list[str]]:
    """Decide whether a sharded weight set is complete, and explain the verdict.

    ``complete`` is ``True`` when an index document is present and every shard it
    names is listed in the repository, or when there is a single unsharded weight
    file. When an index exists but could not be read (offline, oversized, network
    failure) the ``-of-000NN`` shard-count convention is used as a fallback and
    the note says so.
    """
    listed = {entry.name for entry in weights}
    notes: list[str] = []

    if not index_paths:
        if len(weights) == 1:
            return True, [f"Single unsharded {label} file."]
        expected = _expected_shards(sorted(listed))
        notes.append(
            f"No {label} index document is listed, so the shard set cannot be proven complete."
        )
        if expected is not None:
            notes.append(f"File names announce {expected} shard(s); {len(weights)} are listed.")
        return False, notes

    notes.append("Weight index: " + ", ".join(sorted(index_paths)) + ".")
    complete = True
    for index_path in sorted(index_paths):
        shards = None if shard_indexes is None else shard_indexes.get(index_path)
        if shards is None:
            expected = _expected_shards(sorted(listed))
            if expected is not None and expected == len(weights):
                notes.append(
                    f"{index_path} was not read; all {expected} shards named by the "
                    "`-of-` convention are listed."
                )
                continue
            complete = False
            notes.append(f"{index_path} could not be read, so its shard set is unverified.")
            continue
        missing = sorted({PurePosixPath(name).name for name in shards} - listed)
        if missing:
            complete = False
            notes.append(f"{index_path} names {len(missing)} missing shard(s): {missing[0]} …")
        else:
            notes.append(f"{index_path} names {len(set(shards))} shard(s), all listed.")
    return complete, notes


def detect_formats(
    files: Sequence[RepoFile],
    *,
    shard_indexes: Mapping[str, Sequence[str]] | None = None,
    gguf_total_file_size: int | None = None,
) -> list[FormatObservation]:
    """Classify a repository's file inventory into weight-format observations.

    A single repository may legitimately carry several formats at once (an
    OpenVINO export that kept its safetensors, a GGUF repo that also ships the
    original ``.bin`` files), so this returns a *list*, ordered
    OpenVINO IR → safetensors → PyTorch → GGUF.

    Detection keys on file *suffixes*, never on a filename regex: the
    ``nDimensional/Qwen3.5-9B-Uncensored-Safetensors`` fixture publishes its
    shards as ``model.safetensors-00001-of-00004.safetensors``, which no
    conventional ``model-\\d+-of-\\d+`` pattern would match.

    ``shard_indexes`` maps an index document's repository path to the shard file
    names it references; :meth:`HuggingFaceAdapter.inspect` fills it by fetching
    those (kilobyte-sized) JSON documents. It is optional so that this function
    stays pure and unit-testable.

    Index documents are listed in the observation's notes but excluded from
    ``files``/``total_bytes``, so ``total_bytes`` is exactly the weight payload
    the repository publishes in that format.

    Each observation also carries ``primary_bytes``: the size of *one* usable
    model. For OpenVINO IR, safetensors and PyTorch checkpoints every listed
    file belongs to that one model — a four-shard safetensors model genuinely
    needs all four shards — so the two counters are equal. Only GGUF, which
    conventionally publishes one file per quantization of the same model,
    distinguishes them; ``gguf_total_file_size`` carries the Hub's own figure
    for that repository's primary model file when it reported one (see
    :func:`_gguf_primary_bytes`).
    """
    observations: list[FormatObservation] = []
    by_path = {entry.path: entry for entry in files}
    consumed: set[str] = set()

    # --- OpenVINO IR -------------------------------------------------------
    xmls = [
        entry
        for entry in files
        if entry.suffix == XML_SUFFIX and entry.name.startswith(OPENVINO_IR_PREFIX)
    ]
    if xmls:
        ir_files: list[RepoFile] = []
        missing: list[str] = []
        for xml in xmls:
            ir_files.append(xml)
            bin_path = xml.path[: -len(XML_SUFFIX)] + PYTORCH_BIN_SUFFIX
            companion = by_path.get(bin_path)
            if companion is None:
                missing.append(bin_path)
            else:
                ir_files.append(companion)
        consumed.update(entry.path for entry in ir_files)
        total, unknown = _total_bytes(ir_files)
        notes = [f"{len(xmls)} OpenVINO IR submodel(s)."]
        if missing:
            notes.append(f"Missing weight file(s): {', '.join(sorted(missing))}.")
        notes.extend(_size_note(unknown))
        observations.append(
            FormatObservation(
                format=ModelFormat.OPENVINO_IR,
                files=sorted(entry.path for entry in ir_files),
                total_bytes=total,
                # Every submodel XML/BIN pair belongs to the one exported model.
                primary_bytes=total,
                complete=not missing,
                notes=notes,
            )
        )

    # --- safetensors -------------------------------------------------------
    safetensors = [entry for entry in files if entry.suffix == SAFETENSORS_SUFFIX]
    if safetensors:
        index_paths = [
            entry.path for entry in files if entry.name.endswith(SAFETENSORS_INDEX_SUFFIX)
        ]
        complete, notes = _shard_completeness(
            safetensors, index_paths, shard_indexes, label="safetensors"
        )
        total, unknown = _total_bytes(safetensors)
        notes.extend(_size_note(unknown))
        consumed.update(entry.path for entry in safetensors)
        observations.append(
            FormatObservation(
                format=ModelFormat.SAFETENSORS,
                files=sorted(entry.path for entry in safetensors),
                total_bytes=total,
                # Sharded weights are one model: every shard must be downloaded.
                primary_bytes=total,
                complete=complete,
                notes=notes,
            )
        )

    # --- PyTorch .bin ------------------------------------------------------
    torch_bins = [
        entry
        for entry in files
        if entry.suffix == PYTORCH_BIN_SUFFIX and entry.path not in consumed
    ]
    if torch_bins:
        index_paths = [entry.path for entry in files if entry.name.endswith(PYTORCH_INDEX_SUFFIX)]
        complete, notes = _shard_completeness(
            torch_bins, index_paths, shard_indexes, label="PyTorch"
        )
        total, unknown = _total_bytes(torch_bins)
        notes.extend(_size_note(unknown))
        consumed.update(entry.path for entry in torch_bins)
        observations.append(
            FormatObservation(
                format=ModelFormat.PYTORCH_BIN,
                files=sorted(entry.path for entry in torch_bins),
                total_bytes=total,
                # As with safetensors, a sharded checkpoint is a single model.
                primary_bytes=total,
                complete=complete,
                notes=notes,
            )
        )

    # --- GGUF --------------------------------------------------------------
    ggufs = [entry for entry in files if entry.suffix == GGUF_SUFFIX]
    if ggufs:
        total, unknown = _total_bytes(ggufs)
        primary = _gguf_primary_bytes(ggufs, gguf_total_file_size)
        notes = [
            f"{len(ggufs)} self-contained GGUF file(s); the byte total covers every "
            "quantization published in this repository, not one runnable variant."
        ]
        if gguf_total_file_size is not None and gguf_total_file_size > 0:
            notes.append(f"One runnable model is {primary} byte(s), per the Hub's GGUF metadata.")
        elif primary <= 0 < total:
            notes.append(
                "The size of one runnable model is unknown: no GGUF file here is "
                "both a model (rather than a companion) and sized by the Hub."
            )
        elif primary < total:
            notes.append(
                f"One runnable model is estimated at {primary} byte(s): "
                "the largest non-companion GGUF file."
            )
        projectors = sorted(entry.name for entry in ggufs if _is_gguf_companion(entry))
        if projectors:
            notes.append(f"Multimodal projector file(s) present: {', '.join(projectors)}.")
        notes.extend(_size_note(unknown))
        observations.append(
            FormatObservation(
                format=ModelFormat.GGUF,
                files=sorted(entry.path for entry in ggufs),
                total_bytes=total,
                primary_bytes=primary,
                complete=True,
                notes=notes,
            )
        )

    return observations


def _iter_strings(value: object) -> Iterator[str]:
    """Yield every string reachable in an ``auto_map`` entry (str, list or dict)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _module_file(target: str) -> str | None:
    """Turn ``"owner/repo--modeling_x.XModel"`` into ``"modeling_x.py"``."""
    reference = target.rsplit("--", 1)[-1].strip()
    module = reference.rpartition(".")[0]
    if not module:
        return None
    return module.replace(".", "/") + PYTHON_SUFFIX


def detect_remote_code(
    config: Mapping[str, object] | None,
    files: Sequence[RepoFile],
) -> tuple[bool, list[str]]:
    """Decide whether loading this repository needs ``trust_remote_code``.

    Returns ``(requires_remote_code, remote_code_files)``. The requirement is
    driven by ``config.json``: an ``auto_map`` block means Transformers will
    import Python from the repository. Custom modelling modules that follow the
    Transformers naming convention are reported alongside the modules the
    ``auto_map`` explicitly names.

    This function only *reports*. Enabling ``trust_remote_code`` is always an
    explicit caller decision — nothing in this module ever sets it.
    """
    listed = {entry.path for entry in files}
    payload: Mapping[str, object] = config or {}
    auto_map = payload.get("auto_map")

    referenced: set[str] = set()
    if isinstance(auto_map, Mapping):
        for value in auto_map.values():
            for target in _iter_strings(value):
                module = _module_file(target)
                if module is not None and module in listed:
                    referenced.add(module)

    required = bool(isinstance(auto_map, Mapping) and auto_map) or bool(referenced)
    if not required:
        return False, []

    custom = {
        entry.path
        for entry in files
        if entry.suffix == PYTHON_SUFFIX and entry.name.startswith(CUSTOM_CODE_PREFIXES)
    }
    return True, sorted(referenced | custom)


def _compact_gguf_metadata(raw: object) -> dict[str, object]:
    """Keep GGUF metadata scalars; elide multi-kilobyte values such as templates."""
    if not isinstance(raw, Mapping):
        return {}
    compact: dict[str, object] = {}
    for key, value in raw.items():
        name = str(key)
        if value is None or isinstance(value, (bool, int, float)):
            compact[name] = value
            continue
        text = value if isinstance(value, str) else str(value)
        if len(text) > GGUF_METADATA_MAX_VALUE_CHARS:
            compact[name] = f"<omitted: {len(text)} characters>"
        else:
            compact[name] = redact_text(text)
    return compact


def _attach_notes(observations: list[FormatObservation], notes: Sequence[str]) -> None:
    """Record repository-level inspection notes on the most relevant observation.

    :class:`~yonga.domain.models.ModelMetadata` has no free-text notes field, so
    notes that describe the repository as a whole (for example "there is no
    ``config.json``") ride along on the first format observation, which is the
    one the CLI already prints. A repository with no recognised weights at all
    gets an ``UNKNOWN`` observation purely to carry them.
    """
    if not notes:
        return
    if not observations:
        observations.append(
            FormatObservation(format=ModelFormat.UNKNOWN, notes=[redact_text(n) for n in notes])
        )
        return
    observations[0].notes.extend(redact_text(note) for note in notes)


# --------------------------------------------------------------------------- #
# Offline: metadata from the local Hub cache
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _CachedSibling:
    """One file of a cached snapshot, shaped like ``huggingface_hub``'s ``RepoSibling``."""

    rfilename: str
    size: int | None
    lfs: dict[str, int] | None = None


@dataclass(frozen=True)
class _CachedModelInfo:
    """The part of ``ModelInfo`` a cached snapshot can answer, read from disk alone.

    Offline mode builds this instead of calling ``model_info``, so
    :meth:`HuggingFaceAdapter.inspect` reads the same fields either way. What
    only the Hub knows -- gated or private status, downloads, likes, the last
    commit date -- is simply absent, and the extraction helpers already treat an
    absent field as "not reported".
    """

    sha: str
    siblings: list[_CachedSibling]
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    card_data: dict[str, object] | None = None
    safetensors: dict[str, object] | None = None
    gguf: dict[str, object] | None = None


def _safe_relative(name: str) -> bool:
    """True when *name* is a relative path that stays inside the directory it names."""
    parts = PurePosixPath(name).parts
    return bool(parts) and not PurePosixPath(name).is_absolute() and ".." not in parts


def _snapshot_files(snapshot: Path) -> list[_CachedSibling]:
    """List the files of one cached snapshot with their sizes, reading no file.

    A snapshot holds symlinks into ``blobs/``. The blob's name is the file's
    Hub etag: a 64-character SHA-256 for an LFS file, a 40-character git SHA-1
    otherwise, which is how the LFS column survives offline. A dangling link
    (its blob was deleted) is a file that is not cached, so it is left out.
    """
    siblings: list[_CachedSibling] = []
    for directory, _subdirs, names in os.walk(snapshot):
        for name in names:
            path = Path(directory) / name
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
                blob = Path(os.path.realpath(path)).name if path.is_symlink() else ""
            except OSError:
                continue
            siblings.append(
                _CachedSibling(
                    rfilename=path.relative_to(snapshot).as_posix(),
                    size=size,
                    lfs={"size": size} if len(blob) == 64 else None,
                )
            )
    siblings.sort(key=lambda entry: entry.rfilename)
    return siblings


def _card_metadata(snapshot: Path) -> dict[str, object]:
    """The YAML front matter of a cached ``README.md``, or ``{}``."""
    path = snapshot / README_FILENAME
    try:
        if path.stat().st_size > MAX_METADATA_FILE_BYTES:
            return {}
        data = metadata_load(path)
    except Exception:  # a missing, unreadable or malformed card is just "no card"
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _safetensors_header(path: Path) -> Mapping[str, object]:
    """Read one safetensors file's JSON header: eight length bytes, then the JSON."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path.name} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"{path.name} declares an implausible header length")
        payload = json.loads(handle.read(length))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} has no JSON object header")
    return payload


def _local_safetensors_parameters(snapshot: Path) -> dict[str, object] | None:
    """Count parameters per dtype from cached safetensors headers, as the Hub does.

    The Hub's own figure is computed from the same headers, from
    ``model.safetensors`` or else from the shards ``model.safetensors.index.json``
    names; this reads the same layouts, so a profile's size band is judged on
    the same number online and offline. Only headers are read, never weights.
    A shard that is not cached means no count at all, because a partial sum
    would put a large model into a small model's band.
    """
    single = snapshot / SAFETENSORS_SINGLE_FILENAME
    index = snapshot / SAFETENSORS_INDEX_FILENAME
    try:
        if single.is_file():
            shards = [single]
        elif index.is_file() and index.stat().st_size <= MAX_METADATA_FILE_BYTES:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weight_map, Mapping):
                return None
            names = {value for value in weight_map.values() if isinstance(value, str)}
            if not names or not all(_safe_relative(name) for name in names):
                return None
            shards = [snapshot / name for name in sorted(names)]
        else:
            return None
        counts: dict[str, int] = {}
        for shard in shards:
            if not shard.is_file():
                return None
            for name, tensor in _safetensors_header(shard).items():
                if name == "__metadata__" or not isinstance(tensor, Mapping):
                    continue
                dtype, shape = tensor.get("dtype"), tensor.get("shape")
                if not isinstance(dtype, str) or not isinstance(shape, list):
                    return None
                if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
                    return None
                counts[dtype] = counts.get(dtype, 0) + math.prod(shape)
    except (OSError, ValueError, AttributeError, struct.error):
        return None
    if not counts:
        return None
    return {"total": sum(counts.values()), "parameters": counts}


def _local_gguf_metadata(snapshot: Path, siblings: Sequence[_CachedSibling]) -> dict[str, object]:
    """The identity facts the Hub reports for a GGUF repository, from a cached header.

    Online, the Hub's ``gguf`` summary names the architecture a GGUF-only
    repository (no ``config.json``) is identified by. Offline the same name is
    read from the header of the largest cached model file -- a few megabytes,
    never the weights.
    """
    models = [
        entry
        for entry in siblings
        if entry.rfilename.endswith(GGUF_SUFFIX)
        and not _is_gguf_companion(RepoFile(path=entry.rfilename, size=entry.size))
    ]
    if not models:
        return {}
    largest = max(models, key=lambda entry: entry.size or 0)
    try:
        header = read_local_header(snapshot / largest.rfilename)
    except (OSError, GGUFFormatError):
        return {}
    architecture = header.architecture
    if not architecture:
        return {}
    found: dict[str, object] = {"architecture": architecture}
    context = header.metadata.get(f"{architecture}.context_length")
    if isinstance(context, int) and not isinstance(context, bool):
        found["context_length"] = context
    return found


# --------------------------------------------------------------------------- #
# Download progress
# --------------------------------------------------------------------------- #


def _progress_tqdm_class(
    ctx: JobContext,
    *,
    repo_id: str,
    revision: str,
    min_interval: float = PROGRESS_MIN_INTERVAL_SECONDS,
) -> type[Any]:
    """Build a ``tqdm`` subclass that mirrors Hub byte progress into a sink.

    ``snapshot_download`` aggregates every worker thread into one byte counter
    and lets the caller supply the class used to build it, which is the supported
    way to observe download progress without reimplementing the transfer. The
    returned class emits throttled :data:`ProgressKind.DOWNLOAD` events and turns
    a cancellation request into :class:`yonga.domain.diagnostics.CancelledError`.
    """

    class _YongaProgressTqdm(hf_tqdm):
        """Hub progress bar that reports into this job's progress sink."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Remember whether this bar counts bytes, then build the real bar."""
            self._yonga_bytes = kwargs.get("unit") == "B"
            self._yonga_seen = float(kwargs.get("initial") or 0)
            self._yonga_last = 0.0
            # ``huggingface_hub``'s tqdm subclass is untyped upstream.
            super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]

        def _yonga_emit(self, *, force: bool) -> None:
            """Emit a download event, at most once per ``min_interval`` seconds."""
            if not self._yonga_bytes:
                return
            now = time.monotonic()
            if not force and now - self._yonga_last < min_interval:
                return
            self._yonga_last = now
            raw_total = getattr(self, "total", None)
            total = int(raw_total) if isinstance(raw_total, (int, float)) and raw_total else None
            ctx.emit(
                ProgressKind.DOWNLOAD,
                f"Downloading {repo_id}",
                current=int(self._yonga_seen),
                total=total,
                status="downloading",
                repo_id=repo_id,
                revision=revision,
                unit="bytes",
            )

        def update(self, n: float | None = 1) -> bool | None:
            """Account for ``n`` more bytes, emit progress and honour cancellation."""
            if ctx.check_cancelled():
                raise CancelledError(
                    diagnostic(
                        CODE_CANCELLED,
                        FailureCategory.USER_CANCELLED,
                        "Download cancelled",
                        f"The download of {repo_id} was cancelled.",
                        context={"repo_id": repo_id, "revision": revision},
                    )
                )
            self._yonga_seen += float(n or 0)
            self._yonga_emit(force=False)
            result: object = super().update(n)
            return bool(result) if result is not None else None

        def close(self) -> None:
            """Flush a final progress event, then close the underlying bar."""
            self._yonga_emit(force=True)
            super().close()

    return _YongaProgressTqdm


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class HuggingFaceAdapter:
    """Read-only Hugging Face Hub client implementing the ``HubClient`` Protocol.

    Every method pins an immutable commit SHA before it touches bytes, keeps
    downloads inside the standard Hub cache, and converts upstream failures into
    typed :class:`~yonga.domain.diagnostics.YongaError` values whose messages are
    redacted. The adapter holds no credential of its own: ``huggingface_hub``
    resolves the token from ``HF_TOKEN`` or the stored login, and this class only
    ever observes whether one exists.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api: HfApi | None = None,
    ) -> None:
        """Create an adapter.

        ``settings`` supplies offline mode and the optional ``HF_HOME`` override;
        ``api`` allows tests (and a future proxy-aware caller) to inject a
        pre-configured :class:`huggingface_hub.HfApi`.
        """
        self._settings: Settings = settings if settings is not None else Settings()
        self._api: HfApi = api if api is not None else HfApi()

    # --- environment ---------------------------------------------------

    @property
    def settings(self) -> Settings:
        """The settings this adapter was built with."""
        return self._settings

    @property
    def api(self) -> HfApi:
        """The underlying ``huggingface_hub`` API client."""
        return self._api

    @property
    def offline(self) -> bool:
        """``True`` when yonga or ``huggingface_hub`` is configured for offline use."""
        return bool(self._settings.offline) or bool(huggingface_hub.constants.HF_HUB_OFFLINE)

    @property
    def cache_dir(self) -> Path | None:
        """Explicit Hub cache directory to use, or ``None`` to let the Hub decide."""
        hf_home = self._settings.hf_home
        if hf_home is None:
            return None
        return hf_home / "hub"

    def _effective_cache_dir(self) -> Path:
        """The directory snapshots actually land in, for reporting purposes."""
        configured = self.cache_dir
        if configured is not None:
            return configured
        return Path(huggingface_hub.constants.HF_HUB_CACHE)

    # --- revision ------------------------------------------------------

    def resolve_revision(self, model: ModelIdentity) -> str:
        """Resolve a branch, tag or SHA to the immutable commit SHA behind it.

        A mutable branch name is never enough: two runs a day apart would prepare
        different weights under the same name. The resolved SHA is what every
        later download for the job uses.

        Offline, the answer comes from the local cache alone (see
        :meth:`_cached_revision`); the Hub is never contacted.
        """
        if self.offline:
            return self._cached_revision(model, operation="revision resolution")[0]
        info = self._model_info(model, files_metadata=False, operation="revision resolution")
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha.strip():
            return sha.strip()
        raise SourceResolutionError(
            _hub_diagnostic(
                CODE_REVISION_UNRESOLVED,
                FailureCategory.DOWNLOAD_FAILURE,
                "Revision could not be pinned",
                f"The Hub did not report a commit SHA for {model}, so the "
                "revision cannot be pinned and the job would not be reproducible.",
                error=None,
                model=model,
                operation="revision resolution",
                remediation=["Retry, or pass an explicit --revision with a full commit SHA."],
            )
        )

    def _pinned_revision(self, model: ModelIdentity) -> str:
        """Return the SHA to use, resolving a mutable reference when necessary."""
        if is_commit_sha(model.revision):
            return str(model.revision)
        return self.resolve_revision(model)

    # --- offline: the local Hub cache ----------------------------------

    def _repo_cache_folder(self, model: ModelIdentity, *, operation: str) -> Path:
        """The cache folder ``huggingface_hub`` keeps *model*'s snapshots in."""
        try:
            validate_repo_id(model.repo_id)
        except HFValidationError as error:
            raise translate_hub_error(error, model=model, operation=operation) from error
        return self._effective_cache_dir() / repo_folder_name(
            repo_id=model.repo_id, repo_type="model"
        )

    def _offline_missing(
        self,
        model: ModelIdentity,
        *,
        operation: str,
        message: str,
        remediation: Sequence[str] = (),
    ) -> SourceResolutionError:
        """A ``hub.offline.missing`` failure that says what the cache does hold."""
        return SourceResolutionError(
            _hub_diagnostic(
                CODE_OFFLINE_MISSING,
                FailureCategory.DOWNLOAD_FAILURE,
                "Not available offline",
                f"Offline mode is active and {message}",
                error=None,
                model=model,
                operation=operation,
                remediation=[
                    *remediation,
                    "Re-run without offline mode (unset YONGA_OFFLINE / HF_HUB_OFFLINE) "
                    "to populate the cache.",
                    "Or point HF_HOME at a cache that already contains this revision.",
                ],
            )
        )

    def _cached_revision(self, model: ModelIdentity, *, operation: str) -> tuple[str, list[str]]:
        """Resolve *model*'s revision to a cached snapshot's SHA without any network.

        Returns ``(sha, notes)``. A commit SHA must have a snapshot; a branch or
        tag is looked up in the cache's ``refs/`` mapping, exactly as
        ``huggingface_hub`` does with ``local_files_only``. yonga downloads by
        commit SHA, and the Hub writes no ``refs/`` entry for such a download,
        so when nothing was asked for and the cache holds exactly one snapshot,
        that snapshot is used and a note says so. Several unrecorded snapshots
        are refused rather than guessed between.
        """
        folder = self._repo_cache_folder(model, operation=operation)
        snapshots = folder / "snapshots"
        revision = model.revision or DEFAULT_REVISION
        ref = str(model)
        notes: list[str] = []

        sha: str | None = None
        if is_commit_sha(revision):
            sha = revision
        elif _safe_relative(revision):
            try:
                recorded = (folder / "refs" / revision).read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                recorded = ""
            sha = recorded if is_commit_sha(recorded) else None

        if sha is None:
            try:
                cached = sorted(
                    entry.name
                    for entry in snapshots.iterdir()
                    if entry.is_dir() and is_commit_sha(entry.name)
                )
            except OSError:
                cached = []
            if len(cached) == 1 and model.revision is None:
                sha = cached[0]
                notes.append(
                    f"Offline: the local cache records no '{revision}' reference for "
                    f"{model.repo_id}, so its only cached snapshot, {sha[:12]}, was used."
                )
            elif cached:
                shown = ", ".join(cached[:5])
                raise self._offline_missing(
                    model,
                    operation=operation,
                    message=(
                        f"the local Hugging Face cache holds {len(cached)} snapshot(s) of "
                        f"{model.repo_id} ({shown}) but no record of which one "
                        f"'{revision}' is, so none can be chosen without the Hub."
                    ),
                    remediation=[
                        f"Name one with its full commit SHA: {model.repo_id}@<sha> "
                        "(or --revision <sha>)."
                    ],
                )

        if sha is None or not (snapshots / sha).is_dir():
            raise self._offline_missing(
                model,
                operation=operation,
                message=(
                    f"{ref} is not in the local Hugging Face cache "
                    f"({self._effective_cache_dir()}), so it cannot be used without "
                    "network access."
                ),
            )
        return sha, notes

    def _local_model_info(
        self, model: ModelIdentity, *, operation: str
    ) -> tuple[_CachedModelInfo, list[str]]:
        """Build ``model_info``'s answer from the cached snapshot, never the network."""
        sha, notes = self._cached_revision(model, operation=operation)
        snapshot = self._repo_cache_folder(model, operation=operation) / "snapshots" / sha
        siblings = _snapshot_files(snapshot)
        card = _card_metadata(snapshot)
        licence = card.get("license")
        if isinstance(licence, list):
            licence = next((item for item in licence if isinstance(item, str)), None)
        info = _CachedModelInfo(
            sha=sha,
            siblings=siblings,
            tags=_string_list(card.get("tags")),
            pipeline_tag=_optional_str(card.get("pipeline_tag")),
            library_name=_optional_str(card.get("library_name")),
            card_data={"license": licence} if isinstance(licence, str) else None,
            safetensors=_local_safetensors_parameters(snapshot),
            gguf=_local_gguf_metadata(snapshot, siblings) or None,
        )
        notes.append(
            f"Offline: built from the local Hugging Face cache at {sha[:12]}. Only files "
            "present in the cache are listed, and what only the Hub reports (gated or "
            "private status, downloads, likes) is not shown."
        )
        return info, notes

    # --- metadata ------------------------------------------------------

    def _model_info(
        self,
        model: ModelIdentity,
        *,
        files_metadata: bool,
        operation: str,
    ) -> Any:
        """Call ``HfApi.model_info`` and translate any failure."""
        try:
            return self._api.model_info(
                model.repo_id,
                revision=model.revision,
                files_metadata=files_metadata,
            )
        except Exception as error:
            raise translate_hub_error(
                error, model=model, operation=operation, offline=self.offline
            ) from error

    def _fetch_small_json(
        self,
        repo_id: str,
        revision: str,
        entry: RepoFile,
    ) -> tuple[dict[str, object] | None, str | None]:
        """Download one small JSON file from a pinned revision and parse it.

        Returns ``(payload, note)``. Failures never raise: inspection has to keep
        working on a repository whose optional metadata is unreachable, so the
        reason is returned as a note instead.
        """
        if entry.size is not None and entry.size > MAX_METADATA_FILE_BYTES:
            return None, (
                f"{entry.path} is {entry.size} bytes, larger than the "
                f"{MAX_METADATA_FILE_BYTES}-byte metadata limit, and was not fetched."
            )
        try:
            if self.offline:
                # Offline, *revision* is a snapshot _cached_revision found and the
                # entry came from that snapshot's own listing: read it in place.
                located: str | Path = (
                    self._repo_cache_folder(ModelIdentity(repo_id=repo_id), operation="inspection")
                    / "snapshots"
                    / revision
                    / entry.path
                )
            else:
                located = self._api.hf_hub_download(
                    repo_id,
                    filename=entry.path,
                    revision=revision,
                    cache_dir=self.cache_dir,
                )
            payload = json.loads(Path(located).read_text(encoding="utf-8"))
        except Exception as error:
            return None, f"{entry.path} could not be read: {redact_text(str(error))}"
        if not isinstance(payload, dict):
            return None, f"{entry.path} is not a JSON object."
        return payload, None

    def _fetch_shard_indexes(
        self,
        repo_id: str,
        revision: str,
        files: Sequence[RepoFile],
        notes: list[str],
    ) -> dict[str, list[str]]:
        """Fetch every weight index document and return its shard file names.

        Index documents are a few tens of kilobytes; reading them is what makes
        "is this shard set complete?" an answer rather than a guess. No weight
        shard is downloaded.
        """
        indexes: dict[str, list[str]] = {}
        for entry in files:
            if not (
                entry.name.endswith(SAFETENSORS_INDEX_SUFFIX)
                or entry.name.endswith(PYTORCH_INDEX_SUFFIX)
            ):
                continue
            payload, note = self._fetch_small_json(repo_id, revision, entry)
            if note is not None:
                notes.append(note)
            if payload is None:
                continue
            weight_map = payload.get("weight_map")
            if isinstance(weight_map, Mapping):
                indexes[entry.path] = sorted(
                    {value for value in weight_map.values() if isinstance(value, str)}
                )
            else:
                notes.append(f"{entry.path} has no weight_map object.")
        return indexes

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        """Collect repository metadata without downloading any weight shard.

        One ``model_info`` call supplies the resolved SHA, licence, gated/private
        flags, tags and the complete file inventory (with sizes when
        ``with_sizes`` is set). ``config.json`` and any weight index documents —
        kilobytes, not gigabytes — are then fetched from the *pinned* SHA to
        identify the architecture, spot ``auto_map`` remote code and prove shard
        completeness.

        A repository without ``config.json`` (the ``HauhauCS`` GGUF fixture is
        one) is inspected successfully: ``model_type`` stays ``None``, the
        repository tags and any GGUF metadata carry the architecture hint, and a
        note records why.

        Offline, no request is made at all: the same fields are read from the
        cached snapshot (:meth:`_local_model_info`), and a snapshot holding no
        weights raises ``hub.offline.missing`` rather than describing a
        repository by its ``config.json`` alone.
        """
        info: Any
        notes: list[str]
        if self.offline:
            info, notes = self._local_model_info(model, operation="inspection")
        else:
            info = self._model_info(model, files_metadata=with_sizes, operation="inspection")
            notes = []

        sha = getattr(info, "sha", None)
        resolved = sha.strip() if isinstance(sha, str) and sha.strip() else model.revision
        files = _repo_files(info)
        tags = _string_list(getattr(info, "tags", None))

        config: dict[str, object] = {}
        architectures: list[str] = []
        model_type: str | None = None
        config_entry = next((entry for entry in files if entry.path == CONFIG_FILENAME), None)
        if config_entry is None:
            notes.append(
                "This repository has no config.json, so the architecture cannot be read "
                "from a model configuration. Use the repository tags "
                f"({', '.join(tags[:6]) or 'none'}) or the GGUF metadata instead."
            )
        elif resolved is None:
            notes.append("config.json was not fetched: the revision could not be pinned.")
        else:
            payload, note = self._fetch_small_json(model.repo_id, resolved, config_entry)
            if note is not None:
                notes.append(note)
            if payload is None:
                # Fall back to the summary the Hub itself derived from config.json.
                summary = getattr(info, "config", None)
                if isinstance(summary, Mapping):
                    architectures = _string_list(summary.get("architectures"))
                    raw_type = summary.get("model_type")
                    model_type = raw_type.strip() if isinstance(raw_type, str) else None
                    notes.append("Architecture taken from the Hub's config summary.")
            else:
                config = payload
                architectures = _string_list(payload.get("architectures"))
                raw_type = payload.get("model_type")
                model_type = raw_type.strip() if isinstance(raw_type, str) else None

        shard_indexes: dict[str, list[str]] = {}
        if resolved is not None:
            shard_indexes = self._fetch_shard_indexes(model.repo_id, resolved, files, notes)

        # Read before detection: the Hub's GGUF metadata is what tells a
        # multi-quantization repository apart from a single-model one.
        gguf_metadata = _compact_gguf_metadata(getattr(info, "gguf", None))
        formats = detect_formats(
            files,
            shard_indexes=shard_indexes,
            gguf_total_file_size=_optional_int(gguf_metadata.get(GGUF_TOTAL_FILE_SIZE_KEY)),
        )
        if self.offline and not any(observation.files for observation in formats):
            # An online inspect caches config.json and the weight index, so a
            # snapshot of metadata alone is common. It is not a weightless model.
            raise self._offline_missing(
                model,
                operation="inspection",
                message=(
                    f"the local Hugging Face cache holds {len(files)} file(s) of "
                    f"{model.repo_id} at {str(resolved)[:12]} but none of its model "
                    "weights, so it cannot be inspected or prepared without network access."
                ),
            )
        _attach_notes(formats, notes)

        requires_remote_code, remote_code_files = detect_remote_code(config, files)

        return ModelMetadata(
            identity=model,
            resolved_revision=resolved,
            exists=True,
            gated=bool(getattr(info, "gated", False)),
            private=bool(getattr(info, "private", False)),
            license=_license_of(info, tags),
            pipeline_tag=_optional_str(getattr(info, "pipeline_tag", None)),
            tags=tags,
            library_name=_optional_str(getattr(info, "library_name", None)),
            architectures=architectures,
            model_type=model_type,
            config=config,
            files=files,
            formats=formats,
            requires_remote_code=requires_remote_code,
            remote_code_files=remote_code_files,
            parameter_count=_parameter_count(info),
            parameter_dtypes=_parameter_dtypes(info),
            gguf_metadata=gguf_metadata,
            downloads=_optional_int(getattr(info, "downloads", None)),
            likes=_optional_int(getattr(info, "likes", None)),
            last_modified=_isoformat(getattr(info, "last_modified", None)),
        )

    #: Search result fields, in one request: identity for registry matching
    #: (``config`` for Transformers repos, ``gguf`` for GGUF-only ones), a format
    #: hint (``library_name``, ``tags``) and ranking signals. No file is fetched.
    SEARCH_EXPAND: tuple[str, ...] = (
        "config",
        "gguf",
        "pipeline_tag",
        "tags",
        "library_name",
        "downloads",
        "likes",
        "gated",
        "safetensors",
        "lastModified",
    )

    def search(self, query: str, *, limit: int = 20) -> list[ModelMetadata]:
        """Search the Hub and describe each hit from the search response alone.

        One ``list_models`` call carries everything used here, so a search costs
        one request however many results come back -- no ``config.json`` is
        downloaded per hit. The metadata is therefore thinner than
        :meth:`inspect`'s: there is no file inventory, so ``formats`` is empty
        and the format is only hinted at by tags and the library name.
        """
        if self.offline:
            # Search is a Hub query; there is no local index to answer it from,
            # so this is not a cache miss and must not be worded as one.
            raise SourceResolutionError(
                _hub_diagnostic(
                    CODE_OFFLINE_SEARCH,
                    FailureCategory.DOWNLOAD_FAILURE,
                    "Search needs the network",
                    "Offline mode is active, and searching the Hugging Face Hub is a "
                    "network request: there is no local copy of the Hub to search.",
                    error=None,
                    model=None,
                    operation="search",
                    remediation=[
                        "Run the search without offline mode (no --offline, and unset "
                        "YONGA_OFFLINE / HF_HUB_OFFLINE).",
                        "A model already in the local cache can still be inspected offline: "
                        "yonga --offline inspect MODEL",
                    ],
                )
            )
        try:
            results = list(
                self._api.list_models(
                    search=query,
                    limit=max(1, limit),
                    sort="downloads",
                    expand=list(self.SEARCH_EXPAND),
                )
            )
        except Exception as error:
            raise translate_hub_error(error, operation="search", offline=self.offline) from error
        return [_search_hit_metadata(info) for info in results]

    # --- download ------------------------------------------------------

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: JobContext | None = None,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Materialise a pinned revision in the Hub cache and return its path.

        The returned path is the Hub's own snapshot directory: nothing is copied,
        and the exporter is pointed straight at it (``docs/MODEL_PIPELINE.md``
        section 6). Offline mode passes ``local_files_only=True`` and fails with
        ``hub.offline.missing`` when the revision is not fully cached.
        """
        revision = self._pinned_revision(source)
        pinned = source.with_revision(revision)
        offline = self.offline

        if ctx is not None:
            if ctx.check_cancelled():
                raise CancelledError(
                    diagnostic(
                        CODE_CANCELLED,
                        FailureCategory.USER_CANCELLED,
                        "Download cancelled",
                        f"The download of {source.repo_id} was cancelled before it started.",
                        context={"repo_id": source.repo_id, "revision": revision},
                    )
                )
            ctx.logger.info(
                "hub.download.start",
                f"downloading {source.repo_id}",
                repo_id=source.repo_id,
                revision=revision,
                offline=offline,
            )
            ctx.emit(
                ProgressKind.DOWNLOAD,
                f"Downloading {source.repo_id} at {revision[:12]}",
                status="started",
                repo_id=source.repo_id,
                revision=revision,
                offline=offline,
            )

        tqdm_class = (
            None
            if ctx is None
            else _progress_tqdm_class(ctx, repo_id=source.repo_id, revision=revision)
        )

        try:
            located = self._api.snapshot_download(
                source.repo_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=offline,
                allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
                ignore_patterns=list(ignore_patterns) if ignore_patterns is not None else None,
                tqdm_class=tqdm_class,
            )
        except Exception as error:
            raise translate_hub_error(
                error, model=pinned, operation="snapshot download", offline=offline
            ) from error

        if not isinstance(located, str):
            raise SourceResolutionError(
                _hub_diagnostic(
                    CODE_OFFLINE_MISSING,
                    FailureCategory.DOWNLOAD_FAILURE,
                    "Snapshot was not materialised",
                    f"snapshot_download did not return a local path for {pinned}.",
                    error=None,
                    model=pinned,
                    operation="snapshot download",
                    remediation=["Re-run the command; the Hub cache may be in a partial state."],
                )
            )

        snapshot = Path(located)
        if ctx is not None:
            ctx.logger.info(
                "hub.download.done",
                f"snapshot ready for {source.repo_id}",
                repo_id=source.repo_id,
                revision=revision,
                path=str(snapshot),
            )
            ctx.emit(
                ProgressKind.DOWNLOAD,
                f"{source.repo_id} is ready in the Hugging Face cache",
                status="completed",
                repo_id=source.repo_id,
                revision=revision,
                path=str(snapshot),
            )
        return snapshot

    # --- authentication ------------------------------------------------

    def gguf_header(self, model: ModelIdentity, filename: str) -> GGUFHeader:
        """Read one GGUF file's header -- metadata and tensor types -- not its weights.

        A copy already in the Hub cache is read locally. Otherwise only the
        header is fetched, with HTTP range requests of a few megabytes, so
        deciding whether a 9 GB file can be imported costs about 10 MB. Offline,
        an uncached file raises ``hub.offline.missing``.
        """
        revision = model.revision
        cached = huggingface_hub.try_to_load_from_cache(
            model.repo_id, filename, revision=revision, cache_dir=self._cache_dir()
        )
        if isinstance(cached, str) and Path(cached).is_file():
            try:
                return read_local_header(Path(cached))
            except (OSError, GGUFFormatError) as error:
                raise self._gguf_unreadable(model, filename, error) from error
        if self.offline:
            raise translate_hub_error(
                FileNotFoundError(filename),
                model=model,
                operation=f"reading the header of {filename}",
                offline=True,
            )
        url = huggingface_hub.hf_hub_url(model.repo_id, filename, revision=revision)
        session = huggingface_hub.get_session()
        headers = huggingface_hub.utils.build_hf_headers()  # type: ignore[attr-defined]

        def read(offset: int, length: int) -> bytes:
            response = session.get(
                url,
                headers={**headers, "Range": f"bytes={offset}-{offset + length - 1}"},
                follow_redirects=True,
                timeout=60.0,
            )
            response.raise_for_status()
            content: bytes = response.content
            # A server that ignores Range sends the whole file; keep only the slice.
            return content[:length] if response.status_code == 200 and offset == 0 else content

        try:
            return parse_gguf_header(read)
        except GGUFFormatError as error:
            raise self._gguf_unreadable(model, filename, error) from error
        except Exception as error:
            raise translate_hub_error(
                error, model=model, operation=f"reading the header of {filename}"
            ) from error

    def _cache_dir(self) -> str | None:
        home = getattr(self._settings, "hf_home", None)
        return str(Path(home) / "hub") if home else None

    @staticmethod
    def _gguf_unreadable(model: ModelIdentity, filename: str, error: BaseException) -> YongaError:
        return SourceResolutionError(
            diagnostic(
                "source.gguf_header_unreadable",
                FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
                "The GGUF file's header could not be read",
                redact_text(f"{model.repo_id}/{filename}: {error}"),
                remediation=["The file may be damaged or not GGUF; try another file."],
            )
        )

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report authentication state without ever handling the token value.

        Only the *presence* of a credential is observed, plus the user name the
        Hub reports for it. :class:`~yonga.domain.hardware.HuggingFaceAuthInfo`
        has no field that could hold a token and forbids extra fields, so a
        credential cannot leak through this path even by accident.
        """
        offline = self.offline
        cache_dir = self._effective_cache_dir()
        has_token = bool(get_token())

        if not has_token:
            return HuggingFaceAuthInfo(
                logged_in=False,
                token_source=None,
                offline=offline,
                cache_dir=cache_dir,
            )

        source = _token_source()
        if offline:
            return HuggingFaceAuthInfo(
                logged_in=True,
                user=None,
                token_source=source,
                offline=True,
                cache_dir=cache_dir,
            )

        try:
            identity = self._api.whoami()
        except Exception as error:
            rejected = _status_code(error) in (401, 403)
            return HuggingFaceAuthInfo(
                logged_in=not rejected,
                user=None,
                token_source=source,
                offline=offline,
                cache_dir=cache_dir,
            )

        name: object = None
        if isinstance(identity, Mapping):
            name = identity.get("name") or identity.get("fullname")
        return HuggingFaceAuthInfo(
            logged_in=True,
            user=str(name) if isinstance(name, str) and name.strip() else None,
            token_source=source,
            offline=offline,
            cache_dir=cache_dir,
        )


# --------------------------------------------------------------------------- #
# ``model_info`` field extraction (defensive: mocks and Hubs both vary)
# --------------------------------------------------------------------------- #


def _optional_str(value: object) -> str | None:
    """Return a non-empty stripped string, or ``None``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_int(value: object) -> int | None:
    """Return an ``int``, or ``None`` when the Hub omitted the field."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _string_list(value: object) -> list[str]:
    """Return the non-empty strings in a sequence, preserving order."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _isoformat(value: object) -> str | None:
    """Render a ``datetime`` (or an already-formatted string) as ISO 8601."""
    formatter = getattr(value, "isoformat", None)
    if callable(formatter):
        rendered = formatter()
        return rendered if isinstance(rendered, str) else None
    return _optional_str(value)


def _sibling_size(sibling: object) -> int | None:
    """Return a sibling's size, falling back to its LFS metadata."""
    size = getattr(sibling, "size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return size
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    lfs_size = getattr(lfs, "size", None)
    if isinstance(lfs_size, int) and not isinstance(lfs_size, bool) and lfs_size >= 0:
        return lfs_size
    if isinstance(lfs, Mapping):
        raw = lfs.get("size")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            return raw
    return None


def _repo_files(info: object) -> list[RepoFile]:
    """Turn ``ModelInfo.siblings`` into the domain's file inventory."""
    siblings = getattr(info, "siblings", None) or []
    files: list[RepoFile] = []
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None)
        if not isinstance(path, str) or not path.strip():
            continue
        files.append(
            RepoFile(
                path=path.strip(),
                size=_sibling_size(sibling),
                lfs=getattr(sibling, "lfs", None) is not None,
            )
        )
    files.sort(key=lambda entry: entry.path)
    return files


def _license_of(info: object, tags: Sequence[str]) -> str | None:
    """Read the licence from the model card, falling back to a ``license:`` tag."""
    card = getattr(info, "card_data", None)
    if card is not None:
        value = getattr(card, "license", None)
        if value is None and isinstance(card, Mapping):
            value = card.get("license")
        text = _optional_str(value)
        if text is not None:
            return text
    for tag in tags:
        if tag.startswith("license:"):
            return _optional_str(tag.split(":", 1)[1])
    return None


def _parameter_count(info: object) -> int | None:
    """Total parameter count, from safetensors metadata or GGUF metadata."""
    safetensors = getattr(info, "safetensors", None)
    total = getattr(safetensors, "total", None) if safetensors is not None else None
    if total is None and isinstance(safetensors, Mapping):
        total = safetensors.get("total")
    count = _optional_int(total)
    if count is not None:
        return count
    gguf = getattr(info, "gguf", None)
    if isinstance(gguf, Mapping):
        return _optional_int(gguf.get("total"))
    return None


def _parameter_dtypes(info: object) -> dict[str, int]:
    """Parameter count per stored dtype, from the safetensors metadata, if reported."""
    safetensors = getattr(info, "safetensors", None)
    raw = getattr(safetensors, "parameters", None) if safetensors is not None else None
    if raw is None and isinstance(safetensors, Mapping):
        raw = safetensors.get("parameters")
    if not isinstance(raw, Mapping):
        return {}
    found: dict[str, int] = {}
    for dtype, count in raw.items():
        value = _optional_int(count)
        if isinstance(dtype, str) and value is not None and value >= 0:
            found[dtype] = value
    return found


def _token_source() -> str:
    """Name *where* a credential was found. Never the credential itself."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(name, "").strip():
            return f"environment:{name}"
    token_path = getattr(huggingface_hub.constants, "HF_TOKEN_PATH", None)
    if isinstance(token_path, str) and Path(token_path).exists():
        return "token-file"
    return "huggingface_hub"
