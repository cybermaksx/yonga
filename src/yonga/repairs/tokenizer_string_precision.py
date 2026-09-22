"""Repair ``ov-tokenizer-precision-string-casing``: ``precision="STRiNG"`` -> ``"STRING"``.

## The defect

OpenVINO 2026.4.0 with openvino-tokenizers 2026.4.0.0 serialises a string tensor
port as::

    <port id="0" precision="STRiNG" names="string_input">

while writing the element type correctly on the ``<data>`` element of the very same
layer::

    <data shape="?" element_type="string" />

Reading that file back with the same release fails::

    Check 'is_valid_type_idx(type_idx)' failed at src/core/src/type/element_type.cpp:115:
    Unsupported element type: STRiNG

This is a live round-trip defect in the current stack, not a historical quirk: every
freshly exported ``openvino_tokenizer.xml`` / ``openvino_detokenizer.xml`` on this
machine contains it. Correcting only the port precision makes ``read_model`` and
``compile_model(..., "CPU")`` both succeed.

## Why a textual substitution, verified by an XML parse

The obvious implementation — parse with ``ElementTree``, fix the attribute, write the
tree back — is *not* acceptable here. ``ElementTree.write`` re-serialises the whole
document: it would renormalise attribute quoting and ordering, drop comments, rewrite
the XML declaration and change indentation. Those bytes are upstream's, this repair
has no business touching them, and a diff nobody can read is a diff nobody audits.

So the parse is used as a *guard*, not as the transformation:

1. parse the file and confirm it really is an OpenVINO IR net (root ``<net>`` with a
   ``<layers>`` child);
2. count how many ``<port>`` elements carry ``precision="STRiNG"``;
3. count the textual occurrences of the ``precision="STRiNG"`` attribute;
4. only when those two counts are *equal* is a substitution performed — that equality
   is what proves every textual occurrence is a port precision attribute and not a
   comment, a layer name, or element text.

The substitution itself is a regex anchored on the attribute name and quoting, the
post-substitution count is verified against the pre-count, and the result is checked
to differ from the original in exactly one byte per occurrence (``i`` -> ``I``).

``element_type="string"`` is lower case, never matches the pattern, and its count is
asserted to be unchanged. Generic case-normalisation of precision values is expressly
forbidden by ``SPEC.md`` section 13 and is not implemented here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from xml.etree import ElementTree

from yonga.domain.artifacts import MANIFEST_DIRNAME
from yonga.domain.diagnostics import FailureCategory, RepairError, diagnostic
from yonga.repairs.base import (
    Repair,
    RepairContext,
    RepairMatch,
    RepairResult,
    RepairSafety,
    register_repair,
)
from yonga.util.hashing import sha256_bytes

__all__ = [
    "BAD_PRECISION",
    "DEFAULT_FILENAMES",
    "GOOD_PRECISION",
    "IMPLEMENTATION_NAME",
    "REPAIR_ID",
    "TokenizerStringPrecisionRepair",
    "count_bad_precision",
]

REPAIR_ID = "ov-tokenizer-precision-string-casing"
"""Stable repair identifier recorded in the artifact manifest."""

IMPLEMENTATION_NAME = "tokenizer_string_precision"
"""The name registry rules use in ``repair.implementation``."""

BAD_PRECISION = "STRiNG"
"""The invalid mixed-case value emitted by the serialiser."""

GOOD_PRECISION = "STRING"
"""The value OpenVINO's element-type table accepts."""

DEFAULT_FILENAMES: tuple[str, ...] = ("openvino_tokenizer.xml", "openvino_detokenizer.xml")
"""Files this repair considers when the registry rule names none."""

#: The exact attribute occurrence. Anchored on whitespace before the attribute name so
#: a hypothetical ``foo_precision="STRiNG"`` cannot match, and on a back-referenced
#: quote character so an unbalanced quote cannot be spliced.
_PRECISION_ATTR = re.compile(r'(?<=\s)precision=(?P<quote>["\'])' + BAD_PRECISION + r"(?P=quote)")

#: Used only to assert that the lower-case element type is left alone.
_ELEMENT_TYPE_STRING = re.compile(r'(?<=\s)element_type=(?P<quote>["\'])string(?P=quote)')

#: Subdirectory of the artifact's tool-owned metadata directory holding backups.
_BACKUP_SUBDIR = "repairs"

#: A repair only ever considers files matching this shape, whatever the registry says.
_ALLOWED_NAME = re.compile(r"^openvino_[A-Za-z0-9_.-]*\.xml$")


def count_bad_precision(text: str) -> int:
    """Return how many ``precision="STRiNG"`` attribute occurrences ``text`` holds."""
    return len(_PRECISION_ATTR.findall(text))


def _count_element_type_string(text: str) -> int:
    """Return how many ``element_type="string"`` occurrences ``text`` holds."""
    return len(_ELEMENT_TYPE_STRING.findall(text))


def _iter_ports(root: ElementTree.Element) -> Iterator[ElementTree.Element]:
    """Yield every ``<port>`` element in an OpenVINO IR document."""
    return root.iter("port")


@register_repair
class TokenizerStringPrecisionRepair(Repair):
    """Rewrites the invalid ``precision="STRiNG"`` port attribute to ``"STRING"``.

    Scope is deliberately tiny: only ``openvino_*.xml`` files directly inside a
    tool-owned artifact directory, only the ``precision`` attribute of ``<port>``
    elements, and only when an XML parse confirms every textual occurrence is such an
    attribute.
    """

    id: ClassVar[str] = REPAIR_ID
    implementation_name: ClassVar[str] = IMPLEMENTATION_NAME
    description: ClassVar[str] = (
        'Rewrite the invalid OpenVINO IR port precision "STRiNG" to "STRING" in generated '
        'tokenizer/detokenizer XML, leaving element_type="string" untouched.'
    )
    safety: ClassVar[RepairSafety] = RepairSafety.AUTOMATIC

    # --- detection ---------------------------------------------------------------

    def candidate_names(self, context: RepairContext) -> tuple[str, ...]:
        """Return the file names to examine, honouring the registry rule's list.

        Names coming from registry data are filtered to bare ``openvino_*.xml`` file
        names: a rule may narrow or extend the set within that shape, but it can never
        point the repair at an arbitrary path.
        """
        requested = context.rule.files if context.rule is not None else ()
        names = requested or DEFAULT_FILENAMES
        allowed: list[str] = []
        for raw in names:
            name = raw.strip()
            if not _ALLOWED_NAME.match(name):
                context.logger.warning(
                    "repair.filename_rejected",
                    f"{self.id}: ignoring out-of-scope file name {name!r}",
                    repair=self.id,
                    filename=name,
                )
                continue
            if name not in allowed:
                allowed.append(name)
        return tuple(allowed)

    def detect(self, artifact: Path, context: RepairContext) -> RepairMatch | None:
        """Find tokenizer XML files carrying the invalid port precision.

        Returns ``None`` for an already repaired artifact, which is what makes
        re-running preparation idempotent.
        """
        root = context.assert_tool_owned(artifact)
        counts: dict[str, int] = {}
        files: list[Path] = []
        element_type_counts: dict[str, int] = {}
        rejected: list[str] = []
        inspected: list[str] = []

        for name in self.candidate_names(context):
            path = root / name
            if not path.is_file():
                continue
            context.assert_tool_owned(path)
            inspected.append(name)
            text = _read_text(path)
            if text is None:
                rejected.append(f"{name}: not valid UTF-8 text")
                continue
            textual = count_bad_precision(text)
            if textual == 0:
                continue
            verdict = _verify_ir_document(text, textual)
            if verdict is not None:
                rejected.append(f"{name}: {verdict}")
                context.logger.warning(
                    "repair.context_mismatch",
                    f"{self.id}: {name} matched textually but not structurally ({verdict})",
                    repair=self.id,
                    filename=name,
                )
                continue
            counts[name] = textual
            element_type_counts[name] = _count_element_type_string(text)
            files.append(path)

        if not counts:
            return None

        details: dict[str, object] = {
            "pattern": f'precision="{BAD_PRECISION}"',
            "replacement": f'precision="{GOOD_PRECISION}"',
            "element_type_string_counts": element_type_counts,
            "inspected_files": inspected,
            "openvino_version": context.openvino_version or "unknown",
            "tokenizers_version": context.tokenizers_version or "unknown",
        }
        if rejected:
            details["rejected_files"] = rejected

        return RepairMatch(
            repair_id=self.id,
            files=tuple(files),
            counts=counts,
            details=details,
            rule_id=context.rule.id if context.rule is not None else None,
        )

    # --- application -------------------------------------------------------------

    def apply(self, match: RepairMatch, artifact: Path, context: RepairContext) -> RepairResult:
        """Back up, substitute and atomically rewrite each matched file.

        Handed a stale match for an already repaired artifact this writes nothing,
        creates no backup directory and returns ``applied=False``.
        """
        root = context.assert_tool_owned(artifact)
        backup_dir = root / MANIFEST_DIRNAME / _BACKUP_SUBDIR / self.id
        before: dict[str, str] = {}
        after: dict[str, str] = {}
        changed: list[str] = []
        unchanged: list[str] = []
        element_type_counts: dict[str, int] = {}
        written: list[tuple[Path, bytes]] = []

        try:
            for name in match.names:
                path = context.assert_tool_owned(root / name)
                if not path.is_file():
                    unchanged.append(name)
                    continue
                original = path.read_bytes()
                text = original.decode("utf-8")
                pre_count = count_bad_precision(text)
                before[name] = sha256_bytes(original)
                if pre_count == 0:
                    # Idempotent path: nothing to do, so nothing is backed up.
                    after[name] = before[name]
                    unchanged.append(name)
                    continue

                repaired = _substitute(text, pre_count=pre_count, name=name)
                payload = repaired.encode("utf-8")
                _assert_minimal_diff(original, payload, pre_count=pre_count, name=name)
                element_type_counts[name] = _count_element_type_string(repaired)
                after[name] = sha256_bytes(payload)
                changed.append(name)

                if context.dry_run:
                    continue

                backup_dir.mkdir(parents=True, exist_ok=True)
                (backup_dir / path.name).write_bytes(original)
                _atomic_write(path, payload)
                written.append((path, original))
                context.logger.info(
                    "repair.file_rewritten",
                    f"{self.id}: {name} ({pre_count} occurrence(s))",
                    repair=self.id,
                    filename=name,
                    occurrences=pre_count,
                    sha256_before=before[name],
                    sha256_after=after[name],
                )
        except Exception:
            _rollback(written)
            raise

        applied = bool(changed) and not context.dry_run
        detail: dict[str, object] = dict(match.details)
        detail.update(
            {
                "occurrences": match.total_occurrences,
                "changed_files": changed,
                "unchanged_files": unchanged,
                "element_type_string_counts_after": element_type_counts,
            }
        )
        if context.dry_run:
            detail["dry_run"] = True
        if not changed:
            detail["no_op"] = True

        return RepairResult(
            repair_id=self.id,
            description=self.description,
            safety=self.safety,
            applied=applied,
            files=match.names,
            before_hashes=before,
            after_hashes=after,
            backup_dir=backup_dir if applied else None,
            detail=detail,
            rule_id=match.rule_id,
            applied_at=datetime.now(UTC) if applied else None,
            artifact_dir=root,
        )

    # --- validation --------------------------------------------------------------

    def validate(self, result: RepairResult, context: RepairContext) -> RepairResult:
        """Assert the post-conditions and, when possible, prove them with OpenVINO.

        The textual post-condition (no ``precision="STRiNG"`` remains, the
        ``element_type="string"`` count is unchanged) is mandatory. Runtime validation
        needs OpenVINO, which is an optional dependency: when it is missing the repair
        is *not* failed, the skip is recorded instead.
        """
        root = context.assert_tool_owned(result.artifact_dir or context.artifact_dir)
        if context.dry_run or not result.applied:
            return result.with_detail(validation="skipped: no files were modified")

        checked: list[str] = []
        for name in result.files:
            path = root / name
            if not path.is_file():
                continue
            text = _read_text(path)
            if text is None:
                raise _validation_error(f"{name} is no longer readable as UTF-8 text after repair.")
            remaining = count_bad_precision(text)
            if remaining:
                raise _validation_error(
                    f'{name} still contains {remaining} occurrence(s) of precision="'
                    f'{BAD_PRECISION}" after the repair.'
                )
            checked.append(name)

        expected = result.detail.get("element_type_string_counts")
        actual = result.detail.get("element_type_string_counts_after")
        if isinstance(expected, dict) and isinstance(actual, dict):
            for name, count in actual.items():
                if expected.get(name) != count:
                    raise _validation_error(
                        f'{name}: the number of element_type="string" attributes changed '
                        f"from {expected.get(name)} to {count}; the repair must never touch them."
                    )

        result = result.with_detail(postcondition_checked_files=checked)
        return self._validate_with_openvino(result, root, context)

    def _validate_with_openvino(
        self,
        result: RepairResult,
        root: Path,
        context: RepairContext,
    ) -> RepairResult:
        """Read and CPU-compile the repaired tokenizer IRs with the real runtime.

        OpenVINO and openvino-tokenizers are imported here, never at module scope, so
        ``import yonga.cli`` works on a host with neither installed.
        """
        try:
            import openvino
        except ImportError as exc:
            context.logger.info(
                "repair.runtime_validation_skipped",
                f"{self.id}: OpenVINO is not installed; runtime validation skipped",
                repair=self.id,
            )
            return result.with_detail(
                runtime_validation="skipped",
                runtime_validation_reason=f"openvino is not importable: {exc}",
            )

        extension_loaded = True
        extension_note = ""
        try:
            # Importing this package patches ``openvino.Core`` so the tokenizer
            # operations resolve; a real tokenizer IR cannot be read without it.
            import openvino_tokenizers  # noqa: F401
        except ImportError as exc:
            extension_loaded = False
            extension_note = f"openvino_tokenizers is not importable: {exc}"

        targets = _validation_targets(root, result.files)
        core = openvino.Core()
        read_ok: list[str] = []
        compiled_ok: list[str] = []
        for path in targets:
            name = path.name
            try:
                model = core.read_model(path)
            except Exception as exc:
                raise _validation_error(
                    f"{name} still cannot be read by OpenVINO after the repair: {exc}",
                    category=FailureCategory.TOKENIZER_IR_INCOMPATIBLE,
                ) from exc
            read_ok.append(name)
            try:
                core.compile_model(model, "CPU")
            except Exception as exc:
                raise _validation_error(
                    f"{name} was read but could not be compiled on CPU after the repair: {exc}",
                    category=FailureCategory.TOKENIZER_IR_INCOMPATIBLE,
                ) from exc
            compiled_ok.append(name)

        context.logger.info(
            "repair.runtime_validation_ok",
            f"{self.id}: {len(compiled_ok)} IR(s) read and CPU-compiled after repair",
            repair=self.id,
            files=compiled_ok,
        )
        detail: dict[str, object] = {
            "runtime_validation": "ok",
            "openvino_version": str(openvino.get_version()),
            "read_ok": read_ok,
            "cpu_compile_ok": compiled_ok,
            "tokenizer_extension_loaded": extension_loaded,
        }
        if extension_note:
            detail["tokenizer_extension_note"] = extension_note
        return result.with_detail(**detail)


# --- helpers -------------------------------------------------------------------------


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 without newline translation, or ``None`` if it is not text.

    ``Path.read_text`` applies universal newline translation, which would silently
    rewrite CRLF line endings on save. Bytes are read and decoded explicitly so the
    repair can guarantee it changes nothing but the attribute value.
    """
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _verify_ir_document(text: str, textual_count: int) -> str | None:
    """Return why ``text`` may not be repaired, or ``None`` when it may.

    The XML parse is a guard: it proves the document is an OpenVINO IR net and that
    every textual occurrence of the pattern is a ``<port>`` precision attribute, so
    the substitution that follows cannot touch a comment, a name or element text.
    """
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        return f"not well-formed XML ({exc})"
    if root.tag != "net":
        return f"root element is <{root.tag}>, not an OpenVINO IR <net>"
    if root.find("layers") is None:
        return "no <layers> element: not an OpenVINO IR document"
    port_count = sum(1 for port in _iter_ports(root) if port.get("precision") == BAD_PRECISION)
    if port_count == 0:
        return f'no <port precision="{BAD_PRECISION}"> element'
    if port_count != textual_count:
        return (
            f"{textual_count} textual occurrence(s) but only {port_count} on <port> elements; "
            "the pattern appears in a context this repair does not understand"
        )
    return None


def _substitute(text: str, *, pre_count: int, name: str) -> str:
    """Replace every ``precision="STRiNG"`` attribute, asserting the count matches."""
    replaced, count = _PRECISION_ATTR.subn(
        lambda m: f"precision={m.group('quote')}{GOOD_PRECISION}{m.group('quote')}",
        text,
    )
    if count != pre_count:
        raise _validation_error(
            f"{name}: expected to replace {pre_count} occurrence(s) but replaced {count}."
        )
    remaining = count_bad_precision(replaced)
    if remaining:
        raise _validation_error(
            f"{name}: {remaining} occurrence(s) of the invalid precision survived the substitution."
        )
    return replaced


def _assert_minimal_diff(original: bytes, payload: bytes, *, pre_count: int, name: str) -> None:
    """Assert the rewrite changed exactly one byte per occurrence and nothing else.

    ``STRiNG`` and ``STRING`` have the same length and differ in a single character,
    so a correct repair produces a file of identical size differing in exactly
    ``pre_count`` bytes. Anything else means the substitution escaped its scope.
    """
    if len(original) != len(payload):
        raise _validation_error(
            f"{name}: the repair changed the file length from {len(original)} to {len(payload)} "
            "bytes; only the attribute value may change."
        )
    differing = sum(1 for left, right in zip(original, payload, strict=True) if left != right)
    if differing != pre_count:
        raise _validation_error(
            f"{name}: expected exactly {pre_count} differing byte(s) after the repair, "
            f"found {differing}."
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically, preserving the file mode.

    The temporary file is created in the same directory so the final rename is a
    same-filesystem, atomic operation: a crash mid-write can never leave a truncated
    tokenizer IR behind.
    """
    mode: int | None = None
    with suppress(OSError):
        mode = path.stat().st_mode & 0o7777
    tmp = path.with_name(f"{path.name}.yonga-repair-{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            tmp.chmod(mode)
        tmp.replace(path)
    finally:
        if tmp.exists():
            with suppress(OSError):
                tmp.unlink()


def _rollback(written: Sequence[tuple[Path, bytes]]) -> None:
    """Restore the original bytes of every file already rewritten in this attempt."""
    for path, original in reversed(written):
        with suppress(OSError):
            _atomic_write(path, original)


def _validation_targets(root: Path, repaired: Sequence[str]) -> tuple[Path, ...]:
    """Return the IR files runtime validation should exercise.

    Both the tokenizer and the detokenizer are checked whenever they exist, not only
    the files that needed repairing: the pair is loaded together at runtime, so
    proving one of them readable proves nothing useful.
    """
    targets: dict[Path, None] = {}
    for name in (*repaired, *DEFAULT_FILENAMES):
        path = root / name
        if path.is_file() and path.suffix == ".xml":
            targets.setdefault(path, None)
    return tuple(targets)


def _validation_error(
    message: str,
    *,
    category: FailureCategory = FailureCategory.TOKENIZER_IR_INCOMPATIBLE,
) -> RepairError:
    """Build the typed error raised when a repair post-condition fails."""
    return RepairError(
        diagnostic(
            "repair.postcondition_failed",
            category,
            "Repair post-condition failed",
            message,
            remediation=[
                "Restore the originals from the artifact's .yonga/repairs backup directory.",
                "Re-run preparation with --debug and attach the job log to a bug report.",
            ],
            context={"repair": REPAIR_ID},
        )
    )
