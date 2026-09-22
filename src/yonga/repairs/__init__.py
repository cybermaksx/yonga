"""Narrow, versioned, registry-selected repairs for generated artifacts.

Importing this package registers every built-in repair implementation, so
:func:`~yonga.repairs.base.get_repair` and
:func:`~yonga.repairs.base.known_implementations` resolve without further setup.

Nothing here imports OpenVINO at module scope: the runtime is reached only inside
:meth:`TokenizerStringPrecisionRepair.validate`, and its absence downgrades runtime
validation to a recorded skip rather than failing the repair.
"""

from __future__ import annotations

from yonga.repairs.base import (
    REPAIR_IMPLEMENTATIONS,
    REPAIR_VERSION_DISTRIBUTIONS,
    ConfirmCallback,
    Repair,
    RepairContext,
    RepairMatch,
    RepairResult,
    RepairRuleView,
    RepairSafety,
    apply_repairs,
    get_repair,
    huggingface_cache_roots,
    installed_repair_versions,
    known_implementations,
    register_repair,
)
from yonga.repairs.tokenizer_string_precision import (
    BAD_PRECISION,
    DEFAULT_FILENAMES,
    GOOD_PRECISION,
    IMPLEMENTATION_NAME,
    REPAIR_ID,
    TokenizerStringPrecisionRepair,
    count_bad_precision,
)

__all__ = [
    "BAD_PRECISION",
    "DEFAULT_FILENAMES",
    "GOOD_PRECISION",
    "IMPLEMENTATION_NAME",
    "REPAIR_ID",
    "REPAIR_IMPLEMENTATIONS",
    "REPAIR_VERSION_DISTRIBUTIONS",
    "ConfirmCallback",
    "Repair",
    "RepairContext",
    "RepairMatch",
    "RepairResult",
    "RepairRuleView",
    "RepairSafety",
    "TokenizerStringPrecisionRepair",
    "apply_repairs",
    "count_bad_precision",
    "get_repair",
    "huggingface_cache_roots",
    "installed_repair_versions",
    "known_implementations",
    "register_repair",
]
