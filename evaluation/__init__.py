"""Offline evaluation: dataset identity, metrics, calibration, benchmarking.

**Nothing in the serving path imports this package.** The dependency runs one
way — ``evaluation`` imports ``inference`` and ``contracts``, never the reverse —
because these are the tools you use to decide whether a model is fit to ship,
not the code that ships. A guard test asserts the direction, so a convenient
import cannot quietly drag scipy or a benchmark harness into a worker.

Why a separate top-level package rather than more modules under ``inference``:
that package's dependency list is its whole point, and it is enforced by a test
that fails on any import outside ``contracts`` plus numpy, cv2, torch and
onnxruntime. Evaluation needs none of that restraint and should not have to
argue for it.

What this ``__init__`` deliberately does **not** re-export: ``benchmark``,
``calibration_protocol`` and ``band``. All three pull in numpy, and importing
them here would make ``import evaluation`` — currently numpy-free — drag it in
for anyone who only wanted ``MediaItem``. Import those three by module.

A note on naming, because it has bitten this repository twice. A top-level
``datasets/`` package would shadow the HuggingFace ``datasets`` library the same
way ``queue/`` shadowed the stdlib module — silently, at import time, with an
error that names the wrong culprit. Dataset code therefore lives at
``evaluation.dataset``, a submodule, where no shadowing is possible.
"""

from evaluation.dataset import (
    Compression,
    IdentitySplitter,
    LeakageReport,
    MediaItem,
    ResolutionBucket,
    Split,
    SplitAssignment,
    detect_leakage,
    group_by_slice,
    slice_counts,
)
from evaluation.protocol import (
    DatasetRole,
    EvaluationProtocol,
    ProtocolViolation,
    RoleAssignment,
    RoleSelection,
    select,
    validate_protocol,
    verify_run_selection,
)
from evaluation.split_lock import (
    LockIdentityDrift,
    SplitLock,
    SplitLockError,
    create_lock,
    detect_drift,
    extend_lock,
    verify_lock,
)

__all__ = [
    "Compression",
    "DatasetRole",
    "EvaluationProtocol",
    "LockIdentityDrift",
    "ProtocolViolation",
    "RoleSelection",
    "IdentitySplitter",
    "LeakageReport",
    "MediaItem",
    "ResolutionBucket",
    "RoleAssignment",
    "Split",
    "SplitAssignment",
    "SplitLock",
    "SplitLockError",
    "create_lock",
    "detect_drift",
    "detect_leakage",
    "extend_lock",
    "group_by_slice",
    "select",
    "slice_counts",
    "validate_protocol",
    "verify_lock",
    "verify_run_selection",
]
