"""SCRFD Phase 4 prerequisite P5 — tests do not run detection through the
implicit default detector.

**Why.** ``detect_faces(img)`` with no policy, and ``ImagePipeline()`` with no
``face_selection``, both resolve ``default_face_selection()``. Today that is
Haar, which ships inside the ``opencv-python`` wheel. When the shipped default
becomes SCRFD, detection through it needs a licensed, non-vendored ONNX asset
(``docs/SCRFD_ASSET.md``) — and, independently, the suite's synthetic face
fixtures were tuned for Haar and are not faces to a CNN. Every test that
reached detection implicitly would break at cutover for reasons unrelated to
what it tests.

**The rule this suite holds.** A test that runs detection says which detector
it means: :func:`detector_fixtures.haar_selection` for mechanics and Haar
behaviour, or a named entry in :data:`SHIPPED_DEFAULT_DETECTION` when the
shipped detector *is* the subject. That allowlist is also the list of tests
the cutover must give the real asset (or retire), so it is kept exact.

Construct-only identity tests (``ImagePipeline(...).model_version``) are not
detection and are not restricted — they are meant to follow the default.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from inference import preprocess as pre  # noqa: E402
from tests.detector_fixtures import HAAR_DETECTOR, haar_selection  # noqa: E402

TESTS = Path(__file__).resolve().parent

SHIPPED_DEFAULT_DETECTION: dict[tuple[str, str], str] = {
    ("test_face_selection_identity.py", "test_omitting_the_policy_uses_the_shipped_one"): (
        "asserts that omitting the policy reproduces the shipped one"
    ),
    ("test_image_decode_identity.py", "_analyse_image"): (
        "the shipped image bundle's identity, end to end"
    ),
    ("test_worker_inference_integration.py", "_gate_passing_image"): (
        "selects a fixture the production engine registry's detector accepts"
    ),
}
"""Tests that run detection through the shipped default on purpose.

Each needs the real SCRFD asset, or a rewrite, at cutover. Tests that drive
detection through ``workers.engine_registry`` (the production engines) are the
same kind of test but reach the default inside production code, where this
static check cannot see them — see ``docs/SCRFD_PHASE4_CUTOVER_DESIGN.md``."""


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _has_keyword(node: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in node.keywords)


def _string_text(fn: ast.AST) -> str:
    parts = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
    return "\n".join(parts)


def _implicit_default_detection(path: Path) -> list[tuple[str, str]]:
    """``(function, why)`` for each top-level function in ``path`` that runs
    detection without naming a detector."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for top in tree.body:
        if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(top) if isinstance(n, ast.Call)]
        if any(_call_name(c) == "detect_faces" and not _has_keyword(c, "policy") for c in calls):
            found.append((top.name, "detect_faces() without policy="))
        analyses = any(
            isinstance(n, ast.Attribute) and n.attr == "analyze_image" for n in ast.walk(top)
        )
        bare = any(
            _call_name(c) == "ImagePipeline" and not _has_keyword(c, "face_selection")
            for c in calls
        )
        if analyses and bare:
            found.append((top.name, "ImagePipeline() without face_selection= is analysed"))
        # Subprocess scripts are strings; the parser cannot see calls in them.
        if "ImagePipeline()" in _string_text(top):
            found.append((top.name, "a script string analyses a bare ImagePipeline()"))
    return found


def _violations() -> list[str]:
    out = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:  # holds the patterns as probe text
            continue
        for function, why in _implicit_default_detection(path):
            if (path.name, function) not in SHIPPED_DEFAULT_DETECTION:
                out.append(f"{path.name}::{function} — {why}")
    return out


def test_no_test_runs_detection_through_the_implicit_default():
    violations = _violations()
    assert not violations, (
        "these tests run face detection without naming a detector, so they will "
        "silently start needing the SCRFD asset (and stop matching their Haar-tuned "
        "fixtures) at cutover. Pass policy=haar_selection() / "
        "face_selection=haar_selection(), or — if the shipped detector is the "
        "subject — add the function to SHIPPED_DEFAULT_DETECTION:\n  "
        + "\n  ".join(violations)
    )


def test_every_shipped_default_entry_is_still_needed():
    """A stale allowlist entry hides the next real violation under its name."""
    stale = []
    for (filename, function), _why in SHIPPED_DEFAULT_DETECTION.items():
        path = TESTS / filename
        hits = {f for f, _ in _implicit_default_detection(path)} if path.exists() else set()
        if function not in hits:
            stale.append(f"{filename}::{function}")
    assert not stale, f"no longer detects through the default; remove from the allowlist: {stale}"


def test_the_tripwire_catches_each_form_it_claims_to():
    """The check is only worth its allowlist if it actually fires.

    It is function-granular: a bare ``ImagePipeline()`` anywhere in a function
    that also analyses is flagged, even if the bare one is only read for its
    version. Conservative on purpose — identity tests never analyse.
    """
    import tempfile

    source = (
        "def test_a():\n    pre.detect_faces(img)\n\n"
        "def test_b():\n    e = ImagePipeline()\n    e.analyze_image(s, d)\n\n"
        "def test_c():\n    run('ImagePipeline().analyze_image(s,d)')\n\n"
        "def test_ok_pinned():\n"
        "    pre.detect_faces(img, policy=p)\n"
        "    ImagePipeline(face_selection=p).analyze_image(s, d)\n\n"
        "def test_ok_identity_only():\n"
        "    ImagePipeline().model_version\n"
    )
    path = Path(tempfile.mkdtemp(prefix="dfp-tripwire-")) / "test_probe.py"
    path.write_text(source, encoding="utf-8")
    flagged = {f for f, _ in _implicit_default_detection(path)}
    assert flagged == {"test_a", "test_b", "test_c"}, flagged


def test_the_pinned_policy_is_haar():
    p = haar_selection()
    assert p.detector == HAAR_DETECTOR
    assert p.detector_asset == pre.cascade_digest()
    assert (p.scale_factor, p.min_neighbors, p.detect_min_px) == (1.1, 5, 32)
    assert p.confidence_threshold is None and p.nms_iou_threshold is None


def test_pinning_is_output_identical_while_the_shipped_default_is_haar():
    """The pinning moved nothing: same policy, same identity, same boxes.

    Vacuous once the shipped default stops being Haar — by then the two are
    meant to differ, and :func:`test_the_pinned_policy_is_haar` is what holds.
    """
    shipped = pre.default_face_selection()
    if shipped.detector != HAAR_DETECTOR:
        return
    pinned = haar_selection()
    assert pinned == shipped
    assert pinned.identity == shipped.identity
    assert pinned.differences(shipped) == []


def _face_tile() -> np.ndarray:
    img = np.full((480, 480, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (240, 240), (110, 140), 0, 0, 360, (180, 150, 130), -1)
    for cx in (200, 280):
        cv2.circle(img, (cx, 205), 16, (40, 30, 30), -1)
    cv2.ellipse(img, (240, 300), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_pinned_detection_needs_no_scrfd_configuration():
    """Haar-pinned detection must run with ``$DFP_SCRFD_MODEL_PATH`` unset —
    true of every machine without the licensed asset."""
    saved = os.environ.pop("DFP_SCRFD_MODEL_PATH", None)
    pre._scrfd_detector_for.cache_clear()
    try:
        boxes = pre.detect_faces(_face_tile(), policy=haar_selection())
        assert isinstance(boxes, list)
    finally:
        if saved is not None:
            os.environ["DFP_SCRFD_MODEL_PATH"] = saved
        pre._scrfd_detector_for.cache_clear()


def test_the_engines_report_a_version_without_a_detector_asset():
    """Computing the shipped identity must not depend on a file being present.

    ``workers.engine_registry.engine_versions`` constructs both pipelines to
    read their versions, and ``verify_version_alignment`` compares the result
    with ``STUB_ENGINE_VERSION`` at worker boot and across the suite. If the
    shipped policy's ``detector_asset`` were hashed from ``$DFP_SCRFD_MODEL_PATH``
    at construction, the published identity would differ between a machine
    with the asset and one without, and no single literal could match both.
    The SCRFD default must therefore *name* its asset digest as a constant and
    verify the file against it when detecting (``_scrfd_detector_for`` already
    does the second half).
    """
    from workers.engine_registry import engine_versions

    saved = os.environ.pop("DFP_SCRFD_MODEL_PATH", None)
    try:
        versions = engine_versions()
    finally:
        if saved is not None:
            os.environ["DFP_SCRFD_MODEL_PATH"] = saved
    assert versions["image"] == versions["video"]


def test_a_pinned_pipeline_analyses_a_face_without_scrfd_configuration():
    from contracts.enums import MediaKind
    from contracts.job import JobSpec
    from datetime import UTC, datetime
    from uuid import uuid4
    import hashlib

    from inference import ImagePipeline

    ok, buf = cv2.imencode(".png", cv2.cvtColor(_face_tile(), cv2.COLOR_RGB2BGR))
    assert ok
    data = buf.tobytes()
    spec = JobSpec(
        job_id=uuid4(),
        media_kind=MediaKind.IMAGE,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/png",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )
    saved = os.environ.pop("DFP_SCRFD_MODEL_PATH", None)
    try:
        bundle = asyncio.run(
            ImagePipeline(face_selection=haar_selection()).analyze_image(spec, data)
        )
    finally:
        if saved is not None:
            os.environ["DFP_SCRFD_MODEL_PATH"] = saved
    assert bundle.media.kind is MediaKind.IMAGE


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:1500]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
