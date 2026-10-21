"""SCRFD Integration Program, Phase 2 — the SCRFD detector, wired but not
defaulted. See `docs/SCRFD_PHASE2_DESIGN.md` and
`docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`.

WHAT THIS BATCH DOES
---------------------
Introduces `SCRFDDetector` (a second `FaceDetector` implementation,
`inference/preprocess.py`), two new `FaceSelectionPolicy` fields
(`confidence_threshold`/`nms_iou_threshold`, both `None` under Haar),
`scrfd_asset_digest()` (mirrors `cascade_digest()`), and `detect_faces`'s
dispatch on `policy.detector`.

WHAT THIS BATCH DOES NOT DO
----------------------------
`default_face_selection()` still returns a Haar policy — nothing in any
shipped pipeline ever selects SCRFD. No real SCRFD asset is fetched or
committed (`docs/SCRFD_ASSET_AUDIT.md`'s licensing gate); every test below
runs against a small, synthetic ONNX graph this suite builds itself,
matching SCRFD's real output *contract* (verified against the authors' own
`detection/scrfd/tools/scrfd.py`, `docs/SCRFD_ASSET_AUDIT.md` §1.5) without
containing any of their weights.

WHY A SYNTHETIC FIXTURE, NOT A REAL MODEL
-------------------------------------------
Real SCRFD weights are licensed non-commercial-research-only
(`docs/SCRFD_ASSET_AUDIT.md` §2.2) and are not vendored into this
repository (`.gitignore`'s own `*.onnx` entry, predating this program).
A synthetic graph built from `Constant`-plus-broadcast-`Add` nodes lets
every test below run **real `onnxruntime.InferenceSession` inference** —
exercising the actual session-construction, provider-pinning, and
threading code paths — while keeping every expected output a value this
suite chose and can hand-verify, independent of any licensed asset.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import threading as _threading
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from inference import preprocess as pre  # noqa: E402

# --------------------------------------------------------------------------
# Synthetic SCRFD-shaped ONNX fixtures
# --------------------------------------------------------------------------

_INPUT_HW = 32
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2


def _build_model_bytes(
    score_values: dict[int, np.ndarray], bbox_values: dict[int, np.ndarray]
) -> bytes:
    """A 6-output SCRFD-shaped graph (`score_8/16/32` + `bbox_8/16/32`,
    matching `tools/scrfd.py`'s `len(outputs)==6` case) whose outputs are the
    given constants, plus zero times the input's own `ReduceSum` — a valid
    ONNX graph that consumes its declared input while being fully
    deterministic, chosen values regardless of the pixels fed to it."""
    input_name = "input.1"
    graph_input = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [1, 3, _INPUT_HW, _INPUT_HW]
    )
    nodes = [helper.make_node("ReduceSum", [input_name], ["input_sum"], keepdims=0)]
    initializers = [numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_const")]
    nodes.append(helper.make_node("Mul", ["input_sum", "zero_const"], ["zero_scalar"]))

    score_outs, bbox_outs = [], []
    for stride in _STRIDES:
        sc = np.asarray(score_values[stride], dtype=np.float32)
        bb = np.asarray(bbox_values[stride], dtype=np.float32)
        initializers.append(numpy_helper.from_array(sc, f"score_{stride}_const"))
        initializers.append(numpy_helper.from_array(bb, f"bbox_{stride}_const"))
        nodes.append(
            helper.make_node("Add", [f"score_{stride}_const", "zero_scalar"], [f"score_{stride}"])
        )
        nodes.append(
            helper.make_node("Add", [f"bbox_{stride}_const", "zero_scalar"], [f"bbox_{stride}"])
        )
        score_outs.append(
            helper.make_tensor_value_info(f"score_{stride}", TensorProto.FLOAT, list(sc.shape))
        )
        bbox_outs.append(
            helper.make_tensor_value_info(f"bbox_{stride}", TensorProto.FLOAT, list(bb.shape))
        )

    graph = helper.make_graph(
        nodes,
        "scrfd_synthetic_fixture",
        [graph_input],
        score_outs + bbox_outs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _empty_stride_tensors() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    scores, bboxes = {}, {}
    for stride in _STRIDES:
        feat = _INPUT_HW // stride
        n = feat * feat * _NUM_ANCHORS
        scores[stride] = np.zeros(n, dtype=np.float32)
        bboxes[stride] = np.zeros((n, 4), dtype=np.float32)
    return scores, bboxes


def _write_fixture(path: str, detections: list[tuple[int, int, float, list[float]]]) -> None:
    """``detections``: ``(stride, flat_anchor_index, score, distance_raw[4])``.
    Everything else is score 0 (never passes any real threshold)."""
    scores, bboxes = _empty_stride_tensors()
    for stride, idx, score, dist in detections:
        scores[stride][idx] = score
        bboxes[stride][idx] = dist
    Path(path).write_bytes(_build_model_bytes(scores, bboxes))


def _build_input_dependent_model(path: str, *, scale: float, base_dist: float) -> None:
    """Stride-8, flat index 18 only: score fixed high; the *left/top/right/
    bottom* distance is ``base_dist + scale * ReduceSum(input)`` — a real,
    hand-computable function of the input's own pixel content, for the
    concurrency test (Test E). ``ReduceSum``/``Mul``/``Add`` are real ops
    executed by real `onnxruntime` for every call — not stubbed."""
    input_name = "input.1"
    graph_input = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [1, 3, _INPUT_HW, _INPUT_HW]
    )
    nodes = [helper.make_node("ReduceSum", [input_name], ["input_sum"], keepdims=0)]
    initializers = [numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale_const")]
    nodes.append(helper.make_node("Mul", ["input_sum", "scale_const"], ["scaled_sum"]))

    score_outs, bbox_outs = [], []
    for stride in _STRIDES:
        feat = _INPUT_HW // stride
        n = feat * feat * _NUM_ANCHORS
        score_base = np.zeros(n, dtype=np.float32)
        bbox_base = np.zeros((n, 4), dtype=np.float32)
        if stride == 8:
            score_base[18] = 0.95
            bbox_base[18] = [base_dist, base_dist, base_dist, base_dist]
        initializers.append(numpy_helper.from_array(score_base, f"score_{stride}_const"))
        initializers.append(numpy_helper.from_array(bbox_base, f"bbox_{stride}_const"))
        nodes.append(
            helper.make_node("Identity", [f"score_{stride}_const"], [f"score_{stride}"])
        )
        if stride == 8:
            nodes.append(
                helper.make_node("Add", [f"bbox_{stride}_const", "scaled_sum"], [f"bbox_{stride}"])
            )
        else:
            nodes.append(
                helper.make_node("Identity", [f"bbox_{stride}_const"], [f"bbox_{stride}"])
            )
        score_outs.append(
            helper.make_tensor_value_info(
                f"score_{stride}", TensorProto.FLOAT, list(score_base.shape)
            )
        )
        bbox_outs.append(
            helper.make_tensor_value_info(
                f"bbox_{stride}", TensorProto.FLOAT, list(bbox_base.shape)
            )
        )

    graph = helper.make_graph(
        nodes,
        "scrfd_input_dependent_fixture",
        [graph_input],
        score_outs + bbox_outs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    Path(path).write_bytes(model.SerializeToString())


def _tmp_path(name: str) -> str:
    return str(Path(tempfile.mkdtemp(prefix="dfp-scrfd-fixture-")) / name)


def _scrfd_policy(model_path: str, **overrides) -> pre.FaceSelectionPolicy:
    base = replace(
        pre.default_face_selection(),
        detector="scrfd-10g-kps",
        detector_asset=pre.scrfd_asset_digest(model_path),
        confidence_threshold=0.5,
        nms_iou_threshold=0.4,
    )
    return replace(base, **overrides) if overrides else base


def _random_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (_INPUT_HW, _INPUT_HW, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# A — the abstraction was actually implemented against, not just declared
# --------------------------------------------------------------------------


def test_scrfd_detector_satisfies_the_face_detector_protocol():
    detector = pre.SCRFDDetector(_tmp_path("unused.onnx"))
    assert isinstance(detector, pre.FaceDetector), (
        "SCRFDDetector no longer structurally satisfies FaceDetector — check "
        "the detect() method's name and arity"
    )


# --------------------------------------------------------------------------
# B — finds a detection on a controlled, hand-verified synthetic fixture
# --------------------------------------------------------------------------


def test_scrfd_detector_finds_a_face_on_the_shipped_fixture():
    """Flat index 18 (stride 8) is row 2, col 1 -> center (8, 16); distance
    [1,1,1,1] (raw) * stride 8 = 16 px each side -> box (0, 8, 16, 16). Hand
    -derived and cross-checked against a live run before being pinned here."""
    model_path = _tmp_path("single_detection.onnx")
    _write_fixture(model_path, [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0])])
    detector = pre.SCRFDDetector(model_path)
    policy = _scrfd_policy(model_path)
    boxes = detector.detect(_random_image(), policy=policy)
    assert boxes == [pre.FaceBox(0, 8, 16, 16)], boxes


def test_scrfd_detector_and_detect_faces_agree():
    """The dispatcher (detect_faces) must not alter what the detector itself
    returns, other than the total-order sort every detector already gets."""
    model_path = _tmp_path("single_detection.onnx")
    _write_fixture(model_path, [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0])])
    policy = _scrfd_policy(model_path)
    direct = pre.SCRFDDetector(model_path).detect(_random_image(), policy=policy)
    import os

    old = os.environ.get("DFP_SCRFD_MODEL_PATH")
    os.environ["DFP_SCRFD_MODEL_PATH"] = model_path
    try:
        pre._scrfd_detector_for.cache_clear()
        via_dispatch = pre.detect_faces(_random_image(), policy=policy)
    finally:
        if old is None:
            os.environ.pop("DFP_SCRFD_MODEL_PATH", None)
        else:
            os.environ["DFP_SCRFD_MODEL_PATH"] = old
        pre._scrfd_detector_for.cache_clear()
    assert direct == via_dispatch == [pre.FaceBox(0, 8, 16, 16)]


# --------------------------------------------------------------------------
# C — confidence_threshold is read from the policy, not hard-coded
# --------------------------------------------------------------------------


def test_scrfd_detector_respects_confidence_threshold():
    model_path = _tmp_path("two_detections.onnx")
    _write_fixture(
        model_path,
        [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0]), (8, 20, 0.6, [1.0, 1.0, 1.0, 1.0])],
    )
    detector = pre.SCRFDDetector(model_path)

    lenient = _scrfd_policy(model_path, confidence_threshold=0.5)
    boxes_lenient = detector.detect(_random_image(), policy=lenient)
    assert len(boxes_lenient) == 2, boxes_lenient

    strict = _scrfd_policy(model_path, confidence_threshold=0.7)
    boxes_strict = detector.detect(_random_image(), policy=strict)
    assert len(boxes_strict) == 1, boxes_strict


def test_scrfd_detector_raises_on_a_haar_shaped_policy():
    """confidence_threshold/nms_iou_threshold are None under Haar — asking
    SCRFDDetector to score with a Haar policy is a caller error, not a
    silently-picked default (docs/SCRFD_PHASE2_SPEC_RESOLUTION.md Decision 6)."""
    model_path = _tmp_path("single_detection.onnx")
    _write_fixture(model_path, [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0])])
    detector = pre.SCRFDDetector(model_path)
    haar_shaped = pre.default_face_selection()
    try:
        detector.detect(_random_image(), policy=haar_shaped)
    except ValueError as exc:
        assert "confidence_threshold" in str(exc)
        return
    raise AssertionError("SCRFDDetector accepted a policy with None thresholds")


# --------------------------------------------------------------------------
# D — free-aspect boxes: structurally unreachable under Haar, live under SCRFD
# --------------------------------------------------------------------------


def test_scrfd_boxes_are_free_aspect():
    model_path = _tmp_path("free_aspect.onnx")
    _write_fixture(model_path, [(8, 18, 0.95, [1.0, 2.0, 3.0, 4.0])])
    detector = pre.SCRFDDetector(model_path)
    policy = _scrfd_policy(model_path)
    boxes = detector.detect(_random_image(), policy=policy)
    assert len(boxes) == 1
    assert boxes[0].width != boxes[0].height, boxes[0]


# --------------------------------------------------------------------------
# E — one InferenceSession per thread; concurrent calls do not cross-contaminate
# --------------------------------------------------------------------------


def test_the_onnx_session_is_not_shared_unsafely():
    """R15's shape, one layer up: a shared, mutable native-runtime object
    under concurrent ``asyncio.to_thread`` calls silently returning another
    call's answer. Each thread feeds a distinctly-filled image; the box each
    thread sees must depend on *that thread's own* image, not another's."""
    scale, base_dist = 0.0001, 2.0
    model_path = _tmp_path("input_dependent.onnx")
    _build_input_dependent_model(model_path, scale=scale, base_dist=base_dist)
    detector = pre.SCRFDDetector(model_path)
    policy = _scrfd_policy(model_path)

    def expected(fill_value: int) -> pre.FaceBox:
        mean, std = pre.SCRFDDetector._INPUT_MEAN, pre.SCRFDDetector._INPUT_STD
        normalized = (float(fill_value) - mean) / std
        reduce_sum = 3 * _INPUT_HW * _INPUT_HW * normalized
        d = (base_dist + scale * reduce_sum) * 8
        cx, cy = 8.0, 16.0
        x1, y1, x2, y2 = cx - d, cy - d, cx + d, cy + d
        return pre.FaceBox(int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))

    fill_values = [32 * k for k in range(8)]
    results: dict[int, list[list[pre.FaceBox]]] = {}
    errors: list[tuple[int, Exception]] = []

    def worker(v: int) -> None:
        try:
            image = np.full((_INPUT_HW, _INPUT_HW, 3), fill_value=v, dtype=np.uint8)
            for _ in range(20):
                results.setdefault(v, []).append(detector.detect(image, policy=policy))
        except Exception as exc:  # noqa: BLE001
            errors.append((v, exc))

    threads = [_threading.Thread(target=worker, args=(v,)) for v in fill_values]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for v, runs in results.items():
        exp = [expected(v)]
        for run in runs:
            assert run == exp, (
                f"fill_value={v}: expected {exp}, got {run} — cross-thread "
                "contamination in SCRFDDetector's session handling"
            )


# --------------------------------------------------------------------------
# F — the dispatch cache never conflates two different assets
# --------------------------------------------------------------------------


def test_two_different_assets_are_never_conflated_by_the_dispatch_cache():
    import os

    path_a = _tmp_path("asset_a.onnx")
    path_b = _tmp_path("asset_b.onnx")
    _write_fixture(path_a, [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0])])
    _write_fixture(path_b, [(8, 18, 0.95, [2.0, 2.0, 2.0, 2.0])])
    digest_a = pre.scrfd_asset_digest(path_a)
    digest_b = pre.scrfd_asset_digest(path_b)
    assert digest_a != digest_b, "the two fixtures happened to hash identically; adjust them"

    old = os.environ.get("DFP_SCRFD_MODEL_PATH")
    pre._scrfd_detector_for.cache_clear()
    try:
        os.environ["DFP_SCRFD_MODEL_PATH"] = path_a
        detector_a = pre._scrfd_detector_for(digest_a)
        os.environ["DFP_SCRFD_MODEL_PATH"] = path_b
        detector_b = pre._scrfd_detector_for(digest_b)
        assert detector_a is not detector_b

        # stale env var pointing at path_b while asking for digest_a's
        # detector must raise, not silently hand back a wrong detector.
        try:
            pre._scrfd_detector_for.cache_clear()
            pre._scrfd_detector_for(digest_a)
        except pre.UnreadableMedia as exc:
            assert "digest" in str(exc)
        else:
            raise AssertionError("stale $DFP_SCRFD_MODEL_PATH was not detected")
    finally:
        if old is None:
            os.environ.pop("DFP_SCRFD_MODEL_PATH", None)
        else:
            os.environ["DFP_SCRFD_MODEL_PATH"] = old
        pre._scrfd_detector_for.cache_clear()


# --------------------------------------------------------------------------
# G — the asset digest is a pure content hash, pinned against a known file
# --------------------------------------------------------------------------


def test_scrfd_asset_digest_matches_a_known_file():
    import hashlib

    path = _tmp_path("digest_fixture.onnx")
    _write_fixture(path, [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0])])
    expected = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    assert pre.scrfd_asset_digest(path) == expected


def test_scrfd_asset_digest_falls_back_on_a_missing_file():
    assert pre.scrfd_asset_digest("/does/not/exist.onnx") == "unreadable"


# --------------------------------------------------------------------------
# H — default_face_selection() is still Haar — Phase 2 wires, does not default
# --------------------------------------------------------------------------


def test_default_face_selection_is_still_haar():
    assert pre.default_face_selection().detector == "opencv-haar-frontalface-default"


# --------------------------------------------------------------------------
# I — the execution provider is hardcoded, never selected from the environment
# --------------------------------------------------------------------------


def test_cpu_execution_provider_is_hardcoded_not_selected():
    src = inspect.getsource(pre.SCRFDDetector)
    assert "CPUExecutionProvider" in src, "SCRFDDetector no longer pins a provider explicitly"
    # Checks the call syntax specifically (not the bare name), so the class's
    # own docstring can keep honestly explaining, in prose, that this call
    # never happens without tripping over its own explanation.
    assert "get_available_providers()" not in src, (
        "SCRFDDetector calls onnxruntime.get_available_providers() — this reintroduces "
        "the default-provider-selection hazard docs/SCRFD_PHASE2_SPEC_RESOLUTION.md "
        "Decision 5 exists to prevent"
    )


# --------------------------------------------------------------------------
# J — the model path is never a FaceSelectionPolicy field
# --------------------------------------------------------------------------


def test_model_path_is_not_a_face_selection_policy_field():
    names = {f.name for f in fields(pre.FaceSelectionPolicy)}
    assert "model_path" not in names
    assert not any("path" in name for name in names), (
        f"FaceSelectionPolicy grew a path-shaped field {names}; "
        "docs/SCRFD_PHASE2_SPEC_RESOLUTION.md Decision 4 keeps the asset "
        "location out of the identity-bearing dataclass, mirroring "
        "ModelSpec.weights_uri's own exclusion from weights_identity"
    )


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
