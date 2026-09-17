"""R6 — decode determinism: what holds, what does not, and what is merely visible.

WHAT BATCH 8 MEASURED
---------------------
The same bytes, decoded by the two video backends this OpenCV build registers,
produce **no identical frames**: mean |delta| 2.2/255 on photographic content,
12.6 on noise, 85-96% of pixels changed. The difference is a near-identity
colour matrix with a per-channel bias around -1.2 plus a non-affine residual —
a levels and chroma-upsampling disagreement, not float noise.

Nothing recorded which decoder ran. ``effective_model_version`` composes weights,
calibration and the quality gate; the decoder is upstream of all three.

WHAT THESE TESTS ARE FOR
------------------------
Three jobs, and the third is the unusual one.

1. **Pin the guarantees that hold.** One build decodes repeatably; the backend
   is now chosen by name rather than by an environment variable nobody records;
   ``decode_environment`` distinguishes two decoders.

2. **Pin the amplifiers**, so nobody weakens them by accident. A one-frame error
   in ``frame_count`` moves 60% of the sampled indices; ``CAP_PROP_POS_MSEC``
   read after a seek is not a timestamp.

3. **Pin the negative results.** Two hypotheses in the audit's R6 did not
   reproduce: SIMD dispatch does not change ``resize``/``cvtColor``/``imdecode``
   output, and OpenCV and Pillow decode JPEG identically. Those are pinned so
   that if a future build *does* diverge, a test says so rather than the finding
   quietly becoming true again in the dark.

Full measurements and the report: ``python tests/decode_study.py`` and
``docs/DECODE_DETERMINISM.md``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from inference import preprocess as pre  # noqa: E402
from inference.decode_env import (  # noqa: E402
    PREFERRED_VIDEO_BACKENDS,
    VIDEO_BACKEND_ENV,
    DecodeEnvironment,
    decode_environment,
    decode_fingerprint,
    open_video,
    resolve_video_backend,
)
from inference.models import ModelSpec  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
TMP = Path(tempfile.mkdtemp(prefix="dfp-decode-test-"))


def _clip(name: str = "t.mp4", *, frames: int = 40, size: int = 192, fourcc: str = "avc1") -> Path:
    """Smooth, compressible content. Cached on disk across tests in one run."""
    path = TMP / name
    if path.exists():
        return path
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 25, (size, size))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(frames):
        r = 128 + 100 * np.sin((xx + i * 3) / 21.0)
        g = 128 + 100 * np.cos((yy - i * 2) / 17.0)
        b = 128 + 100 * np.sin((xx + yy + i * 4) / 29.0)
        writer.write(np.clip(np.dstack([b, g, r]), 0, 255).astype(np.uint8))
    writer.release()
    return path


def _backend_id(name: str) -> int | None:
    for backend in cv2.videoio_registry.getBackends():
        if cv2.videoio_registry.getBackendName(backend) == name:
            return int(backend)
    return None


def _second_backend() -> tuple[str, int] | None:
    """A registered backend that is not the preferred one and can open a clip.

    Returns ``None`` rather than failing when the build has only one usable
    video backend — on such a build the cross-decoder tests have no subject, and
    a test that cannot run must say so rather than pass silently.
    """
    clip = _clip()
    preferred = PREFERRED_VIDEO_BACKENDS[0]
    for backend in cv2.videoio_registry.getStreamBackends():
        name = cv2.videoio_registry.getBackendName(backend)
        if name == preferred:
            continue
        cap = cv2.VideoCapture(str(clip), int(backend))
        usable = cap.isOpened() and cap.grab()
        cap.release()
        if usable:
            return name, int(backend)
    return None


def _frames_via(clip: Path, api: int, limit: int = 24) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(clip), api)
    out: list[np.ndarray] = []
    while len(out) < limit and cap.grab():
        ok, frame = cap.retrieve()
        if not ok:
            break
        out.append(pre.to_rgb(frame))
    cap.release()
    return out


# --------------------------------------------------------------------------
# 1. Guarantees that hold on one build
# --------------------------------------------------------------------------


def test_decoding_is_repeatable_within_one_build():
    """The control. If this failed, no other measurement would mean anything."""
    data = _clip().read_bytes()
    seen = {
        (
            tuple(ex.indices),
            tuple(ex.timestamps_ms),
            hashlib.sha256(ex.frames.tobytes()).hexdigest(),
        )
        for ex in (pre.extract_frames(data, n_frames=12, target_size=96) for _ in range(4))
    }
    assert len(seen) == 1, f"extract_frames produced {len(seen)} distinct results"


def test_image_decode_and_transforms_are_repeatable():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    jpg = buf.tobytes()
    for fn in (
        lambda: pre.decode_image(jpg),
        lambda: pre.resize(img, 128),
        lambda: pre.to_rgb(img),
        lambda: pre.normalize(img),
    ):
        digests = {hashlib.sha256(np.ascontiguousarray(fn()).tobytes()).hexdigest() for _ in range(4)}
        assert len(digests) == 1


def test_normalisation_is_elementwise_and_therefore_order_free():
    """No reduction, so no summation order to depend on. Stated as a property."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    whole = pre.normalize(img)
    halves = np.concatenate([pre.normalize(img[:32]), pre.normalize(img[32:])])
    assert np.array_equal(whole, halves), "normalize is not elementwise"


# --------------------------------------------------------------------------
# 2. The backend is named, not inherited
# --------------------------------------------------------------------------


def test_the_video_backend_is_resolved_by_name():
    api, name, how = resolve_video_backend()
    assert how in ("env", "preferred", "fallback")
    assert isinstance(name, str) and name
    if how == "preferred":
        assert name == PREFERRED_VIDEO_BACKENDS[0]
        assert api == _backend_id(name)


def test_open_video_uses_the_resolved_backend():
    api, name, _ = resolve_video_backend()
    cap = open_video(_clip())
    try:
        assert cap.isOpened(), "the resolved backend cannot open a clip it wrote"
        if api != cv2.CAP_ANY:
            assert cap.getBackendName() == name, (
                f"asked for {name}, got {cap.getBackendName()}"
            )
    finally:
        cap.release()


def test_the_environment_variable_selects_the_backend_in_a_child_process():
    """The mechanism, exercised the only way it can be: a fresh interpreter.

    ``resolve_video_backend`` is cached, and the variable is read at first use,
    so this cannot be tested by mutating ``os.environ`` in place — which is
    precisely the property that makes the setting safe: it cannot change
    mid-process.
    """
    other = _second_backend()
    if other is None:
        print("        (only one usable video backend on this build; not exercised)")
        return
    name, _ = other
    script = (
        f"import sys; sys.path.insert(0, r'{ROOT}');"
        "from inference.decode_env import resolve_video_backend;"
        "print(resolve_video_backend()[1])"
    )
    env = dict(os.environ)
    env[VIDEO_BACKEND_ENV] = name
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.strip() == name, f"asked for {name}, got {proc.stdout.strip()!r}"


def test_an_unknown_backend_name_falls_back_and_says_so():
    script = (
        f"import sys; sys.path.insert(0, r'{ROOT}');"
        "from inference.decode_env import resolve_video_backend;"
        "print(resolve_video_backend()[2])"
    )
    env = dict(os.environ)
    env[VIDEO_BACKEND_ENV] = "NOT_A_REAL_BACKEND"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.strip() == "fallback", (
        "an unknown backend must be recorded as a fallback, not silently honoured"
    )


# --------------------------------------------------------------------------
# 3. The decode environment is captured and discriminating
# --------------------------------------------------------------------------


def test_the_decode_environment_records_what_was_measured_to_matter():
    env = decode_environment()
    assert env.opencv_version == cv2.__version__
    assert env.video_backend
    assert env.platform_system and env.platform_machine
    assert env.cpu_baseline and env.cpu_dispatch
    assert len(env.digest) == 8
    assert env.digest == decode_fingerprint()
    assert env.digest in env.describe()


def test_the_digest_changes_when_the_backend_changes():
    """The whole point of the digest, checked directly rather than through a run."""
    base = decode_environment()
    from dataclasses import replace

    other = replace(base, video_backend=base.video_backend + "-OTHER")
    assert other.digest != base.digest
    assert any("video_backend" in d for d in base.differences(other))


def test_provenance_only_fields_are_outside_the_digest():
    """``backend_source`` and ``python_version`` do not change a decoded pixel.

    They are reported by ``differences`` because an auditor wants them, and
    excluded from the digest because a digest that moves on a Python patch
    release would report "not comparable" for runs that are.
    """
    from dataclasses import replace

    base = decode_environment()
    for field, value in (("backend_source", "env"), ("python_version", "9.9.9")):
        other = replace(base, **{field: value})
        if getattr(base, field) == value:
            continue
        assert other.digest == base.digest, f"{field} must not be in the digest"
        assert any(field in d for d in base.differences(other)), (
            f"{field} must still be reported by differences()"
        )


def test_two_backends_decode_the_same_bytes_differently():
    """R6's central claim, measured rather than asserted.

    Skipped, loudly, on a build with one usable backend — the claim is about
    decoders disagreeing and needs two of them.
    """
    other = _second_backend()
    if other is None:
        print("        (only one usable video backend on this build; not exercised)")
        return
    name, api = other
    preferred_api = _backend_id(PREFERRED_VIDEO_BACKENDS[0])
    if preferred_api is None:
        print("        (preferred backend missing; not exercised)")
        return

    clip = _clip()
    a = _frames_via(clip, preferred_api)
    b = _frames_via(clip, api)
    n = min(len(a), len(b))
    assert n >= 8, f"only {n} frames decoded; fixture is too small to judge"

    identical = sum(1 for i in range(n) if np.array_equal(a[i], b[i]))
    deltas = [np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)) for i in range(n)]
    mean_delta = float(np.mean([d.mean() for d in deltas]))
    assert identical < n, (
        f"{PREFERRED_VIDEO_BACKENDS[0]} and {name} now agree on all {n} frames. "
        "If this is genuinely a fix rather than a fixture that stopped "
        "exercising the difference, R6 should be revisited."
    )
    assert mean_delta > 0.0
    print(f"        ({PREFERRED_VIDEO_BACKENDS[0]} vs {name}: {identical}/{n} identical, "
          f"mean|d|={mean_delta:.4f})")


# --------------------------------------------------------------------------
# 4. The amplifiers
# --------------------------------------------------------------------------


def test_a_one_frame_metadata_error_moves_most_of_the_sampled_indices():
    """Why container metadata is the highest-leverage decode input.

    ``sample_frame_indices`` is uniform over the declared count, so the whole
    grid shifts when the count moves by one. Measured: 29 of 48.
    """
    base = pre.sample_frame_indices(60, 48)
    off_by_one = pre.sample_frame_indices(59, 48)
    differ = sum(1 for a, b in zip(base, off_by_one) if a != b)
    assert differ >= 20, (
        f"only {differ}/48 indices moved; the amplifier this test exists to "
        "pin has gone away, which would be good news worth confirming"
    )
    assert len(base) == len(off_by_one) == 48


def test_the_position_after_a_seek_is_not_a_timestamp():
    """The hazard that justifies the one-pass design, reproduced as a double.

    ``CAP_PROP_POS_MSEC`` read after ``set(POS_FRAMES, i)`` and before ``read()``
    reports the position seeked *to* in the backend's own units, not the
    presentation time of the frame about to be decoded. ``extract_frames`` reads
    it after ``grab()``, which is why ``FrameScore.t_ms`` is correct.
    """
    clip = _clip()
    api = _backend_id(PREFERRED_VIDEO_BACKENDS[0]) or cv2.CAP_ANY
    target = 15

    cap = cv2.VideoCapture(str(clip), api)
    sequential = None
    i = 0
    while cap.grab():
        if i == target:
            sequential = cap.get(cv2.CAP_PROP_POS_MSEC)
            break
        i += 1
    cap.release()

    cap = cv2.VideoCapture(str(clip), api)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    after_seek = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.release()

    assert sequential is not None
    expected = target * 1000.0 / 25.0
    assert abs(sequential - expected) < 1.0, (
        f"the sequential path's timestamp ({sequential}) is not the frame's "
        f"presentation time ({expected})"
    )
    assert abs(after_seek - expected) > 1.0, (
        "the post-seek position now agrees with the presentation time; the "
        "hazard this test documents no longer exists on this build"
    )


def test_extract_frames_reads_the_timestamp_after_grab():
    """The property the previous test's hazard makes necessary, on real output."""
    data = _clip().read_bytes()
    ex = pre.extract_frames(data, n_frames=8, target_size=64)
    assert len(ex.timestamps_ms) == len(ex.indices) == ex.frames.shape[0]
    for index, stamp in zip(ex.indices, ex.timestamps_ms):
        expected = round(index * 1000.0 / ex.meta.fps)
        assert abs(stamp - expected) <= 1, (
            f"frame {index} reports t_ms={stamp}, expected about {expected}"
        )


# --------------------------------------------------------------------------
# 5. The negative results, pinned so they cannot quietly stop being true
# --------------------------------------------------------------------------


def test_cpu_dispatch_does_not_change_pixel_operations():
    """R6 hypothesised SIMD drift. It does not reproduce on this build.

    Pinned rather than dropped: if a future build *does* diverge here, this
    test is where it will be noticed.
    """
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    path = TMP / "dispatch.npy"
    np.save(path, img)
    script = f"""
import sys, hashlib
sys.path.insert(0, r"{ROOT}")
import cv2, numpy as np
from inference import preprocess as pre
img = np.load(r"{path}")
ok, j = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(hashlib.sha256(pre.resize(img, 128).tobytes()).hexdigest()[:16],
      hashlib.sha256(pre.to_rgb(img).tobytes()).hexdigest()[:16],
      hashlib.sha256(cv2.imdecode(j, cv2.IMREAD_COLOR).tobytes()).hexdigest()[:16])
"""
    outs = []
    for extra in ({}, {"OPENCV_CPU_DISABLE": "AVX2,AVX,FP16,SSE4_2,SSE4_1"}):
        env = dict(os.environ)
        env.update(extra)
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, proc.stderr[-400:]
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1], (
        "resize / to_rgb / imdecode now differ across CPU dispatch paths. This "
        "was measured stable in Batch 8; a divergence here is a new finding and "
        "the decode fingerprint's cpu_dispatch field becomes load-bearing."
    )


def test_in_process_optimisation_toggles_do_not_change_pixels():
    rng = np.random.default_rng(6)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    results = {}
    try:
        for optimised in (True, False):
            for threads in (8, 1):
                cv2.setUseOptimized(optimised)
                cv2.setNumThreads(threads)
                results[(optimised, threads)] = (
                    hashlib.sha256(pre.resize(img, 128).tobytes()).hexdigest(),
                    hashlib.sha256(pre.to_rgb(img).tobytes()).hexdigest(),
                )
    finally:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)
    assert len(set(results.values())) == 1, f"dispatch toggles changed output: {results}"


def test_opencv_and_pillow_decode_jpeg_identically():
    """The other half of R6, and it does not reproduce either.

    Two independently built libjpeg-turbo linkages, same IDCT and upsampling
    defaults, bit-identical output — including progressive JPEG, which is where
    decoders most often differ.
    """
    try:
        from PIL import Image
    except ImportError:
        print("        (Pillow not installed; not exercised)")
        return
    import io

    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for params in (
        [cv2.IMWRITE_JPEG_QUALITY, 88],
        [cv2.IMWRITE_JPEG_QUALITY, 40],
        [cv2.IMWRITE_JPEG_QUALITY, 92, cv2.IMWRITE_JPEG_PROGRESSIVE, 1],
    ):
        ok, buf = cv2.imencode(".jpg", img, params)
        assert ok
        data = buf.tobytes()
        a = pre.decode_image(data)
        b = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        assert np.array_equal(a, b), (
            "OpenCV and Pillow now disagree on a JPEG. R6's image half has "
            "become real and the image path needs the same treatment as video."
        )


# --------------------------------------------------------------------------
# 6. What the version string does and does not cover
# --------------------------------------------------------------------------


def test_the_effective_model_version_does_not_cover_the_decoder():
    """The model identity excludes the decoder — **deliberately, since Batch 9**.

    When this test was written in Batch 8 it recorded a gap: nothing anywhere
    named the decoder, so a cached verdict could cross decode environments.
    Batch 9 closed that with a *separate* runtime identity
    (``contracts.version.compose_runtime_version``, and
    ``tests/test_run_identity.py``), and left this string alone on purpose.

    So the assertion is unchanged and its meaning is inverted. It no longer
    documents an omission; it protects one. If the decoder ever leaks into
    ``effective_version``, "did the model get worse" starts answering yes for a
    worker rebuild, and every benchmark comparison in the platform inherits the
    confusion.
    """
    spec = ModelSpec(name="stub", version="stub-0")
    version = spec.effective_version
    env = decode_environment()
    assert env.digest not in version
    assert env.video_backend not in version
    assert env.opencv_version not in version
    # And the parts it *does* cover, so the test says what the version means.
    assert spec.calibration_fingerprint in version


def test_the_decode_fingerprint_is_stable_within_a_process():
    assert decode_fingerprint() == decode_fingerprint()
    assert decode_environment().digest == decode_fingerprint()


DERIVED_FIELDS = ("identity", "digest")
"""Keys ``to_dict`` adds that are computed, not stored.

Both are properties, so neither is a constructor argument. Listing them here
rather than filtering on "not a dataclass field" is deliberate: adding a third
derived key should break this line and make someone confirm it belongs in the
serialised form, rather than being swept up by a rule.
"""


def test_a_decode_environment_round_trips_through_its_dict():
    env = decode_environment()
    payload = env.to_dict()
    assert payload["digest"] == env.digest
    assert payload["identity"] == env.identity
    rebuilt = DecodeEnvironment(
        **{k: v for k, v in payload.items() if k not in DERIVED_FIELDS}
    )
    assert rebuilt.digest == env.digest
    assert rebuilt.identity == env.identity
    assert rebuilt.differences(env) == []


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
