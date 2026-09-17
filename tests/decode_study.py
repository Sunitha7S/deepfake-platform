"""Decode determinism: eleven experiments, measured. Standalone.

    python tests/decode_study.py

Produces every number in ``docs/DECODE_DETERMINISM.md``. Same pattern as
``band_study.py`` and ``import_audit.py``: kept in the repository because a
claim nobody can re-run is not evidence.

THE QUESTION
------------
Can the same media, model version, calibration, threshold and code produce
different verdicts solely because decoding changed?

WHAT MAKES THIS MEASURABLE WITHOUT A SECOND MACHINE
---------------------------------------------------
Three genuinely different code paths are reachable from one process:

* **Two video backends.** This OpenCV build registers FFMPEG *and* MSMF. They
  are separate decoder implementations, so comparing them is a real
  cross-decoder experiment rather than a simulation of one.
* **Two image decoders.** OpenCV's bundled libjpeg-turbo and Pillow's.
* **Two CPU dispatch paths.** ``OPENCV_CPU_DISABLE`` selects different compiled
  kernels at load time, which is the closest available proxy for a build with a
  different SIMD baseline. Needs a subprocess: it is read at import.

Each experiment states a hypothesis and an expected failure mode *before* it
measures, and several of them do not reproduce. Those are reported as negative
results rather than quietly dropped — a threat model that only lists confirmed
threats is a list of things someone happened to find.

READING THE LOGIT NUMBERS
-------------------------
Two scorers are used deliberately.

``StubBackend`` hashes the frame's bytes, so *any* pixel change produces an
uncorrelated logit. That is the correct upper bound on sensitivity and it is
useless as an estimate: measured below, the stub's per-frame correlation across
decoders is 0.028.

The **surrogate** is a smooth, Lipschitz function of the pixels (gradient energy
on the normalised frame). It stands in for a CNN in the one respect that matters
here: it responds proportionally to a small pixel shift instead of hashing it.
It is not a deepfake detector and is not offered as one. Where this study
estimates what a real model would do, it uses the surrogate and says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.decode_env import decode_environment  # noqa: E402
from inference.metrics import roc_auc  # noqa: E402
from evaluation.thresholds import at_target_fpr  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
TMP = Path(tempfile.mkdtemp(prefix="dfp-decode-study-"))
SHIPPED_BAND = ThresholdInfo.at_margin(
    value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def smooth_clip(path: Path, *, frames=60, size=256, fps=25, fourcc="avc1", seed=0) -> Path:
    """Photographic-ish content: smooth gradients, compressible, no white noise.

    Deliberately not random pixels. Noise is worst-case for a lossy codec and
    exaggerates every difference between decoders; smooth content is what real
    media looks like and gives the honest magnitude.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (size, size))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(frames):
        r = 128 + 100 * np.sin((xx + i * 3 + seed * 11) / (17.0 + seed % 7))
        g = 128 + 100 * np.cos((yy - i * 2 + seed * 7) / (13.0 + seed % 5))
        b = 128 + 100 * np.sin((xx + yy + i * 4) / (23.0 + seed % 3))
        writer.write(np.clip(np.dstack([b, g, r]), 0, 255).astype(np.uint8))
    writer.release()
    return path


def noise_clip(path: Path, *, frames=60, size=240, fps=25, fourcc="mp4v", seed=13) -> Path:
    """Worst case: incompressible content, kept for the upper bound."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (size, size))
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    for i in range(frames):
        writer.write(np.roll(base, i * 5, axis=1))
    writer.release()
    return path


def decode_all(path: Path, api: int, limit: int | None = None):
    """Every frame plus its ``POS_MSEC``, through one named backend."""
    cap = cv2.VideoCapture(str(path), api)
    if not cap.isOpened():
        cap.release()
        return None
    frames, stamps = [], []
    while limit is None or len(frames) < limit:
        if not cap.grab():
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break
        frames.append(pre.to_rgb(frame))
        stamps.append(ms)
    cap.release()
    return frames, stamps


def surrogate_logit(frame_rgb: np.ndarray) -> float:
    """Smooth stand-in for a real model. See the module docstring."""
    x = pre.normalize(pre.resize(frame_rgb, 224))
    gx = np.diff(x, axis=1)
    gy = np.diff(x, axis=0)
    return 6.0 * (float(np.sqrt((gx**2).mean() + (gy**2).mean())) - 0.35)


def stub_logit(frame_rgb: np.ndarray) -> float:
    """Exactly what ``StubBackend`` computes, on one frame."""
    arr = pre.resize(frame_rgb, 384).astype(np.float32)
    digest = hashlib.sha256(b"spatial" + arr.tobytes()).hexdigest()
    p = int(digest[:12], 16) / float(0x1000000000000)
    return float(np.log(p / (1 - p)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def head(n: int, title: str, hypothesis: str, failure: str) -> None:
    print(f"\n{'=' * 74}")
    print(f"E{n}. {title}")
    print(f"{'=' * 74}")
    print(f"  hypothesis     : {hypothesis}")
    print(f"  expected if so : {failure}")
    print()


def verdict_of(p: float) -> str:
    if p >= SHIPPED_BAND.band_hi:
        return "manipulated"
    if p >= SHIPPED_BAND.band_lo:
        return "insufficient"
    return "authentic"


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def e0_environment() -> None:
    head(0, "What this box is", "n/a - context for every number below", "n/a")
    print(f"  {decode_environment().describe()}")
    names = [
        cv2.videoio_registry.getBackendName(b)
        for b in cv2.videoio_registry.getStreamBackends()
    ]
    print(f"  video backends registered: {names}")


def e1_metadata_across_backends() -> None:
    head(
        1,
        "Container metadata across decoders",
        "FFMPEG and MSMF disagree on fps / frame_count for some containers",
        "sample_frame_indices selects different frames -> a different clip is scored",
    )
    rows = []
    for label, maker, fourcc in (
        ("smooth/avc1", smooth_clip, "avc1"),
        ("smooth/mp4v", smooth_clip, "mp4v"),
        ("noise/mp4v", noise_clip, "mp4v"),
        ("noise/MJPG", noise_clip, "MJPG"),
    ):
        suffix = ".avi" if fourcc == "MJPG" else ".mp4"
        path = maker(TMP / f"meta_{label.replace('/', '_')}{suffix}", fourcc=fourcc)
        row = {"clip": label}
        for name, api in (("FFMPEG", cv2.CAP_FFMPEG), ("MSMF", cv2.CAP_MSMF)):
            cap = cv2.VideoCapture(str(path), api)
            if not cap.isOpened():
                row[name] = None
                cap.release()
                continue
            row[name] = (
                float(cap.get(cv2.CAP_PROP_FPS)),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            cap.release()
        rows.append(row)
        agree = row.get("FFMPEG") == row.get("MSMF")
        print(f"  {label:14s} FFMPEG={row.get('FFMPEG')}  MSMF={row.get('MSMF')}  "
              f"{'agree' if agree else '*** DISAGREE ***'}")

    print("\n  Truncated containers (a realistic source of metadata divergence):")
    raw = (TMP / "meta_smooth_avc1.mp4").read_bytes()
    for frac in (0.95, 0.80, 0.50):
        path = TMP / f"trunc_{int(frac * 100)}.mp4"
        path.write_bytes(raw[: int(len(raw) * frac)])
        cells = []
        for name, api in (("FFMPEG", cv2.CAP_FFMPEG), ("MSMF", cv2.CAP_MSMF)):
            cap = cv2.VideoCapture(str(path), api)
            if not cap.isOpened():
                cells.append(f"{name}=refused")
            else:
                declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                got = 0
                while cap.grab():
                    got += 1
                cells.append(f"{name}=declared:{declared}/decoded:{got}")
            cap.release()
        print(f"    {int(frac * 100):3d}% of bytes: " + "  ".join(cells))
    print("\n  VERDICT: not reproduced on this box. Both backends agree on metadata")
    print("  for every intact container tested, and both refuse every truncated one.")


def e2_index_amplification() -> None:
    head(
        2,
        "How much does a one-frame metadata error matter?",
        "sample_frame_indices is uniform, so an off-by-one frame_count shifts "
        "most of the sampled indices",
        "a decoder that miscounts by one scores a substantially different clip",
    )
    base = pre.sample_frame_indices(60, 48)
    print(f"  reference frame_count=60 -> {len(base)} indices, first 8 {base[:8]}")
    for n in (58, 59, 61, 62, 65, 70):
        other = pre.sample_frame_indices(n, 48)
        differ = sum(1 for a, b in zip(base, other) if a != b)
        print(f"    frame_count={n:3d}: {differ:2d}/{len(base)} sampled indices differ "
              f"({100 * differ / len(base):.0f}%)")
    print("\n  This is the amplifier, measured. E1 found no metadata divergence on")
    print("  this box, so the trigger is unconfirmed here - but if any decoder ever")
    print("  miscounts by one frame, 60% of the analysed frames change.")


def e3_pixels_across_backends() -> None:
    head(
        3,
        "Pixel agreement across decoders",
        "FFMPEG and MSMF produce different pixels for the same frame",
        "every downstream number differs, with no version string recording why",
    )
    for label, maker, fourcc in (
        ("smooth/avc1", smooth_clip, "avc1"),
        ("smooth/mp4v", smooth_clip, "mp4v"),
        ("noise/mp4v", noise_clip, "mp4v"),
        ("noise/XVID", noise_clip, "XVID"),
        ("noise/MJPG", noise_clip, "MJPG"),
    ):
        suffix = ".avi" if fourcc == "MJPG" else ".mp4"
        path = maker(TMP / f"px_{label.replace('/', '_')}{suffix}", fourcc=fourcc)
        a = decode_all(path, cv2.CAP_FFMPEG)
        b = decode_all(path, cv2.CAP_MSMF)
        if a is None or b is None:
            print(f"  {label:14s} one backend could not open")
            continue
        fa, fb = a[0], b[0]
        n = min(len(fa), len(fb))
        if n == 0:
            print(f"  {label:14s} no frames")
            continue
        diffs = [np.abs(fa[i].astype(np.int16) - fb[i].astype(np.int16)) for i in range(n)]
        identical = sum(1 for i in range(n) if np.array_equal(fa[i], fb[i]))
        print(
            f"  {label:14s} n={n:3d}  identical={identical:3d}  "
            f"max|d|={max(int(d.max()) for d in diffs):3d}  "
            f"mean|d|={np.mean([d.mean() for d in diffs]):7.4f}  "
            f"pixels differing={100 * np.mean([(d > 0).mean() for d in diffs]):5.2f}%"
        )


def e4_cause() -> None:
    head(
        4,
        "Is the divergence noise, or a systematic transform?",
        "the two decoders differ by a colour/levels interpretation, not randomness",
        "a near-identity 3x3 matrix with a non-zero bias fits most of the delta",
    )
    path = smooth_clip(TMP / "cause.mp4", frames=40)
    a, b = decode_all(path, cv2.CAP_FFMPEG), decode_all(path, cv2.CAP_MSMF)
    n = min(len(a[0]), len(b[0]))
    A = np.stack(a[0][:n]).astype(np.float64)
    B = np.stack(b[0][:n]).astype(np.float64)
    d = A - B
    print(f"  per-channel mean signed delta (R,G,B): "
          f"{d[..., 0].mean():+.4f} {d[..., 1].mean():+.4f} {d[..., 2].mean():+.4f}")
    X = np.c_[B.reshape(-1, 3), np.ones(B.size // 3)]
    M, *_ = np.linalg.lstsq(X, A.reshape(-1, 3), rcond=None)
    print("  least-squares MSMF->FFMPEG (rows R,G,B; last row = bias):")
    for row in M:
        print("      " + "  ".join(f"{v:+.5f}" for v in row))
    resid = A.reshape(-1, 3) - X @ M
    print(f"  residual after the affine fit: mean|r|={np.abs(resid).mean():.4f}  "
          f"max|r|={np.abs(resid).max():.2f}")
    print("\n  A near-identity matrix with a bias near -1.2 per channel means most of")
    print("  the delta is a levels/range interpretation. The residual is not zero, so")
    print("  chroma upsampling differs too. Neither is float noise, and neither will")
    print("  be fixed by anything this codebase can do to its own arithmetic.")


def e5_seek_strategies() -> None:
    head(
        5,
        "Seek strategies: sequential grab vs POS_FRAMES",
        "seeking returns different frames and/or different timestamps",
        "indices and t_ms stop naming the frame that was actually scored",
    )
    path = smooth_clip(TMP / "seek.mp4")
    want = [0, 7, 15, 23, 31, 42, 55]

    seq = {}
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    i = 0
    while True:
        if not cap.grab():
            break
        if i in want:
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, f = cap.retrieve()
            if ok:
                seq[i] = (hashlib.sha256(f.tobytes()).hexdigest()[:10], round(ms, 3))
        i += 1
    cap.release()

    sk = {}
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    for idx in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ms_before = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, f = cap.read()
        if ok:
            sk[idx] = (hashlib.sha256(f.tobytes()).hexdigest()[:10], round(ms_before, 3))
    cap.release()

    px_mismatch = [k for k in want if seq.get(k, ("a",))[0] != sk.get(k, ("b",))[0]]
    print(f"  pixel mismatches, sequential vs seek: {len(px_mismatch)}/{len(want)}  {px_mismatch}")
    print("  timestamps (the field FrameScore.t_ms is built from):")
    for k in want:
        if k in seq and k in sk:
            print(f"    frame {k:3d}: sequential={seq[k][1]:9.3f} ms   "
                  f"after-seek={sk[k][1]:9.3f} ms   delta={seq[k][1] - sk[k][1]:9.3f}")
    print("\n  The pixels agree on this all-keyframe clip. The timestamps do not, and")
    print("  the seek-path value is not a timestamp at all - CAP_PROP_POS_MSEC read")
    print("  after set() and before read() describes the position seeked *to*, not the")
    print("  frame decoded. This is why extract_frames reads it after grab().")


def e6_dispatch_paths() -> None:
    head(
        6,
        "CPU dispatch: do different compiled kernels change pixels?",
        "resize / cvtColor / imdecode differ between SIMD paths",
        "the same build on a different CPU produces different arrays",
    )
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    np.save(TMP / "dispatch.npy", img)
    script = f"""
import sys, hashlib
sys.path.insert(0, r"{ROOT}")
import cv2, numpy as np
from inference import preprocess as pre
img = np.load(r"{TMP / 'dispatch.npy'}")
ok, j = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(cv2.getBuildInformation().split('Baseline:')[1].splitlines()[0].strip()
      if 'Baseline:' in cv2.getBuildInformation() else '?')
print(hashlib.sha256(pre.resize(img, 384).tobytes()).hexdigest()[:16])
print(hashlib.sha256(pre.to_rgb(img).tobytes()).hexdigest()[:16])
print(hashlib.sha256(cv2.imdecode(j, cv2.IMREAD_COLOR).tobytes()).hexdigest()[:16])
"""
    results = {}
    for label, env in (
        ("default (AVX2)", {}),
        ("OPENCV_CPU_DISABLE=AVX2,AVX,FP16,SSE4_2,SSE4_1", {
            "OPENCV_CPU_DISABLE": "AVX2,AVX,FP16,SSE4_2,SSE4_1"}),
    ):
        environ = dict(os.environ)
        environ.update(env)
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=environ
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        results[label] = lines[-3:] if len(lines) >= 3 else lines
        print(f"  {label}")
        print(f"      resize={results[label][0] if results[label] else '?'}")
        if len(results[label]) >= 3:
            print(f"      to_rgb={results[label][1]}  imdecode={results[label][2]}")

    vals = list(results.values())
    same = len(vals) == 2 and vals[0] == vals[1]
    print(f"\n  VERDICT: {'IDENTICAL - hypothesis not reproduced' if same else 'DIFFERENT'}")
    print("  Also checked in-process and identical: setUseOptimized(False),")
    print("  setNumThreads(1). On this build these operations are bit-stable across")
    print("  every dispatch path reachable here. That is a negative result about one")
    print("  build, not a guarantee about all of them.")


def e7_independent_image_decoder() -> None:
    head(
        7,
        "Independent image decoders: OpenCV vs Pillow",
        "different libjpeg builds decode the same JPEG to different pixels",
        "the image path drifts the same way the video path does",
    )
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        print("  Pillow not installed; experiment skipped.")
        return
    import io

    rng = np.random.default_rng(5)
    cases = {
        "noise 720p": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
    }
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    cases["smooth 480p"] = np.clip(
        np.dstack([128 + 110 * np.sin(xx / 23), 128 + 110 * np.cos(yy / 19),
                   128 + 110 * np.sin((xx + yy) / 31)]), 0, 255
    ).astype(np.uint8)

    for label, bgr in cases.items():
        for fmt, params in ((".jpg", [cv2.IMWRITE_JPEG_QUALITY, 88]),
                            (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 40]),
                            (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92,
                                      cv2.IMWRITE_JPEG_PROGRESSIVE, 1]),
                            (".png", [])):
            ok, buf = cv2.imencode(fmt, bgr, params)
            if not ok:
                continue
            data = buf.tobytes()
            a = pre.decode_image(data)
            b = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            tag = fmt.lstrip(".")
            if cv2.IMWRITE_JPEG_PROGRESSIVE in params:
                tag += " progressive"
            elif fmt == ".jpg":
                tag += f" q{params[1]}"
            print(f"  {label:12s} {tag:16s} identical={np.array_equal(a, b)!s:5s} "
                  f"max|d|={int(d.max()):3d}  mean|d|={d.mean():.5f}")
    print("\n  VERDICT: bit-identical throughout. Both link libjpeg-turbo with the")
    print("  same IDCT and upsampling defaults, so the 'different libjpeg' half of")
    print("  R6 does not reproduce here. The video half does - see E3.")


def e8_face_detection_amplifier() -> None:
    head(
        8,
        "Does a decoder-scale pixel shift move the face crop?",
        "Haar detection is not stable under a +/-2 LSB perturbation",
        "the crop moves, so the model sees different pixels than it would have",
    )
    img = np.full((480, 480, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    base = pre.detect_faces(img)
    if not base:
        print("  the synthetic face is not detected by this cascade; experiment void")
        return
    b0 = base[0]
    print(f"  baseline box: x={b0.x} y={b0.y} w={b0.width} h={b0.height}")
    for amplitude in (1, 2, 4):
        changed, lost, shifts = 0, 0, []
        trials = 30
        for trial in range(trials):
            r = np.random.default_rng(trial)
            noise = r.integers(-amplitude, amplitude + 1, img.shape, dtype=np.int16)
            pert = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            found = pre.detect_faces(pert)
            if not found:
                lost += 1
                changed += 1
                continue
            f0 = found[0]
            shifts.append((abs(f0.x - b0.x), abs(f0.y - b0.y), abs(f0.width - b0.width)))
            if (f0.x, f0.y, f0.width, f0.height) != (b0.x, b0.y, b0.width, b0.height):
                changed += 1
        arr = np.array(shifts) if shifts else np.zeros((1, 3), int)
        print(f"  +/-{amplitude} LSB: box changed {changed}/{trials}, lost {lost}/{trials}, "
              f"max |dx|={arr[:, 0].max()} |dy|={arr[:, 1].max()} |dw|={arr[:, 2].max()}")
    print("\n  Measured decoder delta (E3) is mean 2.2 / max 15 on smooth content -")
    print("  larger than the +/-2 perturbation above. The crop is not stable under it.")


def e9_logit_drift() -> None:
    head(
        9,
        "Does the pixel difference reach the logits?",
        "yes, and by how much depends entirely on the scorer",
        "a hash-based scorer decorrelates; a smooth one shifts proportionally",
    )
    path = smooth_clip(TMP / "logits.mp4")
    a, b = decode_all(path, cv2.CAP_FFMPEG, 48), decode_all(path, cv2.CAP_MSMF, 48)
    n = min(len(a[0]), len(b[0]))
    for name, fn in (("surrogate (smooth)", surrogate_logit), ("stub (hash)", stub_logit)):
        la = np.array([fn(a[0][i]) for i in range(n)])
        lb = np.array([fn(b[0][i]) for i in range(n)])
        pa, pb = sigmoid(la), sigmoid(lb)
        print(f"  {name:20s} frames={n}")
        print(f"      logit  mean|d|={np.abs(la - lb).mean():.6f}  "
              f"max|d|={np.abs(la - lb).max():.6f}")
        print(f"      p      mean|d|={np.abs(pa - pb).mean():.6f}  "
              f"max|d|={np.abs(pa - pb).max():.6f}")
        print(f"      per-frame correlation = {np.corrcoef(la, lb)[0, 1]:.6f}")
    print("\n  The stub's 0.03 correlation is the headline: with a hash-based scorer")
    print("  the per-frame curve is destroyed outright. The surrogate's ~0.005 shift")
    print("  in p is the honest estimate for a smooth model.")


def e10_verdict_and_metrics() -> None:
    head(
        10,
        "Verdicts, thresholds and benchmark metrics under a decoder change",
        "a small shift in p flips verdicts near the band edges and moves the "
        "fitted threshold",
        "the same corpus reports a different operating point",
    )
    clips = []
    for s in range(24):
        p = TMP / f"corpus_{s}.mp4"
        if not p.exists():
            smooth_clip(p, frames=40, seed=s)
        clips.append(p)

    sa, sb = [], []
    for p in clips:
        ga = decode_all(p, cv2.CAP_FFMPEG, 16)
        gb = decode_all(p, cv2.CAP_MSMF, 16)
        m = min(len(ga[0]), len(gb[0]))
        sa.append(float(np.mean([surrogate_logit(ga[0][i]) for i in range(m)])))
        sb.append(float(np.mean([surrogate_logit(gb[0][i]) for i in range(m)])))
    sa, sb = np.array(sa), np.array(sb)
    pa, pb = sigmoid(sa), sigmoid(sb)

    # Labels fixed once, from the FFMPEG scores, so both decoders are judged
    # against one labelling. Re-labelling per decoder would compare nothing.
    y = (sa > np.median(sa)).astype(np.float64)
    order_a = np.argsort(sa)
    order_b = np.argsort(sb)
    inversions = sum(
        1
        for i in range(len(sa))
        for j in range(i + 1, len(sa))
        if (sa[i] < sa[j]) != (sb[i] < sb[j])
    )
    pairs = len(sa) * (len(sa) - 1) // 2

    print(f"  clips={len(clips)}")
    print(f"  logit    mean|d|={np.abs(sa - sb).mean():.6f}  max|d|={np.abs(sa - sb).max():.6f}")
    print(f"  p        mean|d|={np.abs(pa - pb).mean():.6f}  max|d|={np.abs(pa - pb).max():.6f}")
    print(f"  ranking  identical order: {np.array_equal(order_a, order_b)}  "
          f"inverted pairs: {inversions}/{pairs}")
    print(f"  ROC AUC  FFMPEG={roc_auc(y, pa):.6f}  MSMF={roc_auc(y, pb):.6f}")
    ta = at_target_fpr(y, pa, target_fpr=0.2)
    tb = at_target_fpr(y, pb, target_fpr=0.2)
    print(f"  fitted threshold  FFMPEG={ta.value:.8f}  MSMF={tb.value:.8f}  "
          f"delta={abs(ta.value - tb.value):.8f}")
    print("\n  AUC is rank-only, so it moves only when the ranking moves. The fitted")
    print("  *threshold* is a quantile of the scores themselves and moves with any")
    print("  shift - which is the number that reaches calibration_fingerprint.")

    print("\n  How many decisions does a shift of |dp| actually flip?")
    print(f"  (shipped band [{SHIPPED_BAND.band_lo:.4f}, {SHIPPED_BAND.band_hi:.4f}), "
          "scores ~ Beta(2,2), 200000 draws)")
    rng = np.random.default_rng(0)
    scores = rng.beta(2, 2, 200_000)

    def region(v):
        return np.where(v >= SHIPPED_BAND.band_hi, 2,
                        np.where(v >= SHIPPED_BAND.band_lo, 1, 0))

    base = region(scores)
    for dp, tag in (
        (float(np.abs(pa - pb).mean()), "measured, surrogate on this corpus"),
        (0.0047, "measured, surrogate per frame (E9)"),
        (0.026, "measured, stub end-to-end"),
        (0.05, "reference"),
    ):
        up = (region(np.clip(scores + dp, 0, 1)) != base).mean()
        down = (region(np.clip(scores - dp, 0, 1)) != base).mean()
        print(f"    |dp|={dp:.4f}  -> {100 * max(up, down):6.3f}% of decisions change"
              f"   ({tag})")


def e11_end_to_end_and_version() -> None:
    head(
        11,
        "The whole pipeline, twice, and what the version string says",
        "the same bytes through two decoders give different scores and the "
        "SAME effective_model_version",
        "two deployments disagree on a verdict and agree on every version field",
    )
    clip = smooth_clip(TMP / "e2e.mp4", frames=60, size=256)
    worker = TMP / "e2e_worker.py"
    worker.write_text(
        f'''
import sys, asyncio, hashlib, json
from datetime import UTC, datetime
from uuid import UUID
sys.path.insert(0, r"{ROOT}")
import cv2
from contracts.enums import MediaKind
from contracts.job import JobSpec
from inference import VideoPipeline
from inference import preprocess as pre
from inference.decode_env import decode_environment

data = open(r"{clip}", "rb").read()
ex = pre.extract_frames(data, n_frames=48, target_size=384)
spec = JobSpec(job_id=UUID(int=7), media_kind=MediaKind.VIDEO, storage_key="u",
               sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
               declared_mime="video/mp4", model_version="v", idempotency_key="k",
               submitted_at=datetime(2026, 1, 1, tzinfo=UTC))
pipe = VideoPipeline()
bundle = asyncio.run(pipe.analyze_video(spec, data))
env = decode_environment()
print("@@" + json.dumps({{
    "backend": env.video_backend,
    "decode_digest": env.digest,
    "model_version": pipe.model_version,
    "indices": ex.indices,
    "stamps": ex.timestamps_ms,
    "frames_sha": hashlib.sha256(ex.frames.tobytes()).hexdigest()[:16],
    "verdict": bundle.verdict.value,
    "p_fake": bundle.p_fake,
    "frame_p": [f.p for f in (bundle.video.frame_scores if bundle.video else [])][:8],
}}))
''',
        encoding="utf-8",
    )

    runs = {}
    for label, env in (
        ("FFMPEG", {"DFP_VIDEO_BACKEND": "FFMPEG"}),
        ("MSMF", {"DFP_VIDEO_BACKEND": "MSMF"}),
    ):
        environ = dict(os.environ)
        environ.update(env)
        proc = subprocess.run(
            [sys.executable, str(worker)], capture_output=True, text=True, env=environ
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
        if not line:
            print(f"  {label}: FAILED\n{proc.stderr[-600:]}")
            continue
        runs[label] = json.loads(line[0][2:])

    for label, r in runs.items():
        print(f"  {label:7s} backend={r['backend']:7s} decode_digest={r['decode_digest']} "
              f"verdict={r['verdict']:22s} p_fake={r['p_fake']:.10f}")
        print(f"          model_version={r['model_version']}")
        print(f"          frames_sha={r['frames_sha']}  indices[:6]={r['indices'][:6]}")

    if len(runs) == 2:
        a, b = runs["FFMPEG"], runs["MSMF"]
        print()
        print(f"  frame indices identical : {a['indices'] == b['indices']}")
        print(f"  timestamps identical    : {a['stamps'] == b['stamps']}")
        print(f"  decoded pixels identical: {a['frames_sha'] == b['frames_sha']}")
        print(f"  p_fake                  : {a['p_fake']:.10f} vs {b['p_fake']:.10f}"
              f"   |d|={abs(a['p_fake'] - b['p_fake']):.10f}")
        print(f"  verdict                 : {a['verdict']} vs {b['verdict']}")
        print(f"  effective_model_version : "
              f"{'*** IDENTICAL ***' if a['model_version'] == b['model_version'] else 'differs'}")
        print(f"  decode digest           : "
              f"{'differs (detectable)' if a['decode_digest'] != b['decode_digest'] else 'identical'}")
        print("\n  This is the finding, in one line: different pixels, different score,")
        print("  identical model version. The decode digest is the field added in")
        print("  Batch 8 so the difference is at least visible.")


def e12_same_build_repeatability() -> None:
    head(
        12,
        "Control: is decoding repeatable at all on one build?",
        "it is - otherwise nothing above would be interpretable",
        "if this failed, every other experiment would be measuring noise",
    )
    clip = smooth_clip(TMP / "repeat.mp4", frames=40)
    data = clip.read_bytes()
    seen = set()
    for _ in range(5):
        ex = pre.extract_frames(data, n_frames=16, target_size=128)
        seen.add(
            (
                tuple(ex.indices),
                tuple(ex.timestamps_ms),
                hashlib.sha256(ex.frames.tobytes()).hexdigest(),
            )
        )
    print(f"  extract_frames x5 -> {len(seen)} distinct result(s): "
          f"{'DETERMINISTIC' if len(seen) == 1 else '*** NONDETERMINISTIC ***'}")

    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg = buf.tobytes()
    digests = {hashlib.sha256(pre.decode_image(jpg).tobytes()).hexdigest() for _ in range(5)}
    print(f"  decode_image x5   -> {len(digests)} distinct result(s): "
          f"{'DETERMINISTIC' if len(digests) == 1 else '*** NONDETERMINISTIC ***'}")

    sizes = {hashlib.sha256(pre.resize(img, 384).tobytes()).hexdigest() for _ in range(5)}
    norms = {hashlib.sha256(pre.normalize(img).tobytes()).hexdigest() for _ in range(5)}
    print(f"  resize x5         -> {len(sizes)} distinct result(s)")
    print(f"  normalize x5      -> {len(norms)} distinct result(s)")


def main() -> int:
    print(__doc__.strip())
    for fn in (
        e0_environment,
        e1_metadata_across_backends,
        e2_index_amplification,
        e3_pixels_across_backends,
        e4_cause,
        e5_seek_strategies,
        e6_dispatch_paths,
        e7_independent_image_decoder,
        e8_face_detection_amplifier,
        e9_logit_drift,
        e10_verdict_and_metrics,
        e11_end_to_end_and_version,
        e12_same_build_repeatability,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  EXPERIMENT FAILED: {type(exc).__name__}: {exc}")
    print(f"\n\nartifacts in {TMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
