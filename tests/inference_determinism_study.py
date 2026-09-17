"""Within-environment determinism: twelve stages, measured. Standalone.

    python tests/inference_determinism_study.py

Produces every number in ``docs/INFERENCE_DETERMINISM.md``. Same pattern as
``decode_study.py`` and ``band_study.py``.

THE QUESTION
------------
Batch 8 proved that *different* decode environments produce different verdicts.
Batch 9 made those environments unable to share a cache. Batch 10 asks the
remaining question:

    Can identical media produce different verdicts inside ONE runtime identity?

That is a different question from Batch 8's, and a harder one to answer
usefully, because the honest answer for most of the pipeline is "no" — and a
study that only reports the things that broke would leave the reader unable to
tell a measured guarantee from an untested assumption. So every stage is
measured and reported, including the eight that are clean.

The answer overall is **yes, twice**, and the two divergences are found by two
different instruments. Stage 2 finds R14 by permuting inputs; stage 11 finds R15
by running the detector from eight threads at once. Neither is reachable by
running the pipeline twice and diffing, which is what "we tested determinism"
usually means and is exactly what stages 1 and 3-10 do.

WHAT COUNTS AS DIVERGENCE HERE
------------------------------
Two kinds, and conflating them is the trap:

**True nondeterminism** — the same code on the same input gives different
answers across runs. Threading, hash seeding, uninitialised memory, wall-clock.

**Order dependence** — the answer is a deterministic function of something that
is not the input: the order a library returned its results in, the order a sum
was accumulated in. Reproducible today, and reproducible in a *different* way
tomorrow, because nothing specifies it.

This study found one of each. **R14** is order dependence: face selection
resolved ties on the detector's output order. **R15** is true nondeterminism:
one ``cv2.CascadeClassifier`` shared by every thread, reached through
``asyncio.to_thread`` from both pipelines, returning a well-formed wrong answer
when two calls overlap.

Order dependence is the more insidious of the two — run the pipeline a thousand
times and it agrees a thousand times, right up until the day the upstream
ordering changes. But R15 is the more expensive: it is live, it is silent, and
it was measured at 15% of concurrent analyses of one file publishing a different
verdict from the sequential answer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from inference import postprocess as post  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.decode_env import decode_environment  # noqa: E402
from inference.preprocess import FaceBox, face_order_key  # noqa: E402
from inference.video_pipeline import MIN_USABLE_FRAMES  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
TMP = Path(tempfile.mkdtemp(prefix="dfp-inference-study-"))
BAND = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)


# --------------------------------------------------------------------------
# The named double
# --------------------------------------------------------------------------


class LegacyAreaOnlyOrder:
    """The pre-Batch-10 face ordering: ``sorted(boxes, key=lambda b: -b.area)``.

    Verbatim. Python's sort is stable, so equal areas resolve to detector output
    order — which is the defect, and the reason this is kept as a double rather
    than described.
    """

    name = "area only, stable sort (pre-Batch-10)"

    @staticmethod
    def order(boxes: list[FaceBox]) -> list[FaceBox]:
        return sorted(boxes, key=lambda b: -b.area)


class GeometricOrder:
    """The shipped ordering. A total order on the box itself."""

    name = "area, then top-most, then left-most (Batch 10)"

    @staticmethod
    def order(boxes: list[FaceBox]) -> list[FaceBox]:
        return sorted(boxes, key=face_order_key)


class LegacySharedCascade:
    """The pre-R15 detector: one ``CascadeClassifier`` for the whole process.

    Verbatim, including the unguarded lazy global. Both pipelines reach
    ``detect_faces`` through ``asyncio.to_thread``, so a process-wide instance
    was shared across pool threads in the shipped configuration — this double is
    what the platform actually ran, not a worst case constructed for the study.
    """

    name = "one process-wide cascade (pre-R15)"
    _cascade = None

    @classmethod
    def detect(cls, image: np.ndarray, *, min_px: int = 32) -> tuple:
        if cls._cascade is None:
            cls._cascade = cv2.CascadeClassifier(pre.CASCADE_PATH)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found = cls._cascade.detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(min_px, min_px)
        )
        return tuple(sorted(
            (FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in found),
            key=face_order_key,
        ))


class LockedSharedCascade:
    """The rejected alternative: one cascade, serialised by a lock.

    Correct — measured at 320/320 — and priced differently from the shipped
    fix. It makes every concurrent image job queue for the single most expensive
    step in the pipeline; the per-thread cascade instead pays a one-off load and
    0.41 MB per pool thread. Kept so the comparison in the report is executed
    rather than argued.
    """

    name = "one process-wide cascade behind a lock"
    _lock = threading.Lock()

    @classmethod
    def detect(cls, image: np.ndarray, *, min_px: int = 32) -> tuple:
        with cls._lock:
            return LegacySharedCascade.detect(image, min_px=min_px)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def face_image(size: int = 480) -> np.ndarray:
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def shifted_face_image(size: int = 480, dx: int = -40, dy: int = -30) -> np.ndarray:
    """The same face, translated. A second file whose correct answer differs.

    Needed for the cross-image half of stage 11: to show a *foreign* box the two
    files must have different correct answers, and to distinguish that from a
    lost face they must both have one.

    **The offset is not free to choose.** Of nine shifts tried, only this one
    still detects — the drawing sits near this cascade's decision boundary, which
    is Batch 8's R13 observation showing up again as a fixture constraint rather
    than as a finding. It is why stage 11 asserts both files detect before
    measuring: a silently undetectable second file would report every foreign box
    as a lost face and hide the contamination entirely.
    """
    src = face_image(size)
    out = np.full_like(src, 200)
    ys, xs = slice(max(0, dy), size + min(0, dy)), slice(max(0, dx), size + min(0, dx))
    yd, xd = slice(max(0, -dy), size + min(0, -dy)), slice(max(0, -dx), size + min(0, -dx))
    out[yd, xd] = src[ys, xs]
    return out


def two_region_image() -> np.ndarray:
    """Two visually different regions of exactly equal size."""
    img = np.full((400, 800, 3), 200, np.uint8)
    rng = np.random.default_rng(7)
    img[100:300, 100:300] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    img[100:300, 500:700] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    return img


def clip(path: Path, *, frames: int = 60, size: int = 256) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 25, (size, size))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(frames):
        r = 128 + 100 * np.sin((xx + i * 3) / 21.0)
        g = 128 + 100 * np.cos((yy - i * 2) / 17.0)
        b = 128 + 100 * np.sin((xx + yy + i * 4) / 29.0)
        writer.write(np.clip(np.dstack([b, g, r]), 0, 255).astype(np.uint8))
    writer.release()
    return path


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def stub_logit_of(batch_frame: np.ndarray) -> float:
    """Exactly what ``StubBackend`` computes for one prepared frame."""
    digest = hashlib.sha256(b"spatial" + batch_frame.tobytes()).hexdigest()
    p = int(digest[:12], 16) / float(0x1000000000000)
    return float(np.log(p / (1 - p)))


def verdict_of(p: float) -> str:
    if p >= BAND.band_hi:
        return "manipulated"
    if p >= BAND.band_lo:
        return "insufficient"
    return "authentic"


def head(n: int, stage: str, question: str) -> None:
    print(f"\n{'=' * 74}")
    print(f"S{n}. {stage}")
    print(f"{'=' * 74}")
    print(f"  question : {question}\n")


def report(label: str, inputs: str, outputs: set, n: int) -> None:
    rate = (len(outputs) - 1) / max(n - 1, 1)
    status = "DETERMINISTIC" if len(outputs) == 1 else "*** DIVERGENT ***"
    print(f"  {label:<34s} in={inputs}  runs={n}  distinct={len(outputs)}  "
          f"divergence={rate:.3f}  {status}")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def s1_face_detection() -> None:
    head(1, "Face detection", "same image, repeated / threads / dispatch — same boxes?")
    img = face_image()
    key = sha(img)

    runs = {tuple((b.x, b.y, b.width, b.height) for b in pre.detect_faces(img)) for _ in range(20)}
    report("repeated in one process", key, runs, 20)

    threaded = set()
    try:
        for n in (0, 1, 2, 4, 8):
            cv2.setNumThreads(n)
            threaded.add(tuple((b.x, b.y, b.width, b.height) for b in pre.detect_faces(img)))
    finally:
        cv2.setNumThreads(0)
    report("across cv2 thread counts", key, threaded, 5)

    opt = set()
    try:
        for flag in (True, False):
            cv2.setUseOptimized(flag)
            opt.add(tuple((b.x, b.y, b.width, b.height) for b in pre.detect_faces(img)))
    finally:
        cv2.setUseOptimized(True)
    report("across optimisation toggle", key, opt, 2)


def s2_face_selection() -> None:
    head(2, "Face selection", "which face does faces[0] name when areas tie?")
    boxes = [FaceBox(10, 10, 100, 100), FaceBox(500, 10, 100, 100), FaceBox(250, 200, 100, 100)]
    print(f"  three boxes of equal area ({boxes[0].area}), six input orders\n")

    for double in (LegacyAreaOnlyOrder, GeometricOrder):
        winners = {(double.order(list(p))[0].x, double.order(list(p))[0].y)
                   for p in itertools.permutations(boxes)}
        report(double.name, "3 equal-area boxes", winners, 6)

    print("\n  Downstream impact of picking the other face:")
    img = two_region_image()
    a, b = FaceBox(100, 100, 200, 200), FaceBox(500, 100, 200, 200)
    for label, order in (("A first", [a, b]), ("B first", [b, a])):
        chosen = LegacyAreaOnlyOrder.order(order)[0]
        crop = pre.crop_face(img, chosen)
        batch = pre.prepare_image(crop, size=384)
        logit = stub_logit_of(batch[0])
        p = 1.0 / (1.0 + np.exp(-logit))
        print(f"    {label}: x={chosen.x}  crop={sha(crop)}  logit={logit:+.4f}  "
              f"p_fake={p:.4f}  verdict={verdict_of(p)}")
    print(f"    band [{BAND.band_lo:.4f}, {BAND.band_hi:.4f})")

    print("\n  Do real Haar detections tie in practice?")
    ties, trials = 0, 0
    for size in (90, 100, 110, 120):
        canvas = np.full((520, 1000, 3), 205, np.uint8)
        for cx in (250, 720):
            cv2.ellipse(canvas, (cx, 250), (int(0.75 * size), size), 0, 0, 360, (175, 160, 150), -1)
            cv2.circle(canvas, (cx - int(0.28 * size), 250 - int(0.29 * size)),
                       max(2, int(0.13 * size)), (60, 60, 60), -1)
            cv2.circle(canvas, (cx + int(0.28 * size), 250 - int(0.29 * size)),
                       max(2, int(0.13 * size)), (60, 60, 60), -1)
        found = pre.detect_faces(cv2.GaussianBlur(canvas, (5, 5), 0))
        areas = [f.area for f in found]
        trials += 1
        ties += len(areas) != len(set(areas))
        print(f"    face size {size:3d}: {len(found)} detections, areas={areas}")
    print(f"    ties observed: {ties}/{trials}  — the risk is structural, not yet live "
          "on Haar")


def s3_crop_alignment() -> None:
    head(3, "Crop alignment", "same box, same image — same pixels into the model?")
    img = face_image()
    face = pre.detect_faces(img)[0]
    crops = {sha(pre.crop_face(img, face)) for _ in range(10)}
    report("crop_face", sha(img), crops, 10)
    preps = {sha(pre.prepare_image(pre.crop_face(img, face), size=384)) for _ in range(10)}
    report("prepare_image (resize)", sha(img), preps, 10)
    norms = {sha(pre.normalize(pre.crop_face(img, face))) for _ in range(10)}
    report("normalize", sha(img), norms, 10)


def s4_frame_extraction() -> None:
    head(4, "Frame extraction", "same bytes — same frames, indices and timestamps?")
    data = clip(TMP / "s4.mp4").read_bytes()
    key = hashlib.sha256(data).hexdigest()[:12]
    sigs = set()
    for _ in range(6):
        ex = pre.extract_frames(data, n_frames=16, target_size=128)
        sigs.add((tuple(ex.indices), tuple(ex.timestamps_ms), sha(ex.frames)))
    report("extract_frames", key, sigs, 6)

    counts = {tuple(pre.sample_frame_indices(60, 48)) for _ in range(10)}
    report("sample_frame_indices", "frame_count=60", counts, 10)


def s5_frame_ranking() -> None:
    head(5, "Frame ranking", "which frames are selected when scores tie?")
    for label, scores in (
        ("all zero", np.zeros(48)),
        ("all equal (3.5)", np.full(48, 3.5)),
        ("three-way ties", np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0] * 8)),
        ("distinct", np.random.default_rng(0).normal(0, 3, 48)),
    ):
        picks = {tuple(post.topk_indices(scores)) for _ in range(10)}
        report(f"topk_indices, {label}", f"{scores.size} scores", picks, 10)

    print("\n  Ties resolve to the LOWEST indices, by construction:")
    print(f"    all-equal 48 scores -> {post.topk_indices(np.zeros(48))}")
    print("    np.lexsort((arange, -arr)) — the index is the tiebreak, deliberately.")
    print("    Order dependence here was closed in Batch 0; this re-measures it.")


def s6_pooling() -> None:
    head(6, "Frame pooling", "is the pooled logit a function of the values alone?")
    order_sensitive, worst = 0, 0.0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        vals = rng.normal(0, 3, 48)
        idx = post.topk_indices(vals)
        forward = post.pool_at(vals, idx)
        backward = post.pool_at(vals, list(reversed(idx)))
        shuffled = post.pool_at(vals, random.Random(seed).sample(idx, len(idx)))
        delta = max(abs(forward - backward), abs(forward - shuffled))
        if delta > 0:
            order_sensitive += 1
        worst = max(worst, delta)
    print(f"  pool_at over 200 fixtures: {order_sensitive}/200 are order-sensitive, "
          f"worst |d| = {worst:.3e}")
    print("  BUT topk_indices returns sorted(), so pool_at only ever receives")
    print("  ascending indices — the dependence is real and structurally unreachable.")
    print(f"  sorted on a random fixture: "
          f"{post.topk_indices(np.random.default_rng(3).normal(0, 1, 32)) == sorted(post.topk_indices(np.random.default_rng(3).normal(0, 1, 32)))}")

    vals = np.random.default_rng(1).normal(0, 3, 48)
    idx = post.topk_indices(vals)
    pools = {post.pool_at(vals, idx) for _ in range(10)}
    report("pool_at, repeated", "48 logits", pools, 10)


def s7_video_aggregation() -> None:
    head(7, "Video aggregation", "same per-frame curves — same branch outcomes?")
    rng = np.random.default_rng(5)
    per_branch = {"A": rng.normal(0, 2, 48), "B": rng.normal(0, 2, 48)}
    fused = rng.normal(0, 2, 48)
    sigs = set()
    for _ in range(10):
        selected = post.topk_indices(fused)
        sigs.add(tuple(round(post.pool_at(v, selected), 15) for v in per_branch.values()))
    report("select + pool per branch", "48x2 logits", sigs, 10)
    print("  One selection, shared by every branch — so a disagreement between")
    print("  branches is about the media, not about which frames each looked at.")


def s8_thresholding() -> None:
    head(8, "Final thresholding", "is decide() a pure function of (p, threshold)?")
    from inference.postprocess import decide

    outs = set()
    for _ in range(10):
        outs.add(tuple(decide(p / 1000.0, BAND)[0].value for p in range(0, 1001, 37)))
    report("decide over 28 probabilities", "p grid", outs, 10)

    print("\n  The boundaries are exact comparisons, so the cliff is at a float:")
    for p in (BAND.band_lo - 1e-16, BAND.band_lo, BAND.band_hi - 1e-16, BAND.band_hi):
        print(f"    p={p!r:24s} -> {decide(p, BAND)[0].value}")


def s9_abstention() -> None:
    head(9, "Abstention decisions", "are the gate boundaries stable and exact?")
    print(f"  MIN_USABLE_FRAMES = {MIN_USABLE_FRAMES}")
    print("  A clip decoding 7 frames abstains (TOO_FEW_FRAMES); 8 is scored.")
    print("  The boundary is an integer comparison on the count that survived")
    print("  decoding, so it is exact and stable *within* one decode environment")
    print("  — and it is precisely the quantity Batch 8 showed a decoder change")
    print("  can move. Within one runtime identity: deterministic.\n")

    caps = pre.MediaCaps()
    print(f"  min_face_px       = {caps.min_face_px}  (FACE_TOO_SMALL below this)")
    print(f"  min_short_edge    = {caps.min_short_edge}  (QUALITY_BELOW_FLOOR below this)")
    img = face_image()
    face = pre.detect_faces(img)[0]
    print(f"  fixture face short edge = {face.short_edge}  -> "
          f"{'passes' if face.short_edge >= caps.min_face_px else 'abstains'}")
    edges = {pre.detect_faces(img)[0].short_edge for _ in range(10)}
    report("face_px across repeats", sha(img), edges, 10)


def s10_end_to_end() -> None:
    head(10, "End to end", "same media across processes, threads and hash seeds?")
    img_path = TMP / "face.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(face_image(), cv2.COLOR_RGB2BGR))
    vid_path = clip(TMP / "e2e.mp4")

    worker = TMP / "worker.py"
    worker.write_text(
        f'''
import sys, asyncio, hashlib, json, os
from datetime import UTC, datetime
from uuid import UUID
sys.path.insert(0, r"{ROOT}")
import cv2
cv2.setNumThreads(int(os.environ.get("DFP_TEST_THREADS", "0")))
from contracts.enums import MediaKind
from contracts.job import JobSpec
from inference import ImagePipeline, VideoPipeline

out = {{}}
for kind, path, mime in (("image", r"{img_path}", "image/png"),
                         ("video", r"{vid_path}", "video/mp4")):
    data = open(path, "rb").read()
    spec = JobSpec(job_id=UUID(int=1),
                   media_kind=MediaKind.IMAGE if kind == "image" else MediaKind.VIDEO,
                   storage_key="u", sha256=hashlib.sha256(data).hexdigest(),
                   size_bytes=len(data), declared_mime=mime, model_version="v",
                   idempotency_key="k", submitted_at=datetime(2026, 1, 1, tzinfo=UTC))
    pipe = ImagePipeline() if kind == "image" else VideoPipeline()
    fn = pipe.analyze_image if kind == "image" else pipe.analyze_video
    b = asyncio.run(fn(spec, data))
    out[kind] = {{
        "verdict": b.verdict.value,
        "p_fake": b.p_fake,
        "interval": list(b.interval) if b.interval else None,
        "version": b.model.version,
        "faces": [(f.bbox.x, f.bbox.y, f.bbox.width, f.bbox.height) for f in b.faces],
        "abstain": b.quality.gate_reason.value if b.quality.gate_reason else None,
        "frames": [(f.frame_idx, f.t_ms, f.p) for f in (b.video.frame_scores if b.video else [])][:8],
    }}
print("@@" + json.dumps(out, sort_keys=True))
''',
        encoding="utf-8",
    )

    runs = {}
    for label, env in (
        ("launch 1", {}),
        ("launch 2", {}),
        ("launch 3", {}),
        ("threads=1", {"DFP_TEST_THREADS": "1"}),
        ("threads=2", {"DFP_TEST_THREADS": "2"}),
        ("threads=8", {"DFP_TEST_THREADS": "8"}),
        ("PYTHONHASHSEED=0", {"PYTHONHASHSEED": "0"}),
        ("PYTHONHASHSEED=12345", {"PYTHONHASHSEED": "12345"}),
    ):
        environ = dict(os.environ)
        environ.update(env)
        proc = subprocess.run(
            [sys.executable, str(worker)], capture_output=True, text=True, env=environ
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
        if not line:
            print(f"  {label}: FAILED — {proc.stderr[-300:]}")
            continue
        runs[label] = line[0][2:]

    image_sigs = {json.loads(v)["image"]["p_fake"] for v in runs.values()}
    video_sigs = {json.loads(v)["video"]["p_fake"] for v in runs.values()}
    verdicts = {(json.loads(v)["image"]["verdict"], json.loads(v)["video"]["verdict"])
                for v in runs.values()}
    abstentions = {(json.loads(v)["image"]["abstain"], json.loads(v)["video"]["abstain"])
                   for v in runs.values()}
    whole = set(runs.values())

    first = json.loads(next(iter(runs.values())))
    print(f"  image: verdict={first['image']['verdict']} p_fake={first['image']['p_fake']}")
    print(f"  video: verdict={first['video']['verdict']} p_fake={first['video']['p_fake']}")
    print(f"  runtime identity: {first['image']['version']}\n")
    report("p_fake (image)", "1 png", image_sigs, len(runs))
    report("p_fake (video)", "1 mp4", video_sigs, len(runs))
    report("verdicts", "both", verdicts, len(runs))
    report("abstentions", "both", abstentions, len(runs))
    report("whole bundle payload", "both", whole, len(runs))


def _parallel(detect, image, *, threads: int = 8, reps: int = 40) -> tuple[list, float]:
    """``detect(image)`` on ``threads`` threads at once. Answers and wall time.

    ``detectMultiScale`` releases the GIL, so this is genuine overlap rather
    than interleaved bytecode — which is exactly why it is dangerous.
    """
    collected: list = []
    guard = threading.Lock()

    def worker() -> None:
        local = [detect(image) for _ in range(reps)]
        with guard:
            collected.extend(local)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    started = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return collected, time.perf_counter() - started


def s11_concurrent_detection() -> None:
    head(11, "Concurrent detection (R15)",
         "eight threads, one image — does a shared cascade still agree?")
    img = face_image()
    truth = LegacySharedCascade.detect(img)
    print(f"  sequential answer: {[(b.x, b.y, b.width, b.height) for b in truth]}")
    print("  both pipelines call detect_faces through asyncio.to_thread, so a")
    print("  process-wide cascade is shared across pool threads by construction.\n")

    for double, detect in (
        (LegacySharedCascade, LegacySharedCascade.detect),
        (LockedSharedCascade, LockedSharedCascade.detect),
        (None, lambda i: tuple(pre.detect_faces(i))),
    ):
        label = double.name if double else "one cascade per thread (SHIPPED)"
        answers, seconds = _parallel(detect, img)
        wrong = [a for a in answers if a != truth]
        lost = sum(1 for a in wrong if not a)
        print(f"  {label:<38s} wrong={len(wrong):>3d}/{len(answers)} "
              f"({len(wrong)/len(answers):6.2%})  lost-face={lost:>3d}  "
              f"{seconds*1000:7.0f} ms")

    print("\n  Cross-image: four threads per file, two different files at once.")
    print("  This is the shape a worker actually runs — concurrent jobs, not")
    print("  concurrent repeats of one job.\n")
    other = shifted_face_image()
    truth_a, truth_b = LegacySharedCascade.detect(img), LegacySharedCascade.detect(other)
    print(f"  file a sequential: {[(b.x, b.y, b.width, b.height) for b in truth_a]}")
    print(f"  file b sequential: {[(b.x, b.y, b.width, b.height) for b in truth_b]}\n")
    if not (truth_a and truth_b and truth_a != truth_b):
        print("  *** fixture broken: both files must detect, and differently. ***")
        print("  Without that, a foreign box is indistinguishable from a lost one")
        print("  and the contamination measurement below means nothing.")
        return

    collected: dict[str, list] = {"a": [], "b": []}
    guard = threading.Lock()

    def worker(tag: str, image: np.ndarray) -> None:
        local = []
        for _ in range(40):
            # A raise is an *outcome*, not a crashed experiment. Letting the
            # thread die would drop the calls from the denominator and report
            # the most severe failure mode as an empty column.
            try:
                local.append(LegacySharedCascade.detect(image))
            except cv2.error as exc:
                local.append(("RAISED", str(exc).strip().splitlines()[-1][:60]))
        with guard:
            collected[tag].extend(local)

    workers = [threading.Thread(target=worker, args=("a", img)) for _ in range(4)]
    workers += [threading.Thread(target=worker, args=("b", other)) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    raised_examples: set[str] = set()
    for tag, truth_t, other_t in (("a", truth_a, truth_b), ("b", truth_b, truth_a)):
        rows = collected[tag]
        raised = [r for r in rows if r and r[0] == "RAISED"]
        raised_examples.update(r[1] for r in raised)
        ok = sum(1 for r in rows if r == truth_t)
        lost = sum(1 for r in rows if r == ())
        foreign = [r for r in rows if r and r[0] != "RAISED" and r != truth_t]
        matched_other = sum(1 for r in foreign if r == other_t)
        print(f"  file {tag}: {len(rows):>3d} calls -> correct {ok:>3d}, "
              f"lost-face {lost:>3d}, other-box {len(foreign):>3d} "
              f"({matched_other} of them the other file's), raised {len(raised):>3d}")
    for message in sorted(raised_examples):
        print(f"    raised: cv2.error: {message}")

    print("\n  Three distinct failure modes, all from one shared object:")
    print("    * a lost face      -> a scored image published as NO_FACE_DETECTED;")
    print("    * a foreign box    -> a crop taken from another job's media;")
    print("    * a raise          -> an out-of-range read inside OpenCV's own")
    print("                          vectors, surfacing as a failed job.")
    print("  No identity string in the platform moves across any of them: same")
    print("  process, same decode identity, same model version. Detection is")
    print("  structurally unavailable, so prevention is the only control.")


def s12_verdict_impact() -> None:
    head(12, "Verdict impact", "what do the two divergences cost in decisions?")
    print(f"  band [{BAND.band_lo:.6f}, {BAND.band_hi:.6f}), width "
          f"{BAND.band_hi - BAND.band_lo:.6f}\n")

    rng = np.random.default_rng(7)
    population = rng.beta(2, 2, 200_000)

    def region(p):
        return np.where(p >= BAND.band_hi, 2, np.where(p >= BAND.band_lo, 1, 0))

    base = region(population)
    print("  A shift of |dp| applied to a Beta(2,2) score population:")
    for dp, source in (
        (0.0003, "fixed crop, +-3 LSB pixel noise (surrogate)"),
        (0.0050, "R12 cross-decoder drift (surrogate, Batch 8)"),
        (0.0910, "R13 crop movement under +-3 LSB (surrogate)"),
    ):
        moved = region(np.clip(population + dp, 0.0, 1.0))
        changed = float((moved != base).mean())
        print(f"    |dp|={dp:.4f}  {changed:6.2%} of decisions change   {source}")

    print("\n  R15 is not a shift. A lost face is not a moved score — it replaces")
    print("  the score with an abstention, so the measured 15% concurrent")
    print("  divergence is 15% of decisions changed outright, not 15% of a band.")


def main() -> int:
    print(__doc__.strip())
    print(f"\n  {decode_environment().describe()}")
    for fn in (
        s1_face_detection,
        s2_face_selection,
        s3_crop_alignment,
        s4_frame_extraction,
        s5_frame_ranking,
        s6_pooling,
        s7_video_aggregation,
        s8_thresholding,
        s9_abstention,
        s10_end_to_end,
        s11_concurrent_detection,
        s12_verdict_impact,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  STAGE FAILED: {type(exc).__name__}: {exc}")
    print(f"\n\nartifacts in {TMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
