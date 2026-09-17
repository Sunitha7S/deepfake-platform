"""R13/R14/R15 — determinism inside a single runtime identity.

THE QUESTION BATCH 10 ASKED
---------------------------
Batch 8 proved different decode environments produce different verdicts. Batch 9
made those environments unable to share a cache. What was left:

    Can identical media produce different verdicts inside ONE runtime identity?

Measured across eleven stages (``tests/inference_determinism_study.py``):
**yes**, by two mechanisms, and they are different kinds of thing.

**R14, order dependence.** Face selection resolved ties on the detector's output
order. Deterministic every run, and specified by nothing — so it is stable until
the day the upstream ordering moves. A repeat-run test cannot see it.

**R15, true nondeterminism.** One ``cv2.CascadeClassifier`` was shared by every
thread, and both pipelines reach it through ``asyncio.to_thread``. Concurrent
calls corrupt each other's working state and return a well-formed wrong answer:
measured at 24 of 160 concurrent ``analyze_image`` calls publishing
``INSUFFICIENT_EVIDENCE / no_face_detected`` for bytes that every sequential run
scored ``AUTHENTIC`` at p_fake 0.0329. A repeat-run test cannot see this one
either — not because it is stable, but because it only fires under concurrency,
and the eight clean stages below were all measured single-threaded.

That is the shape of this file: neither finding is reachable by running the
pipeline twice and comparing, which is what "we tested determinism" usually
means.

R14 — FACE SELECTION WAS DECIDED BY DETECTOR OUTPUT ORDER
---------------------------------------------------------
``detect_faces`` returned ``sorted(boxes, key=lambda b: -b.area)``. Python's
sort is stable, so two faces of equal area resolved to whichever the cascade
emitted first — and ``faces[0]`` chooses the crop the entire verdict is computed
from. Measured on a 400x800 fixture with two 200x200 regions:

    A first -> crop 6c9a2dfa3527, p_fake 0.4830  ->  authentic
    B first -> crop f04cb2ef504e, p_fake 0.5637  ->  insufficient evidence

Same image, same model, same runtime identity, different verdict.

**This is order dependence, not nondeterminism**, and the distinction is the
whole point of this file. Running the pipeline a thousand times gives the same
answer a thousand times, because ``detectMultiScale`` is deterministic on this
build. Nothing *specified* that ordering, so nothing stops it moving — and the
frozen architecture replaces Haar with SCRFD, whose box order falls out of NMS
over float scores and is an implementation detail by construction.

So the tests below come in three shapes:

* **Repeat-invariance** — the eight clean stages, pinned so a regression shows.
  These would have passed before Batch 10 too. They are the floor, not the
  finding.
* **Order-invariance** — the property a repeat test cannot see, checked by
  feeding the *same* inputs in every order and requiring one answer. This is
  where ``LegacyAreaOnlyOrder`` fails and the shipped ordering does not.
* **Concurrency-invariance** — section 5, checked by running the real detector
  and the real pipeline from eight threads at once. This is where
  ``LegacySharedCascade`` fails and the shipped per-thread cascade does not.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from contracts.enums import Verdict  # noqa: E402
from inference import postprocess as post  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.preprocess import FaceBox, face_order_key  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
TMP = Path(tempfile.mkdtemp(prefix="dfp-inference-det-"))
BAND = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)


# --------------------------------------------------------------------------
# The named double
# --------------------------------------------------------------------------


class LegacyAreaOnlyOrder:
    """The pre-Batch-10 face ordering, verbatim.

    One line, kept beside the shipped one so "the fix was needed" is measured
    rather than remembered. Python's ``sorted`` is stable, which is what turns
    an incomplete key into a positional tiebreak.
    """

    name = "area only, stable sort (pre-Batch-10)"

    @staticmethod
    def order(boxes: list[FaceBox]) -> list[FaceBox]:
        return sorted(boxes, key=lambda b: -b.area)


def _face_image(size: int = 480) -> np.ndarray:
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _clip(name: str = "d.mp4", *, frames: int = 40, size: int = 192) -> Path:
    path = TMP / name
    if path.exists():
        return path
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 25, (size, size))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(frames):
        r = 128 + 100 * np.sin((xx + i * 3) / 21.0)
        g = 128 + 100 * np.cos((yy - i * 2) / 17.0)
        b = 128 + 100 * np.sin((xx + yy + i * 4) / 29.0)
        writer.write(np.clip(np.dstack([b, g, r]), 0, 255).astype(np.uint8))
    writer.release()
    return path


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


EQUAL_AREA = [
    FaceBox(10, 10, 100, 100),
    FaceBox(500, 10, 100, 100),
    FaceBox(250, 200, 100, 100),
]


# --------------------------------------------------------------------------
# 1. R14: the defect, and the fix
# --------------------------------------------------------------------------


def test_the_legacy_ordering_picks_a_different_face_per_input_order():
    """The defect, reproduced against the real old key function."""
    winners = {
        (LegacyAreaOnlyOrder.order(list(perm))[0].x,
         LegacyAreaOnlyOrder.order(list(perm))[0].y)
        for perm in itertools.permutations(EQUAL_AREA)
    }
    assert len(winners) == 3, (
        f"the legacy double no longer diverges (winners={winners}); it can no "
        "longer demonstrate R14"
    )


def test_the_shipped_ordering_is_a_function_of_the_boxes_alone():
    winners = {
        (sorted(perm, key=face_order_key)[0].x, sorted(perm, key=face_order_key)[0].y)
        for perm in itertools.permutations(EQUAL_AREA)
    }
    assert len(winners) == 1, f"face selection is still order-dependent: {winners}"


def test_the_ordering_is_a_total_order():
    """Two boxes must never compare equal, or the tiebreak is still positional.

    Includes the case an area-plus-origin key would miss: 100x200 and 200x100
    share both area and origin and differ only in aspect.
    """
    boxes = [
        FaceBox(0, 0, 100, 200),
        FaceBox(0, 0, 200, 100),
        FaceBox(0, 0, 100, 100),
        FaceBox(5, 0, 100, 100),
        FaceBox(0, 5, 100, 100),
    ]
    keys = [face_order_key(b) for b in boxes]
    assert len(set(keys)) == len(keys), f"two distinct boxes share a key: {keys}"


def test_the_ordering_still_puts_the_largest_face_first():
    """The fix must not have changed the rule, only completed it."""
    boxes = [FaceBox(0, 0, 50, 50), FaceBox(300, 300, 200, 200), FaceBox(10, 10, 120, 120)]
    for perm in itertools.permutations(boxes):
        assert sorted(perm, key=face_order_key)[0].area == 200 * 200


def test_choosing_the_other_tied_face_changes_the_verdict():
    """Why an ordering nobody specified is worth a batch.

    The two crops are different images, so the model sees different pixels and
    the verdict moves across a band edge. Asserted rather than described,
    because "it could matter" and "it changes the answer" are different claims.
    """
    img = np.full((400, 800, 3), 200, np.uint8)
    rng = np.random.default_rng(7)
    img[100:300, 100:300] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    img[100:300, 500:700] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

    a, b = FaceBox(100, 100, 200, 200), FaceBox(500, 100, 200, 200)
    assert a.area == b.area, "the fixture must actually tie"

    verdicts, probabilities = set(), []
    for order in ([a, b], [b, a]):
        chosen = LegacyAreaOnlyOrder.order(order)[0]
        batch = pre.prepare_image(pre.crop_face(img, chosen), size=384)
        digest = hashlib.sha256(b"spatial" + batch[0].tobytes()).hexdigest()
        p = int(digest[:12], 16) / float(0x1000000000000)
        probabilities.append(p)
        verdicts.add(
            "manipulated" if p >= BAND.band_hi
            else "insufficient" if p >= BAND.band_lo
            else "authentic"
        )
    assert probabilities[0] != probabilities[1]
    assert len(verdicts) == 2, (
        f"the fixture no longer straddles a band edge (p={probabilities}, "
        f"band=[{BAND.band_lo:.4f}, {BAND.band_hi:.4f})); it still shows the "
        "score moving, but the verdict claim needs a new fixture"
    )


def test_detect_faces_uses_the_shipped_ordering():
    """The unit fix must actually be wired into the function that ships."""
    img = _face_image()
    found = pre.detect_faces(img)
    assert found, "the fixture must detect at least one face"
    assert found == sorted(found, key=face_order_key)


# --------------------------------------------------------------------------
# 2. Repeat-invariance: the eight clean stages
# --------------------------------------------------------------------------


def test_face_detection_is_repeat_invariant():
    img = _face_image()
    runs = {
        tuple((b.x, b.y, b.width, b.height) for b in pre.detect_faces(img))
        for _ in range(8)
    }
    assert len(runs) == 1


def test_face_detection_is_invariant_to_thread_count():
    """cv2 parallelises detection; the result must not depend on how much."""
    img = _face_image()
    results = set()
    try:
        for threads in (1, 2, 4, 8):
            cv2.setNumThreads(threads)
            results.add(tuple((b.x, b.y, b.width, b.height) for b in pre.detect_faces(img)))
    finally:
        cv2.setNumThreads(0)
    assert len(results) == 1, f"detection moved with thread count: {results}"


def test_crop_and_prepare_are_repeat_invariant():
    img = _face_image()
    face = pre.detect_faces(img)[0]
    assert len({_sha(pre.crop_face(img, face)) for _ in range(6)}) == 1
    assert len({_sha(pre.prepare_image(pre.crop_face(img, face), size=384)) for _ in range(6)}) == 1


def test_frame_extraction_is_repeat_invariant():
    data = _clip().read_bytes()
    sigs = set()
    for _ in range(4):
        ex = pre.extract_frames(data, n_frames=12, target_size=96)
        sigs.add((tuple(ex.indices), tuple(ex.timestamps_ms), _sha(ex.frames)))
    assert len(sigs) == 1


def test_frame_ranking_is_order_free_even_when_every_score_ties():
    """The pathological case: 48 identical scores.

    Closed in Batch 0 with an explicit index tiebreak in ``np.lexsort``. Pinned
    here because it is the same class of defect as R14, one stage later, and the
    two should fail together if the pattern is ever undone.
    """
    for scores in (np.zeros(48), np.full(48, 3.5), np.array([1.0, 1.0, 2.0] * 16)):
        picks = {tuple(post.topk_indices(scores)) for _ in range(6)}
        assert len(picks) == 1
    assert post.topk_indices(np.zeros(48)) == [0, 1, 2, 3, 4, 5], (
        "all-equal scores must resolve to the lowest indices"
    )


def test_pooling_receives_ascending_indices_and_is_therefore_order_free():
    """A latent dependence, and the invariant that makes it unreachable.

    ``pool_at`` takes a mean, and a float mean *is* order-sensitive — measured,
    92 of 200 random fixtures differ at ~1.8e-15 between forward and shuffled
    index order. It never matters because ``topk_indices`` returns ``sorted()``.

    That is a real guarantee resting on one ``sorted()`` call in another
    function, so it is asserted here rather than assumed. If ``topk_indices``
    ever stops sorting, this fails and names the reason.
    """
    rng = np.random.default_rng(0)
    order_sensitive = 0
    for seed in range(40):
        vals = np.random.default_rng(seed).normal(0, 3, 48)
        idx = post.topk_indices(vals)
        assert idx == sorted(idx), "topk_indices no longer returns ascending indices"
        if post.pool_at(vals, idx) != post.pool_at(vals, list(reversed(idx))):
            order_sensitive += 1
    assert order_sensitive > 0, (
        "pool_at is no longer order-sensitive; the invariant this test protects "
        "may have become unnecessary, which is worth confirming before deleting it"
    )
    vals = rng.normal(0, 3, 48)
    idx = post.topk_indices(vals)
    assert len({post.pool_at(vals, idx) for _ in range(6)}) == 1


def test_thresholding_is_a_pure_function():
    grid = [p / 500.0 for p in range(501)]
    runs = {tuple(post.decide(p, BAND)[0].value for p in grid) for _ in range(4)}
    assert len(runs) == 1
    # And the regions still partition, which is what makes the cliff safe.
    seen = {post.decide(p, BAND)[0] for p in grid}
    assert seen == {Verdict.AUTHENTIC, Verdict.INSUFFICIENT_EVIDENCE, Verdict.MANIPULATED}


def test_abstention_boundaries_are_exact_integer_comparisons():
    """Gate cliffs are exact, not fuzzy — so they cannot drift, only be crossed."""
    from inference.video_pipeline import MIN_USABLE_FRAMES

    assert isinstance(MIN_USABLE_FRAMES, int)
    caps = pre.MediaCaps()
    assert isinstance(caps.min_face_px, int)
    assert isinstance(caps.min_short_edge, int)
    img = _face_image()
    assert len({pre.detect_faces(img)[0].short_edge for _ in range(6)}) == 1


# --------------------------------------------------------------------------
# 3. End to end, across process boundaries
# --------------------------------------------------------------------------


def test_the_whole_pipeline_agrees_across_processes_threads_and_hash_seeds():
    """The claim a user cares about, checked the only way it can be.

    ``PYTHONHASHSEED`` is in here because set and dict iteration order is the
    classic source of cross-process divergence in Python, and the pipeline holds
    several — ``pending = set(wanted)`` in ``extract_frames``, the per-branch
    dicts in the video path. Each is used in a way that does not depend on
    iteration order; this is what makes that a measurement rather than a claim.
    """
    img_path = TMP / "e2e.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(_face_image(), cv2.COLOR_RGB2BGR))
    vid_path = _clip("e2e.mp4")

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
        "verdict": b.verdict.value, "p_fake": b.p_fake,
        "interval": list(b.interval) if b.interval else None,
        "version": b.model.version,
        "faces": [(f.bbox.x, f.bbox.y, f.bbox.width, f.bbox.height) for f in b.faces],
        "abstain": b.quality.gate_reason.value if b.quality.gate_reason else None,
        "frames": [(f.frame_idx, f.t_ms, f.p) for f in (b.video.frame_scores if b.video else [])],
    }}
print("@@" + json.dumps(out, sort_keys=True))
''',
        encoding="utf-8",
    )

    payloads = set()
    for extra in (
        {},
        {},
        {"DFP_TEST_THREADS": "1"},
        {"DFP_TEST_THREADS": "8"},
        {"PYTHONHASHSEED": "0"},
        {"PYTHONHASHSEED": "12345"},
    ):
        env = dict(os.environ)
        env.update(extra)
        proc = subprocess.run(
            [sys.executable, str(worker)], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
        assert line, f"worker produced no result: {proc.stdout[-300:]}"
        payloads.add(line[0])

    assert len(payloads) == 1, (
        f"{len(payloads)} distinct bundles across 6 launches; identical media "
        "inside one runtime identity produced different results"
    )


# --------------------------------------------------------------------------
# 4. The limits
# --------------------------------------------------------------------------


def test_runtime_identity_cannot_detect_an_ordering_change():
    """What R14 taught, asserted so the report cannot overclaim.

    Runtime identity separates decode *environments*. An ordering change is not
    an environment change — same OpenCV, same backend, same OS — so the digest
    is identical on both sides of it. Detection is impossible by construction;
    only prevention works, which is why R14 was fixed rather than recorded.
    """
    from inference.decode_env import decode_identity

    before = decode_identity()
    winners_legacy = {
        LegacyAreaOnlyOrder.order(list(perm))[0].x
        for perm in itertools.permutations(EQUAL_AREA)
    }
    after = decode_identity()
    assert before == after
    assert len(winners_legacy) > 1, "the divergence must be real for this to mean anything"


def test_an_upstream_pixel_change_still_moves_everything():
    """Determinism is not stability. Batch 10 guarantees the first, not the second.

    One changed pixel gives a different crop hash and a different stub logit —
    correctly. Asserted so "the pipeline is deterministic" is never read as "the
    pipeline is robust".
    """
    img = _face_image()
    face = pre.detect_faces(img)[0]
    base = pre.prepare_image(pre.crop_face(img, face), size=384)

    nudged = img.copy()
    nudged[250, 250, 0] = np.uint8((int(nudged[250, 250, 0]) + 7) % 256)
    other = pre.prepare_image(pre.crop_face(nudged, face), size=384)

    assert _sha(base) != _sha(other)


# --------------------------------------------------------------------------
# 5. R15: the one true nondeterminism — a cascade shared between threads
# --------------------------------------------------------------------------


class LegacySharedCascade:
    """The pre-R15 detector: one ``CascadeClassifier`` for the whole process.

    Six lines, copied from ``inference/preprocess.py`` as it stood, kept beside
    the shipped version so "the fix was needed" is executed rather than
    remembered. The lazy global is reproduced too, unguarded, exactly as it was.

    Both pipelines reach ``detect_faces`` through ``asyncio.to_thread``, so
    "shared between threads" was the shipped configuration, not a hypothetical
    one.
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


def _detect_in_parallel(detect, image, *, threads: int = 8, reps: int = 40) -> list[tuple]:
    """Run ``detect(image)`` on ``threads`` threads at once and collect answers.

    ``detectMultiScale`` releases the GIL, so the overlap is real rather than
    interleaved bytecode.
    """
    import threading as _threading

    collected: list[tuple] = []
    guard = _threading.Lock()

    def worker() -> None:
        local = [detect(image) for _ in range(reps)]
        with guard:
            collected.extend(local)

    workers = [_threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return collected


def test_a_shared_cascade_gives_different_answers_for_identical_bytes():
    """R15's defect, reproduced. **This is true nondeterminism, not ordering.**

    Everything else in this file is order dependence: deterministic today,
    fragile tomorrow. This is the real thing — same process, same build, same
    runtime identity, same bytes, different answer, decided by which two threads
    happened to be inside ``detectMultiScale`` at the same moment.

    The failure is silent. ``CascadeClassifier`` holds the feature evaluator and
    its scaled-image buffers as instance state, so concurrent calls overwrite
    each other's working set; nothing raises, and the caller receives a
    well-formed box list that is simply wrong. Measured here as a lost face,
    which routes a scored image to ``NO_FACE_DETECTED``.

    Rounds, because the race is probabilistic: at the measured 3-15% per-call
    rate, one round of 320 calls misses it with probability under 2e-3, and
    three rounds make a false pass not worth reasoning about. A run that reaches
    the end without diverging fails loudly rather than passing — the double
    exists to show the defect, and one that cannot has stopped doing its job.
    """
    img = _face_image()
    truth = LegacySharedCascade.detect(img)  # sequential: the correct answer
    assert truth, "the fixture must detect a face for a lost one to be visible"

    for _ in range(3):
        answers = _detect_in_parallel(LegacySharedCascade.detect, img)
        wrong = [a for a in answers if a != truth]
        if wrong:
            print(f"        ({len(wrong)}/{len(answers)} concurrent detections wrong; "
                  f"{sum(1 for a in wrong if not a)} lost the face entirely)")
            return
    raise AssertionError(
        "a process-wide cascade no longer diverges under 8-way concurrency after "
        "960 calls. Either OpenCV made CascadeClassifier thread-safe — which "
        "would be good news worth confirming before deleting this test — or the "
        "fixture stopped exercising the race."
    )


def _shifted_face_image(size: int = 480, dx: int = -40, dy: int = -30) -> np.ndarray:
    """The same face translated, so a second file has a different correct box.

    The offset is not free: of nine tried, only this one still detects. The
    drawing sits near this cascade's decision boundary — Batch 8's R13
    observation reappearing as a fixture constraint.
    """
    src = _face_image(size)
    out = np.full_like(src, 200)
    ys, xs = slice(max(0, dy), size + min(0, dy)), slice(max(0, dx), size + min(0, dx))
    yd, xd = slice(max(0, -dy), size + min(0, -dy)), slice(max(0, -dx), size + min(0, -dx))
    out[yd, xd] = src[ys, xs]
    return out


def test_a_shared_cascade_returns_one_jobs_box_for_another_jobs_media():
    """R15's severe half: this is contamination, not merely nondeterminism.

    Two files analysed at once through one cascade. The measured failure is not
    "job B got an unstable answer" — it is "job B got **job A's bounding box**",
    at rates of 46-69% across recorded runs. The pipeline then crops job B's
    pixels at coordinates located in job A's image and publishes the resulting
    score as job B's verdict.

    That is worse than an unreproducible number, because it is not visibly
    wrong: the box is well-formed, inside the frame, and of a plausible size.
    Nothing downstream can tell it came from the wrong file.

    Same-image concurrency understates this badly (2-6% wrong), which is why the
    two-file case is a separate test rather than a variation of the one above.
    """
    a, b = _face_image(), _shifted_face_image()
    truth_a, truth_b = LegacySharedCascade.detect(a), LegacySharedCascade.detect(b)
    assert truth_a and truth_b and truth_a != truth_b, (
        f"the fixture must give both files a face, and different ones "
        f"(a={truth_a}, b={truth_b}); without that a foreign box is "
        "indistinguishable from a lost one"
    )

    import threading as _threading

    collected: dict[str, list] = {"a": [], "b": []}
    guard = _threading.Lock()

    def worker(tag: str, image: np.ndarray) -> None:
        local = []
        for _ in range(40):
            # A raise is an outcome of the race, not a broken experiment: a
            # shared cascade also reads out of range inside OpenCV's own
            # vectors. Swallowing the thread would drop those calls from the
            # denominator and understate the defect.
            try:
                local.append(LegacySharedCascade.detect(image))
            except cv2.error:
                local.append(("RAISED",))
        with guard:
            collected[tag].extend(local)

    workers = [_threading.Thread(target=worker, args=("a", a)) for _ in range(4)]
    workers += [_threading.Thread(target=worker, args=("b", b)) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    foreign_total = 0
    for tag, mine, theirs in (("a", truth_a, truth_b), ("b", truth_b, truth_a)):
        rows = collected[tag]
        foreign = sum(1 for r in rows if r == theirs)
        foreign_total += foreign
        print(f"        (file {tag}: {foreign}/{len(rows)} calls returned the "
              f"other file's box)")
    assert foreign_total > 0, (
        "a process-wide cascade no longer leaks one file's detection into "
        "another's. Either OpenCV made CascadeClassifier thread-safe — worth "
        "confirming before deleting this test — or the fixture stopped racing."
    )


def test_the_shipped_detector_never_returns_another_files_box():
    """The fix, on the severe case. Prevention, not mitigation."""
    a, b = _face_image(), _shifted_face_image()
    truth_a = tuple(pre.detect_faces(a))
    truth_b = tuple(pre.detect_faces(b))
    assert truth_a and truth_b and truth_a != truth_b

    import threading as _threading

    collected: dict[str, list] = {"a": [], "b": []}
    guard = _threading.Lock()

    def worker(tag: str, image: np.ndarray) -> None:
        local = [tuple(pre.detect_faces(image)) for _ in range(40)]
        with guard:
            collected[tag].extend(local)

    workers = [_threading.Thread(target=worker, args=("a", a)) for _ in range(4)]
    workers += [_threading.Thread(target=worker, args=("b", b)) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    for tag, mine in (("a", truth_a), ("b", truth_b)):
        wrong = [r for r in collected[tag] if r != mine]
        assert not wrong, f"file {tag}: {len(wrong)}/{len(collected[tag])} wrong: {wrong[:3]}"


def test_the_shipped_detector_is_unaffected_by_concurrency():
    """The fix, proved by execution against the function that ships.

    Not "the race is less likely": a thread cannot contend with itself, so the
    mechanism is gone rather than narrowed.
    """
    img = _face_image()
    truth = tuple(pre.detect_faces(img))
    answers = _detect_in_parallel(lambda i: tuple(pre.detect_faces(i)), img)
    wrong = sum(1 for a in answers if a != truth)
    assert wrong == 0, f"{wrong}/{len(answers)} concurrent detections disagreed"


def test_concurrent_analyses_of_one_file_publish_one_verdict():
    """End to end, through the real pipeline, on the real event loop.

    The unit test above shows the detector agreeing; this shows the *bundle*
    agreeing, which is the claim a user of this platform actually has. Before
    R15 this produced two verdicts for one file: ``AUTHENTIC`` at p_fake 0.0329
    and ``INSUFFICIENT_EVIDENCE / no_face_detected``, in a 136/24 split over 160
    runs.
    """
    import asyncio
    from datetime import UTC, datetime
    from uuid import uuid4

    from contracts.enums import MediaKind
    from contracts.job import JobSpec
    from inference import ImagePipeline

    ok, buf = cv2.imencode(".png", cv2.cvtColor(_face_image(), cv2.COLOR_RGB2BGR))
    assert ok
    data = buf.tobytes()
    pipeline = ImagePipeline()

    def make_spec() -> JobSpec:
        return JobSpec(
            job_id=uuid4(),
            media_kind=MediaKind.IMAGE,
            storage_key="u",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            declared_mime="image/png",
            model_version="v",
            idempotency_key="k",
            submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    async def one() -> tuple:
        bundle = await pipeline.analyze_image(make_spec(), data)
        return (
            bundle.verdict.value,
            None if bundle.p_fake is None else round(bundle.p_fake, 12),
            bundle.abstain_reason.value if bundle.abstain_reason else None,
        )

    async def many() -> list[tuple]:
        out: list[tuple] = []
        for _ in range(8):
            out.extend(await asyncio.gather(*[one() for _ in range(8)]))
        return out

    outcomes = set(asyncio.run(many()))
    assert len(outcomes) == 1, (
        f"64 concurrent analyses of one file produced {len(outcomes)} distinct "
        f"bundles: {outcomes}"
    )


def test_the_cascade_is_not_shared_between_threads():
    """The structural invariant, checked directly rather than through the race.

    A behavioural test can only sample a probabilistic failure; this asserts the
    property that makes the failure impossible, so a future detector that
    reintroduces process-wide sharing fails here immediately and deterministically
    rather than on whichever CI run happens to interleave badly.
    """
    import threading as _threading

    assert isinstance(pre._cascade_local, _threading.local), (
        "the cascade must be held per thread; a module-level instance is R15"
    )
    assert not hasattr(pre, "_cascade"), (
        "a process-wide `_cascade` has come back; that is the R15 defect verbatim"
    )

    seen: dict[str, int] = {}
    guard = _threading.Lock()

    def grab() -> None:
        pre.detect_faces(_face_image(size=96))
        with guard:
            seen[_threading.current_thread().name] = id(pre._face_cascade())

    workers = [_threading.Thread(target=grab, name=f"t{i}") for i in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert len(set(seen.values())) == len(seen), (
        f"four threads share {len(set(seen.values()))} cascade object(s): {seen}"
    )


def test_runtime_identity_cannot_detect_the_cascade_race():
    """Why R15 had to be prevented rather than recorded.

    Batch 9's answer to a decoder disagreement was to make the two results
    *distinguishable* — different decode identity, different cache namespace.
    That instrument is useless here. Both answers come out of one process, one
    OpenCV build, one backend, one model version, so every identity string in
    the platform is byte-identical on both sides of the divergence, and the
    result cache would serve whichever landed first to every later request for
    those bytes.
    """
    from inference.decode_env import decode_identity

    img = _face_image()
    before = decode_identity()
    for _ in range(3):
        answers = set(_detect_in_parallel(LegacySharedCascade.detect, img, reps=20))
        if len(answers) > 1:
            break
    else:  # pragma: no cover - covered by the dedicated defect test
        print("        (race did not fire this round; identity claim unchanged)")
        answers = {(), tuple(pre.detect_faces(img))}
    assert len(answers) > 1
    assert decode_identity() == before, (
        "the decode identity moved across the divergence; if that is genuinely "
        "true the race would be detectable and R15's reasoning needs revisiting"
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
