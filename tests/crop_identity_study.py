"""R25 — crop geometry inputs outside every identity string. Batch 21 investigation.
Standalone, read-only, changes nothing.

    python tests/crop_identity_study.py

Same pattern as ``decode_study.py``, ``crop_geometry_study.py``,
``decode_identity_completeness_study.py``. Produced the numbers Batch 21's
architecture report rests on.

*** R25 IS CLOSED (Batch 22). *** Sections 1-3 below still measure the OPEN
state (the hole, its magnitude, and R28) as the evidence B22 was built against;
after B22 the section-1 "identical identity" lines now print False. Section 4
reads the SHIPPED post-B22 identity from live code. See ``docs/CROP_IDENTITY.md``,
``docs/CROP_IDENTITY_B22_PLAN.md`` and ``tests/test_crop_identity.py``.

THE QUESTION
------------
Batch 15 recorded R25: three preprocessing knobs decide what pixels the model
sees and none of them is in any identity string —

  1. ``ModelSpec.input_size``      — the square resize target (whole field of view)
  2. the interpolation rule         — ``INTER_AREA if h>size or w>size else INTER_LINEAR``
  3. the crop clipping/centring rule — window ∩ image, undocumented, unnamed

Batch 15 deferred the fix to the SCRFD migration on the grounds that closing it
"spends the cache generation twice". The pre-Batch-21 feasibility study argued
that reasoning holds for knob 3 only. This study re-derives the hole from shipped
code and measures each knob independently.

WHAT THIS DOES NOT DO
---------------------
It does not change any identity composition. E4/E5 *project* the post-fix
identity by calling ``fingerprint`` with the extra keys, to show the migration
shape — they do not patch ``models.py`` or ``preprocess.py``.

DOUBLES
-------
    ScriptedBox        a square Haar-shaped box at a chosen position
    ProjectedWeightsId  weights_identity with input_size folded in  (E4)
    ProjectedPolicyId   FaceSelectionPolicy.identity with interp+clip folded in (E4)
Everything else is shipped code.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # Windows consoles default to cp1252; this study prints a few non-ASCII glyphs.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.version import (  # noqa: E402
    compose_model_version,
    compose_runtime_version,
    fingerprint,
    model_identity_of,
    parse_model_version,
)
from inference import preprocess as pre  # noqa: E402
from inference.decode_env import decode_identity  # noqa: E402
from inference.models import ModelSpec, StubBackend  # noqa: E402
from inference.preprocess import FaceBox  # noqa: E402

POLICY = pre.default_face_selection()
CAPS = pre.DEFAULT_CAPS
MARGIN = POLICY.crop_margin


def head(n: int, title: str, q: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n   {q}\n{'=' * 78}")


def _face_tile(size: int = 480) -> np.ndarray:
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _canvas_with_face(w: int, h: int, box: FaceBox) -> np.ndarray:
    """A grey canvas with the face tile scaled into ``box``."""
    img = np.full((h, w, 3), 128, np.uint8)
    tile = cv2.resize(_face_tile(), (box.width, box.height), interpolation=cv2.INTER_AREA)
    img[box.y : box.y + box.height, box.x : box.x + box.width] = tile
    return img


_STUB = StubBackend(ModelSpec(name="s", version="stub-0", backend="stub"))


def _stub_logit(tensor: np.ndarray) -> float:
    out = _STUB.score(tensor.astype(np.float32))
    first = next(iter(out))
    return float(out[first][0])


def _p_from_logit(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# The shipped operating band, for the decisions-changed table.
_BAND = ModelSpec(name="s", version="stub-0", backend="stub").threshold
BAND_LO, BAND_HI = _BAND.band_lo, _BAND.band_hi


def _decisions_changed(dp: float, n: int = 200_000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    pop = rng.beta(2, 2, n)

    def region(p):
        return np.where(p >= BAND_HI, 2, np.where(p >= BAND_LO, 1, 0))

    base = region(pop)
    moved = region(np.clip(pop + dp, 0.0, 1.0))
    return float((moved != base).mean())


# --------------------------------------------------------------------------


def s1_the_hole() -> None:
    head(1, "The hole (pre-Batch-22)",
         "different tensor per knob — the identity that did NOT move before B22 (see s4)")

    spec = ModelSpec(name="stub-image", version="stub-0", backend="stub")
    ev = (
        spec.with_gate(CAPS.gate_fingerprint)
        .with_face_selection(POLICY.identity)
        .effective_version
    )
    rv = compose_runtime_version(model=ev, decode=decode_identity())
    print(f"  effective_version : {ev}")
    print(f"  runtime version   : {rv}")
    print(f"  content-cache key : by-content/<sha256>/{rv}.json\n")

    # ---- knob 1: input_size --------------------------------------------
    box = FaceBox(x=140, y=120, width=200, height=200)
    img = _canvas_with_face(480, 480, box)
    crop = pre.crop_face(img, box, policy=POLICY)
    t384 = pre.prepare_image(crop, size=384)
    t224 = pre.prepare_image(crop, size=224)
    l384, l224 = _stub_logit(t384), _stub_logit(t224)
    print("  knob 1  ModelSpec.input_size 384 vs 224")
    print(f"    tensor shape      {t384.shape} vs {t224.shape}")
    print(f"    stub p_fake       {_p_from_logit(l384):.6f} vs {_p_from_logit(l224):.6f}"
          f"   |dp|={abs(_p_from_logit(l384) - _p_from_logit(l224)):.6f}")
    s2 = ModelSpec(name="stub-image", version="stub-0", backend="stub", input_size=224)
    print(f"    weights_identity  384: {spec.weights_identity}")
    print(f"                      224: {s2.weights_identity}")
    print(f"    pre-B22 both were stub-0-w7f45fada; B22 folded input_size into -w, so")
    print(f"    weights_identity now differs: {spec.weights_identity != s2.weights_identity};"
          f"  effective_version differs: {spec.effective_version != s2.effective_version}")

    # ---- knob 2: interpolation rule -----------------------------------
    # A mixed-scale crop: one axis larger than 384, one smaller.
    mixed = _canvas_with_face(500, 300, FaceBox(x=150, y=50, width=200, height=200))
    area = cv2.resize(mixed, (384, 384), interpolation=cv2.INTER_AREA)
    linear = cv2.resize(mixed, (384, 384), interpolation=cv2.INTER_LINEAR)
    dpx = float(np.abs(area.astype(int) - linear.astype(int)).mean())
    la = _stub_logit(area[np.newaxis].astype(np.float32))
    ll = _stub_logit(linear[np.newaxis].astype(np.float32))
    print("\n  knob 2  interpolation rule  INTER_AREA vs INTER_LINEAR (500x300 -> 384)")
    print(f"    mean |pixel delta|  {dpx:.3f} / 255")
    print(f"    stub p_fake         {_p_from_logit(la):.6f} vs {_p_from_logit(ll):.6f}"
          f"   |dp|={abs(_p_from_logit(la) - _p_from_logit(ll)):.6f}")
    print("    the rule and the choice appear in FaceSelectionPolicy.identity: "
          f"{'interp' in str(POLICY.to_dict())}")

    # ---- knob 3: clip rule -------------------------------------------
    # Square box flush with the right border -> shipped crop is non-square.
    rb = FaceBox(x=280, y=140, width=200, height=200)
    rimg = _canvas_with_face(480, 480, rb)
    clipped = pre.crop_face(rimg, rb, policy=POLICY)

    def _pad_to_square(image, box, margin=MARGIN):
        h, w = image.shape[:2]
        side = int(max(box.width, box.height) * margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0, y0 = cx - side // 2, cy - side // 2
        x1, y1 = x0 + side, y0 + side
        inner = image[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        pl, pt = max(0, -x0), max(0, -y0)
        prr, pb = max(0, x1 - w), max(0, y1 - h)
        if pl or prr or pt or pb:
            inner = cv2.copyMakeBorder(inner, pt, pb, pl, prr, cv2.BORDER_REPLICATE)
        return inner

    padded = _pad_to_square(rimg, rb)
    tc = pre.prepare_image(clipped, size=384)
    tp = pre.prepare_image(padded, size=384)
    lc, lp = _stub_logit(tc), _stub_logit(tp)
    print("\n  knob 3  clip rule  intersect-image vs pad-to-square (right-border box)")
    print(f"    clipped crop shape {clipped.shape[:2]}  (anisotropy "
          f"{max(clipped.shape[0]/clipped.shape[1], clipped.shape[1]/clipped.shape[0]):.4f})")
    print(f"    padded  crop shape {padded.shape[:2]}")
    print(f"    stub p_fake        {_p_from_logit(lc):.6f} vs {_p_from_logit(lp):.6f}"
          f"   |dp|={abs(_p_from_logit(lc) - _p_from_logit(lp)):.6f}")
    print(f"    crop_shape names only how 'side' is derived: {POLICY.crop_shape!r}")
    print("    -> pre-B22 both border behaviours were honestly 'square-longest-edge' "
          "and no identity separated them;")
    print(f"       B22 added crop_clip_rule={POLICY.crop_clip_rule!r} to "
          "FaceSelectionPolicy.identity (a label closure — see s4).")


def s2_magnitude() -> None:
    head(2, "Magnitude", "decisions changed against the shipped band (stub = upper bound)")
    print(f"  band [{BAND_LO:.6f}, {BAND_HI:.6f})  width {BAND_HI - BAND_LO:.6f}\n")

    box = FaceBox(x=140, y=120, width=200, height=200)
    img = _canvas_with_face(480, 480, box)
    crop = pre.crop_face(img, box, policy=POLICY)

    # knob 1
    dp1 = abs(_p_from_logit(_stub_logit(pre.prepare_image(crop, size=384)))
              - _p_from_logit(_stub_logit(pre.prepare_image(crop, size=224))))
    # knob 2
    mixed = _canvas_with_face(500, 300, FaceBox(x=150, y=50, width=200, height=200))
    dp2 = abs(_p_from_logit(_stub_logit(
        cv2.resize(mixed, (384, 384), interpolation=cv2.INTER_AREA)[np.newaxis].astype(np.float32)))
        - _p_from_logit(_stub_logit(
        cv2.resize(mixed, (384, 384), interpolation=cv2.INTER_LINEAR)[np.newaxis].astype(np.float32))))
    # knob 3
    rb = FaceBox(x=280, y=140, width=200, height=200)
    rimg = _canvas_with_face(480, 480, rb)
    clipped = pre.crop_face(rimg, rb, policy=POLICY)
    h, w = rimg.shape[:2]
    side = int(max(rb.width, rb.height) * MARGIN)
    cx, cy = rb.x + rb.width // 2, rb.y + rb.height // 2
    x0, y0 = cx - side // 2, cy - side // 2
    inner = rimg[max(0, y0):min(h, y0 + side), max(0, x0):min(w, x0 + side)]
    padded = cv2.copyMakeBorder(inner, 0, 0, 0, max(0, x0 + side - w), cv2.BORDER_REPLICATE)
    dp3 = abs(_p_from_logit(_stub_logit(pre.prepare_image(clipped, size=384)))
              - _p_from_logit(_stub_logit(pre.prepare_image(padded, size=384))))

    print("  knob                         |dp| (stub)   decisions changed")
    for name, dp in (("1 input_size 384->224", dp1),
                     ("2 interpolation rule ", dp2),
                     ("3 clip rule (border) ", dp3)):
        print(f"    {name}     {dp:.4f}          {_decisions_changed(dp):6.2%}")
    print("\n  Reference points (surrogate, prior batches):")
    print(f"    R12 cross-decoder 0.0050  -> {_decisions_changed(0.0050):6.2%}")
    print(f"    R13 crop movement 0.0910  -> {_decisions_changed(0.0910):6.2%}")
    print("  The stub hashes bytes, so its |dp| is the decorrelated upper bound;")
    print("  the honest per-knob figure needs the smooth surrogate and real weights.")


def s3_the_or_bug() -> None:
    head(3, "The 'or' bug", "does the rule area-filter an axis it is enlarging?")
    src = _canvas_with_face(500, 300, FaceBox(x=150, y=50, width=200, height=200))
    size = 384
    h, w = src.shape[:2]
    rule = "INTER_AREA" if (h > size or w > size) else "INTER_LINEAR"
    print(f"  crop {w}x{h} -> {size}x{size}")
    print(f"    width  {w} > {size}: {w > size}   height {h} > {size}: {h > size}")
    print(f"    rule picks: {rule}   (applied to BOTH axes, including the "
          f"{'height' if h < size else 'width'} axis being ENLARGED {h}->{size})")
    # what INTER_AREA vs INTER_LINEAR does on the upsampled axis alone
    up_area = cv2.resize(src, (w, size), interpolation=cv2.INTER_AREA)
    up_linear = cv2.resize(src, (w, size), interpolation=cv2.INTER_LINEAR)
    d = float(np.abs(up_area.astype(int) - up_linear.astype(int)).mean())
    print(f"    on the enlarged axis only: mean |AREA - LINEAR| = {d:.3f}/255")
    print("    preprocess.resize docstring: 'INTER_AREA when shrinking' — but the")
    print("    'or' makes it fire whenever EITHER axis exceeds the target. R28,")
    print("    DEFERRED by Batch 22 (fixing the predicate moves pixels). The B22")
    print("    interpolation_rule name records this behaviour verbatim.")


def s4_the_shipped_identity() -> None:
    head(4, "The shipped identity (Batch 22)", "what the knobs moved, read from live code")
    spec = ModelSpec(name="stub-image", version="stub-0", backend="stub")
    bound = spec.with_gate(CAPS.gate_fingerprint).with_face_selection(POLICY.identity)

    print("  knob 1  input_size -> weights_identity (folded into the -w fingerprint,")
    print("          NOT a new -i token, so parse_model_version is untouched)")
    print(f"    pre-B22 : stub-0-w7f45fada")
    print(f"    shipped : {spec.weights_identity}")
    print(f"    a 224 build: {ModelSpec(name='x', version='stub-0', backend='stub', input_size=224).weights_identity}")

    print("\n  knobs 2+3  interpolation_rule + crop_clip_rule -> FaceSelectionPolicy.identity")
    print(f"    interpolation_rule = {POLICY.interpolation_rule!r}")
    print(f"    crop_clip_rule     = {POLICY.crop_clip_rule!r}")
    print(f"    pre-B22 .f. : 19559387")
    print(f"    shipped .f. : {POLICY.identity}")

    rt = compose_runtime_version(model=bound.effective_version, decode=decode_identity())
    parts = parse_model_version(rt)
    print(f"\n  effective_version : {bound.effective_version}")
    print(f"  STUB_ENGINE_VERSION: {rt}")
    print(f"    parse round-trips              : {parts.compose() == bound.effective_version}")
    print(f"    model_identity_of(runtime) ok  : {model_identity_of(rt) == bound.effective_version}")
    print(f"    calibration/gate unmoved       : "
          f"{parts.calibration == '739c3fec' and parts.gate == 'ae232cba'}")


def s5_shipped_pixels_unchanged() -> None:
    head(5, "Shipped pixels", "did recording the knobs in the identity change any output?")
    box = FaceBox(x=140, y=120, width=200, height=200)
    img = _canvas_with_face(480, 480, box)
    crop = pre.crop_face(img, box, policy=POLICY)
    t = pre.prepare_image(crop, size=384)
    print(f"  crop shape {crop.shape[:2]}   tensor {t.shape}   dtype {t.dtype}")
    print("  Batch 22 records input_size / interpolation_rule / crop_clip_rule in the")
    print("  identity; it changes no line of resize(), crop_face() or prepare_image().")
    print("  Byte-identical output -> no verdict moves -> no recalibration. Cost is")
    print("  exactly one cache generation (B14/B20 profile). Proven by")
    print("  tests/test_crop_identity.py::test_the_shipped_pixels_did_not_move.")


def s6_video_shares_knobs() -> None:
    head(6, "Video path", "which knobs does the video path share?")
    src = Path(__file__).resolve().parent.parent / "inference" / "preprocess.py"
    text = src.read_text(encoding="utf-8")
    vp = (Path(__file__).resolve().parent.parent / "inference" / "video_pipeline.py").read_text("utf-8")
    print(f"  extract_frames resizes each frame with resize(): "
          f"{'resize(rgb, target_size)' in text}")
    print(f"  video_pipeline passes target_size=self._spec.input_size: "
          f"{'target_size=self._spec.input_size' in vp}")
    print("  -> knob 1 (input_size) and knob 2 (the resize interpolation rule) are")
    print("     shared with the image path. Knob 3 (crop clip) is image-only —")
    print("     the video path has no crop (R18/R24).")
    print("  NOTE: R25 closed the IMAGE crop path only. The video path's own")
    print("  frame-selection / pooling parameters (n_frames, min_usable_frames,")
    print("  topk depth, the 'uniform' sampling label) were R29 — CLOSED in Batch")
    print("  23 via VideoSelectionPolicy (preprocess.py). selection_identity(face,")
    print("  video) now feeds the .f. segment from BOTH pipelines.")
    from inference import preprocess as _pre
    _f, _v = _pre.default_face_selection(), _pre.default_video_selection()
    print(f"    video_selection_identity() = {_pre.video_selection_identity()}")
    print(f"    selection_identity(face, video) [the .f. segment] = "
          f"{_pre.selection_identity(_f, _v)}")
    print("  See REPRODUCIBILITY_AUDIT.md Batch 23, docs/VIDEO_SELECTION_IDENTITY.md,")
    print("  and tests/test_video_selection_identity.py.")


def main() -> int:
    print(__doc__.strip())
    for fn in (s1_the_hole, s2_magnitude, s3_the_or_bug,
               s4_the_shipped_identity, s5_shipped_pixels_unchanged, s6_video_shares_knobs):
        fn()
    print("\n" + "=" * 78)
    print("R25 CLOSED by Batch 22: the 3 crop-geometry knobs now reach the model")
    print("identity — input_size folded into weights_identity (functionally threaded),")
    print("interpolation_rule + crop_clip_rule in FaceSelectionPolicy.identity (label")
    print("closures pinned by re-derivation tests). Image crop path only; the video")
    print("frame-selection / pooling parameters were R29 — CLOSED in Batch 23. s4")
    print("reads the shipped post-B22 identity from live code.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
