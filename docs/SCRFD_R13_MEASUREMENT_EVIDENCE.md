# R13 Acceptance Measurement — SCRFD-10G-KPS (Evidence Record)

**Date run:** 2026-10-17. **Verdict: NO-GO.** SCRFD does not clear R13's
acceptance criterion, in either of two independently seeded runs. The shipped
default detector stays Haar; no cutover; no version axis moves.

## 1. Criterion

`REPRODUCIBILITY_AUDIT.md`, R13: *"Under ±4 LSB uniform noise on real
photographs, across ≥ 200 images: no face lost, and the box short edge moving
≤ 1%. Amplification below 10×, from the measured 310×."* Amplification gated
on the 95% cluster-bootstrap CI's upper bound
(`docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §4.4).

## 2. Results

| Check | Seed 0 | Seed 1 | Gate |
|---|---|---|---|
| Images (N) × realisations (K) | 300 × 11 (3,201 trials) | 300 × 11 (3,201 trials) | ≥ 200 → PASS |
| Faces lost under noise | 2 | 3 | 0 → **FAIL** |
| Max short-edge move | 27.27% (CI 8.50–27.27%) | 25.40% (CI 9.00–25.40%) | ≤ 1% → **FAIL** |
| Amplification, documented pooled statistic | 1.00× (CI 1.00–1.00) | 1.00× (CI 1.00–1.00) | < 10× → PASS, **not informative** (§4.1) |
| Amplification, within-image (diagnostic) | 26.74× (CI 14.15–38.75) | 25.57× (CI 12.88–38.43) | would FAIL |
| Clean images with no detection | 9 | 9 | tracked, excluded |
| Top-face identity switches | 1.13% of trials | 1.00% of trials | diagnostic |
| Same-face short-edge move | median 0%, p99 5.5%, 13.2% > 1% | median 0%, p99 6.0%, 12.8% > 1% | diagnostic |

The two seeds agree on every check's outcome and their CIs overlap
substantially (plan §5.4 cross-check satisfied).

**Latency** (CPU, `CPUExecutionProvider`, 640×640 letterbox, same machine and
images): SCRFD median 247.9 / 252.4 ms, p95 302.1 / 297.3 ms over 3,501 calls
per run; Haar baseline median 213.5 / 208.4 ms, p95 451.6 / 476.5 ms. SCRFD is
~17% slower at the median and has a tighter tail.

## 3. Diagnosis

1. **Face loss is one near-threshold face.** All five lost trials are a
   single 84×90 face in one image. A threshold sweep on the clean image finds it
   at `confidence_threshold` ≤ 0.50 and not at ≥ 0.52, so its score sits in
   [0.50, 0.52); ±4 LSB noise pushes it below the shipped 0.5 floor in 2–3 of
   11 draws.
2. **Box movement is mostly one-pixel quantisation.** Boxes are integer
   pixels; for a face with short edge under 100 px a 1-px change is already
   > 1%. Same-face movement by short edge (seed 0; seed 1 matches within ~2
   points):

   | Short edge | Trials | > 1% | 0 px | 1 px | ≥ 2 px |
   |---|---|---|---|---|---|
   | 32–64 px | 803 | 20.7% | 79.3% | 18.4% | 2.2% |
   | 64–128 px | 834 | 21.7% | 70.0% | 25.3% | 4.7% |
   | 128–256 px | 792 | 5.6% | 65.2% | 28.7% | 6.2% |
   | ≥ 256 px | 770 | 4.0% | 41.8% | 40.1% | 18.1% |

   Read literally on integer boxes, "≤ 1%" demands zero-pixel movement for
   every face under 100 px. Separately, real multi-pixel outliers exist (max
   16–18 px), so even a pixel-tolerant reading would not pass on maximum.
3. **Most amplification is selection, not localisation.** Excluding trials
   where noise changed *which* face ranks first, within-image amplification
   falls from 26.7× / 25.6× to **10.5× / 11.6×**. Localisation jitter alone is
   ~11×: about 28× better than Haar's 310×, and still above 10×.

## 4. Methodology findings made during the run

1. **The documented amplification statistic is blind on a real corpus.** It
   pools raw scores across images, so between-image variance (measured at
   100.0% / 99.9% of the fixed / free pooled spread) enters both sides and pins
   the ratio at 1.00× whatever the detector does. The within-image statistic
   (one-way ANOVA within-group spread) is the like-for-like generalisation of
   the original single-fixture 310× and is reported alongside; both are in
   `tests/scrfd_detector_study.py` with tests demonstrating the blind spot. The
   gate itself was not changed — re-specifying it is a decision for the
   project owner (§6).
2. **The study now tracks the face the pipeline scores.** `FaceDetector.detect`
   promises no order and SCRFD emits confidence order, so the study previously
   compared the most *confident* face before and after noise. It now applies
   `face_order_key`, as `detect_faces` does. The 186-image pilot predated this
   fix and is superseded; the two runs above include it.

## 5. Provenance and reproduction

| Input | Identity |
|---|---|
| Model | `det_10g.onnx` from InsightFace `cheetah_l.zip` (model-zoo GitHub release). SHA-256 `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`, equal to the value in the package's upstream `manifest.json`; CRC32-checked on extraction. The archive-level digest GitHub publishes (`sha256:df81a058…bc8d`) was **not** checked: the download stalled at ~62 MB of 212 MB and the detector member (bytes 35.5–52.3 MB) was read from that partial file. Completing the download and checking it remains open. Dynamic input `[1,3,?,?]`, opset 11, 9 outputs (score/bbox/kps × 3 strides) |
| Model licence | Package `MODEL.LICENSE`: `cheetah_l-public-v1`, grant `non-commercial`, signed (signature not verifiable — no published key). Used locally for non-commercial evaluation only; weights not committed or redistributed |
| Corpus source | WIDER FACE validation split (CC BY-NC-ND 4.0), Hugging Face `CUHK-CSE/wider_face` revision `db171f1b`. `WIDER_val.zip` SHA-256 `f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a` and `wider_face_split.zip` SHA-256 `c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c`, both equal to the published values; `wider_face_val_bbx_gt.txt` SHA-256 `5ca83488f72758ca4826c35212610a8abf04a7af7b17eb407d6284007baba3b6` |
| Corpus | `tests/scrfd_corpus.py`, n = 300, seed 0, min face 32 px; manifest digest `c5907efee86cbb2f518bc9cded15aa1cbff2247d27cc53469c56707bb6abfb27`. Coverage: face-size buckets 76/76/74/74, single/multi 150/150, extreme illumination 33, atypical pose 49, heavy blur 41 |
| Detector config | `scrfd-10g-kps`, `confidence_threshold` 0.5, `nms_iou_threshold` 0.4 |
| Environment | Python 3.12.0, numpy 1.26.4, OpenCV 4.8.1, onnxruntime 1.30.0, Windows 11 |
| Code | the commit that adds this record: `inference/preprocess.py` and `tests/scrfd_detector_study.py` are byte-identical to what ran (only documents changed after the runs) |
| Evidence files (kept outside the repository) | `r13_seed0.json` `d29d8c12…04f5`, `r13_seed1.json` `7b65307a…3158`, `r13_seed0.txt` `07bc27ca…cdbf`, `r13_seed1.txt` `183b1623…d7d6` |

```bash
python tests/scrfd_corpus.py --images <WIDER_val/images> \
  --annotations <wider_face_val_bbx_gt.txt> --out <corpus_r13>
for seed in 0 1; do
  DFP_SCRFD_MODEL_PATH=<det_10g.onnx> DFP_SCRFD_STUDY_PHOTOS=<corpus_r13> \
  DFP_SCRFD_STUDY_SEED=$seed DFP_SCRFD_STUDY_JSON=r13_seed$seed.json \
  python tests/scrfd_detector_study.py
done
```

Licence constraints on publication: aggregate figures (this document) only.
No WIDER images, crops or noised variants are committed or published
(NoDerivatives); no model weights are redistributed.

## 6. Options — decisions for the project owner, none taken here

1. **Keep Haar (current state).** R13 stays Open with this measurement
   attached.
2. **Re-specify the amplification gate** to the within-image statistic. It is
   the honest generalisation of the original measurement; on this evidence it
   would still fail (~26×, or ~11× excluding top-face switches).
3. **Re-specify the box criterion** in pixels (e.g. ≤ 1 px) or exclude faces
   below a size. This changes the bar; it must be decided before, not after, a
   fresh measurement. Multi-pixel outliers (≤ 18 px) would still need
   explaining.
4. **Change the configuration and re-measure on a fresh, disjoint corpus**
   (e.g. a different `confidence_threshold`, or tie-breaking for near-equal
   faces to reduce top-face switches). Tuning against this corpus would fit
   the test.

R13 remains **Open**.
