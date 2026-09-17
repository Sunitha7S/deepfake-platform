# Video selection identity — R29, Batch 23

> **Shipped state after B23:** model identity
> `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c`, engine literal
> `…f.e629642c.d.a97ccb68`. The `.f.` segment is now the **composite** selection
> identity: `fingerprint({"face": <face policy>, "video": <video policy>})`.
> `default_face_selection().identity` is **unchanged** (`251d22e9`);
> `default_video_selection().identity` is `db8e6a0d`; the composite is `e629642c`.
> One cache generation, both media namespaces. No `CONTRACT_VERSION` /
> `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI-structural move.
> Output byte-identical. Proven by `tests/test_video_selection_identity.py`.

## 1. The gap

The result cache is keyed on `(sha256, model_version)`. Until Batch 23
`model_version` composed weights, calibration, gate, face selection and decode —
and none of those move when the video path changes how many frames it samples,
what its abstention floor is, or how deep the top-k pool runs. The video path has
no face stage (R18): it samples frames and resizes them whole. The parameters
that decide *which* frames the model sees and *how* they become a verdict reached
no identity string:

| Parameter | Pre-B23 home | In an identity? |
| --- | --- | --- |
| `n_frames` | `VideoPipeline.__init__` default `48` | no |
| `min_usable_frames` | `VideoPipeline.__init__` default `8` — the `TOO_FEW_FRAMES` gate, the `min_face_px` / R19 shape, and *not* a `MediaCaps` field | no |
| pooling depth `k` | the literal `arr.size // 8` in `topk_indices` | no |
| the sampling rule | `sample_frame_indices`, published as `VideoAnalysis.sampling = "uniform"` — a label bound to nothing | no |

Two `VideoPipeline` instances that disagree on these produced a byte-identical
`model_version` while sampling a different frame set and able to publish a
different `p_fake`, or — via `min_usable_frames` — to flip `manipulated` →
`insufficient_evidence`. This is the video analogue of R17 (face selection) and
R25 (crop geometry).

## 2. `VideoSelectionPolicy`

A `@dataclass(frozen=True)` in `inference/preprocess.py`, every field required,
one factory `default_video_selection()` — the Batch 12 `FaceSelectionPolicy`
template, applied a fourth time (after R17, R19, R25).

| Field | Shipped | Closure | Held to the code by |
| --- | --- | --- | --- |
| `n_frames` | 48 | **functional** — `VideoPipeline` passes it to `sample_frame_indices` | it *is* the parameter |
| `min_usable_frames` | 8 | **functional** — the pipeline reads it and abstains | same |
| `pool_divisor` | 8 | **functional** — the pipeline computes `k = max(1, n // divisor)` and passes it explicitly to `topk_indices` | same |
| `sampling_rule` | `"uniform"` | **label** — also the published `VideoAnalysis.sampling` | `test_the_sampling_rule_name_matches_sample_frame_indices` re-implements `step = frame_count / n; min(fc-1, int(i*step))` and compares index lists |
| `pooling_rule` | `"topk_mean"` | **label** — also the published `VideoAnalysis.pooling` | `test_the_pooling_rule_and_divisor_match_the_pipeline` |
| `batch_size` | *excluded* | — | `test_batch_size_is_output_neutral` — two batch sizes → byte-identical bundle |

`__post_init__` validates `n_frames >= 1`, `1 <= min_usable_frames <= n_frames`,
`pool_divisor >= 1` — the ad-hoc guards, centralised.

**The labels are stronger closures than B22's `interpolation_rule`.** B22's label
is not on the wire at all, held only by a re-derivation test. Here the label is
*also* the published `VideoAnalysis` value (sourced from the policy), so "the wire
value equals the identity input" is true by construction, and the re-derivation
test additionally holds the identity input to the code.

### `batch_size` is excluded, and that is proven

The uint8→float32 widening is elementwise and `topk_indices` / `pool_at` run on
the full `scores` array after every batch; only progress-event fractions depend on
the batch size. Two batch sizes produce a byte-identical bundle. Following the
`cpu_baseline` / `decode_env` pattern, a knob that cannot move a published number
stays out of the identity.

Neutrality is **backend-dependent** (R32). It holds only while all three of:
(a) `batch_clips` yields contiguous non-overlapping slices with **no padding**;
(b) the model backend's `score()` is **row-independent**; and (c) `per_frame`
reconstruction is **order- and value-preserving**. A static-shape backend
(ONNX / TensorRT) pads the final batch and breaks (a); a cross-frame backend
(temporal attention — the `pooling_rule="attention"` future) breaks (b). Either
re-opens the exclusion and forces `batch_size` into `VideoSelectionPolicy` — a
further `.f.` move and cache generation. `test_batch_size_is_output_neutral` is
the behavioural tripwire; `test_the_batch_size_exclusion_note_is_complete` (PA11)
holds the `VideoSelectionPolicy` docstring to naming (a)/(b)/(c) and the trigger.

## 3. Why the `.f.` composite, not a new `.v.` segment

Three routes were weighed:

| Route | `contracts/version.py` | Single literal | Verdict |
| --- | --- | --- | --- |
| **composite `.f.`** — `selection_identity(face, video) = fingerprint({"face": …, "video": …})`, both pipelines call it | **untouched** — `.f.` is still one opaque 8-hex fingerprint | **preserved** | **chosen** |
| new `.v.` segment | a new marker in `_MODEL_MARKERS`, a fifth `ModelVersionParts` field, a new "pre-B23" parser era to keep forever | preserved | rejected — larger surface for a legibility gain `differences()` already provides |
| per-kind configured version | untouched | **two literals** | rejected — `verify_version_alignment` requires one string for both engines; touches `/health`, the idempotency key, every fixture |

Route A's cost is **legibility** — an operator seeing `.f.` move cannot tell from
the string alone whether the face or the video policy changed. `FaceSelectionPolicy.
differences()` / `VideoSelectionPolicy.differences()` and `verify_version_alignment`'s
per-engine detail name the mismatch. This is the same cost B22 accepted when it
folded `input_size` into `-w`.

Route A's other cost: the **image** cache namespace regenerates for a video-only
addition (a false miss — compute, not correctness). Precedented exactly by
**Batch 20**, whose `.d.` demuxer change regenerated the image namespace although
image decoding uses no demuxer, and settled by `docs/RUN_IDENTITY.md` §3.

`selection_identity(face, video)` is a hash of two hashes and is **not
injective** over the `(face, video)` pair — two distinct policy pairs could in
principle collide on the 8-hex `.f.` value. Acceptable for the same reason every
`fingerprint()` is (`contracts/version.py` `FINGERPRINT_LENGTH`): a 32-bit change
*detector* across the handful of policies a deployment holds, not a security
boundary. The composition adds no collision risk beyond what
`FaceSelectionPolicy.identity` and `VideoSelectionPolicy.identity` carry
individually.

**The honest-named reader.** B23 adds `contracts.version.selection_identity_of`
and `is_selection_pinned`; the pre-B23 names `face_selection_identity_of` /
`is_face_selection_pinned` are kept as permanent aliases with the identical
body, and `ModelSpec.with_selection` aliases `with_face_selection`. The pipelines
call `.with_selection(selection_identity(...))`. `face_selection_identity_of`
carries an explicit banner: for a B23+ string it returns the **composite**, never
`default_face_selection().identity` — new code should call
`selection_identity_of`. No parser change. `contracts/version.py`'s module
docstring, the `.f.` marker docstring and both reader docstrings name the video
half (pinned by `test_the_f_marker_and_public_readers_document_both_policies`).

## 4. Both pipelines carry it

`verify_version_alignment` requires **every** engine to report the one configured
`Settings.model_version`. So `ImagePipeline` carries `VideoSelectionPolicy`
without consuming it, the exact mirror of `VideoPipeline` carrying
`FaceSelectionPolicy` since Batch 12. `ImagePipeline().model_version ==
VideoPipeline().model_version`, pinned by
`test_image_and_video_engines_report_the_same_version` and
`test_the_image_pipeline_carries_the_video_segment_without_consuming_it`.

## 5. Legacy-scalar shim

`VideoPipeline(n_frames=…, batch_size=…, min_usable_frames=…)` still works
(deprecated). When a scalar is passed it is reconstructed into the policy via
`replace()` *before* the identity is computed, so `VideoPipeline(n_frames=9).
model_version != VideoPipeline().model_version` and the identity is always honest.
~10 study / test call sites need no edit.

## 6. N4 — the stale `"uniform+adaptive"` default

`contracts/video.py` declared `sampling: str = Field(default="uniform+adaptive")`
and the fixtures emitted it, while the live bundle hard-coded `"uniform"` and
nothing adaptive runs. B23 makes the live value authoritative: `sampling_rule =
"uniform"`, sourced from the policy; the contract default, `contracts/fixtures.py`,
`fixtures_out/video_*.json` and `tests/stub_handlers.py` corrected to match. A
`Field(default=)` value change — `python -m backend.openapi --update-snapshot`
produced a zero structural diff.

## 7. OR1 — the startup alignment gate

`workers.engine_registry.verify_version_alignment` existed since B9 and was called
only from tests. B23 makes it a deploy-time gate:

- **Worker:** `workers.runner.verify_engine_alignment_at_startup(configured=None)`
  runs from `run_with_shutdown` before the loop, fail-closed. The **only** escape
  hatch is the environment variable `DFP_SKIP_VERSION_ALIGNMENT` (truthy set
  `{"1","true","yes","on"}`, logged `WARNING` every boot) — the worker reads it
  directly from `os.environ` because it may not import `backend` (the dependency
  DAG). There is **no** `Settings.skip_version_alignment` field: on the env-only
  worker it would be dead config (nothing reads it). A wired field returns in
  **B24/OR1-b** together with `backend/worker_main.py`, the composition root that
  passes `get_settings().model_version` into the gate's `configured` parameter.
  Until B24, the configured literal reaches the gate only as `$DFP_MODEL_VERSION`;
  with neither, the gate still refuses to start on engines that disagree with
  each other or a malformed version, and
  `test_configured_model_version_matches_the_engines` stays the pre-merge guard
  for the `backend.config` default.
- **API:** `backend.app.lifespan` gains a stdlib-only format check on
  `settings.model_version` — `is_composed` + `is_environment_pinned` + a
  `parse_model_version(...).compose()` round-trip. The API process must stay free
  of cv2 / numpy, so it cannot run the full engine comparison; it rejects the
  historically-real failure (`"1.0.0-dev"`).

## 8. Migration

| Quantity | Before | After |
| --- | --- | --- |
| `default_face_selection().identity` | `251d22e9` | `251d22e9` — unchanged |
| `default_video_selection().identity` | — | `db8e6a0d` |
| `selection_identity(...)` — the `.f.` value | `251d22e9` (face only) | `e629642c` (composite) |
| `weights_identity` / `+c.` / `.g.` / `.d.` | `stub-0-wacdd9240` / `739c3fec` / `ae232cba` / `a97ccb68` | unchanged |
| model identity | `…f.251d22e9` | `…f.e629642c` |
| `STUB_ENGINE_VERSION` | `…f.251d22e9.d.a97ccb68` | `…f.e629642c.d.a97ccb68` |
| `VideoAnalysis.sampling` (wire) | `"uniform"` code / `"uniform+adaptive"` fixtures | `"uniform"` everywhere |

**One cache generation**, both content-cache namespaces, idempotency keys in
lockstep. Old stored bundles still load (`ModelInfo.version` is an opaque `str`).
`parse_model_version` stays total — `.f.` is still one fingerprint, no new era.

**Version-constant sweep (no bump).** Every carrier of a model version —
`ModelInfo.version` (`contracts/common.py`, `Field(description=…)` only, no
`pattern`/`min_length`/`max_length`), `BenchmarkRun.effective_model_version`
(`evaluation/benchmark.py`, bare `str`), `JobSpec.model_version`,
`Settings.model_version` — is an unconstrained `str`. `CONTRACT_VERSION`,
`RUN_SCHEMA_VERSION`, `CALIBRATION_WITNESS_VERSION` (digest excludes `.f.` — R31),
`LOCK_VERSION` (the split lock carries no model identity) govern nothing whose
shape a `.f.` value change touches. B23 bumps no version constant.

**Fixtures.** `fixtures_out/video_*.json` carry `model.version == "1.0.0-dev"`
(`contracts/fixtures.py` `MODEL_VERSION`, a synthetic spec — **not**
`STUB_ENGINE_VERSION`); the `.f.` move does not touch it. `python -m
contracts.fixtures` changes only the `sampling` value.

### Rollback

- **Code rollback** (revert `STUB_ENGINE_VERSION` to `…f.251d22e9.d.a97ccb68`,
  revert the `inference/` + `contracts/` changes): reverted engines recompute the
  face-only `.f.` `251d22e9`; the worker gate and `verify_version_alignment` pass
  against the reverted literal. Clean. The N4 `contracts/video.py` default revert
  is inert (nothing reads the `Field(default=)`). `selection_identity_of` /
  `is_selection_pinned` / `ModelSpec.with_selection` disappear; no external
  caller depends on them within the B23 window.
- **Orphaned cache entries** written under `…f.e629642c…` become unreachable —
  storage cost, not correctness; they expire under `result_ttl_s` (7 days).
  Idempotency keys the same. Identical to the B20/B22 rollback profile.
- **Partial rollback** (code without the literal, or vice versa): the worker gate
  refuses to start (engines report X, configured literal Y) — the gate working
  as designed, a loud failure instead of a silent broken cache. Make code and
  literal consistent.
- **Re-applying B23** restores `…f.e629642c…`; entries orphaned by the rollback
  become reachable again if still within TTL. No corruption in either direction.

### Rollout (R34)

B23 moves `STUB_ENGINE_VERSION`, and OR1's worker gate is fail-closed. In a
pod-by-pod rolling update, a worker whose configured literal disagrees with its
engine build refuses to start during the window. Two safe paths:

1. **Coordinated worker restart** at cutover (land code + literal together).
2. **Hold `DFP_SKIP_VERSION_ALIGNMENT=true`** on every worker for the window and
   remove it after. This bypasses the gate and reduces OR1's deploy behaviour to
   **exactly B20/B22**: each worker starts against whichever literal it is
   configured with, the API serves reads under its literal, and whichever
   generation the API's literal does not match produces false cache **misses**
   (re-runs) — never wrong verdicts. OR1 is **not worse** than B20/B22 during
   the window; it is strictly better outside it. The every-boot `WARNING` log is
   the audit trace.

Deploy runbook: set `DFP_MODEL_VERSION` to the new `STUB_ENGINE_VERSION` in the
worker and API environment before rollout (until B24/OR1-b wires the
config-default path into the full check).

## 9. What B23 does not fix

The video path still has no face stage. `n_frames` is now fingerprinted but the
*quality* of uniform sampling (R6-a/R6-b — a `frame_count` miscount still shifts
every sampled index) is not improved. `min_usable_frames` entering the identity
does not make a 3-frame verdict trustworthy — it makes two deployments that
disagree on the floor *distinguishable*. R24 (whole-frame squash) is untouched.
R28 stays out: its predicate fix moves shipped pixels and would forfeit B23's
output-identical guarantee — it rides the SCRFD / R23 batch.
