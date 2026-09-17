# Face-selection identity: naming the policy that chooses the pixels

> **Batch 22 (R25) added two fields** to `FaceSelectionPolicy`:
> `interpolation_rule` (`"area-if-either-axis-exceeds-target-else-linear"`) and
> `crop_clip_rule` (`"intersect-image"`) — both label closures over `resize` /
> `crop_face`, held to the behaviour by re-derivation tests (the `ORDERING_RULE`
> pattern this doc describes). The `.f.` fingerprint moved `19559387` →
> `251d22e9`; Batch 22 also folded `ModelSpec.input_size` into `weights_identity`
> (`w7f45fada` → `wacdd9240`). No pixels moved, one cache generation. See
> `docs/CROP_IDENTITY.md`.
>
> **Batch 23 (R29) broadened the `.f.` segment.** It no longer names the
> face-selection policy alone — it names the *selection* policy: face selection
> (image + video), plus frame selection and pooling (video), combined by
> `inference.preprocess.selection_identity(face, video) = fingerprint({"face": …,
> "video": …})`, which **both** pipelines compute. `default_face_selection().identity`
> is unchanged (`251d22e9`); the `.f.` fingerprint is now the composite
> `e629642c`. `face_selection_identity_of(version)` returns that composite for a
> B23+ string; `face_selection_identity()` (this module) stays face-only.
> `contracts/version.py` is untouched. See `docs/VIDEO_SELECTION_IDENTITY.md`.

**Status:** S8 Batch 12. **R17 closed** — measured, fixed, and pinned by named
doubles. R13 unchanged (open, deferred to SCRFD). R16, R18–R22 unchanged (open;
this batch touches none of them and says so in §8).
**Contract impact:** none. `CONTRACT_VERSION` 1.0.0, `EvidenceBundle`,
`ThresholdInfo`, `ModelInfo`, the OpenAPI snapshot and every serving route
unchanged. One configured literal moved, deliberately and loudly (§6).
**Suite:** 40 suites, 1047 tests, zero failures.

---

## 1. Phase 1 — every variable that influences image face selection

Traced exhaustively from the four call sites the Batch 11 review located.
"Fingerprinted" means *before this batch*.

| # | Variable | Shipped value | Where it lived | Fingerprinted |
|---|---|---|---|---|
| 1 | Detector implementation | `opencv-haar-frontalface-default` | `preprocess.CASCADE_PATH` | **no** |
| 2 | Detector asset (the cascade XML) | sha256 `0f7d4527844eb514` | an OpenCV data file on disk | **no** |
| 3 | `scaleFactor` | `1.1` | literal inside `detect_faces` | **no** |
| 4 | `minNeighbors` | `5` | literal inside `detect_faces` | **no** |
| 5 | Detection floor (`minSize`) | `32` | `detect_faces(min_px=32)` default arg | **no** |
| 6 | Ordering rule / tie-breaks | `(-area, y, x, w, h)` | `face_order_key` | **no** |
| 7 | Crop expansion (margin) | `1.3` | `crop_face(margin=1.3)` default arg | **no** |
| 8 | Crop shape | `max(w,h)` square, centred | `crop_face` body | **no** |
| 9 | Gate floor `min_face_px` | `64` | `MediaCaps` | yes — gate fingerprint |
| 10 | Image floor `min_short_edge` | `64` | `MediaCaps` | yes — gate fingerprint |

**Eight of ten were invisible.** Five of those eight were hard-coded literals or
default arguments — values nobody thinks of as configuration precisely because
they never appear in a config file.

### 1.1 The hidden defaults, named

Three deserve calling out because they are the ones a reviewer would not have
gone looking for:

**The two face floors are different numbers and only one was covered.**
`detect_min_px = 32` decides which boxes *exist*. `MediaCaps.min_face_px = 64`
decides which of them is large enough to score. A box between the two is
detected, becomes `faces[0]`, and is then gated as `FACE_TOO_SMALL` — so the pair
*jointly* decides the abstention, and fingerprinting one without the other was a
half fingerprint that looked complete.

**The crop margin was a default argument.** `crop_face(image, box)` is how every
caller in the repo invokes it. The `1.3` reached the pixels of every scored image
and appeared nowhere else.

**The detector's asset is not the detector's version.** OpenCV owns the cascade
XML and may reship a re-trained one in a patch release. `cv2.__version__` would
move — but no part of the *model* identity reads it. That lives in the decode
identity, which a model comparison deliberately strips.

---

## 2. Phase 2 — which of them reach what

Measured, `tests/face_selection_study.py` stage 2. One image, one model, one
decoder; one knob varied at a time.

| Variable | Selected face | Crop bytes | `p_fake` | Verdict | Cache key | Repro artifacts |
|---|---|---|---|---|---|---|
| detector implementation | yes | yes | yes | yes | **was blind** | not recorded |
| detector asset | yes | yes | yes | yes | **was blind** | not recorded |
| `scale_factor` | **yes — measured** | yes | yes | **yes — measured** | **was blind** | not recorded |
| `min_neighbors` | **yes — measured** | yes | yes | **yes — measured** | **was blind** | not recorded |
| `detect_min_px` | **yes — measured** | yes | yes | **yes — measured** | **was blind** | not recorded |
| ordering rule | **yes — measured** | yes | yes | yes | **was blind** | not recorded |
| `crop_margin` | no | **yes — measured** | **yes — measured** | **yes — measured** | **was blind** | not recorded |
| crop shape | no | yes | yes | yes | **was blind** | not recorded |

`FaceTrack` contents follow the selected face directly: `bbox` is `faces[0]`'s
box and `quality.face_px` is its short edge, so every row that moves the
selection moves the published track too.

### 2.1 The measurements

**Crop margin — the headline.** Same image, same model, same everything else:

```
crop_margin 1.0  crop=567a77aac7  p=0.889291  manipulated
crop_margin 1.1  crop=c8481af845  p=0.119862  authentic
crop_margin 1.2  crop=aaa2f1db36  p=0.659734  insufficient_evidence
crop_margin 1.3  crop=379c10e869  p=0.013562  authentic              <- shipped
crop_margin 1.4  crop=3ef98815ce  p=0.494578  authentic
```

**All three verdicts are reachable by changing one default argument.**

**Detector parameters — the face is lost, not moved.**

```
scale_factor  1.05 / 1.2 / 1.3   -> 0 boxes -> INSUFFICIENT_EVIDENCE / no_face_detected
scale_factor  1.1                -> 1 box   -> AUTHENTIC at 0.013562   <- shipped
min_neighbors 8 / 12             -> 0 boxes -> INSUFFICIENT_EVIDENCE / no_face_detected
min_neighbors 3 / 5              -> 1 box   -> AUTHENTIC at 0.013562
detect_min_px 320                -> 0 boxes -> INSUFFICIENT_EVIDENCE / no_face_detected
```

This is the same failure shape R15 produced by a data race: a *lost* face, not a
moved score. A verdict is replaced by an abstention, which no interval widens for
and no fingerprint recorded.

**Ordering rule — a different face entirely.** On a wide box and a square one:

```
area-desc (shipped)  faces[0] = (100,100,400,100)
short-edge-desc      faces[0] = (700,100,190,190)
```

### 2.2 What the stub can and cannot establish

`StubBackend` derives its logits from a hash of the crop's bytes, so the
*direction* of every score change above is arithmetic, not judgement. The study
says so and this report repeats it.

What the stub does establish is the part an identity argument needs: **the crop
bytes change**, and a scorer that is a function of its input therefore changes
its answer. The crop digest is the hard measurement; the verdict flip is the
consequence made visible. A real model would move by a different amount in a
different direction, and it would still move.

---

## 3. Phase 3 — can two runs diverge under identical identities?

**Yes, and it was proved against the real cache.** `tests/face_selection_study.py`
stage 4 and `tests/test_face_selection_identity.py` §1.

Two deployments differing only in `crop_margin`, 1.3 against 1.2:

| Identity | 1.3 | 1.2 | Same? |
|---|---|---|---|
| weights identity | `stub-0-w7f45fada` | `stub-0-w7f45fada` | **yes** |
| calibration fingerprint | `739c3fec` | `739c3fec` | **yes** |
| gate fingerprint | `ae232cba` | `ae232cba` | **yes** |
| decode identity | `c20a064a` | `c20a064a` | **yes** |
| **model identity (before)** | `stub-0-w7f45fada+c.739c3fec.g.ae232cba` | *identical* | **yes** |
| **runtime identity (before)** | `…+c.739c3fec.g.ae232cba.d.c20a064a` | *identical* | **yes** |
| benchmark identity (before) | model identity | *identical* | **yes** |
| published verdict | `authentic`, p 0.013562 | `insufficient_evidence` | **no** |

Every identity string in the platform agreed. The verdicts did not.

The consequence, executed against `InMemoryResultSink` — the production key
derivation, not a restatement:

```
BEFORE, SelectionBlindVersion (the named double):
  a margin-1.2 deployment looks up its key -> HIT: authentic
  served p_fake=0.013562, computed under margin 1.3, when its own crop abstains

AFTER, against the real sink:
  same policy  -> hit
  other policy -> miss
```

---

## 4. Phase 4 — the minimum extension

A fourth segment in the **model** identity, `.f.`, carrying a fingerprint of one
frozen policy object.

```
stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.d662e22d.d.c20a064a
└──────────────┘  └───────┘  └───────┘  └───────┘  └───────┘
     weights      calibration   gate    face select  decode
└─────────────────── model identity ──────────────┘
└──────────────────────── runtime identity ──────────────────┘
```

### 4.1 The structural half, which matters more than the hash

`FaceSelectionPolicy` is a frozen dataclass with **no defaults on any field**,
and `detect_faces` / `crop_face` take *the policy object* rather than loose
`min_px` / `margin` keywords.

That is the actual fix. Hashing a set of constants some other code happens to
use is a fingerprint that can drift from the behaviour it claims to name. Reading
both the behaviour and the fingerprint off one object means **a caller cannot
change one without changing the other**. The no-defaults rule closes the loop:
hidden defaults were the defect, so a policy cannot be constructed without
stating every knob, and `default_face_selection()` is the single place the
shipped numbers live.

`test_the_policy_has_no_defaults` and `test_every_policy_field_moves_the_identity`
both iterate the dataclass rather than a hand-written list, so a knob added later
without reaching the identity fails there instead of silently joining the set of
things nothing records.

### 4.2 The ordering rule is named, not hashed

`ORDERING_RULE = "area-desc,y-asc,x-asc,width-asc,height-asc"` — a name, not a
hash of `face_order_key`'s bytecode. Bytecode moves with every Python release and
every comment edit, which would dump the result cache on changes that cannot move
a box.

A name is only sound if something holds it to the behaviour, so
`test_the_ordering_rule_name_matches_the_ordering_rule` re-derives the order the
name describes and compares it against `face_order_key` on boxes chosen to
separate every component of the key.

### 4.3 API

```python
# inference/preprocess.py
FaceSelectionPolicy            # frozen, eight required fields
default_face_selection()       # the shipped policy; lru_cached (reads the cascade)
face_selection_identity()      # its fingerprint, the one-liner
cascade_digest()               # sha256[:16] of the cascade asset

# contracts/version.py  (stdlib only)
compose_model_version(..., face_selection=...) -> str
parse_model_version(v) -> ModelVersionParts(weights, calibration, gate, face_selection)
face_selection_identity_of(v) -> str | None
is_face_selection_pinned(v)   -> bool

# inference/models.py
ModelSpec.face_selection_fingerprint
ModelSpec.with_face_selection(fp)     # exact parallel of with_gate
```

`FACE_SELECTION_MARKER` is `.f.`, **not** `.s.`: `selection` is already taken in
this codebase by *dataset* selection — `CalibrationWitness.selection_witness`
names the corpus a temperature was fitted on — and two things called "selection"
in one identity vocabulary is how an auditor reads the wrong one.

---

## 5. Phase 5 — alternatives evaluated

| Option | Covers | Misses | Verdict |
|---|---|---|---|
| **Runtime identity only** (a second `.d.`-style segment outside the model) | cache, idempotency, `/health` | **benchmark comparison** — a policy change would read as environment noise, and `regression_gate` strips it | rejected |
| **Benchmark identity only** (a `BenchmarkRun` field) | regression gates | **the cache** — i.e. the live defect | rejected |
| **Calibration identity only** (fold into `calibration_fingerprint`) | cache and benchmark, incidentally | **meaning.** A crop margin is not a calibration. Folding it in makes "the temperature changed" and "the detector changed" the same event — the conflation Batch 9 explicitly rejected for the decoder (`RUN_IDENTITY` §5.4) | rejected |
| **Fold into the gate fingerprint** | everything, with the smallest diff | **meaning**, again. The gate decides *whether* an input is scored; selection decides *which face* and *what pixels*. Different owners (deployment config vs shipped code), different lifecycles, and SCRFD moves one without touching the other. An operator diffing two versions must see which | rejected |
| **Dedicated face-selection segment in the model identity** | cache, idempotency, `/health`, alignment, benchmark, forensics — all by inheritance | nothing | **shipped** |
| **Full detector fingerprint** | everything above, plus the cascade asset | nothing — the asset digest **is** included | shipped, as a subset of the above |
| **Policy version only** (a hand-maintained string like `"selection-v2"`) | nothing reliably | the numbers it names. A human must remember to bump it, and the failure mode is silent | rejected |

The "full detector fingerprint" row deserves a note: it is not a competing
option, it is a question about *depth*, and the answer is yes. `cascade_digest()`
hashes the 930 127-byte XML in 46.8 ms, once per process — the same order as the
45–73 ms the cascade *load* already costs per thread, and paid once rather than
per thread.

### 5.1 Why the model identity rather than the runtime identity

Face selection is **code this deployment ships**; the decoder is **environment it
inherits**. That distinction is the whole of Batch 9's design, and it decides
this cleanly: a selection-policy change genuinely *is* a model change for the
purposes of "did the model get worse", because the same weights cropping
differently score differently. So it belongs in `effective_version`, where
`BenchmarkRun.effective_model_version` picks it up for free.

Putting it in the model identity buys every consumer at once, with no new
plumbing: runtime identity is `model + decode`, so the cache key, the idempotency
key, `/health` and `verify_version_alignment` all inherit it.

### 5.2 Why the video engine carries a segment it does not consume

`VideoPipeline` detects no faces at all (R18) — `detect_faces` appears zero times
in it — so no selection policy can move a video verdict today. It carries the
segment anyway, for a reason about the platform rather than about the pipeline:
`Settings.model_version` is **one** string that keys the result cache for both
media kinds, and `verify_version_alignment` requires every engine to report it.
An image-only segment would either fail startup or force a per-kind configured
version, which is a serving change this brief forbids.

The cost is a video cache namespace that changes when a policy video does not use
changes. That is a **false miss — compute, not correctness**, and `RUN_IDENTITY`
§3 already settles which way a cache key breaks a tie. When R18 gives video a
face stage, the segment is already correct.
`test_the_video_pipeline_carries_the_segment_without_consuming_it` pins both
halves, and fails loudly if the video path grows a face stage — at which point
the test says so and asks to be replaced, the way Batch 9's benchmark tripwire
did and Batch 11 honoured.

---

## 6. Phase 6 — migration impact

### 6.1 What moves

| Artifact | Before | After |
|---|---|---|
| `STUB_ENGINE_VERSION` | `stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a` | `…g.ae232cba**.f.d662e22d**.d.c20a064a` |
| Version string length | 49 | 60 characters |
| `ModelSpec.effective_version` | weights + calibration + gate | **+ face selection** |
| `pipeline.model_version` | model + decode | inherits the segment |
| `detect_faces` signature | `(image, *, min_px=32)` | `(image, *, policy=None)` |
| `crop_face` signature | `(image, box, *, margin=1.3)` | `(image, box, *, policy=None)` |

Both signature changes are additive in practice: **no caller in the repo passed
`min_px` or `margin`**, verified before the change, and
`test_omitting_the_policy_uses_the_shipped_one` asserts the no-argument form is
byte-identical to the explicit one.

### 6.2 Artifact by artifact

| Artifact | Impact |
|---|---|
| **Result cache** | One-time full miss. Namespaces do not overlap, so nothing stale is served; old objects stay readable by job id and only the content pointer is orphaned. Cost is one re-analysis per distinct file. Same shape as Batch 9's, and back-filling was rejected for the same reason: it would assert that historical results were produced under *this* policy. |
| **Job idempotency keys** | `sha256:model_version`, computed per submission. In-flight duplicates re-enqueue once. No stored state. |
| **Benchmark reports** | Readable, no schema bump. `RUN_SCHEMA_VERSION` stays `1.2.0`. A pre-Batch-12 run and a post-Batch-12 run now compare as **different models**, which is correct — the crop changed, so it is not a like-for-like comparison. |
| **Calibration witnesses** | **Untouched.** `CALIBRATION_WITNESS_VERSION` unchanged, no stored witness re-digests, no `compute_witness` signature change. A selection change does not move `calibration_fingerprint` (§5, and asserted). |
| **Stored bundles** | Readable. `ModelInfo.version` is a `str` and got eleven characters longer. |
| **Split locks** | Untouched. |
| **OpenAPI snapshot** | Unchanged. `test_openapi`'s drift guard passes with no snapshot edit. |
| **`CONTRACT_VERSION`** | **Unchanged at 1.0.0.** No contract field changed shape or meaning; asserted by `test_the_contract_version_did_not_move`. |

### 6.3 Backward compatibility

`parse_model_version` is total on both eras. A pre-Batch-12 string parses to
`face_selection=None`, which is not an error — it means *this version makes no
claim about which face it chose*, and `is_face_selection_pinned` reports exactly
that. Nothing needs migrating to be readable. Verified against the literal that
actually shipped, not one constructed to parse:

```
parse('stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a')
  weights = stub-0-w7f45fada   calibration = 739c3fec
  gate    = ae232cba           face_selection = None
```

The decode segment stays **last**, so the model identity is still a prefix of the
runtime identity and `model_identity_of` is still one `partition`
(`test_the_decode_segment_stays_last`).

---

## 7. Guarantees gained

1. **A cached result cannot cross face-selection policies.** Proved against the
   real sink with a named double for the old composition (§3).
2. **A selection change is visible in the deployment identity** — `/health`, the
   alignment check and every log line carry it.
3. **A worker whose selection policy differs from the configured one fails at
   startup**, instead of silently sharing a namespace
   (`test_alignment_fails_when_the_configured_policy_is_stale`).
4. **A benchmark comparison across a policy change reports different models**,
   because `effective_model_version` carries the segment.
5. **The behaviour and the fingerprint are read off one object**, so they cannot
   drift apart — the structural guarantee, stronger than the hash (§4.1).
6. **There are no unfingerprinted selection knobs left**, enforced by iterating
   the dataclass rather than a list.
7. **The detector's asset is hashed**, so a re-trained cascade shipped under the
   same filename moves the identity.
8. **Behaviour is unchanged.** Boxes and crop bytes are byte-identical to the
   pre-Batch-12 literals, proved against `LegacyHardcodedSelection` verbatim.
9. **Both face floors are covered** — `detect_min_px` in the face segment,
   `min_face_px` in the gate — in different fingerprints, on purpose.

## 8. Guarantees NOT gained

1. **Identity does not make two policies agree.** It makes them
   distinguishable. Same shape of guarantee as Batch 9's about decoders and the
   split lock's about corpora, and asserted rather than assumed
   (`test_identity_does_not_make_two_policies_agree`).
2. **R16 is untouched.** The platform still publishes one face's verdict as the
   media's. The policy is now *named and reproducible*; it is not *right*. Batch
   11 said R14 made the choice reproducible rather than correct — this batch made
   it auditable rather than correct.
3. **R13 is untouched.** Crop instability under ±4 LSB is a property of the
   placeholder detector, and a fingerprint cannot separate two runs that share
   one. Still deferred to SCRFD with its acceptance criterion.
4. **R18–R22 are untouched.** The video path still has no face stage; the gate
   still ranks on area and tests short edge (R19); `face_px`'s description is
   still wrong (R20); `faces[]` still has no consistency invariant (R21).
5. **The video engine's segment is carried, not consumed.** §5.2. It over-keys
   the video cache today, deliberately.
6. **The policy is not published in the bundle.** `FaceSelectionPolicy.to_dict()`
   exists for forensics and nothing writes it into an `EvidenceBundle` — that
   would be a contract change. A stored result names its policy by *fingerprint*
   only, so reconstructing the eight fields needs the deployment that produced
   it. Same limitation the gate fingerprint has had since Batch 0.
7. **Nothing forces the forensic record to be read.** Standing complaint since
   R4.
8. **SCRFD is not implemented**, and this batch deliberately makes no room for it
   beyond a `detector` field that will change value.

---

## 9. Measured impact

| Measurement | Value |
|---|---|
| Selection variables total | 10 |
| Unfingerprinted before this batch | **8** |
| Unfingerprinted after | **0** |
| Verdict-flipping knobs measured | 4 of 4 tried (`scale_factor`, `min_neighbors`, `detect_min_px`, `crop_margin`) |
| Distinct verdicts reachable by crop margin alone | **3 of 3** |
| `p_fake` spread across margins 1.0–2.0, one image | 0.0136 → 0.8893 |
| Cross-policy false cache hits before | possible; proved (§3) |
| Cross-policy false cache hits after | 0 |
| Version string length | 49 → 60 characters |
| Cache namespaces, homogeneous fleet | 1 → 1 |
| Pre-Batch-12 cache entries invalidated | all — one-time, no data loss |
| Cascade hash cost | 46.8 ms, once per process |
| Contract changes | 0 |
| Schema changes | 0 |
| `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` bumps | 0 |
| Suite | 39 → **40 suites**, 1015 → **1047 tests**, zero failures |

---

## 10. R17 status

**R17 — the face-selection policy and crop margin are outside every identity
string: CLOSED.**

| Half | Status |
|---|---|
| Cache identity | **Closed.** Proved against the real sink with a named double. |
| Idempotency key | **Closed.** Inherits the runtime identity. |
| Deployment identity | **Closed.** `/health`, alignment, logs. |
| Benchmark identity | **Closed.** `effective_model_version` carries the segment. |
| Calibration identity | **Closed by design** — deliberately excluded; a crop margin is not a calibration, and the model identity already separates the two runs. |
| Forensic identity | **Partial.** The bundle carries the fingerprint, not the eight fields. §8.6. |

---

## 11. Where things live

| what | where |
|---|---|
| The policy object, its identity and the cascade digest | `inference/preprocess.py` |
| Composition, parsing, the marker | `contracts/version.py` |
| The spec binding | `ModelSpec.with_face_selection` |
| Where the pipelines bind it | `{Image,Video}Pipeline.__init__` |
| The configured literal and its warning | `backend/config.py` |
| Proof, doubles, and the asserted limits | `tests/test_face_selection_identity.py` |
| The measurements this rests on | `tests/face_selection_study.py` |
| The review that raised R17 | `docs/MULTI_FACE_SEMANTICS.md` |
| The precedent this follows | `docs/RUN_IDENTITY.md` |
