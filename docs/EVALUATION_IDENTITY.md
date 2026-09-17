# Evaluation identity — S8 Batch 11 (closing R12)

> **Verdict: close it, asymmetrically.** The benchmark half is a real defect with
> a measured, bidirectional cost to release decisions, and it closes cleanly with
> an additive field that breaks no stored artifact. The calibration half is
> weaker than it looks — the fitted values already distinguish two decoders in
> most cases — but it has one reachable blind spot, and closing it costs a digest
> rule change rather than a data migration.
>
> **A naive `CALIBRATION_WITNESS_VERSION` bump would make every stored witness
> unreadable.** That is proved by execution in §6 and is the reason this batch
> changes *how* the digest is computed rather than only *what* goes into it.

Companion to [`docs/RUN_IDENTITY.md`](RUN_IDENTITY.md), which closed the same
finding on the serving side. This is the evaluation side.

---

## 1. Where benchmark identity is stored

`evaluation/benchmark.py`, in `BenchmarkRun` — a frozen dataclass serialised to
JSONL by `append_history` and read back by `read_runs` / `BenchmarkRun.from_dict`.
`RUN_SCHEMA_VERSION` is the axis; it was `1.1.0` before this batch.

The identity-bearing fields, and where each comes from:

| Field | Source | Names |
| --- | --- | --- |
| `effective_model_version` | **caller argument** | weights + calibration + gate |
| `dataset_fingerprint` | computed from the scored items | the evaluated set |
| `split_lock_hash`, `lock_generation` | `split_lock` argument | the frozen split |
| `protocol`, `protocol_role`, `selection_witness` | `RoleSelection` only | how items were chosen |
| `calibration` (full witness dict), `calibration_fingerprint` | `calibration` argument | the calibration consumed |
| `schema_version` | the constant | the row's shape |

**`effective_model_version` is passed in, not read off the predictor** — the
docstring is explicit that only the caller holds all four components. Batch 9
deliberately routed the *model* identity here, not the runtime identity, so that
"did the model get worse" is not answered yes by a worker rebuild.

**The evaluation package never decodes anything.** `run_benchmark` calls
`predictor.predict(items)` and nothing else touches media. `StubPredictor` hashes
the media id. The decode environment is therefore entirely upstream, inside
whatever the caller passed as `predictor` — a fact that decides §5's design.

## 2. Where calibration identity is stored

Three places, in decreasing detail:

1. **`CalibrationWitness`** (`evaluation/calibration_protocol.py`,
   `CALIBRATION_WITNESS_VERSION`, `1.1.0` before this batch). The full record:
   protocol, role, corpus fingerprint, sample count, temperature, both NLLs,
   threshold, margin, target FPR, the selection witness, the split lock hash, and
   a digest binding all of them. `created_at` and `schema_version` sit outside the
   digest.
2. **`inference.models.CalibrationSpec.witness`** — the digest only, carried into
   the serving path. `is_protocol_compliant` is `witness is not None`.
3. **`BenchmarkRun.calibration`** — the whole serialised witness, embedded in the
   run, so a report stands alone months later.

The digest is computed by `CalibrationWitness.compute_witness`, and it includes
`"schema": CALIBRATION_WITNESS_VERSION`. That one line is what makes §6's
migration hazard real.

## 3. Is cross-decoder benchmark comparison currently possible?

**Yes. Proved by execution.** Two runs, identical items, identical model
identity, scored by a predictor that differs only in its decode environment
(`DecoderSensitivePredictor`, applying Batch 8's measured per-item |dp| = 0.005):

```
FFMPEG run: AUC 0.996125  FPR 0.000000  data d7f63f2e
MSMF   run: AUC 0.996575  FPR 0.000000  data d7f63f2e
AUC delta: +0.000450

regression gate: PASS
  note  both runs report the same effective model version; if the models
        really differ, something that changes a score is missing from the
        version composition

>>> gate.passed = True; failures = 0
```

The gate passes, and its one note points at the *opposite* conclusion — it
suggests the version composition is incomplete for the **model**, when what
actually changed was the environment.

Serialising both runs and diffing every field:

```
fields present  : 23
fields differing: ['pr_auc', 'roc_auc']
any 'decode' field? NO
```

**Only the measured numbers differ.** There is no field in which the cause could
be recorded, so the delta is attributable to nothing.

### 3.1 How much does it matter? A negative result first

The obvious worry — that decoder drift swamps the AUC gate — **does not
reproduce**. With an i.i.d. per-item perturbation of 0.005 over n = 2000:

| skill | AUC (FFMPEG) | AUC (MSMF) | \|dAUC\| | % of the 0.005 allowance |
| --- | --- | --- | --- | --- |
| 0.10 | 0.688926 | 0.688947 | 0.000021 | 0.4% |
| 0.20 | 0.835002 | 0.834967 | 0.000035 | 0.7% |
| 0.30 | 0.924473 | 0.924469 | 0.000004 | 0.1% |
| 0.50 | 0.990032 | 0.990049 | 0.000017 | 0.3% |
| 0.80 | 0.999885 | 0.999887 | 0.000002 | 0.0% |

Worst case 0.7% of budget. AUC is a rank statistic averaged over ~n²/4 pairs, so
i.i.d. noise averages out. A regression-masking search across skill drops of
0.02–0.05 found **no case** where the decoder flipped the gate's verdict through
AUC.

### 3.2 Where it actually bites: the threshold-dependent metrics

Batch 8 did not measure noise. It measured a **systematic** difference — a
near-identity colour matrix with a per-channel bias around −1.2. A systematic
score shift is close to rank-preserving, so AUC barely moves; but the threshold
is a *fixed number*, and a shifted distribution crosses it. Same fixture, a
uniform bias instead of noise:

```
baseline: AUC 0.924507  coverage 0.8975  FPR 0.0680  risk 0.1242

    bias       dAUC   dCoverage       dFPR  %FPR budget      dRisk   gate
   0.001  +0.000000     +0.0000    +0.0010       20.1%    -0.0006   PASS
   0.005  +0.000000     +0.0000    +0.0012       24.6%    -0.0017   PASS
   0.010  +0.000000     -0.0010    +0.0035       69.3%    -0.0027   PASS
   0.020  -0.000019     -0.0005    +0.0086      172.6%    -0.0049   FAIL
   0.050  -0.000153     +0.0010    +0.0166      331.2%    -0.0085   FAIL
```

Two things follow, and they are the case for closing this.

**At Batch 8's measured magnitude (0.005), a decoder change silently consumes
24.6% of the false-positive regression budget.** A genuine regression of 75% of
budget then passes. The gate's tolerance is being spent on something it does not
know about.

**At 0.020 the gate fails and blames the model, in those words:**

```
bias 0.02: gate FAILS -> reported as a MODEL regression:
    selective FPR rose 0.0086 (0.0680 -> 0.0766); detection was bought
    with false accusations
...but the model did not change. Only the decoder did.
```

That message is actively misleading. An engineer reading it goes looking for a
training change that does not exist.

**The combination is the worst available one:** the metric that survives a
decoder change (AUC) is the one the gate leans on hardest, and the metric that
moves (selective FPR) is the one it attributes to the model.

## 4. Is cross-decoder calibration comparison currently possible?

**Yes, but the finding is weaker than the benchmark one and the report should say
so.** Two witnesses fitted under two decoders:

```
FFMPEG witness digest: 092895e2  T=1.234567
MSMF   witness digest: 2ef3afec  T=1.239012
digests differ: True
```

They *are* distinguishable — because the decoder changed the logits, so the
fitted temperature changed, and the temperature is inside the digest. Detection
mostly works already, by accident of what the digest covers.

Two gaps remain.

**The blind spot.** When two fits converge to the same quantised values, the
witnesses are byte-identical:

```
FFMPEG: 092895e2
MSMF  : 092895e2
IDENTICAL: True  <-- two environments, one witness
```

Quantisation to 6 decimals makes this more reachable than it sounds: two fits
differing at the 7th decimal collapse to one witness.

**The question cannot be asked.** `verify_calibration_witness` takes `protocol`,
`selection`, `items` and `split_lock_hash`. There is no parameter through which
an auditor could supply the decode environment they expect, so even where the
digests *do* differ, nothing says the decoder is why.

The live risk is a temperature fitted under one decoder applied under another.
The serving cache is safe (Batch 9 put the decode identity in the runtime version),
but the calibration itself is silently mis-applied: `T` was fitted to map
FFMPEG-decoded logits to probabilities and is scaling MSMF-decoded ones.

## 5. Is a decode identity field sufficient?

**Yes, and it is the correct decomposition.**

`BenchmarkRun` already carries `effective_model_version`, which is the *model*
identity. Adding `decode_identity` beside it makes the runtime identity
recoverable by composition:

    compose_runtime_version(model=run.effective_model_version,
                            decode=run.decode_identity)

So the decomposed pair is **strictly more informative** than the composed string:
you can always build the runtime identity from it, and you cannot go the other
way without agreeing on a parse.

**Which digest.** `decode_identity()` (OpenCV version, backend, libavcodec, OS,
machine) rather than `decode_fingerprint()` (which adds CPU baseline and
dispatch). Batch 8's E6 measured `resize`/`to_rgb`/`imdecode` byte-identical
under `OPENCV_CPU_DISABLE`, so the CPU fields do not change a decoded pixel, and
including them would make two machines in one fleet permanently incomparable for
no correctness gain. The forensic digest remains available for the *diagnosis*
after a refusal; it is the wrong thing to gate on.

## 6. Is a full runtime identity field required?

**No, and storing one would be actively harmful.**

If the runtime identity went into `effective_model_version`, then two runs from
one model on two workers would report different model versions. Every
model-versus-model comparison would become a decoder comparison, and
`regression_gate`'s "both runs report the same effective model version" note
would stop firing when it should. That is precisely the conflation Batch 9
created two separate strings to avoid, and its reasoning transfers unchanged:

> "Benchmark identity deliberately still takes the model identity, so 'did the
> model get worse' is not answered yes by a worker rebuild."

A separate field keeps both questions askable. One combined field answers neither
cleanly.

### 6.1 The field is passed in, never auto-detected

`run_benchmark` must **not** call `decode_identity()` itself, for the same reason
it does not compute `effective_model_version`: the harness process is not
necessarily the process that produced the scores. Offline scoring, cached logits,
a scoring job on a GPU box and a report assembled on a laptop are all ordinary,
and in every one of them the harness's own environment is the wrong answer.
Recording it would be a **false claim**, which is worse than no claim — the
distinction `RUN_IDENTITY.md` established when it refused a sentinel for
"unpinned".

## 7. What schema version changes are required?

| Constant | From | To | Nature |
| --- | --- | --- | --- |
| `RUN_SCHEMA_VERSION` | 1.1.0 | **1.2.0** | additive optional field |
| `CALIBRATION_WITNESS_VERSION` | 1.1.0 | **1.2.0** | additive field **inside the digest** |
| `CONTRACT_VERSION` | 1.0.0 | 1.0.0 | untouched |
| `LOCK_VERSION` | 1.1.0 | 1.1.0 | untouched |

**Justification for `RUN_SCHEMA_VERSION`.** The field is optional with a `None`
default, so nothing forces it. The bump is still required, because
`BenchmarkRun.from_dict` refuses unknown fields — a 1.2.0 row read by a 1.1.0
build must fail loudly rather than be partially read, and the version is how that
build knows. Precedent: Batch 4 bumped 1.0.0 → 1.1.0 for exactly this shape.

**Justification for `CALIBRATION_WITNESS_VERSION`.** Different and larger,
because the new field goes **inside** the binding digest. It has to: a decode
identity recorded beside the digest could be edited without invalidating
anything, and a provenance field nothing protects is decoration. `created_at` is
outside for a reason that does not apply here — a timestamp is not an input to
the numbers, and the decode environment is.

## 8. Do any existing artifacts become unreadable?

**Benchmark history: no.** `from_dict` fills absent fields with `None`, which
reads as *not recorded*. Verified: a 1.1.0 row loads under the 1.2.0 build with
`decode_identity=None`. Forward incompatibility (a 1.2.0 row on a 1.1.0 build) is
refused loudly, which is the existing and correct behaviour.

**Calibration witnesses: yes — under a naive bump — and this is the crux of the
batch.** `compute_witness` puts `"schema": CALIBRATION_WITNESS_VERSION` inside
the digest, so changing the constant changes the digest of *unchanged fields*.
Executed against the real class with the constant temporarily set to `1.2.0`:

```
stored 1.1.0 witness digest : 092895e2
digest under schema 1.1.0   : 092895e2   (== stored: True)
digest under schema 1.2.0   : 54adb0e2
digests differ              : True

simulating it against the real class:
  reload FAILED: ProtocolViolation
    calibration witness '092895e2' does not match its own fields
    (expected '54adb0e2'); this witness was assembled or edited by hand
    rather than produced by fit_calibration()
```

Every stored witness becomes **unreadable**, not merely unverifiable —
`__post_init__` raises, so `from_dict` cannot return one at all. And the error
message accuses the operator of hand-editing, which is the worst possible
diagnosis to hand someone whose file is fine.

## 9. Is migration possible?

**Yes for readability, no for back-fill, and the two must not be confused.**

**Readability — implemented.** `compute_witness` takes the schema version as a
parameter, defaulting to the current one, and verification dispatches on the
payload's *own* `schema_version`. A 1.1.0 witness is then verified under the
1.1.0 rule (no decode key) and continues to validate; a 1.2.0 witness is verified
under the 1.2.0 rule (decode key present). Proved in §6's last block:

```
a 1.1.0 payload verifies under the 1.1.0 rule : True
```

This is a genuine migration: old artifacts stay readable **and stay
cryptographically verifiable**, rather than being grandfathered past the check.

**Back-fill — impossible, and refused.** Writing the current environment's decode
identity into a historical witness or run would assert that the fit happened
under *this* environment, which nobody observed. That is the identical claim
Batch 9 refused when it declined to back-fill cache entries:

> "Back-filling old entries with the current decode identity was rejected: it
> would assert that historical results were produced by *this* environment, which
> is the unfounded claim the batch exists to remove."

So historical rows and witnesses keep `decode_identity=None`, and every consumer
reports that as **"not recorded"** rather than as agreement. `None` versus `None`
is a note, never a pass.

## 10. Correctness-critical or reproducibility-critical?

**Benchmark half: reproducibility-critical, with a measured bidirectional effect
on release decisions.** No user-visible verdict is wrong — Batch 9 already
stopped the cache serving across decoders. What is wrong is the *gate*:

- a decoder change consumes 24.6% of the FPR budget at Batch 8's measured
  magnitude, so a real regression of the remaining 75% ships;
- a larger decoder change fails the gate and reports a model regression that did
  not happen, in a message naming false accusations.

A release gate that can be wrong in both directions for a reason it cannot name
is not a reproducibility nicety.

**Calibration half: reproducibility-critical, shading into correctness.** A
temperature fitted under one decoder and applied under another produces
miscalibrated probabilities at serving time — a genuinely wrong `p_fake`, not
merely an untraceable one. It is second-order (the fitted values usually differ
enough to be noticed) and it is the reason this half is closed too rather than
deferred.

**Neither is correctness-critical in the R15 sense** — nothing here produces an
answer computed from the wrong file. Rated below R15, above R14.

---

## 11. What was implemented

Everything below is additive. No serving route, no `EvidenceBundle`, no
`ThresholdInfo`, no `CONTRACT_VERSION`, no OpenAPI change.

**`BenchmarkRun.decode_identity: str | None = None`**, populated from a
`decode_identity=` argument to `run_benchmark`. `RUN_SCHEMA_VERSION` → 1.2.0.

**`regression_gate` refuses a cross-decoder comparison**, on the same footing as
a cross-dataset one and short-circuiting for the same reason — every subsequent
number inherits the contamination:

| incumbent | challenger | outcome |
| --- | --- | --- |
| recorded `X` | recorded `X` | silent, comparison proceeds |
| recorded `X` | recorded `Y` | **failure**, comparison refused |
| `None` | anything | **note** — the question cannot be answered |

The three-state treatment is R4's "unsealed is a third state, not a clean one",
applied here: no evidence is not evidence of sameness.

**`CalibrationWitness.decode_identity`**, inside the digest, with
`compute_witness(..., schema_version=...)` so 1.1.0 payloads verify under the
1.1.0 rule. `CALIBRATION_WITNESS_VERSION` → 1.2.0.

**`verify_calibration_witness(..., decode_identity=...)`** — the parameter §4 said
was missing, so an auditor can finally ask the question.

**`BenchmarkReport.validate()`** cross-checks the calibration's decode identity
against the benchmark run's, mirroring the split-lock check it already performs.

**Named doubles**, per the batch convention: `LegacyDecodeBlindRun` and
`LegacySchemaPinnedDigest` reproduce the pre-batch behaviour verbatim, each test
demonstrating the defect before proving the fix.

## 12. Guarantees gained

1. A benchmark comparison across decode environments is **refused**, not silently
   averaged into a model delta.
2. A run **records** the environment that produced its scores, or records that it
   does not know — and the two are distinguishable.
3. A calibration witness **binds** its decode environment, so editing it is
   detected.
4. An auditor can **ask** whether a calibration matches an expected environment.
5. Historical runs and witnesses remain **readable and verifiable**; the digest
   rule is versioned rather than replaced.
6. The runtime identity is **recoverable** from a run without storing it, keeping
   model-versus-model comparison decoder-blind by design.

## 13. Guarantees NOT gained

1. **Two decoders still disagree.** This makes them distinguishable, not equal —
   the same guarantee, and the same limit, as Batch 9's.
2. **Nothing forces a caller to pass `decode_identity`.** It is optional, because
   the harness cannot honestly infer it (§6.1). A run that omits it is a run whose
   comparisons carry a note instead of a guarantee.
3. **Back-fill is impossible.** Historical artifacts stay `None` forever.
4. **Equal-length logits from the wrong corpus** remains open, unchanged from
   Batch 6.
5. **The magnitudes are synthetic.** §3.2's biases are a model of Batch 8's
   measured levels difference, not a measurement of one on a real corpus — there
   is still no dataset and no trained model.
6. **`decode_identity` excludes the CPU fields** on the strength of one build's
   measurement, so two genuinely different-decoding machines could share an
   identity if OpenCV ever gains a CPU-dependent decode path.
