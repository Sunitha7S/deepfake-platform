# Cross-dataset evaluation protocol

Which corpus plays which role, what may be reported, and why the obvious
arrangement is wrong.

Implementation: [`evaluation/protocol.py`](../evaluation/protocol.py).
Tests: [`tests/test_evaluation_protocol.py`](../tests/test_evaluation_protocol.py).

## The number everyone reports is the one that means least

FF++ in-dataset AUC above 0.99 has been a solved problem for years. It survives
in papers because it is easy, and it predicts nothing about production: a
detector that has learned FaceSwap's blending seam scores 0.99 on FF++ and falls
to near chance on a diffusion-generated face it has never seen.

This project's honest targets, agreed at design review, are **Celeb-DF v2 AUC
0.90–0.93** and **DFDC 0.75–0.80**. Above 95% "real-world accuracy" is not
achievable and claiming it is a sign that the measurement is in-domain.

## The five roles

| Role | What it does | Why it is separate |
| --- | --- | --- |
| `TRAIN` | Fits the weights. | — |
| `VALIDATION` | Chooses the checkpoint, learning rate, architecture. | Sees the model thousands of times and is used up by the end of training. It is not a benchmark and must never be reported as one. |
| `CALIBRATION` | Fits the temperature, selects the threshold and band. | Touches no weights and is *still* a way to overfit an operating point. A threshold chosen on the benchmark reports the best FPR that corpus can produce, not the one production will see. |
| `PRIMARY_BENCHMARK` | The headline. | Cross-dataset, wholly held out, measured rarely. |
| `SECONDARY_BENCHMARK` | The others. | Reported alongside, never instead. |

Exactly one dataset holds each of the first four. One headline benchmark is on
purpose: with two, whichever looks better gets quoted, and the choice is made
after the numbers are in.

## The rule that does the work

> **A benchmark dataset contributes to nothing.** Not a training split, not a
> validation split, not a calibration set, not a slice used to pick a band width.

This is stricter than "don't train on test", and deliberately so. The usual rule
is satisfied by fitting a temperature on Celeb-DF and then reporting Celeb-DF
AUC — which happens constantly, and which `validate_protocol` refuses.

Once any part of a corpus has influenced any fitted quantity, the rest of it is
no longer measuring generalisation to unseen data. It is measuring
generalisation to unseen *files* from a distribution the model has already been
tuned against, and those are different numbers by tens of AUC points.

## Why calibration has its own role

It could have been folded into validation, and folding it in is a mistake worth
naming. The validation split selects checkpoints, so its logits are optimistic —
the chosen checkpoint is the one that happened to do best on them. A temperature
fitted there is fitted to that optimism and comes out close to 1.0, reporting a
well-calibrated model that is overconfident everywhere else.

Calibration wants data that selected nothing. `validate_protocol` warns when
calibration and validation share a split.

## The protocol this platform uses

`evaluation.protocol.FFPP_CELEBDF`:

```
train                ffpp     [train]   c23 and c40 together; both qualities of one
                                        source share an identity component
validation           ffpp     [val]     checkpoint and hyper-parameters only; never reported
calibration          ffpp     [test]    temperature and threshold
primary_benchmark    celebdf  [all]     the headline; target AUC 0.90–0.93
secondary_benchmark  dfdc     [all]     harder and noisier; target 0.75–0.80
```

The interesting choice is **calibration on FF++ test rather than FF++ val**. It
buys a temperature fitted to logits from a split that selected no checkpoint.

What it costs is narrower than it first looks, and the distinction matters:

- Temperature scaling is monotone, so **FF++ test AUC is untouched** by having
  been calibrated on. It remains a legitimate in-domain ranking number, and that
  is what `in_domain` means in a report.
- What is spent is every **threshold-dependent** number on that split. Coverage,
  selective risk and selective FPR measured there are best-case, because the
  threshold was chosen to make them so. Report those from the benchmark corpora
  only.

## Adding a future dataset

```python
protocol = FFPP_CELEBDF.with_secondary("dfd-2026q1", note="quarterly holdout")
report = validate_protocol(protocol)
assert report.valid, report.describe()
```

Added as *secondary*, not promoted to primary. The primary is the number that
gates releases, and swapping it changes the meaning of the entire history — a
model that "improved" across a primary-benchmark change has not been measured,
it has been re-scoped.

Validation runs on the extended protocol on the same terms as the original, so a
new corpus that already contributes to a fitting role is rejected the same way
an old one would be
(`test_adding_a_dataset_that_is_already_fitted_on_is_rejected`).

The design intent recorded at review is that the release gate is a
**self-assembled modern-generator holdout, refreshed quarterly**, with Celeb-DF
kept for comparability with published work. That holdout enters as a secondary
benchmark and is promoted only by an explicit, dated decision.

## Selecting items for a role

```python
items_for_training = select(protocol, DatasetRole.TRAIN, items, lock.assignments)
```

`select` takes a plain `media_id → Split` mapping so a `SplitLock`, a
`SplitAssignment.by_media` or a hand-built dict all work.

Items with no assignment are **excluded, not defaulted**. There is no safe
default: guessing train poisons the weights and guessing test poisons the
benchmark.

## Reporting conventions

Every reported number must be accompanied by `REQUIRED_REPORT_FIELDS`.
`check_report(payload)` returns what is missing, and
`evaluation.benchmark.BenchmarkReport` assembles a complete payload from run
artifacts alone:

```python
report = BenchmarkReport(primary=celebdf_run, in_domain=ffpp_run, protocol=protocol.name)
assert not report.validate(), report.describe()
```

It takes **two** runs because `primary_benchmark` and `in_domain` are required
together and no single run holds both. `validate()` also refuses a pair of runs
from different models, and a pair measured on the same corpus.

Threshold-dependent numbers in the payload come from the primary run only — see
the note below on what calibrating on FF++ test does and does not cost.

| Field | Why it is mandatory |
| --- | --- |
| `protocol`, `split_lock_hash`, `dataset_fingerprint` | What was measured, on which frozen split. Without these a number cannot be compared with any other number. |
| `effective_model_version` | Weights + calibration + gate. The bare weights version hides a recalibration, and a recalibration moves every probability. |
| `primary_benchmark` **and** `in_domain`, together | Quoting either alone is the standard way to mislead. In-domain alone sells a solved problem as an achievement; cross-dataset alone hides that the model may simply be weak everywhere. |
| `threshold`, `band_width`, `coverage`, `selective_risk`, `selective_fpr` | A three-state verdict has no accuracy. Coverage is how often it answered at all, and a system can improve every rate by answering less. Since Batch 7 `band_width` is a *derived* probability width — the band's parameter is a log-odds margin, and both come out of one derivation over the calibration split. See [THRESHOLD_BAND_DESIGN.md](THRESHOLD_BAND_DESIGN.md). |
| `worst_slice` | The average is not the product. Low resolution and heavy compression are where a detector fails and where users need it. |
| `n` | An AUC without a sample size is not a measurement. |

## Slices

Reports are sliced along four axes (`evaluation.dataset.SLICE_DIMENSIONS`):

- `dataset` — the corpus, always available.
- `compression` — normalised to `none` / `light` / `heavy` / `unknown`, not
  corpus-native labels. The point of a slice is to compare *across* corpora:
  FF++ c40 and a re-shared WhatsApp upload are the same regime, and a slice keyed
  on the raw label would never put them in one bucket. `Compression.from_ffpp`
  and `.from_crf` hold the mappings.
- `resolution` — keyed on the **short side** of the frame, because a face crop's
  usable detail is set by the smaller dimension. Bucketing on pixel count would
  rank 2560×1080 above 1920×1080 despite identical facial detail.
- `generator` — a free string, deliberately not an enum. Generators appear faster
  than this file can be edited, and an enum would mean a new one either crashes
  the index or is recorded as `OTHER`, which is the same as not recording it.
  `slice_counts` exists partly so a typo shows up as a slice of one.

`Compression.UNKNOWN` and `None` are different states. `None` means nobody
recorded it; `UNKNOWN` means someone looked and could not tell.

Items with nothing recorded on an axis are **dropped** from `group_by_slice` by
default rather than forming an "unknown" cohort — a group whose only shared
property is that nobody measured it invites a conclusion it cannot support.
`slice_counts` always shows the remainder, so a dimension recorded on 4% of the
corpus is visibly useless rather than quietly producing a confident slice AUC.
