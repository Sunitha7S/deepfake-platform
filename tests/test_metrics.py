"""Metric tests, against hand-computed values.

Every expected number below was worked out by hand rather than taken from a
previous run of the code. A test that asserts what the implementation currently
does is a change detector, not a correctness check — it passes just as happily
when the implementation is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.metrics import (  # noqa: E402
    accuracy,
    confusion_matrix,
    equal_error_rate,
    f1,
    precision,
    recall,
    roc_auc,
    summary,
)

# The worked example used throughout.
#
#   y_true   0     0     1     1
#   y_score  0.1   0.4   0.35  0.8
#
# At threshold 0.5:  predictions 0, 0, 0, 1
#   tn=2 fp=0 fn=1 tp=1
#   accuracy  = 3/4 = 0.75
#   precision = 1/1 = 1.0
#   recall    = 1/2 = 0.5
#   f1        = 2*1*0.5/1.5 = 0.6666...
#
# AUC by pairs: positives {0.35, 0.8}, negatives {0.1, 0.4}
#   0.35 > 0.1  win     0.35 < 0.4  loss
#   0.80 > 0.1  win     0.80 > 0.4  win
#   3 wins of 4 pairs = 0.75
Y = np.array([0, 0, 1, 1])
S = np.array([0.1, 0.4, 0.35, 0.8])


def test_confusion_matrix_hand_computed():
    cm = confusion_matrix(Y, S, threshold=0.5)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (2, 0, 1, 1)
    assert cm.as_array().tolist() == [[2, 0], [1, 1]]
    assert cm.total == 4
    assert cm.threshold == 0.5


def test_accuracy_hand_computed():
    assert accuracy(Y, S, threshold=0.5) == 0.75


def test_precision_hand_computed():
    assert precision(Y, S, threshold=0.5) == 1.0


def test_recall_hand_computed():
    assert recall(Y, S, threshold=0.5) == 0.5


def test_f1_hand_computed():
    assert abs(f1(Y, S, threshold=0.5) - (2 / 3)) < 1e-12


def test_roc_auc_hand_computed():
    assert roc_auc(Y, S) == 0.75


def test_rates_hand_computed():
    cm = confusion_matrix(Y, S, threshold=0.5)
    assert cm.fpr == 0.0
    assert cm.tpr == 0.5


def test_threshold_changes_the_answer():
    """A metric reported without its threshold is a number without units."""
    assert accuracy(Y, S, threshold=0.3) == 0.75
    cm = confusion_matrix(Y, S, threshold=0.3)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 0, 2)


# --------------------------------------------------------------------------
# Special cases
# --------------------------------------------------------------------------


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0
    assert accuracy(y, s, threshold=0.5) == 1.0
    assert precision(y, s, threshold=0.5) == 1.0
    assert recall(y, s, threshold=0.5) == 1.0
    assert f1(y, s, threshold=0.5) == 1.0


def test_perfectly_inverted_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(y, s) == 0.0


def test_complete_ties_give_exactly_one_half():
    """The tie-handling that matters.

    A saturated detector produces many identical scores. Treating those as
    arbitrary orderings inflates AUC by a small, flattering, entirely fictional
    amount — which is what a trapezoid over a sampled curve does.
    """
    assert roc_auc(np.array([0, 1, 0, 1]), np.full(4, 0.5)) == 0.5
    assert roc_auc(np.array([0, 0, 0, 1]), np.full(4, 0.9)) == 0.5


def test_partial_ties_are_averaged_not_ordered():
    #   y      0     1     1     0
    #   s      0.5   0.5   0.9   0.1
    # positives {0.5, 0.9}, negatives {0.5, 0.1}
    #   0.5 vs 0.5 tie = 0.5   0.5 vs 0.1 win
    #   0.9 vs 0.5 win         0.9 vs 0.1 win
    #   (0.5 + 1 + 1 + 1) / 4 = 0.875
    y = np.array([0, 1, 1, 0])
    s = np.array([0.5, 0.5, 0.9, 0.1])
    assert abs(roc_auc(y, s) - 0.875) < 1e-12


def test_all_positive_labels():
    y, s = np.ones(5), np.linspace(0.1, 0.9, 5)
    assert recall(y, s, threshold=0.5) == 0.6
    assert precision(y, s, threshold=0.5) == 1.0
    try:
        roc_auc(y, s)
    except ValueError as exc:
        assert "both classes" in str(exc)
        return
    raise AssertionError("AUC was computed with one class")


def test_all_negative_labels():
    #   y      0     0     0     0     0
    #   s      0.1   0.3   0.5   0.7   0.9
    # at threshold 0.5 three are predicted positive, all wrongly:
    #   tn=2 fp=3 fn=0 tp=0  →  accuracy 2/5 = 0.4
    y, s = np.zeros(5), np.linspace(0.1, 0.9, 5)
    cm = confusion_matrix(y, s, threshold=0.5)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (2, 3, 0, 0)
    assert accuracy(y, s, threshold=0.5) == 0.4
    assert cm.fpr == 0.6
    assert recall(y, s, threshold=0.5) == 0.0, "no positives means nothing to recall"
    assert precision(y, s, threshold=0.5) == 0.0
    assert f1(y, s, threshold=0.5) == 0.0


def test_no_predicted_positives_gives_zero_not_nan():
    y = np.array([0, 1])
    s = np.array([0.1, 0.2])
    assert precision(y, s, threshold=0.9) == 0.0
    assert f1(y, s, threshold=0.9) == 0.0


def test_empty_input_is_rejected():
    for fn in (accuracy, precision, recall, f1):
        try:
            fn(np.array([]), np.array([]))
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted empty input")


def test_length_mismatch_is_rejected():
    try:
        accuracy(np.array([0, 1]), np.array([0.5]))
    except ValueError as exc:
        assert "mismatch" in str(exc)
        return
    raise AssertionError("mismatched lengths were accepted")


def test_non_binary_labels_are_rejected():
    try:
        accuracy(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))
    except ValueError as exc:
        assert "0/1" in str(exc)
        return
    raise AssertionError("non-binary labels were accepted")


# --------------------------------------------------------------------------
# Derived
# --------------------------------------------------------------------------


def test_equal_error_rate_on_symmetric_data():
    y = np.concatenate([np.zeros(100), np.ones(100)])
    s = np.concatenate([np.linspace(0.0, 0.6, 100), np.linspace(0.4, 1.0, 100)])
    eer, threshold = equal_error_rate(y, s)
    assert 0.0 <= eer <= 0.5
    assert 0.0 <= threshold <= 1.0


def test_eer_is_zero_under_perfect_separation():
    y = np.concatenate([np.zeros(50), np.ones(50)])
    s = np.concatenate([np.zeros(50), np.ones(50)])
    eer, _ = equal_error_rate(y, s)
    assert eer == 0.0


def test_summary_reports_its_threshold():
    out = summary(Y, S, threshold=0.5)
    assert out["threshold"] == 0.5
    assert out["n"] == 4
    assert out["accuracy"] == 0.75
    assert out["roc_auc"] == 0.75
    assert out["tp"] == 1 and out["tn"] == 2


def test_auc_is_invariant_to_monotone_rescaling():
    """The property that makes AUC threshold-free — and the reason calibration
    cannot improve it."""
    from inference.calibration import sigmoid

    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([-2.0, -0.5, 0.3, 2.0, -1.0, 1.0])
    assert abs(roc_auc(y, s) - roc_auc(y, np.asarray(sigmoid(s / 3.0)))) < 1e-12


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
