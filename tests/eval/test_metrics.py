import numpy as np
import pytest

from eval import metrics as M


NEG = np.linspace(0.1, 1.0, 10)  # 0.1, 0.2, ..., 1.0


def test_threshold_at_rate_counts_errors_exactly():
    t = M.threshold_at_rate(NEG, 0.1)
    assert np.mean(NEG >= t) == pytest.approx(0.1)
    assert np.mean(NEG >= M.threshold_at_rate(NEG, 0.0)) == 0.0
    # a rate between two achievable values rounds down to the stricter one
    assert np.mean(NEG >= M.threshold_at_rate(NEG, 0.25)) == pytest.approx(0.2)


def test_tar_at_far_known_answer():
    tar, t = M.tar_at_far([0.95, 0.5], NEG, 0.1)
    assert tar == 0.5
    assert 0.9 < t <= 0.95


def test_roc_separated_scores_reach_perfect_point():
    far, tar, thr = M.roc([0.8, 0.9], [0.1, 0.2])
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0)
    perfect = (far == 0) & (tar == 1)
    assert perfect.any()
    assert far[-1] == 1.0 and tar[-1] == 1.0


def test_best_accuracy_threshold_separated():
    t, acc = M.best_accuracy_threshold([0.8, 0.9], [0.1, 0.2])
    assert acc == 1.0
    assert M.verification_accuracy([0.8, 0.9], [0.1, 0.2], t) == 1.0


def test_fnir_at_fpir_nist_definition():
    non = [0.1, 0.2, 0.3, 0.4]
    mated = [0.9, 0.35, 0.8]
    correct = [True, True, False]  # third probe: rank-1 is the wrong identity
    fnir, t = M.fnir_at_fpir(mated, correct, non, 0.25)
    assert np.mean(np.array(non) >= t) == 0.25
    assert fnir == pytest.approx(1 / 3)


def test_failed_templates_count_as_misses():
    fnir, _ = M.fnir_at_fpir([0.9, -np.inf], [True, False], [0.1, 0.2], 0.5)
    assert fnir == 0.5


def test_fnir_fpir_at_threshold():
    fnir, fpir = M.fnir_fpir_at_threshold([0.9, 0.3], [True, True], [0.1, 0.6], 0.5)
    assert fnir == 0.5 and fpir == 0.5


def test_det_curve_is_monotonic_tradeoff():
    rng = np.random.default_rng(0)
    mated = rng.normal(0.7, 0.1, 500)
    non = rng.normal(0.2, 0.1, 500)
    fpir, fnir, thr = M.det_curve(mated, np.ones(500, bool), non)
    assert np.all(np.diff(thr) < 0)
    assert np.all(np.diff(fpir) >= 0)
    assert np.all(np.diff(fnir) <= 0)


def test_ece_and_brier_known_answers():
    labels = np.array([0, 1] * 50)
    assert M.ece(np.full(100, 0.5), labels) == pytest.approx(0.0)
    assert M.ece(np.ones(10), np.zeros(10)) == pytest.approx(1.0)
    assert M.brier(np.ones(10), np.zeros(10)) == pytest.approx(1.0)
    assert M.brier(np.full(100, 0.5), labels) == pytest.approx(0.25)


def test_reliability_bins_counts_sum_to_n():
    p = np.random.default_rng(1).random(200)
    bins = M.reliability_bins(p, (p > 0.5).astype(int), n_bins=10)
    assert sum(b[2] for b in bins) == 200


def test_bootstrap_resamples_identities_and_is_seeded():
    # identity A has 3 records (all 1), identity B has 1 record (0): resampling identities, not
    # records, means the mean over records can be 0 (only B drawn) or 1 (only A drawn).
    units = [[1, 1, 1], [0]]

    def mean_over_records(us):
        flat = [x for u in us for x in u]
        return float(np.mean(flat))

    point, lo, hi = M.bootstrap_ci(mean_over_records, units, n=500, seed=3)
    assert point == pytest.approx(0.75)
    assert lo == 0.0 and hi == 1.0
    assert M.bootstrap_ci(mean_over_records, units, n=500, seed=3) == (point, lo, hi)


def test_bootstrap_constant_statistic_has_zero_width():
    point, lo, hi = M.bootstrap_ci(lambda us: 0.42, [[1], [2], [3]], n=50)
    assert point == lo == hi == 0.42
