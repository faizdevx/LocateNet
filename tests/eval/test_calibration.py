import json

import numpy as np
import pytest

from eval import metrics as M
from eval.calibrate import fit_isotonic, fit_platt, prior_weights
from services.calibration import Calibrator, adjust_prior


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0.65, 0.1, 400), rng.normal(0.15, 0.1, 1600)])
    y = np.concatenate([np.ones(400, int), np.zeros(1600, int)])
    return s, y, y.astype(bool)


def test_isotonic_matches_sklearn_and_is_monotonic(data, tmp_path):
    s, y, m = data
    spec, iso = fit_isotonic(s, y, prior_weights(m, 0.1))
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({**spec, "prior": 0.1}))
    cal = Calibrator.from_json(path)
    held_out = np.random.default_rng(1).uniform(-0.3, 1.1, 500)
    ours = cal.predict_many(held_out)
    assert np.allclose(ours, iso.predict(held_out), atol=1e-9)
    grid = np.linspace(-1, 1, 400)
    out = cal.predict_many(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_platt_matches_sklearn(data):
    from sklearn.linear_model import LogisticRegression
    s, y, m = data
    w = prior_weights(m, 0.1)
    spec = fit_platt(s, y, w)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y, sample_weight=w)
    held_out = np.linspace(-0.2, 1.0, 50)
    assert np.allclose(Calibrator({**spec, "prior": 0.1}).predict_many(held_out),
                       lr.predict_proba(held_out.reshape(-1, 1))[:, 1], atol=1e-9)


def test_calibration_improves_ece_over_raw(data):
    s, y, m = data
    w = prior_weights(m, 0.1)
    spec, _ = fit_isotonic(s, y, w)
    p = Calibrator({**spec, "prior": 0.1}).predict_many(s)
    raw = np.clip((s - s.min()) / (s.max() - s.min()), 0, 1)
    assert M.ece(p, y, weights=w) < M.ece(raw, y, weights=w)


def test_prior_weights_give_requested_mix():
    m = np.array([True] * 10 + [False] * 990)
    w = prior_weights(m, 0.1)
    assert w[m].sum() / w.sum() == pytest.approx(0.1)


def test_bayes_prior_adjustment():
    assert adjust_prior(0.5, 0.1, 0.1) == 0.5
    # at p=0.5 under a 10% prior, the likelihood ratio is 9; under a 50% prior that gives 0.9
    assert float(adjust_prior(0.5, 0.1, 0.5)) == pytest.approx(0.9)
    cal = Calibrator({"type": "platt", "a": 10.0, "b": -5.0, "prior": 0.1})
    assert cal.predict(0.5, prior=0.5) == pytest.approx(0.9)
    assert cal.predict(None) is None
