import json

import numpy as np
import pytest

from services import recalibration
from services.feedback import suppression_decision, template_growth_check
from services.field_metrics import calibration_by_bucket, rate, review_latency_hours, weekly_precision, wilson_ci


def unit(*xs):
    v = np.array(xs, dtype="float32")
    return v / np.linalg.norm(v)


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert all(x != x for x in wilson_ci(0, 0))  # nan when n = 0
    lo, hi = wilson_ci(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.0 < hi < 0.08


def test_insufficient_data_hides_rates():
    small = rate(9, 10)
    assert small["enough"] is False and small["rate"] is None and small["ci"] is None
    big = rate(27, 30)
    assert big["enough"] and big["rate"] == pytest.approx(0.9) and big["ci"][0] < 0.9 < big["ci"][1]


def test_weekly_precision_and_calibration_buckets():
    rows = [{"review_decision": "confirm_match", "created_at": "2026-10-19 10:00:00", "similarity": 0.55,
             "calibrated_likelihood": 0.8}] * 30 + \
           [{"review_decision": "false_positive", "created_at": "2026-10-20 10:00:00", "similarity": 0.33,
             "calibrated_likelihood": 0.05}] * 10 + \
           [{"review_decision": "ignore", "created_at": "2026-10-20 10:00:00", "similarity": 0.33}]
    w = weekly_precision(rows)
    assert list(w) == ["2026-W43"] and w["2026-W43"]["n"] == 40 and w["2026-W43"]["rate"] == pytest.approx(0.75)
    c = calibration_by_bucket(rows, [0.312, 0.4, 0.5, 0.634, 1.01])
    assert c["0.50-0.63"]["rate"] == 1.0 and c["0.50-0.63"]["mean_predicted"] == pytest.approx(0.8)
    assert c["0.31-0.40"]["enough"] is False and c["0.31-0.40"]["rate"] is None


def test_review_latency():
    rows = [{"created_at": "2026-10-20 10:00:00", "review_timestamp": "2026-10-20 12:00:00"},
            {"created_at": "2026-10-20 10:00:00", "review_timestamp": None}]
    assert review_latency_hours(rows) == (2.0, 2.0, 1)


GOOD = dict(is_owner_or_admin=True, n_existing_sighting_templates=0, max_sighting_templates=3, face_px=80,
            det_score=0.9, det_score_min=0.6, min_face_px=32, embedding=unit(1, 0.1, 0),
            enrolment_embeddings=[unit(1, 0, 0)], match_threshold=0.312)


@pytest.mark.parametrize("change,word", [
    ({"is_owner_or_admin": False}, "officer"),
    ({"n_existing_sighting_templates": 3}, "already has 3"),
    ({"det_score": 0.3}, "unclear"),
    ({"face_px": 20}, "too small"),
    ({"embedding": unit(0, 1, 0)}, "not similar enough"),
])
def test_anti_poisoning_guard(change, word):
    assert template_growth_check(**GOOD) == (True, None)
    ok, reason = template_growth_check(**{**GOOD, **change})
    assert not ok and word in reason


def test_suppression_downgrades_only_when_closer_to_a_rejected_face():
    probe = unit(1, 0.05, 0)
    assert suppression_decision(probe, 0.4, [], 0.6) == (False, None)
    down, neg = suppression_decision(probe, 0.4, [unit(1, 0, 0)], 0.6)
    assert down and neg > 0.99
    assert not suppression_decision(probe, 0.999, [unit(1, 0, 0)], 0.6)[0]   # closer to the case itself
    assert not suppression_decision(probe, 0.4, [unit(0, 1, 0)], 0.6)[0]     # rejected face is someone else


def test_recalibration_minimum_n_guards():
    assert recalibration.check_minimums([1] * 150 + [0] * 49)[0] is False      # < 200 reviewed
    ok, why = recalibration.check_minimums([1] * 190 + [0] * 20)
    assert not ok and "30" in why                                             # too few false positives
    assert recalibration.check_minimums([1] * 170 + [0] * 30) == (True, None)


def test_recalibration_proposal_and_atomic_approve(tmp_path):
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0.6, 0.08, 150), rng.normal(0.36, 0.04, 80)])
    y = np.concatenate([np.ones(150), np.zeros(80)])
    cal = tmp_path / "calibrator.json"
    cal.write_text(json.dumps({"type": "platt", "a": 29.0, "b": -13.0, "prior": 0.1}))
    proposal, reason = recalibration.build_proposal(s, y, json.loads(cal.read_text()))
    assert reason is None and proposal["status"] == "pending" and proposal["calibrator"]["a"] > 0
    path = recalibration.write_proposal(str(tmp_path), proposal)
    name = path.replace("\\", "/").rsplit("/", 1)[1]
    done = recalibration.decide(str(tmp_path), name, True, "boss", str(cal))
    assert done["status"] == "approved" and done["decided_by"] == "boss"
    assert json.loads(cal.read_text())["a"] == pytest.approx(proposal["calibrator"]["a"])
    assert (tmp_path / done["previous_calibrator_backup"]).exists()
    with pytest.raises(ValueError):
        recalibration.decide(str(tmp_path), name, True, "boss", str(cal))     # not pending any more
    with pytest.raises(ValueError):
        recalibration.decide(str(tmp_path), "../x.json", True, "boss", str(cal))
