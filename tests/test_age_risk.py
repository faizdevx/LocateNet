import io
from datetime import date
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app as app_module
from services import age_risk, ages

from conftest import csrf_token, login, make_user

RESULTS = {"fgnet_1to1": {
    "all": {"6-10": {"fnmr": 0.14, "fnmr_ci": [0.11, 0.17]}},
    "child_enrolment": {"6-10": {"fnmr": 0.17, "fnmr_ci": [0.14, 0.2]}},
    "adult_enrolment": {"6-10": {"fnmr": 0.01, "fnmr_ci": [0.0, 0.02]}}}}


def test_age_arithmetic():
    today = date(2026, 10, 22)
    assert ages.parse_date("2019-05") == date(2019, 5, 1)
    assert ages.parse_date("garbage") is None
    assert ages.current_age(dob="2010-10-22", today=today) == pytest.approx(16, abs=0.01)
    assert ages.current_age(age_reported=12, age_reported_on="2024-10-22", today=today) == pytest.approx(14, abs=0.01)
    assert ages.age_at_photo("2016-10-22", dob="2010-10-22", today=today) == pytest.approx(6, abs=0.01)
    assert ages.photo_years_old("2016-10-22", today=today) == pytest.approx(10, abs=0.01)
    assert ages.photo_years_old(None, age_at=5, now_age=13) == 8


def test_expected_miss_uses_the_right_table_and_bucket():
    assert age_risk.expected_miss(RESULTS, 8, age_at_photo=6)["fnmr"] == 0.17
    assert age_risk.expected_miss(RESULTS, 8, age_at_photo=30)["fnmr"] == 0.01
    assert age_risk.expected_miss(RESULTS, 8)["table"] == "all"
    assert age_risk.expected_miss(RESULTS, None) is None and age_risk.expected_miss(None, 8) is None


def test_warning_only_above_threshold_and_number_comes_from_results():
    est = age_risk.expected_miss(RESULTS, 8, 6)
    assert age_risk.warning_text(4, est, 5) is None
    text = age_risk.warning_text(8, est, 5)
    assert "~8 years old" in text and "~17%" in text and "recent photo" in text


def test_case_page_and_alert_show_the_age_gap_banner(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "AGE_RESULTS", RESULTS)
    app_module.search_svc.rebuild_face_index([])
    make_user("owner")
    login(client, "owner")
    emb = np.random.default_rng(3).normal(size=512).astype("float32")
    app_module.face_svc.faces = [SimpleNamespace(bbox=np.array([5, 5, 60, 60], "float32"), embedding=emb, det_score=0.99)]
    img = io.BytesIO(cv2.imencode(".jpg", np.full((80, 80, 3), 120, dtype="uint8"))[1].tobytes())
    photo_year = date.today().year - 8
    r = client.post("/register_case", data={"person_name": "Kid", "city": "Delhi", "dob": f"{photo_year - 6}-01-01",
                                            "images": [(img, "kid.jpg")], "photo_dates": [f"{photo_year}-{date.today().month:02d}"],
                                            "csrf_token": csrf_token(client, "/register_case")},
                    content_type="multipart/form-data")
    app_module.face_svc.faces = []
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Reference photo is ~8 years old" in page and "~17%" in page


def test_tiny_measured_rate_is_worded_as_low_not_zero():
    est = {"bucket": "6-10", "fnmr": 0.0, "fnmr_ci": [0.0, 0.0], "table": "adult_enrolment"}
    text = age_risk.warning_text(10, est, 5)
    assert "low, under 3%" in text and "~0%" not in text
