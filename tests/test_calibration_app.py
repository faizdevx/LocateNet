import io
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app as app_module
from services.calibration import Calibrator

from conftest import csrf_token, db_rows, login, make_user

PLATT = {"type": "platt", "a": 29.0, "b": -13.0, "prior": 0.1}


def face(seed=0):
    emb = np.random.default_rng(seed).normal(size=512).astype("float32")
    return SimpleNamespace(bbox=np.array([5, 5, 60, 60]), embedding=emb, det_score=0.99)


def jpeg():
    return io.BytesIO(cv2.imencode(".jpg", np.full((80, 80, 3), 120, dtype="uint8"))[1].tobytes())


def test_missing_calibrator_file_falls_back_to_none(tmp_path):
    assert app_module.load_calibrator(str(tmp_path / "missing.json")) is None


def test_broken_calibrator_file_falls_back_to_none(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"type": "unknown"}))
    assert app_module.load_calibrator(str(path)) is None


def test_estimate_uses_calibrator_when_present(monkeypatch):
    monkeypatch.setattr(app_module, "calibrator", Calibrator(PLATT))
    expected = Calibrator(PLATT).predict(0.5, prior=app_module.config.CALIBRATION_PRIOR)
    assert app_module.estimate_likelihood(0.5) == pytest.approx(expected)
    monkeypatch.setattr(app_module, "calibrator", None)
    assert app_module.estimate_likelihood(0.5) is None


@pytest.mark.parametrize("with_calibrator", [True, False])
def test_sighting_stores_similarity_and_likelihood(client, monkeypatch, with_calibrator):
    monkeypatch.setattr(app_module, "calibrator", Calibrator(PLATT) if with_calibrator else None)
    app_module.search_svc.rebuild_face_index([])
    make_user("owner")
    login(client, "owner")
    app_module.face_svc.faces = [face(1)]
    client.post("/register_case", data={"person_name": "P", "city": "Delhi", "image": (jpeg(), "a.jpg"),
                                        "csrf_token": csrf_token(client, "/register_case")},
                content_type="multipart/form-data")

    public = app_module.app.test_client()
    public.post("/report_sighting", data={"name": "R", "mobile": "9876543210", "location": "Delhi",
                                          "upload_type": "image", "image": (jpeg(), "s.jpg"),
                                          "csrf_token": csrf_token(public, "/report_sighting")},
                content_type="multipart/form-data")
    app_module.face_svc.faces = []

    [(sim, lik, prior)] = db_rows("SELECT similarity, calibrated_likelihood, likelihood_prior FROM sighting_faces")
    [(m_sim, m_lik)] = db_rows("SELECT similarity, calibrated_likelihood FROM matches")
    assert sim == pytest.approx(1.0, abs=1e-5) and m_sim == pytest.approx(sim)
    if with_calibrator:
        assert lik == pytest.approx(Calibrator(PLATT).predict(sim, prior=0.1)) and m_lik == pytest.approx(lik)
        assert prior == pytest.approx(0.1)
    else:
        assert lik is None and m_lik is None and prior is None

    page = client.get("/dashboard").get_data(as_text=True)
    assert "Similarity 1.00" in page and "Needs human confirmation" in page
    assert ("Est. match likelihood" in page) == with_calibrator
