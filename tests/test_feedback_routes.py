import io
import json
from types import SimpleNamespace

import cv2
import numpy as np

import app as app_module
import config

from conftest import csrf_token, db_rows, login, make_user


def emb(seed):
    v = np.random.default_rng(seed).normal(size=512).astype("float32")
    return v / np.linalg.norm(v)


def face(e, width=55):
    return SimpleNamespace(bbox=np.array([5, 5, 5 + width, 5 + width], "float32"), embedding=e, det_score=0.99)


def jpeg():
    return io.BytesIO(cv2.imencode(".jpg", np.full((80, 80, 3), 120, dtype="uint8"))[1].tobytes())


def setup_function():
    app_module.search_svc.rebuild_face_index([])
    app_module.reload_matcher()


def register(client, e):
    app_module.face_svc.faces = [face(e)]
    client.post("/register_case", data={"person_name": "P", "city": "Delhi", "images": [(jpeg(), "p.jpg")],
                                        "csrf_token": csrf_token(client, "/register_case")}, content_type="multipart/form-data")
    app_module.face_svc.faces = []
    return db_rows("SELECT id FROM cases ORDER BY id DESC")[0][0]


def sighting(faces):
    app_module.face_svc.faces = faces
    public = app_module.app.test_client()
    public.post("/report_sighting", data={"name": "R", "mobile": "9876543210", "location": "Delhi", "upload_type": "image",
                                          "image": (jpeg(), "s.jpg"), "csrf_token": csrf_token(public, "/report_sighting")},
                content_type="multipart/form-data")
    app_module.face_svc.faces = []
    return db_rows("SELECT id, sighting_id FROM sighting_faces ORDER BY id DESC")[0]


def review(client, face_id, sighting_id, decision, add=False):
    data = {"decision": decision, "csrf_token": csrf_token(client, "/register_case")}
    if add:
        data["add_as_template"] = "1"
    return client.post(f"/sighting/{sighting_id}/review/{face_id}", data=data)


def near(e, noise, seed):
    v = e + noise * np.random.default_rng(seed).normal(size=512).astype("float32") / 22.6
    return v / np.linalg.norm(v)


def test_confirm_and_add_as_template_with_guards(client):
    make_user("owner")
    login(client, "owner")
    base = emb(1)
    case_id = register(client, base)
    fid, sid = sighting([face(near(base, 0.3, 2))])
    review(client, fid, sid, "confirm_match")                   # checkbox off by default: nothing added
    assert db_rows("SELECT COUNT(*) FROM case_embeddings WHERE source = 'confirmed_sighting'") == [(0,)]
    review(client, fid, sid, "confirm_match", add=True)
    assert db_rows("SELECT COUNT(*) FROM case_embeddings WHERE source = 'confirmed_sighting'") == [(1,)]
    assert ("photo_added",) in db_rows("SELECT action FROM audit_log WHERE case_id = ?", (case_id,))


def test_template_growth_refuses_a_different_person(client, monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", -1.0)      # make the stranger alert, then check the guard
    make_user("owner")
    login(client, "owner")
    register(client, emb(1))
    fid, sid = sighting([face(emb(99))])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 0.312)
    review(client, fid, sid, "confirm_match", add=True)
    with client.session_transaction() as s:
        msgs = " ".join(m for _, m in s.get("_flashes", []))
    assert "not similar enough" in msgs
    assert db_rows("SELECT COUNT(*) FROM case_embeddings WHERE source = 'confirmed_sighting'") == [(0,)]


def test_false_positive_becomes_negative_exemplar_and_downgrades_repeats(client, monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 0.0)
    make_user("owner")
    login(client, "owner")
    base = emb(1)
    case_id = register(client, base)
    stranger = near(base, 3.0, 7)                               # loosely similar look-alike
    fid, sid = sighting([face(stranger)])
    review(client, fid, sid, "false_positive")
    assert db_rows("SELECT case_id FROM negative_exemplars") == [(case_id,)]
    fid2, _ = sighting([face(near(stranger, 0.2, 8))])          # the same look-alike comes back
    [(cat, sup, match)] = db_rows("SELECT category, suppressed, match_id FROM sighting_faces WHERE id = ?", (fid2,))
    assert sup == 1 and "previously rejected" in cat and match == case_id   # downgraded, not dropped
    fid3, _ = sighting([face(near(base, 0.2, 9))])              # the real person is not downgraded
    assert db_rows("SELECT suppressed FROM sighting_faces WHERE id = ?", (fid3,)) == [(0,)]


def test_random_qa_sample_is_flagged_and_routed_to_the_case_officer(client, monkeypatch):
    monkeypatch.setattr(config, "AUDIT_SAMPLE_RATE", 1.0)
    make_user("owner")
    login(client, "owner")
    case_id = register(client, emb(1))
    fid, sid = sighting([face(emb(50))])                        # below threshold
    [(qa, match, cand)] = db_rows("SELECT qa_sample, match_id, candidate_id FROM sighting_faces WHERE id = ?", (fid,))
    assert qa == 1 and match == -1 and cand == case_id
    assert "Random QA sample" in client.get("/dashboard").get_data(as_text=True)
    assert review(client, fid, sid, "confirm_match").status_code == 302


def test_model_health_is_admin_only_and_shows_insufficient_data(client, flask_app):
    make_user("boss", role="Admin")
    make_user("owner")
    officer = flask_app.test_client()
    login(officer, "owner")
    assert officer.get("/admin/model-health").status_code == 403
    login(client, "boss")
    page = client.get("/admin/model-health").get_data(as_text=True)
    assert "Model health" in page and ("insufficient data" in page or "no reviewed alerts" in page)
    assert client.get("/admin/model-health/chart/precision.png").mimetype == "image/png"


def test_proposal_approve_and_reject_are_admin_only(client, flask_app, tmp_path, monkeypatch):
    from services import recalibration
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))
    cal = tmp_path / "calibrator.json"
    cal.write_text(json.dumps({"type": "platt", "a": 29.0, "b": -13.0, "prior": 0.1}))
    monkeypatch.setattr(config, "CALIBRATOR_PATH", str(cal))
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0.6, 0.08, 150), rng.normal(0.36, 0.04, 80)])
    y = np.concatenate([np.ones(150), np.zeros(80)])
    names = []
    for _ in range(2):
        p, _ = recalibration.build_proposal(s, y, None)
        names.append(recalibration.write_proposal(str(tmp_path), p).replace("\\", "/").rsplit("/", 1)[1])
    make_user("boss", role="Admin")
    make_user("owner")
    officer = flask_app.test_client()
    login(officer, "owner")
    token = csrf_token(officer, "/register_case")
    assert officer.post(f"/admin/model-health/proposal/{names[0]}",
                        data={"decision": "approve", "csrf_token": token}).status_code == 403
    login(client, "boss")
    token = csrf_token(client, "/register_case")
    client.post(f"/admin/model-health/proposal/{names[0]}", data={"decision": "approve", "csrf_token": token})
    client.post(f"/admin/model-health/proposal/{names[1]}", data={"decision": "reject", "csrf_token": token})
    statuses = {p["name"]: p["status"] for p in recalibration.list_proposals(str(tmp_path))}
    assert statuses[names[0]] == "approved" and statuses[names[1]] == "rejected"
    assert app_module.calibrator.spec["a"] != 29.0               # live calibrator swapped
