import io
from types import SimpleNamespace

import cv2
import numpy as np

import app as app_module

from conftest import csrf_token, db_rows, login, make_user


def face(seed=0, width=55):
    emb = np.random.default_rng(seed).normal(size=512).astype("float32")
    return SimpleNamespace(bbox=np.array([5, 5, 5 + width, 5 + width], dtype="float32"), embedding=emb, det_score=0.99)


def jpeg():
    return io.BytesIO(cv2.imencode(".jpg", np.full((80, 80, 3), 120, dtype="uint8"))[1].tobytes())


def register(client, n_photos=2, faces=None):
    app_module.face_svc.faces = faces or [face(1)]
    data = {"person_name": "P", "city": "Delhi", "csrf_token": csrf_token(client, "/register_case"),
            "images": [(jpeg(), f"p{i}.jpg") for i in range(n_photos)], "photo_dates": ["2019-05", ""][:n_photos],
            "age": "12"}
    r = client.post("/register_case", data=data, content_type="multipart/form-data")
    app_module.face_svc.faces = []
    return r


def index_ids():
    from vector_db.rebuild import index_ids as ids
    return ids(app_module.search_svc)


def setup_function():
    app_module.search_svc.rebuild_face_index([])
    app_module.reload_matcher()


def test_register_with_several_photos_stores_one_vector_per_photo(client):
    make_user("owner")
    login(client, "owner")
    register(client, n_photos=2)
    rows = db_rows("SELECT id, case_id, source, photo_date, age_at_photo FROM case_embeddings ORDER BY id")
    assert len(rows) == 2 and rows[0][2] == "enrolment" and rows[0][3] == "2019-05-01"
    assert rows[0][4] is not None  # age at photo derived from the reported age
    assert index_ids() == {r[0] for r in rows}
    assert app_module.template_state["matcher"].vector_to_case == {r[0]: r[1] for r in rows}


def test_bad_photos_are_rejected_with_a_reason(client):
    make_user("owner")
    login(client, "owner")
    r = register(client, n_photos=1, faces=[face(1, width=20)])
    with client.session_transaction() as s:
        msgs = " ".join(m for _, m in s.get("_flashes", []))
    assert r.status_code == 302 and "too small" in msgs
    assert db_rows("SELECT COUNT(*) FROM cases") == [(0,)]


def test_add_and_deactivate_reference_photo_with_audit_and_permissions(client, flask_app):
    make_user("owner")
    make_user("other")
    login(client, "owner")
    register(client, n_photos=1)
    [(case_id,)] = db_rows("SELECT id FROM cases")

    other = flask_app.test_client()
    login(other, "other")
    assert other.get(f"/case/{case_id}").status_code == 403
    token = csrf_token(other, "/register_case")
    assert other.post(f"/case/{case_id}/photos", data={"csrf_token": token}).status_code == 403

    app_module.face_svc.faces = [face(2)]
    client.post(f"/case/{case_id}/photos", data={"csrf_token": csrf_token(client, "/register_case"),
                                                 "image": (jpeg(), "new.jpg"), "photo_date": "2024-01"},
                content_type="multipart/form-data")
    app_module.face_svc.faces = []
    ids = [r[0] for r in db_rows("SELECT id FROM case_embeddings ORDER BY id")]
    assert len(ids) == 2 and index_ids() == set(ids)

    assert other.post(f"/case/{case_id}/photo/{ids[0]}/deactivate", data={"csrf_token": token}).status_code == 403
    client.post(f"/case/{case_id}/photo/{ids[0]}/deactivate", data={"csrf_token": csrf_token(client, "/register_case")})
    assert index_ids() == {ids[1]}
    # the last active photo cannot be deactivated
    client.post(f"/case/{case_id}/photo/{ids[1]}/deactivate", data={"csrf_token": csrf_token(client, "/register_case")})
    assert index_ids() == {ids[1]}
    actions = [r[0] for r in db_rows("SELECT action FROM audit_log WHERE case_id = ? ORDER BY id", (case_id,))]
    assert actions == ["case_registered", "photo_added", "photo_deactivated"]
    assert client.get(f"/case/{case_id}").status_code == 200


def test_resolving_a_case_removes_all_its_vectors(client):
    make_user("owner")
    login(client, "owner")
    register(client, n_photos=2)
    [(case_id,)] = db_rows("SELECT id FROM cases")
    assert len(index_ids()) == 2
    client.post(f"/resolve/{case_id}", data={"csrf_token": csrf_token(client, "/register_case")})
    assert index_ids() == set()
    assert app_module.template_state["matcher"].vector_to_case == {}
