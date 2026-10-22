"""Shared fixtures. Models are mocked so the tests need no insightface, onnx or torch."""
import importlib.util
import os
import re
import sqlite3
import sys
import tempfile
import types

import bcrypt
import numpy as np
import pytest

# tests/eval needs requirements-eval.txt (pandas, scikit-learn, scipy, PyYAML); skip it when those are
# not installed (the lightweight CI job). Paths here are relative to this conftest's folder.
_EVAL_DEPS = ("pandas", "sklearn", "scipy", "yaml")
collect_ignore_glob = [] if all(importlib.util.find_spec(m) for m in _EVAL_DEPS) else ["eval/*"]

# Everything the app writes (database, FAISS index, crops) goes to a throwaway directory.
_TMP = tempfile.mkdtemp(prefix="locatenet-tests-")
os.environ["LOCATENET_SECRET_KEY"] = "test-secret-key"
os.environ["LOCATENET_DB"] = os.path.join(_TMP, "test.db")
os.chdir(_TMP)

# Stub the heavy libraries before the app imports them.
_insightface = types.ModuleType("insightface")
_insightface_app = types.ModuleType("insightface.app")


class _FakeFaceAnalysis:
    def __init__(self, *args, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        pass

    def get(self, frame):
        return []


_insightface_app.FaceAnalysis = _FakeFaceAnalysis
_insightface.app = _insightface_app
sys.modules["insightface"] = _insightface
sys.modules["insightface.app"] = _insightface_app
sys.modules["ultralytics"] = None  # the video pipeline then reports itself unavailable


class FakeFaceService:
    """Stands in for services.face_service.FaceService."""

    def __init__(self, *args, **kwargs):
        self.faces = []

    def detect_faces(self, frame, pad_retry=None):
        return self.faces

    def get_quality_embedding(self, face):
        return getattr(face, "embedding", None)

    def get_best_face(self, frame):
        return self.faces[0] if self.faces else None

    def get_single_embedding(self, frame):
        face = self.get_best_face(frame)
        return face.embedding if face is not None else None


import services.face_service as _face_module  # noqa: E402

_face_module.FaceService = FakeFaceService

import app as app_module  # noqa: E402
import models  # noqa: E402

# Never let tests write case photos or sighting media into the real project folders.
for _key, _name in (("UPLOAD_FOLDER", "uploads"), ("RESOURCES_FOLDER", "resources")):
    app_module.app.config[_key] = os.path.join(_TMP, _name)
    os.makedirs(app_module.app.config[_key], exist_ok=True)

TABLES = ["matches", "sighting_face_reviews", "sighting_faces", "public_submissions", "case_embeddings", "audit_log",
          "negative_exemplars",
          "cases", "users"]


@pytest.fixture(scope="session")
def flask_app():
    return app_module.app


@pytest.fixture(autouse=True)
def clean_db():
    models.create_db()
    with models.get_db_connection() as conn:
        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
    yield


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


def make_user(username, role="Officer", password="pw"):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    models.add_new_user(username, hashed, username.title(), role)


def csrf_token(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def login(client, username, password="pw"):
    token = csrf_token(client)
    return client.post("/login", data={"username": username, "password": password, "csrf_token": token})


def add_case(officer="officer1", name="Missing Person", city="Aligarh", status="NF"):
    case_id = models.add_case(officer, name, city, "uploads/x.jpg", 1.0, 2.0, [0.1] * 4)
    if status != "NF":
        models.resolve_case(case_id)
    return case_id


def add_match(case_id, sighting_id, location, when):
    """Insert a public submission at `location` and a match for `case_id` detected at `when`."""
    with models.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO public_submissions (id, submitted_by, location, mobile, status, date_submitted) "
            "VALUES (?, 'p', ?, '9876543210', 'NF', ?)",
            (sighting_id, location, when),
        )
        conn.execute(
            "INSERT INTO matches (case_id, sighting_id, confidence, date_detected) VALUES (?, ?, 50, ?)",
            (case_id, sighting_id, when),
        )
        conn.commit()


def unit_vector(seed, dim=512):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype("float32")
    return vec / np.linalg.norm(vec)


def db_rows(query, params=()):
    conn = sqlite3.connect(os.environ["LOCATENET_DB"])
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()
