import numpy as np
import pytest

from pipeline.detection_pipeline import DetectionPipeline
from pipeline.scoring import CATEGORY_HIGH, CATEGORY_NONE


class FakeFace:
    bbox = np.array([5, 5, 40, 40])
    embedding = np.ones(512, dtype="float32")
    det_score = 0.99


class FakeFaceService:
    def detect_faces(self, frame):
        return [FakeFace()]

    def get_quality_embedding(self, face):
        return face.embedding


class FakeSearch:
    def __init__(self, scores, ids):
        self.scores, self.ids = np.array(scores, dtype="float32"), np.array(ids)

    def search_face(self, embedding, top_k=5):
        return self.scores, self.ids


@pytest.fixture
def frame():
    return np.zeros((64, 64, 3), dtype="uint8")


def run(tmp_path, frame, scores, ids, is_active=None):
    pipeline = DetectionPipeline(FakeFaceService(), None, FakeSearch(scores, ids), storage_path=str(tmp_path))
    return pipeline.process_frame(frame, is_active=is_active)


def test_low_score_is_not_a_match(tmp_path, frame):
    [result] = run(tmp_path, frame, [0.2], [1])
    assert result["id"] == -1
    assert result["category"] == CATEGORY_NONE
    assert "confidence" not in result
    assert result["similarity"] == pytest.approx(0.2)


def test_resolved_rank1_falls_back_to_active_rank2(tmp_path, frame):
    [result] = run(tmp_path, frame, [0.9, 0.7], [1, 2], is_active=lambda case_id: case_id == 2)
    assert result["id"] == 2
    assert result["similarity"] == pytest.approx(0.7)
    assert result["category"] == CATEGORY_HIGH


def test_only_resolved_candidates_give_no_match(tmp_path, frame):
    [result] = run(tmp_path, frame, [0.9, 0.8], [1, 2], is_active=lambda case_id: False)
    assert result["id"] == -1
    assert result["category"] == CATEGORY_NONE


def test_minus_one_ids_are_ignored(tmp_path, frame):
    [result] = run(tmp_path, frame, [0.8, 0.0], [-1, -1])
    assert result["id"] == -1
