from vector_db.search_service import SearchService

from conftest import unit_vector


def make_service(tmp_path):
    return SearchService(index_path=str(tmp_path / "vectors"))


def test_add_and_search_returns_closest(tmp_path):
    svc = make_service(tmp_path)
    svc.add_face_embedding(unit_vector(1), 10)
    svc.add_face_embedding(unit_vector(2), 20)
    scores, ids = svc.search_face(unit_vector(1), top_k=2)
    assert ids[0] == 10
    assert scores[0] > 0.99


def test_minus_one_ids_are_filtered_when_index_is_small(tmp_path):
    svc = make_service(tmp_path)
    svc.add_face_embedding(unit_vector(1), 10)
    scores, ids = svc.search_face(unit_vector(1), top_k=5)
    assert list(ids) == [10]
    assert len(scores) == 1


def test_empty_index_returns_nothing(tmp_path):
    scores, ids = make_service(tmp_path).search_face(unit_vector(1))
    assert len(scores) == 0 and len(ids) == 0


def test_remove_face_drops_the_vector(tmp_path):
    svc = make_service(tmp_path)
    svc.add_face_embedding(unit_vector(1), 10)
    svc.add_face_embedding(unit_vector(2), 20)
    assert svc.remove_face(10) == 1
    assert svc.face_index.ntotal == 1
    _, ids = svc.search_face(unit_vector(1), top_k=5)
    assert 10 not in list(ids)


def test_save_and_load_round_trip(tmp_path):
    svc = make_service(tmp_path)
    svc.add_face_embedding(unit_vector(1), 10)
    svc.add_face_embedding(unit_vector(2), 20)
    svc.remove_face(20)

    reloaded = make_service(tmp_path)
    assert reloaded.face_index.ntotal == 1
    _, ids = reloaded.search_face(unit_vector(1), top_k=3)
    assert list(ids) == [10]


def test_rebuild_replaces_the_index(tmp_path):
    svc = make_service(tmp_path)
    svc.add_face_embedding(unit_vector(1), 10)
    assert svc.rebuild_face_index([(5, unit_vector(5)), (6, unit_vector(6))]) == 2
    assert make_service(tmp_path).face_index.ntotal == 2
