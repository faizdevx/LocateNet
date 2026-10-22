import json
from datetime import datetime, timedelta

import models
from scripts.purge_expired import purge

from conftest import add_case, db_rows


def _ts(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def test_purge_removes_only_expired_feedback_data(tmp_path):
    case_id = add_case()
    with models.get_db_connection() as conn:
        for days in (120, 10):
            conn.execute("INSERT INTO negative_exemplars (case_id, embedding, created_at) VALUES (?, ?, ?)",
                         (case_id, json.dumps([0.1] * 4), _ts(days)))
            conn.execute("INSERT INTO sighting_faces (sighting_id, face_crop_path, created_at, qa_sample, embedding) "
                         "VALUES ('s', 'resources/x.jpg', ?, 1, '[0.1]')", (_ts(days),))
            conn.execute("INSERT INTO sighting_faces (sighting_id, face_crop_path, created_at, qa_sample, embedding) "
                         "VALUES ('s', 'resources/y.jpg', ?, 0, '[0.1]')", (_ts(days),))
        conn.commit()
    dry = purge(90, dry_run=True, root=str(tmp_path))
    assert (dry["negative_exemplars"], dry["qa_samples"], dry["face_embeddings_cleared"]) == (1, 1, 2)
    assert db_rows("SELECT COUNT(*) FROM negative_exemplars") == [(2,)]       # dry run changed nothing
    purge(90, root=str(tmp_path))
    assert db_rows("SELECT COUNT(*) FROM negative_exemplars") == [(1,)]
    assert db_rows("SELECT COUNT(*) FROM sighting_faces WHERE qa_sample = 1") == [(1,)]
    # the old regular face keeps its row and decision, only its embedding is cleared
    assert db_rows("SELECT embedding IS NULL FROM sighting_faces WHERE qa_sample = 0 ORDER BY created_at") == [(1,), (0,)]
