import json
import sqlite3

import models

WEEK2_SCHEMA = """
CREATE TABLE cases (id INTEGER PRIMARY KEY AUTOINCREMENT, person_name TEXT, city TEXT, officer TEXT, image_path TEXT,
                    status TEXT, date_reported TEXT, latitude REAL, longitude REAL, face_vector TEXT);
CREATE TABLE users (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT, name TEXT);
CREATE TABLE matches (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, sighting_id TEXT, confidence REAL,
                      is_read INTEGER DEFAULT 0, date_detected TEXT, similarity REAL, calibrated_likelihood REAL,
                      likelihood_prior REAL);
"""


def test_week2_database_migrates_additively(tmp_path, monkeypatch):
    db = tmp_path / "week2.db"
    conn = sqlite3.connect(db)
    conn.executescript(WEEK2_SCHEMA)
    conn.execute("INSERT INTO cases (person_name, city, officer, image_path, status, date_reported, face_vector) "
                 "VALUES ('A', 'Delhi', 'o1', 'uploads/a.jpg', 'NF', '2026-01-01 10:00:00', ?)", (json.dumps([0.1] * 4),))
    conn.execute("INSERT INTO cases (person_name, city, officer, image_path, status, date_reported, face_vector) "
                 "VALUES ('B', 'Delhi', 'o1', 'uploads/b.jpg', 'F', '2026-01-02 10:00:00', '[]')")
    conn.execute("INSERT INTO matches (case_id, sighting_id, confidence) VALUES (1, 's1', 50)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(models, "DB_NAME", str(db))
    assert models.create_db() == 1          # only the case with a real vector is copied
    assert models.create_db() == 0          # idempotent

    rows = models.get_case_embeddings(1)
    assert len(rows) == 1 and rows[0]["source"] == "enrolment" and json.loads(rows[0]["embedding"]) == [0.1] * 4
    assert models.get_case_embeddings(2) == []
    case = models.get_case_by_id(1)
    assert case["face_vector"] and "dob" in case.keys()     # old column kept, new columns added
    with models.get_db_connection() as c:
        assert c.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1   # existing data untouched
    assert [r["id"] for r in models.active_template_rows()] == [rows[0]["id"]]
