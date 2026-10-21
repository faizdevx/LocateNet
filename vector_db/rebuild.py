import json
import logging
import os

from models import get_db_connection


def load_active_vectors():
    """Return (case_id, vector) for every NF case that has a stored face vector."""
    items = []
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, face_vector FROM cases WHERE status = 'NF' AND face_vector IS NOT NULL"
        ).fetchall()
    for row in rows:
        try:
            vector = json.loads(row["face_vector"])
        except (TypeError, ValueError):
            continue
        if vector:
            items.append((row["id"], vector))
    return items


def rebuild_face_index(search_service):
    """Rebuild face_index.bin from cases.face_vector WHERE status='NF'. Returns the vector count."""
    return search_service.rebuild_face_index(load_active_vectors())


def ensure_index_in_sync(search_service):
    """Rebuild the index when the file is missing or its size disagrees with the SQL count."""
    expected = len(load_active_vectors())
    file_missing = not os.path.exists(search_service.face_bin)
    actual = search_service.face_index.ntotal
    if (file_missing and expected > 0) or actual != expected:
        logging.warning(
            "FAISS index out of sync (index=%s, SQL active cases=%s, file_missing=%s). Rebuilding.",
            actual, expected, file_missing,
        )
        rebuild_face_index(search_service)
        return True
    if file_missing:
        search_service.save_index()
    return False
