import json
import logging
import os

import faiss

from models import active_template_rows


def load_active_vectors():
    """(case_embeddings.id, vector) for every active reference photo of a case that is still missing."""
    items = []
    for row in active_template_rows():
        try:
            vector = json.loads(row["embedding"])
        except (TypeError, ValueError):
            continue
        if vector:
            items.append((row["id"], vector))
    return items


def index_ids(search_service):
    index = search_service.face_index
    if index.ntotal == 0:
        return set()
    return {int(i) for i in faiss.vector_to_array(index.id_map)}


def rebuild_face_index(search_service):
    """Rebuild face_index.bin from case_embeddings (one vector per photo id). Returns the vector count."""
    return search_service.rebuild_face_index(load_active_vectors())


def ensure_index_in_sync(search_service):
    """Rebuild when the index file is missing or its vector ids differ from the active photos in SQL
    (this also migrates a Week 2 index, whose ids were case ids)."""
    expected = {vid for vid, _ in load_active_vectors()}
    file_missing = not os.path.exists(search_service.face_bin)
    actual = index_ids(search_service)
    if (file_missing and expected) or actual != expected:
        logging.warning(
            "FAISS index out of sync (index=%s vectors, SQL active photos=%s, file_missing=%s). Rebuilding.",
            len(actual), len(expected), file_missing,
        )
        rebuild_face_index(search_service)
        return True
    if file_missing:
        search_service.save_index()
    return False
