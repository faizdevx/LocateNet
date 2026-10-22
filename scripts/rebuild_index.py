"""Rebuild vector_db/storage/face_index.bin from case_embeddings: one vector per active reference photo of
a case that is still missing (FAISS id = case_embeddings.id)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_db  # noqa: E402
from vector_db.rebuild import rebuild_face_index  # noqa: E402
from vector_db.search_service import SearchService  # noqa: E402


def main():
    create_db()
    count = rebuild_face_index(SearchService())
    print(f"Face index rebuilt with {count} active case vector(s).")


if __name__ == "__main__":
    main()
