import os

import faiss
import numpy as np


class SearchService:
    def __init__(self, face_dim=512, body_dim=512, index_path="vector_db/storage"):
        self.face_dim = face_dim
        self.body_dim = body_dim
        self.index_path = index_path
        self.face_bin = os.path.join(self.index_path, "face_index.bin")
        self.body_bin = os.path.join(self.index_path, "body_index.bin")

        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        # Store real SQL ids directly in FAISS so search results map cleanly to SQLite.
        self.face_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.face_dim))
        self.body_index = faiss.IndexFlatIP(self.body_dim)
        self.load_index()

    def _normalize(self, embedding):
        emb = np.array([embedding], dtype="float32")
        faiss.normalize_L2(emb)
        return emb

    def _is_id_mapped_index(self, index):
        return "IndexIDMap" in type(index).__name__

    def add_face_embedding(self, embedding, person_id):
        emb = self._normalize(embedding)
        ids = np.array([person_id], dtype="int64")
        self.face_index.add_with_ids(emb, ids)
        self.save_index()

    def add_body_embedding(self, embedding):
        emb = self._normalize(embedding)
        self.body_index.add(emb)

    def remove_face(self, person_id):
        """Remove a case vector from the face index and persist the change."""
        removed = self.face_index.remove_ids(np.array([person_id], dtype="int64"))
        self.save_index()
        return int(removed)

    def search_face(self, embedding, top_k=5):
        if self.face_index.ntotal == 0:
            return [], []

        emb = self._normalize(embedding)
        distances, indices = self.face_index.search(emb, top_k)
        # FAISS pads with -1 ids when the index holds fewer than top_k vectors.
        valid = indices[0] != -1
        return distances[0][valid], indices[0][valid]

    def search_body(self, embedding, top_k=3):
        emb = self._normalize(embedding)
        distances, indices = self.body_index.search(emb, top_k)
        return distances, indices

    def rebuild_face_index(self, items):
        """Replace the face index with (case_id, embedding) pairs and persist it."""
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.face_dim))
        for case_id, embedding in items:
            index.add_with_ids(self._normalize(embedding), np.array([case_id], dtype="int64"))
        self.face_index = index
        self.save_index()
        return index.ntotal

    def save_index(self):
        # Errors propagate so callers can keep SQL and FAISS in sync.
        faiss.write_index(self.face_index, self.face_bin)
        faiss.write_index(self.body_index, self.body_bin)

    def load_index(self):
        if os.path.exists(self.face_bin):
            loaded_face_index = faiss.read_index(self.face_bin)
            if self._is_id_mapped_index(loaded_face_index):
                self.face_index = loaded_face_index
                print(f"AI Memory Loaded: {self.face_index.ntotal} faces active.")
            else:
                print("Existing face index uses the old non-ID format. Re-enroll cases to rebuild AI memory.")

        if os.path.exists(self.body_bin):
            self.body_index = faiss.read_index(self.body_bin)
            print(f"Loaded {self.body_index.ntotal} body signatures.")
