class DetectionPipeline:
    def __init__(self, face_service, reid_service, search_service):
        self.face_svc = face_service
        self.reid_service = reid_service
        self.search_svc = search_service
        self.strict_threshold = 0.72

    def process_frame(self, frame):
        faces = self.face_svc.detect_faces(frame)
        results = []

        for face in faces:
            embedding = self.face_svc.get_quality_embedding(face)
            if embedding is None:
                continue

            scores, ids = self.search_svc.search_face(embedding)

            best_score = float(scores[0]) if len(scores) > 0 else 0.0
            best_id = int(ids[0]) if len(ids) > 0 else -1

            identity = "Unknown"
            if best_score >= self.strict_threshold and best_id != -1:
                identity = f"Person_{best_id}"

            results.append({
                "id": best_id,
                "label": identity,
                "confidence": round(best_score * 100, 2),
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": embedding.tolist()
            })

        return results

    def enroll_new_person(self, frame, person_id, name=None):
        face_emb = self.face_svc.get_single_embedding(frame)
        if face_emb is None:
            return False

        self.search_svc.add_face_embedding(face_emb, person_id)

        body_emb = self.reid_service.extract_feature(frame)
        if body_emb is not None:
            self.search_svc.add_body_embedding(body_emb)
            self.search_svc.save_index()

        return True
