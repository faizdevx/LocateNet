
import os
import uuid

import cv2


class DetectionPipeline:
    def __init__(self, face_service, reid_service, search_service, storage_path="resources"):
        self.face_svc = face_service
        self.reid_service = reid_service
        self.search_svc = search_service
        self.storage_path = storage_path
        self.strict_threshold = 0.72

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def process_frame(self, frame):
        """
        Process every detected face, save a crop, and return match metadata.
        """
        results = []
        faces = self.face_svc.detect_faces(frame)

        for face in faces:
            embedding = self.face_svc.get_quality_embedding(face)
            if embedding is None:
                continue

            scores, ids = self.search_svc.search_face(embedding)
            best_score = float(scores[0]) if len(scores) > 0 else 0.0
            best_id = int(ids[0]) if len(ids) > 0 else -1
            percentage = round(best_score * 100, 2)

            if percentage >= 75.0:
                category = "High Confidence Match"
            elif percentage >= 50.0:
                category = "Likely Match"
            elif percentage >= 10.0:
                category = "Potential Match (Check Group)"
            else:
                category = "Unknown (No SQL Match)"

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            y1, y2 = int(max(0, y1)), int(min(frame.shape[0], y2))
            x1, x2 = int(max(0, x1)), int(min(frame.shape[1], x2))

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            crop_filename = f"face_{uuid.uuid4()}.jpg"
            crop_path = os.path.join(self.storage_path, crop_filename)
            cv2.imwrite(crop_path, face_crop)

            label = "Unknown"
            if best_score >= self.strict_threshold and best_id != -1:
                label = f"Person_{best_id}"

            results.append({
                "id": int(best_id),
                "label": label,
                "confidence": float(percentage),
                "category": category,
                "face_crop_path": f"resources/{crop_filename}",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
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
