import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config


class FaceService:
    def __init__(self, det_threshold=0.5, det_size=(640, 640), allowed_modules=None):
        """
        Initialize the high-accuracy InsightFace model for group-friendly detection.
        allowed_modules=None loads every buffalo_l model (production default). The evaluation
        harness passes ["detection", "recognition"], which gives identical embeddings faster.
        """
        self.det_size = tuple(det_size)
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                                allowed_modules=allowed_modules)
        self.app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=det_threshold)
        print(f"Face model loaded (Threshold: {det_threshold}, det_size: {self.det_size})")

    def detect_faces(self, frame, pad_retry=None):
        """
        Detect and embed faces. SCRFD misses faces that fill most of the frame (close-up portraits,
        passport-style enrolment photos). When no face reaches DET_SCORE_MIN and DETECT_PAD_RETRY is
        on, detection is retried once on the image centred in a 2x canvas, and boxes / landmarks are
        mapped back to the original image. See eval/results/age_gap.json (FG-NET failure to enrol).
        """
        if frame is None:
            return []
        faces = self.app.get(frame)
        if pad_retry is None:
            pad_retry = config.DETECT_PAD_RETRY
        if not pad_retry or any(getattr(f, "det_score", 0.0) >= config.DET_SCORE_MIN for f in faces):
            return faces
        h, w = frame.shape[:2]
        canvas = np.zeros((h * 2, w * 2) + frame.shape[2:], dtype=frame.dtype)
        oy, ox = h // 2, w // 2
        canvas[oy:oy + h, ox:ox + w] = frame
        padded = self.app.get(canvas)
        best = max((f.det_score for f in faces), default=0.0)
        if not padded or max(f.det_score for f in padded) <= best:
            return faces
        for f in padded:
            f.bbox = f.bbox - np.array([ox, oy, ox, oy], dtype=f.bbox.dtype)
            if getattr(f, "kps", None) is not None:
                f.kps = f.kps - np.array([ox, oy], dtype=f.kps.dtype)
        return padded

    def get_embeddings(self, frame):
        """
        Return normalized embeddings and metadata for all detected faces.
        """
        faces = self.detect_faces(frame)
        results = []

        for face in faces:
            emb = np.array(face.embedding, dtype="float32")
            norm = np.linalg.norm(emb)
            norm_embedding = emb / norm if norm > 0 else emb

            results.append({
                "embedding": norm_embedding.tolist(),
                "bbox": face.bbox.astype(int).tolist(),
                "confidence": float(face.det_score),
                "gender": "M" if getattr(face, "gender", 0) == 1 else "F",
                "age": getattr(face, "age", None)
            })

        return results

    def get_quality_embedding(self, face):
        # Ignore weak detections because they tend to create noisy identity vectors.
        if getattr(face, "det_score", 0.0) < config.DET_SCORE_MIN:
            return None
        return face.embedding

    def get_best_face(self, frame):
        faces = self.detect_faces(frame)
        if not faces:
            return None

        faces = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )
        for face in faces:
            if self.get_quality_embedding(face) is not None:
                return face
        return None

    def get_single_embedding(self, frame):
        best_face = self.get_best_face(frame)
        if best_face is not None:
            return self.get_quality_embedding(best_face)
        return None

    def compare_embeddings(self, emb1, emb2):
        emb1 = np.array(emb1, dtype="float32").flatten()
        emb2 = np.array(emb2, dtype="float32").flatten()

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1 / norm1, emb2 / norm2))

    def process_uploaded_file(self, file):
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
