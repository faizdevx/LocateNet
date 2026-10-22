import logging
import os
import uuid

import cv2
import numpy as np

import config
from pipeline.scoring import match_fields, rank_cases
from services import photo_quality
from services.track_aggregator import Observation, aggregate, representative, split_track


class VideoPipeline:
    def __init__(
        self,
        face_service,
        reid_service,
        search_service,
        storage_path="resources/video_frames",
        model_name="yolov8n.pt",
        person_conf_threshold=0.35,
        frame_stride=3,
        matcher_provider=None,
    ):
        self.face_service = face_service
        self.reid_service = reid_service
        self.search_service = search_service
        self.storage_path = storage_path
        self.person_conf_threshold = person_conf_threshold
        self.frame_stride = max(1, int(frame_stride))
        self.matcher_provider = matcher_provider or (lambda: None)
        self.model_name = model_name

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self.available = False
        self.init_error = None
        self._load_dependencies()

    def _load_dependencies(self):
        try:
            from ultralytics import YOLO
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self.detector = YOLO(self.model_name)
            self.tracker = DeepSort(max_age=30, n_init=2)
            self.available = True
        except Exception as e:
            self.init_error = str(e)
            self.detector = None
            self.tracker = None

    def _detect_people(self, frame):
        detections = []
        results = self.detector(frame, verbose=False)

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                if cls_id != 0 or conf < self.person_conf_threshold:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        return detections

    def _extract_face_from_track(self, frame, track):
        """One per-frame face observation for a confirmed person track, or None."""
        x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        face = self.face_service.get_best_face(person_crop)
        if face is None:
            return None

        embedding = self.face_service.get_quality_embedding(face)
        if embedding is None:
            return None

        fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
        fx1 += x1
        fy1 += y1
        fx2 += x1
        fy2 += y1
        face_crop = frame[max(0, fy1):min(frame.shape[0], fy2), max(0, fx1):min(frame.shape[1], fx2)].copy()
        if face_crop.size == 0:
            return None
        body_embedding = (
            self.reid_service.extract_feature(person_crop) if self.reid_service is not None else None
        )
        return {
            "embedding": np.asarray(embedding, dtype="float32"),
            "det_score": float(face.det_score),
            "face_px": float(fx2 - fx1),
            "blur_var": photo_quality.blur_variance(frame, (fx1, fy1, fx2, fy2)),
            "crop": face_crop,
            "track_bbox": [x1, y1, x2, y2],
            "face_bbox": [int(fx1), int(fy1), int(fx2), int(fy2)],
            "body_embedding": body_embedding.tolist() if hasattr(body_embedding, "tolist") else body_embedding,
        }

    def _save_crop(self, crop):
        filename = f"video_face_{uuid.uuid4()}.jpg"
        cv2.imwrite(os.path.join(self.storage_path, filename), crop)
        return f"resources/video_frames/{filename}"

    def _score_segment(self, segment, is_active):
        """Aggregate one clean sub-track and search it (best_frame searches every frame)."""
        best, calls = None, 0
        embs = aggregate(segment, config.TRACK_AGG_STRATEGY)
        for emb in embs:
            scores, ids = rank_cases(emb, self.search_service, self.matcher_provider(), config.TOP_K)
            calls += 1
            match = match_fields(scores, ids, is_active)
            if best is None or match["similarity"] > best[0]["similarity"]:
                best = (match, emb)
        return best[0], best[1], calls

    def process_video(self, video_path, max_frames=None, is_active=None):
        if not self.available:
            raise RuntimeError(
                "Video pipeline dependencies are unavailable. Install 'ultralytics' and "
                "'deep-sort-realtime'."
            )

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

        observations = {}
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if max_frames and frame_index > max_frames:
                    break
                if frame_index % self.frame_stride != 0:
                    continue

                detections = self._detect_people(frame)
                tracks = self.tracker.update_tracks(detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    data = self._extract_face_from_track(frame, track)
                    if data is None:
                        continue
                    observations.setdefault(int(track.track_id), []).append(Observation(
                        frame_index, data["embedding"], data["det_score"], data["face_px"], data["blur_var"],
                        crop=data["crop"], meta=data))
        finally:
            capture.release()

        # Track hygiene + aggregation: one search per identity-consistent sub-track.
        out_tracks, n_splits, calls = [], 0, 0
        for track_id, obs in observations.items():
            segments = split_track(obs, config.TRACK_SPLIT_THRESHOLD) if config.ENABLE_TRACK_SPLIT else [obs]
            n_splits += len(segments) - 1
            for n, segment in enumerate(segments):
                match, emb, c = self._score_segment(segment, is_active)
                calls += c
                reps = representative(segment, 3)
                crops = [self._save_crop(o.crop) for o in reps]
                out_tracks.append({
                    "track_id": f"{track_id}" if len(segments) == 1 else f"{track_id}.{n + 1}",
                    "match_id": match["id"],
                    "label": match["label"],
                    "similarity": match["similarity"],
                    "percentage": match["percentage"],
                    "category": match["category"],
                    "person_bbox": reps[0].meta["track_bbox"],
                    "face_bbox": reps[0].meta["face_bbox"],
                    "face_crop_path": crops[0],
                    "extra_crops": crops,
                    "frame_index": reps[0].frame_index,
                    "start_s": round(segment[0].frame_index / fps, 2),
                    "end_s": round(segment[-1].frame_index / fps, 2),
                    "n_frames": len(segment),
                    "face_embedding": np.asarray(emb, dtype="float32").tolist(),
                    "body_embedding": reps[0].meta["body_embedding"],
                })
        if n_splits:
            logging.info("Video %s: %s track split(s) at TRACK_SPLIT_THRESHOLD=%s", os.path.basename(video_path),
                         n_splits, config.TRACK_SPLIT_THRESHOLD)
        return {
            "video_path": video_path,
            "tracks": out_tracks,
            "num_tracks": len(out_tracks),
            "n_track_splits": n_splits,
            "faiss_calls": calls,
            "fps": fps,
        }
