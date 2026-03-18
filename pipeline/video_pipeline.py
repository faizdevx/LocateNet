import os
import uuid

import cv2


class VideoPipeline:
    def __init__(
        self,
        face_service,
        reid_service,
        search_service,
        storage_path="resources/video_frames",
        model_name="yolov8n.pt",
        face_match_threshold=0.72,
        person_conf_threshold=0.35,
        frame_stride=3,
    ):
        self.face_service = face_service
        self.reid_service = reid_service
        self.search_service = search_service
        self.storage_path = storage_path
        self.face_match_threshold = face_match_threshold
        self.person_conf_threshold = person_conf_threshold
        self.frame_stride = max(1, int(frame_stride))
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

        face_crop = frame[max(0, fy1):min(frame.shape[0], fy2), max(0, fx1):min(frame.shape[1], fx2)]
        face_crop_path = None
        if face_crop.size > 0:
            filename = f"video_face_{uuid.uuid4()}.jpg"
            full_path = os.path.join(self.storage_path, filename)
            cv2.imwrite(full_path, face_crop)
            face_crop_path = f"resources/video_frames/{filename}"

        body_embedding = self.reid_service.extract_feature(person_crop)
        return {
            "track_bbox": [x1, y1, x2, y2],
            "face_bbox": [int(fx1), int(fy1), int(fx2), int(fy2)],
            "face_crop_path": face_crop_path,
            "face_embedding": embedding,
            "body_embedding": body_embedding.tolist() if hasattr(body_embedding, "tolist") else body_embedding,
        }

    def process_video(self, video_path, max_frames=None):
        if not self.available:
            raise RuntimeError(
                "Video pipeline dependencies are unavailable. Install 'ultralytics' and "
                "'deep-sort-realtime'."
            )

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        processed_tracks = {}
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if frame_index % self.frame_stride != 0:
                    continue
                if max_frames and frame_index > max_frames:
                    break

                detections = self._detect_people(frame)
                tracks = self.tracker.update_tracks(detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = int(track.track_id)
                    track_data = self._extract_face_from_track(frame, track)
                    if track_data is None:
                        continue

                    scores, ids = self.search_service.search_face(track_data["face_embedding"])
                    best_score = float(scores[0]) if len(scores) > 0 else 0.0
                    best_id = int(ids[0]) if len(ids) > 0 else -1

                    label = "Unknown"
                    if best_score >= self.face_match_threshold and best_id != -1:
                        label = f"Person_{best_id}"

                    current = {
                        "track_id": track_id,
                        "match_id": best_id,
                        "label": label,
                        "confidence": round(best_score * 100, 2),
                        "category": (
                            "High Confidence Match" if best_score * 100 >= 75.0
                            else "Likely Match" if best_score * 100 >= 50.0
                            else "Potential Match (Check Group)" if best_score * 100 >= 10.0
                            else "Unknown (No SQL Match)"
                        ),
                        "person_bbox": track_data["track_bbox"],
                        "face_bbox": track_data["face_bbox"],
                        "face_crop_path": track_data["face_crop_path"],
                        "frame_index": frame_index,
                        "body_embedding": track_data["body_embedding"],
                    }

                    previous = processed_tracks.get(track_id)
                    if previous is None or current["confidence"] >= previous["confidence"]:
                        processed_tracks[track_id] = current
        finally:
            capture.release()

        return {
            "video_path": video_path,
            "tracks": list(processed_tracks.values()),
            "num_tracks": len(processed_tracks),
        }
