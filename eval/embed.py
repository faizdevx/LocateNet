"""Batch embedding extraction through the production FaceService, with an on-disk cache.

Per image we record one of three outcomes:
  ok           face found and det_score >= DET_SCORE_MIN (FaceService.get_quality_embedding accepts it)
  low_quality  face found but det_score < DET_SCORE_MIN (production would reject it)
  no_face      detector found nothing
The embedding of the selected face is stored even for low_quality, so the quality-gate sweep can
re-apply other cut-offs; protocols treat low_quality as a failure unless told otherwise.

Face selection when an image has several faces (LFW often shows bystanders):
  keep the faces whose box area is at least 50% of the largest box, then pick the one whose box
  centre is closest to the image centre. This is the "largest face nearest the centre" rule.
"""
import hashlib
import os
import time

import cv2
import numpy as np

from eval.common import CACHE_DIR, MODEL_NAME, ROOT

STATUS_OK, STATUS_NO_FACE, STATUS_LOW = "ok", "no_face", "low_quality"
EMB_DIM = 512


def path_key(path):
    try:
        rel = os.path.relpath(os.path.abspath(path), ROOT).replace("\\", "/")
    except ValueError:  # different drive (temporary files on Windows)
        rel = os.path.abspath(path).replace("\\", "/")
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:20]


def select_face(faces, image_shape):
    if not faces:
        return None
    areas = np.array([(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces])
    keep = [i for i, a in enumerate(areas) if a >= 0.5 * areas.max()]
    cy, cx = image_shape[0] / 2.0, image_shape[1] / 2.0

    def centre_dist(i):
        b = faces[i].bbox
        return ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2

    return faces[min(keep, key=centre_dist)]


def make_face_service(det_size=(640, 640), det_thresh=0.5):
    """Production FaceService; only the detection + recognition models are loaded (same embeddings)."""
    from services.face_service import FaceService
    return FaceService(det_threshold=det_thresh, det_size=tuple(det_size),
                       allowed_modules=["detection", "recognition"])


def cache_name(dataset, variant=None, det_size=(640, 640)):
    name = f"{dataset}_{MODEL_NAME}"
    if tuple(det_size) != (640, 640):
        name += f"_det{det_size[0]}"
    if variant:
        name += f"_{variant}"
    return name


class EmbeddingCache:
    def __init__(self, name, cache_dir=CACHE_DIR):
        self.path = os.path.join(cache_dir, f"{name}.npz")
        self.records = {}
        if os.path.exists(self.path):
            with np.load(self.path, allow_pickle=False) as data:
                # read each array once: indexing an NpzFile re-reads the whole array every time
                keys, emb, det = data["keys"], data["emb"], data["det_score"]
                status, bbox = data["status"], data["bbox"]
            for i, key in enumerate(keys):
                self.records[str(key)] = (emb[i], float(det[i]), str(status[i]), bbox[i])

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        keys = list(self.records)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp,
                 keys=np.array(keys, dtype="U40"),
                 emb=np.stack([self.records[k][0] for k in keys]).astype("float32") if keys else np.zeros((0, EMB_DIM), "float32"),
                 det_score=np.array([self.records[k][1] for k in keys], dtype="float32"),
                 status=np.array([self.records[k][2] for k in keys], dtype="U12"),
                 bbox=np.stack([self.records[k][3] for k in keys]).astype("float32") if keys else np.zeros((0, 4), "float32"))
        os.replace(tmp, self.path)


def embed_image(face_svc, image):
    """Embed one decoded image through the production code path. Returns (emb, det_score, status, bbox)."""
    nan_box = np.full(4, np.nan, dtype="float32")
    zero = np.zeros(EMB_DIM, dtype="float32")
    if image is None:
        return zero, 0.0, STATUS_NO_FACE, nan_box
    faces = face_svc.detect_faces(image)
    face = select_face(faces, image.shape)
    if face is None:
        return zero, 0.0, STATUS_NO_FACE, nan_box
    accepted = face_svc.get_quality_embedding(face)
    emb = np.asarray(face.embedding, dtype="float32")
    norm = np.linalg.norm(emb)
    emb = emb / norm if norm > 0 else emb  # same L2 normalisation as SearchService
    status = STATUS_OK if accepted is not None else STATUS_LOW
    return emb, float(face.det_score), status, np.asarray(face.bbox, dtype="float32")


def embed_paths(paths, name, face_svc_factory=make_face_service, transform=None, log_every=500, cache_dir=CACHE_DIR):
    """
    Embed every path (cached by path hash). `transform(image, path)` optionally degrades the image
    before detection. Returns dict of aligned arrays: emb, det_score, status, bbox.
    """
    cache = EmbeddingCache(name, cache_dir)
    keys = [path_key(p) for p in paths]
    missing = [(p, k) for p, k in zip(paths, keys) if k not in cache.records]
    if missing:
        face_svc = face_svc_factory()
        start = time.time()
        for i, (p, k) in enumerate(missing, 1):
            image = cv2.imread(p)
            if transform is not None and image is not None:
                image = transform(image, p)
            cache.records[k] = embed_image(face_svc, image)
            if i % log_every == 0:
                rate = i / (time.time() - start)
                print(f"  [{name}] {i}/{len(missing)} embedded ({rate:.1f} img/s)")
                cache.save()
        cache.save()
    recs = [cache.records[k] for k in keys]
    return {
        "emb": np.stack([r[0] for r in recs]) if recs else np.zeros((0, EMB_DIM), "float32"),
        "det_score": np.array([r[1] for r in recs], dtype="float32"),
        "status": np.array([r[2] for r in recs]),
        "bbox": np.stack([r[3] for r in recs]) if recs else np.zeros((0, 4), "float32"),
    }


def fte_summary(status):
    status = np.asarray(status)
    n = int(status.size)
    counts = {s: int(np.sum(status == s)) for s in (STATUS_OK, STATUS_LOW, STATUS_NO_FACE)}
    return {"n_images": n, **counts, "fte_rate": float((n - counts[STATUS_OK]) / n) if n else float("nan")}
