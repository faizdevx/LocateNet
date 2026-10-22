"""Video tracks: per-frame face observations -> clean sub-tracks -> one embedding per sub-track.

Track hygiene: tracker ID switches can glue two people into one track. While walking through a
track's observations in frame order we keep the running mean of the current segment; when a new
face's cosine to that mean drops below `split_threshold`, a new segment starts.

Aggregation strategies (all return L2-normalised embeddings to search with):
  best_frame            every frame is searched and the best score wins (Week 1-2 behaviour);
                        returns all frame embeddings, the caller keeps the max
  mean                  normalised mean of all frames
  quality_weighted_mean mean weighted by frame quality (det score x face size x sharpness,
                        services/photo_quality.quality_score)
  top_n_quality_mean    mean of the n best-quality frames (n = 5)
Pure numpy, no models.
"""
from dataclasses import dataclass

import numpy as np

from services.photo_quality import quality_score

STRATEGIES = ("best_frame", "mean", "quality_weighted_mean", "top_n_quality_mean")


def _norm(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


@dataclass
class Observation:
    frame_index: int
    embedding: np.ndarray
    det_score: float
    face_px: float
    blur_var: float
    crop: object = None          # optional face crop (numpy image) for the officer gallery
    meta: dict = None            # optional extra data (boxes, ...) carried through unchanged

    @property
    def quality(self):
        return quality_score(self.det_score, self.face_px, self.blur_var)


def split_track(observations, split_threshold):
    """Split one tracker track into identity-consistent segments. Returns a list of lists."""
    obs = sorted(observations, key=lambda o: o.frame_index)
    segments, current, total = [], [], None
    for o in obs:
        e = _norm(o.embedding)
        if current and float(_norm(total) @ e) < split_threshold:
            segments.append(current)
            current, total = [], None
        current.append(o)
        total = e.copy() if total is None else total + e
    if current:
        segments.append(current)
    return segments


def aggregate(observations, strategy, top_n=5):
    """Embeddings to search for one segment: (k, d) array; k = 1 except for best_frame."""
    if not observations:
        raise ValueError("empty track")
    embs = _norm(np.stack([o.embedding for o in observations]))
    if strategy == "best_frame":
        return embs
    if strategy == "mean":
        return _norm(embs.mean(axis=0))[None]
    q = np.array([max(o.quality, 1e-6) for o in observations])
    if strategy == "quality_weighted_mean":
        return _norm((embs * q[:, None]).sum(axis=0))[None]
    if strategy == "top_n_quality_mean":
        best = np.argsort(-q)[:top_n]
        return _norm(embs[best].mean(axis=0))[None]
    raise ValueError(f"unknown strategy {strategy}")


def representative(observations, k=3):
    """The k best-quality observations (for the officer's frame gallery), best first."""
    return sorted(observations, key=lambda o: -o.quality)[:k]
