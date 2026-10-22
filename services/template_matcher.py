"""Multi-photo case templates: turn per-photo similarities into one score per case.

A case can have several reference photos (different ages, angles, lighting). FAISS stores one
vector per photo (case_embeddings.id); this module groups the hits by case and scores each case
with one of these strategies:

  max                     highest similarity over the case's photos
  mean_template           cosine to the L2-normalised mean of the case's photos
  quality_weighted_mean   cosine to the quality-weighted mean of the case's photos
  top2_mean               mean of the two best per-photo similarities (single photo: that photo)

Optional recency weighting (Task 3) multiplies each photo's weight by exp(-years_old / tau), where
years_old is how old the photo is at search time; for `max` it subtracts
recency_penalty * (1 - recency weight) from each photo's similarity.

Pure numpy, no models, no database: the app and the evaluation harness feed it rows.
"""
from dataclasses import dataclass, field

import numpy as np

STRATEGIES = ("max", "mean_template", "quality_weighted_mean", "top2_mean")


def _normalize(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


@dataclass
class CaseTemplate:
    case_id: int
    vectors: np.ndarray                      # (n, d), L2-normalised
    qualities: np.ndarray                    # (n,), > 0
    years_old: np.ndarray = None             # (n,) age of each photo in years at search time (nan = unknown)
    vector_ids: list = field(default_factory=list)


def recency_weights(years_old, tau):
    """exp(-years/tau); unknown age -> weight of the median known photo (1.0 if none known)."""
    y = np.asarray(years_old if years_old is not None else [], dtype="float64")
    if tau is None or tau <= 0 or y.size == 0:
        return np.ones(y.size)
    w = np.exp(-np.clip(y, 0, None) / tau)
    known = np.isfinite(w)
    fill = float(np.median(w[known])) if known.any() else 1.0
    return np.where(known, w, fill)


def case_score(template, probe, strategy, tau=None, recency_penalty=0.05):
    """Score one case against one L2-normalised probe embedding."""
    sims = template.vectors @ probe
    rec = recency_weights(template.years_old if template.years_old is not None else np.full(len(sims), np.nan), tau)
    if strategy == "max":
        if tau:
            sims = sims - recency_penalty * (1.0 - rec)
        return float(np.max(sims))
    if strategy == "top2_mean":
        best = np.sort(sims)[::-1][:2]
        return float(np.mean(best))
    if strategy in ("mean_template", "quality_weighted_mean"):
        w = np.ones(len(sims)) if strategy == "mean_template" else np.clip(template.qualities, 1e-6, None)
        if tau:
            w = w * rec
        mean = (template.vectors * w[:, None]).sum(axis=0)
        return float(_normalize(mean) @ probe)
    raise ValueError(f"unknown strategy {strategy}")


class TemplateMatcher:
    """Holds every active case template and ranks cases for a probe."""

    def __init__(self, templates, strategy="max", tau=None, recency_penalty=0.05):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.strategy, self.tau, self.recency_penalty = strategy, tau, recency_penalty
        self.templates = {t.case_id: t for t in templates}
        self.vector_to_case = {vid: t.case_id for t in templates for vid in t.vector_ids}
        self.max_photos = max((len(t.vectors) for t in templates), default=1)
        # in-memory template index for the mean strategies (one row per case)
        self._case_ids = list(self.templates)
        if self._case_ids and strategy in ("mean_template", "quality_weighted_mean"):
            rows = []
            for cid in self._case_ids:
                t = self.templates[cid]
                w = np.ones(len(t.vectors)) if strategy == "mean_template" else np.clip(t.qualities, 1e-6, None)
                if tau:
                    w = w * recency_weights(t.years_old, tau)
                rows.append((t.vectors * w[:, None]).sum(axis=0))
            self._mean_index = _normalize(np.stack(rows))
        else:
            self._mean_index = None

    @classmethod
    def from_rows(cls, rows, strategy="max", tau=None, recency_penalty=0.05):
        """rows: iterable of (vector_id, case_id, embedding, quality, years_old)."""
        grouped = {}
        for vid, cid, emb, q, years in rows:
            grouped.setdefault(int(cid), []).append((int(vid), np.asarray(emb, "float32"), float(q or 1.0),
                                                    np.nan if years is None else float(years)))
        templates = [CaseTemplate(cid, _normalize(np.stack([g[1] for g in items])),
                                  np.array([g[2] for g in items]), np.array([g[3] for g in items]),
                                  [g[0] for g in items]) for cid, items in grouped.items()]
        return cls(templates, strategy, tau, recency_penalty)

    def search_k(self, top_k):
        """How many per-photo FAISS neighbours to fetch so top_k cases are covered."""
        return int(top_k) * int(self.max_photos)

    def rank(self, probe, vector_hits=None, top_k=5):
        """
        Return (scores, case_ids) for the best top_k cases, highest first.
        For max / top2_mean pass vector_hits = (scores, vector_ids) from FAISS; candidate cases are
        the cases of those vectors, re-scored exactly over all of each case's photos.
        The mean strategies search the in-memory template index directly.
        """
        probe = _normalize(probe)
        if self._mean_index is not None:
            sims = self._mean_index @ probe
            order = np.argsort(-sims)
            pairs = [(float(sims[i]), self._case_ids[i]) for i in order]
        else:
            if vector_hits is None:
                candidates = list(self.templates)
            else:
                candidates = []
                for vid in vector_hits[1]:
                    cid = self.vector_to_case.get(int(vid))
                    if cid is not None and cid not in candidates:
                        candidates.append(cid)
            pairs = sorted(((case_score(self.templates[c], probe, self.strategy, self.tau, self.recency_penalty), c)
                            for c in candidates), reverse=True)
        pairs = pairs[:top_k]
        return np.array([p[0] for p in pairs], dtype="float32"), np.array([p[1] for p in pairs], dtype="int64")
