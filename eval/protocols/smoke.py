"""Experiment `smoke`: run the harness end to end on ~20 synthetic images in a temporary directory.

It checks wiring (manifest -> embed cache -> open-set search with the production SearchService
-> metrics with bootstrap -> calibration -> report), NOT accuracy. With face_service: stub the
face model is replaced by a deterministic fake so it runs in CI without model downloads; with
face_service: real it uses buffalo_l (synthetic images contain no faces, so FTE will be 100%).
Nothing is written to eval/results or eval/cache.
"""
import os
import shutil
import tempfile
import types

import cv2
import numpy as np
import pandas as pd

from eval import metrics as M
from eval.calibrate import fit_isotonic, fit_platt, prior_weights
from eval.embed import embed_paths, fte_summary, make_face_service
from eval.protocols import openset


class StubFaceService:
    """Deterministic stand-in: one 'face' per image, embedding = identity centre + noise."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.centres = {}

    def detect_faces(self, image, pad_retry=False):
        ident = int(image[0, 0, 0])  # the synthetic generator writes the identity id into pixel (0,0)
        centre = self.centres.setdefault(ident, np.random.default_rng(ident).normal(size=512))
        emb = centre + 0.4 * self.rng.normal(size=512)
        return [types.SimpleNamespace(bbox=np.array([10, 10, 90, 90], "float32"), det_score=0.9, embedding=emb)]

    def get_quality_embedding(self, face):
        return face.embedding


def synthetic_images(root, cfg):
    rows = []
    n_id, per, n_single = int(cfg.get("n_identities", 6)), int(cfg.get("images_per_identity", 3)), int(cfg.get("n_single", 4))
    for i in range(n_id + n_single):
        for j in range(per if i < n_id else 1):
            img = np.full((100, 100, 3), 200, dtype="uint8")
            img[0, 0] = i + 1
            path = os.path.join(root, f"id{i:02d}", f"{j}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, img)
            rows.append((path, f"id{i:02d}", "all", "{}"))
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])


def extra_smoke_metrics(cfg):
    """Deterministic metrics of the Week 3 components on synthetic vectors (no models)."""
    from services.feedback import suppression_decision
    from services.template_matcher import STRATEGIES, TemplateMatcher
    from services.track_aggregator import Observation, aggregate, split_track
    rng = np.random.default_rng(cfg["seed"])
    centres = rng.normal(size=(8, 64))
    rows, vid = [], 1
    for case, c in enumerate(centres):
        for _ in range(3):
            rows.append((vid, case + 1, c + 0.5 * rng.normal(size=64), float(rng.uniform(0.3, 1.0)), None))
            vid += 1
    probes = [(case + 1, c + 0.5 * rng.normal(size=64)) for case, c in enumerate(centres)]
    out = {}
    for strat in STRATEGIES:
        m = TemplateMatcher.from_rows(rows, strategy=strat)
        hits = 0
        for case, p in probes:
            _, ids = m.rank(p, top_k=1)
            hits += int(len(ids) and ids[0] == case)
        out[f"template_{strat}_rank1"] = hits / len(probes)
    track = [Observation(k, centres[0] + 0.2 * rng.normal(size=64), 0.9, 80, 300) for k in range(6)]
    track += [Observation(k, centres[1] + 0.2 * rng.normal(size=64), 0.9, 80, 300) for k in range(6, 12)]
    out["track_segments"] = float(len(split_track(track, 0.3)))
    out["track_qwm_first_dim"] = float(aggregate(track[:6], "quality_weighted_mean")[0][0])
    down, neg = suppression_decision(centres[2], 0.2, [centres[2] + 0.01], 0.6)
    out["suppression_downgrade"] = float(down)
    return out


def run(cfg):
    tmp = tempfile.mkdtemp(prefix="locatenet-smoke-")
    try:
        manifest = synthetic_images(os.path.join(tmp, "images"), cfg)
        factory = (lambda: StubFaceService(cfg["seed"])) if cfg.get("face_service", "stub") == "stub" else make_face_service
        out = embed_paths(list(manifest["image_path"]), "smoke", face_svc_factory=factory, cache_dir=os.path.join(tmp, "cache"))
        fte = fte_summary(out["status"])
        again = embed_paths(list(manifest["image_path"]), "smoke", face_svc_factory=None, cache_dir=os.path.join(tmp, "cache"))
        assert np.allclose(again["emb"], out["emb"]), "cache round trip changed embeddings"

        emb_of = {p: e for p, e, s in zip(manifest["image_path"], out["emb"], out["status"]) if s == "ok"}
        proto = openset.build_protocol(manifest, out["status"], seed=cfg["seed"], nonmated_frac=0.34)
        checked, metrics = 0, {}
        for split in ("tune", "test"):
            p = proto[split]
            recs = openset.search_probes(openset.gallery_for_size(p, "all"), p["probes"], emb_of, top_k=5)
            if recs.empty:
                continue
            summary = openset.summarise(recs, n_boot=20, seed=cfg["seed"])
            assert "0.01" in summary["fnir_at_fpir"]
            metrics[f"{split}_rank1"] = summary["rank1"]["rate"]
            metrics[f"{split}_fnir_fpir1"] = summary["fnir_at_fpir"]["0.01"]["fnir"]
            metrics[f"{split}_threshold_fpir1"] = summary["fnir_at_fpir"]["0.01"]["threshold"]
            finite = recs[np.isfinite(recs["top_score"])]
            y = (finite["mated"] & (finite["rank"] == 1)).to_numpy(int)
            if 0 < y.sum() < len(y):
                w = prior_weights(finite["mated"].to_numpy(bool), 0.1)
                fit_platt(finite["top_score"].to_numpy(float), y, w)
                fit_isotonic(finite["top_score"].to_numpy(float), y, w)
            checked += 1
        assert checked, "no split produced search records"
        assert M.ece([0.5, 0.5], [0, 1]) == 0.0
        metrics.update(extra_smoke_metrics(cfg))
        out_path = os.environ.get("LOCATENET_SMOKE_METRICS")
        if out_path:
            import json
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
        print(f"smoke OK: {len(manifest)} images, FTE {fte['fte_rate']:.2%}, {checked} splits searched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
