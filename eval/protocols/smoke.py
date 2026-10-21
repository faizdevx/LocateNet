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

    def detect_faces(self, image):
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
        checked = 0
        for split in ("tune", "test"):
            p = proto[split]
            recs = openset.search_probes(openset.gallery_for_size(p, "all"), p["probes"], emb_of, top_k=5)
            if recs.empty:
                continue
            summary = openset.summarise(recs, n_boot=20, seed=cfg["seed"])
            assert "0.01" in summary["fnir_at_fpir"]
            finite = recs[np.isfinite(recs["top_score"])]
            y = (finite["mated"] & (finite["rank"] == 1)).to_numpy(int)
            if 0 < y.sum() < len(y):
                w = prior_weights(finite["mated"].to_numpy(bool), 0.1)
                fit_platt(finite["top_score"].to_numpy(float), y, w)
                fit_isotonic(finite["top_score"].to_numpy(float), y, w)
            checked += 1
        assert checked, "no split produced search records"
        assert M.ece([0.5, 0.5], [0, 1]) == 0.0
        print(f"smoke OK: {len(manifest)} images, FTE {fte['fte_rate']:.2%}, {checked} splits searched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
