"""Experiment `detsize_check`: do the thresholds measured at det_size 640 still hold at 1024?

The thresholds were chosen with embeddings from det_size 640. Changing the detector input size
moves the landmarks slightly, which changes the aligned crop and therefore the embedding. We embed
a seeded sample of LFW images at both sizes and report how much the embeddings move (cosine
between the two embeddings of the same image) and how genuine / impostor scores shift.
"""
import numpy as np

from eval.common import DATA_DIR, metadata, save_result
from eval.datasets import lfw
from eval.embed import STATUS_OK, cache_name, embed_paths, make_face_service


def run(cfg):
    manifest = lfw.load_manifest(DATA_DIR)
    counts = manifest["identity_id"].value_counts()
    multi = sorted(counts[counts >= 2].index)
    rng = np.random.default_rng(cfg["seed"])
    ids = sorted(rng.choice(multi, int(cfg.get("n_identities", 200)), replace=False))
    paths = []
    for i in ids:
        paths += sorted(manifest.loc[manifest["identity_id"] == i, "image_path"])[:2]
    size = int(cfg.get("det_size", 1024))
    base = embed_paths(paths, cache_name("lfw"))
    new = embed_paths(paths, cache_name("lfw", det_size=(size, size)),
                      face_svc_factory=lambda: make_face_service(det_size=(size, size)))
    ok = (base["status"] == STATUS_OK) & (new["status"] == STATUS_OK)
    self_cos = np.sum(base["emb"] * new["emb"], axis=1)[ok]
    a, b = base["emb"][0::2], base["emb"][1::2]
    a2, b2 = new["emb"][0::2], new["emb"][1::2]
    pair_ok = ok[0::2] & ok[1::2]
    gen_640, gen_new = np.sum(a * b, 1)[pair_ok], np.sum(a2 * b2, 1)[pair_ok]
    imp_640 = np.sum(a[pair_ok] * np.roll(b[pair_ok], 1, axis=0), 1)
    imp_new = np.sum(a2[pair_ok] * np.roll(b2[pair_ok], 1, axis=0), 1)
    result = {
        "meta": metadata(cfg, lfw.DATASET_VERSION), "det_size": size, "n_images": len(paths),
        "n_both_enrolled": int(ok.sum()),
        "fte_640": float(np.mean(base["status"] != STATUS_OK)), f"fte_{size}": float(np.mean(new["status"] != STATUS_OK)),
        "self_cosine_mean": float(self_cos.mean()), "self_cosine_p01": float(np.quantile(self_cos, 0.01)),
        "genuine_mean_640": float(gen_640.mean()), f"genuine_mean_{size}": float(gen_new.mean()),
        "impostor_mean_640": float(imp_640.mean()), f"impostor_mean_{size}": float(imp_new.mean()),
        "genuine_mean_shift": float((gen_new - gen_640).mean()),
        "impostor_mean_shift": float((imp_new - imp_640).mean()),
        "max_abs_score_shift": float(max(np.abs(gen_new - gen_640).max(), np.abs(imp_new - imp_640).max())),
    }
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items() if k != "meta"})
    save_result("detsize_check", result)
