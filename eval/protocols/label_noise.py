"""Experiment `label_noise`: list the highest-scoring "impostor" comparisons so a reader can see
which errors come from the data rather than the model.

LFW contains the same person under two names, the same photo filed under two people, and
genuine look-alikes such as identical twins. These pairs sit at the very top of the non-mated
score distribution and decide the FPIR = 0.1% threshold. They are listed, not removed: removing
them after looking at test-split errors would be tuning on the test split.
"""
import numpy as np
import pandas as pd

from eval.common import metadata, save_result
from eval.datasets import lfw
from eval.protocols import openset


def run(cfg):
    n = int(cfg.get("top_n", 8))
    out = {"meta": metadata(cfg, lfw.DATASET_VERSION), "protocol": __doc__.strip().split("\n\n", 1)[1]}
    for split in ("tune", "test"):
        r = pd.read_csv(openset.records_path(split, cfg.get("gallery_size", 1000)))
        top = r[~r["mated"]].sort_values("top_score", ascending=False).head(n)
        out[f"openset_top_nonmated_{split}"] = [
            {"probe_identity": a, "matched_identity": b, "score": float(s)}
            for a, b, s in zip(top["identity_id"], top["top_id"], top["top_score"])]

    manifest, _, emb_of, proto = openset.load_lfw_protocol(cfg)
    test = proto["test"]
    ids = sorted(set(test["gallery_ids"]) | set(test["distractors"]) | set(test["probes"]["identity_id"]))
    imgs = manifest.groupby("identity_id")["image_path"].apply(list).to_dict()
    firsts = [(i, next((p for p in imgs[i] if p in emb_of), None)) for i in ids]
    firsts = [(i, p) for i, p in firsts if p]
    emb = np.stack([emb_of[p] for _, p in firsts])
    sims = emb @ emb.T
    iu = np.triu_indices(len(firsts), 1)
    vals = sims[iu]
    order = np.argsort(-vals)[:n]
    out["impostor_top_pairs_test"] = [
        {"identity_a": firsts[iu[0][k]][0], "identity_b": firsts[iu[1][k]][0], "score": float(vals[k])} for k in order]
    out["n_impostor_pairs_test"] = int(vals.size)
    for row in out["openset_top_nonmated_test"][:4] + out["impostor_top_pairs_test"][:4]:
        print(row)
    save_result("label_noise", out)
