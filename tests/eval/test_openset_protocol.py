import numpy as np
import pandas as pd

from eval.embed import STATUS_OK
from eval.protocols import openset


def synthetic(n_multi=60, n_single=80, per_id=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, emb = [], {}
    for i in range(n_multi + n_single):
        ident = f"id{i:03d}"
        centre = rng.normal(size=512)
        for j in range(per_id if i < n_multi else 1):
            path = f"{ident}/{j}.jpg"
            v = centre + 0.3 * rng.normal(size=512)
            emb[path] = (v / np.linalg.norm(v)).astype("float32")
            rows.append((path, ident, "all", "{}"))
    manifest = pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])
    status = np.array([STATUS_OK] * len(manifest))
    return manifest, status, emb


def test_splits_are_disjoint_by_identity():
    manifest, status, _ = synthetic()
    proto = openset.build_protocol(manifest, status, seed=1)

    def ids(split):
        p = proto[split]
        return set(p["gallery_ids"]) | set(p["distractors"]) | set(p["probes"]["identity_id"])
    assert ids("tune").isdisjoint(ids("test"))
    for split in ("tune", "test"):
        p = proto[split]
        non = set(p["probes"][~p["probes"]["mated"]]["identity_id"])
        assert non.isdisjoint(p["gallery_ids"]) and non.isdisjoint(p["distractors"])
        # enrolment image is never reused as a probe
        assert not set(p["enrol"].values()) & set(p["probes"]["image_path"])


def test_gallery_sizes_and_search_metrics(tmp_path):
    manifest, status, emb = synthetic()
    proto = openset.build_protocol(manifest, status, seed=1)
    p = proto["test"]
    small = openset.gallery_for_size(p, 5)
    assert len(small) == 5
    big = openset.gallery_for_size(p, "all")
    assert len(big) == len(p["gallery_ids"]) + len(p["distractors"])

    recs = openset.search_probes(big, p["probes"], emb, top_k=5)
    summary = openset.summarise(recs, n_boot=50, seed=0, extra_thresholds={"t": 0.5})
    # well separated synthetic identities: near-perfect identification
    assert summary["rank1"]["rate"] > 0.95
    assert summary["fnir_at_fpir"]["0.01"]["fnir"] < 0.1
    lo, hi = summary["fnir_at_fpir"]["0.01"]["fnir_ci"]
    assert lo <= summary["fnir_at_fpir"]["0.01"]["fnir"] <= hi


def test_failed_probe_counts_as_miss():
    manifest, status, emb = synthetic(n_multi=20, n_single=10)
    proto = openset.build_protocol(manifest, status, seed=2)
    p = proto["test"]
    mated_probe = p["probes"][p["probes"]["mated"]].iloc[0]["image_path"]
    emb = dict(emb)
    del emb[mated_probe]
    recs = openset.search_probes(openset.gallery_for_size(p, "all"), p["probes"], emb, top_k=5)
    row = recs[recs["image_path"] == mated_probe].iloc[0]
    assert row["rank"] == 0 and row["top_score"] == -np.inf


def test_identity_aliases_merge_by_name_only():
    from eval.datasets.lfw import identity_aliases
    names = {"Yingfan_Wang", "Wang_Yingfan", "Choi_Sung-hong", "Sung_Hong_Choi", "Jon_Gruden", "John_Gruden", "Mary_Smith"}
    alias = identity_aliases(names, {"John_Gruden": "Jon_Gruden"})
    assert alias == {"Yingfan_Wang": "Wang_Yingfan", "Sung_Hong_Choi": "Choi_Sung-hong", "Jon_Gruden": "John_Gruden"}
