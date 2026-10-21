"""Experiment `openset`: 1:N open-set identification on LFW, mirroring the app's real question
("is this sighting face one of our missing people, and which one?").

Protocol (all splits are by identity, seeded):
  * identities with >= 2 images are split 50/50 into tune and test;
    single-image identities are split 50/50 the same way.
  * inside a split, 30% of the multi-image identities are held out as non-mated; the other 70%
    form the gallery: one enrolment image each (a random image that enrolled), the rest are mated probes.
  * single-image identities of the split are divided in half: one half pads the gallery as
    distractors (gallery sizes 100/500/1000/all), the other half are extra non-mated probes.
  * probes per identity are capped (default 10, seeded) so a few heavily photographed people do
    not dominate the error rates.
  * search uses the production SearchService (FAISS IndexFlatIP + IDMap) in a temporary directory,
    top_k = config.TOP_K. Probes that failed to enrol count as misses (mated) and as no alert (non-mated).
  * CIs: 95% percentile bootstrap over identities (thresholds are re-derived in every resample).
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import config
from eval import metrics as M
from eval.common import CACHE_DIR, DATA_DIR, metadata, plot_path, save_result
from eval.datasets import lfw
from eval.embed import STATUS_OK, cache_name, embed_paths

FPIR_TARGETS = [0.001, 0.01, 0.05]


def build_protocol(manifest, status, seed=0, nonmated_frac=0.3, max_probes=10):
    """Return {split: {gallery_ids, distractor_ids, enrol(dict id->path), probes(DataFrame)}}."""
    rng = np.random.default_rng(seed)
    ok = dict(zip(manifest["image_path"], status == STATUS_OK))
    by_id = manifest.groupby("identity_id")["image_path"].apply(lambda s: sorted(s)).to_dict()
    multi = sorted(i for i, ps in by_id.items() if len(ps) >= 2)
    single = sorted(i for i, ps in by_id.items() if len(ps) == 1)
    multi = list(rng.permutation(multi))
    single = list(rng.permutation(single))
    splits = {"tune": (multi[: len(multi) // 2], single[: len(single) // 2]),
              "test": (multi[len(multi) // 2:], single[len(single) // 2:])}

    out = {}
    for split, (m_ids, s_ids) in splits.items():
        m_ids = list(rng.permutation(m_ids))
        n_non = int(round(nonmated_frac * len(m_ids)))
        non_ids, gal_ids = m_ids[:n_non], m_ids[n_non:]
        half = len(s_ids) // 2
        distractor_ids, single_non_ids = s_ids[:half], s_ids[half:]

        enrol, probes, enrol_failed = {}, [], []
        for ident in gal_ids:
            paths = by_id[ident]
            candidates = [p for p in paths if ok[p]]
            if not candidates:
                enrol_failed.append(ident)
                continue
            e = candidates[int(rng.integers(len(candidates)))]
            enrol[ident] = e
            rest = [p for p in paths if p != e]
            rest = list(rng.permutation(rest))[:max_probes]
            probes += [(p, ident, True) for p in rest]
        for ident in non_ids:
            paths = list(rng.permutation(by_id[ident]))[:max_probes]
            probes += [(p, ident, False) for p in paths]
        for ident in single_non_ids:
            probes.append((by_id[ident][0], ident, False))
        distractors = {i: by_id[i][0] for i in distractor_ids if ok[by_id[i][0]]}
        out[split] = {
            "gallery_ids": [i for i in gal_ids if i in enrol],
            "enrol": enrol,
            "distractors": distractors,
            "enrol_failed": enrol_failed,
            "probes": pd.DataFrame(probes, columns=["image_path", "identity_id", "mated"]),
        }
    return out


def gallery_for_size(proto_split, size, seed=0):
    """Pick gallery identities for a target size: subsample mated identities when size is smaller
    than the number of mated identities, otherwise add distractors. Returns dict id -> enrol path."""
    rng = np.random.default_rng(seed + 17)
    mated = proto_split["gallery_ids"]
    if size != "all" and int(size) <= len(mated):
        keep = sorted(rng.permutation(mated)[: int(size)])
        return {i: proto_split["enrol"][i] for i in keep}
    gallery = {i: proto_split["enrol"][i] for i in mated}
    distractors = sorted(proto_split["distractors"])
    if size != "all":
        distractors = sorted(rng.permutation(distractors)[: int(size) - len(mated)])
    gallery.update({i: proto_split["distractors"][i] for i in distractors})
    return gallery


def search_probes(gallery, probes, emb_of, top_k):
    """Enrol the gallery in a temporary production SearchService and search every probe."""
    from vector_db.search_service import SearchService
    tmp = tempfile.mkdtemp(prefix="locatenet-openset-")
    try:
        ids = {ident: n + 1 for n, ident in enumerate(sorted(gallery))}
        names = {v: k for k, v in ids.items()}
        svc = SearchService(index_path=tmp)
        svc.rebuild_face_index([(ids[i], emb_of[p]) for i, p in gallery.items()])
        rows = []
        for p, ident, mated in probes.itertuples(index=False):
            in_gallery = ident in gallery
            if mated and not in_gallery:
                continue  # its gallery identity was subsampled away at this gallery size
            emb = emb_of.get(p)
            if emb is None:
                rows.append((p, ident, mated, None, -np.inf, 0))
                continue
            scores, found = svc.search_face(emb, top_k=top_k)
            found_names = [names[int(i)] for i in found]
            rank = found_names.index(ident) + 1 if ident in found_names else 0
            top_name = found_names[0] if found_names else None
            top_score = float(scores[0]) if len(scores) else -np.inf
            rows.append((p, ident, mated, top_name, top_score, rank))
        return pd.DataFrame(rows, columns=["image_path", "identity_id", "mated", "top_id", "top_score", "rank"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def to_units(records):
    """One unit per identity: numpy arrays of its mated scores/ranks and non-mated scores."""
    units = []
    for _, g in records.groupby("identity_id"):
        m = g[g["mated"]]
        n = g[~g["mated"]]
        units.append({"ms": m["top_score"].to_numpy(float), "mr": m["rank"].to_numpy(int),
                      "ns": n["top_score"].to_numpy(float)})
    return units


def _cat(us, key):
    return np.concatenate([u[key] for u in us]) if us else np.zeros(0)


def unit_arrays(us):
    ranks = _cat(us, "mr")
    return _cat(us, "ms"), ranks == 1, _cat(us, "ns")


def summarise(records, n_boot=1000, seed=0, extra_thresholds=None):
    """FNIR@FPIR (+threshold), rank-1/5, and FNIR/FPIR at fixed thresholds, all with identity bootstrap CIs."""
    units = to_units(records)
    out = {"n_mated_probes": int(records["mated"].sum()),
           "n_nonmated_probes": int((~records["mated"]).sum()),
           "n_identities": len(units), "fnir_at_fpir": {}, "at_threshold": {}}
    for fpir in FPIR_TARGETS:
        fn = M.bootstrap_ci(lambda us, f=fpir: M.fnir_at_fpir(*unit_arrays(us), f)[0], units, n=n_boot, seed=seed)
        th = M.bootstrap_ci(lambda us, f=fpir: M.threshold_at_rate(_cat(us, "ns"), f), units, n=n_boot, seed=seed)
        out["fnir_at_fpir"][str(fpir)] = {"fnir": fn[0], "fnir_ci": [fn[1], fn[2]],
                                          "threshold": th[0], "threshold_ci": [th[1], th[2]]}
    mated_units = [u for u in units if u["mr"].size]
    for k in (1, 5):
        def rate(us, k=k):
            r = _cat(us, "mr")
            return float(((r >= 1) & (r <= k)).mean()) if r.size else float("nan")
        r = M.bootstrap_ci(rate, mated_units, n=n_boot, seed=seed)
        out[f"rank{k}"] = {"rate": r[0], "ci": [r[1], r[2]]}
    for name, t in (extra_thresholds or {}).items():
        a = M.bootstrap_ci(lambda us, t=t: M.fnir_fpir_at_threshold(*unit_arrays(us), t)[0], units, n=n_boot, seed=seed)
        b = M.bootstrap_ci(lambda us, t=t: M.fnir_fpir_at_threshold(*unit_arrays(us), t)[1], units, n=n_boot, seed=seed)
        out["at_threshold"][name] = {"threshold": t, "fnir": a[0], "fnir_ci": [a[1], a[2]],
                                     "fpir": b[0], "fpir_ci": [b[1], b[2]]}
    t1 = out["fnir_at_fpir"]["0.01"]["threshold"]
    ms, mr = _cat(units, "ms"), _cat(units, "mr")
    if ms.size:
        out["miss_breakdown_at_fpir_0.01"] = {
            "failed_to_enrol": float(np.mean(~np.isfinite(ms))),
            "wrong_identity_at_rank1": float(np.mean(np.isfinite(ms) & (mr != 1))),
            "right_identity_below_threshold": float(np.mean((mr == 1) & (ms < t1))),
        }
    fpir, fnir, thr = M.det_curve(*unit_arrays(units))
    idx = np.unique(np.linspace(0, len(thr) - 1, min(300, len(thr))).astype(int))
    out["det"] = {"fpir": fpir[idx].tolist(), "fnir": fnir[idx].tolist(), "threshold": thr[idx].tolist()}
    return out


def records_path(split, size, tag=""):
    return os.path.join(CACHE_DIR, f"openset_records{tag}_{split}_{size}.csv")


def load_lfw_protocol(cfg):
    manifest = lfw.load_manifest(DATA_DIR)
    out = embed_paths(list(manifest["image_path"]), cache_name("lfw"))
    aliases = {}
    if cfg.get("merge_duplicate_identities", True):
        aliases = lfw.identity_aliases(set(manifest["identity_id"]), cfg.get("identity_aliases"))
        manifest = manifest.copy()
        manifest["identity_id"] = manifest["identity_id"].replace(aliases)
    cfg["_merged_identities"] = aliases
    status = out["status"]
    emb_of = {p: e for p, e, s in zip(manifest["image_path"], out["emb"], status) if s == STATUS_OK}
    proto = build_protocol(manifest, status, seed=cfg["seed"], nonmated_frac=cfg.get("nonmated_frac", 0.3),
                           max_probes=cfg.get("max_probes_per_identity", 10))
    return manifest, out, emb_of, proto


def run(cfg):
    manifest, out, emb_of, proto = load_lfw_protocol(cfg)
    sizes = cfg.get("gallery_sizes", [100, 500, 1000, "all"])
    top_k = int(cfg.get("top_k", config.TOP_K))
    current = {"MATCH_THRESHOLD": config.MATCH_THRESHOLD, "HIGH_CONF_THRESHOLD": config.HIGH_CONF_THRESHOLD}
    result = {"meta": metadata(cfg, lfw.DATASET_VERSION),
              "protocol": __doc__.strip().split("\n\n", 1)[1],
              "top_k": top_k, "fpir_targets": FPIR_TARGETS,
              "merged_duplicate_identities": cfg.get("_merged_identities", {}), "splits": {}}
    for split in ("tune", "test"):
        p = proto[split]
        result["splits"][split] = {"n_gallery_identities": len(p["gallery_ids"]),
                                   "n_distractors_available": len(p["distractors"]),
                                   "n_enrol_failed": len(p["enrol_failed"]), "by_gallery_size": {}}
        for size in sizes:
            gallery = gallery_for_size(p, size, seed=cfg["seed"])
            recs = search_probes(gallery, p["probes"], emb_of, top_k)
            recs.to_csv(records_path(split, size, cfg.get("records_tag", "")), index=False)
            summary = summarise(recs, n_boot=cfg.get("n_boot", 1000), seed=cfg["seed"], extra_thresholds=current)
            summary["gallery_size"] = len(gallery)
            result["splits"][split]["by_gallery_size"][str(size)] = summary
            f1 = summary["fnir_at_fpir"]["0.01"]
            print(f"[{split}] gallery {len(gallery):5d}: FNIR@FPIR=1% {f1['fnir']:.4f} "
                  f"[{f1['fnir_ci'][0]:.4f}, {f1['fnir_ci'][1]:.4f}] thr {f1['threshold']:.3f}; "
                  f"rank-1 {summary['rank1']['rate']:.4f}")
    if not cfg.get("records_tag"):
        plot_det(result, sizes)
        plot_hist(cfg)
    save_result(cfg.get("result_name", "openset_lfw"), result)


def plot_det(result, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for size in sizes:
        s = result["splits"]["test"]["by_gallery_size"][str(size)]
        fpir = np.clip(np.array(s["det"]["fpir"]), 1e-4, 1)
        ax.plot(fpir, np.clip(s["det"]["fnir"], 1e-4, 1), lw=2, label=f"gallery {s['gallery_size']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("FPIR (non-mated searches that raise an alert)")
    ax.set_ylabel("FNIR (mated searches missed at rank 1)")
    ax.set_title("LFW open-set DET, test split (buffalo_l)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path("openset_det.png"), dpi=130)
    plt.close(fig)


def plot_hist(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = cfg.get("hist_gallery_size", 1000)
    recs = pd.read_csv(records_path("test", size))
    recs = recs[np.isfinite(recs["top_score"])]
    mated_ok = recs[recs["mated"] & (recs["rank"] == 1)]["top_score"]
    non = recs[~recs["mated"]]["top_score"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(-0.2, 1.0, 61)
    ax.hist(non, bins=bins, alpha=0.6, density=True, label=f"non-mated (n={len(non)})")
    ax.hist(mated_ok, bins=bins, alpha=0.6, density=True, label=f"mated, correct rank-1 (n={len(mated_ok)})")
    ax.axvline(config.MATCH_THRESHOLD, color="k", ls="--", label=f"MATCH_THRESHOLD = {config.MATCH_THRESHOLD}")
    ax.set_xlabel("Rank-1 cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Rank-1 score distribution, test split, gallery {size}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path("score_hist.png"), dpi=130)
    plt.close(fig)
