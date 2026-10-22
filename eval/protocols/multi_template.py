"""Experiment `multi_template`: does enrolling several photos per case help, and how should their
similarities be combined?

LFW: identities with >= 6 enrolled images, split into tune / test by identity (the Week 2 open-set
partition). In each split 70% are gallery cases and 30% are held out as non-mated; single-image
identities pad the gallery (distractors, 1 photo each) or act as extra non-mated probes. For each
gallery identity the images are shuffled once (seeded): the first k (k = 1, 2, 3, 5, nested) are
enrolled and images 6..15 are the probes, so every k is scored on the same probes.
FG-NET: subjects are split tune / test; each subject's latest third of photos (by age) are the probes,
the earlier photos are the enrolment pool, from which k photos are picked either at random or
age-diverse (spread evenly over the pool's ages, always including the most recent). 30% of the
subjects are held out as non-mated; LFW single-image identities pad the gallery.

Search uses the production path: SearchService (one FAISS vector per photo) + TemplateMatcher with
top_k * MAX_PHOTOS_PER_CASE neighbours, grouped by case and re-scored. Because `max` over more
photos also raises impostor scores, each (strategy, k) gets its OWN threshold at FPIR = 1% chosen
on the tune split, applied unchanged to the test split. FPIR at the deployed MATCH_THRESHOLD is
reported too, to show how much each strategy inflates false alerts at one global threshold.
Photo quality for quality_weighted_mean = detector score x face-size factor (services/photo_quality).
"""
import shutil
import tempfile

import numpy as np
import pandas as pd

import config
from eval import metrics as M
from eval.common import DATA_DIR, metadata, plot_path, save_result
from eval.datasets import fgnet
from eval.embed import STATUS_OK, cache_name, embed_paths
from eval.protocols import openset
from pipeline.scoring import rank_cases
from services import photo_quality
from services.template_matcher import STRATEGIES, TemplateMatcher

KS = [1, 2, 3, 5]


def _quality(det, box):
    width = float(box[2] - box[0]) if np.all(np.isfinite(box)) else 0.0
    return photo_quality.quality_score(det, width, photo_quality.REFERENCE_SHARPNESS)


def search(gallery_rows, probes, emb_of, strategy, max_photos):
    """gallery_rows: list of (vector_id, case_name, path, quality). Returns per-probe records."""
    from vector_db.search_service import SearchService
    names = sorted({r[1] for r in gallery_rows})
    case_id = {n: i + 1 for i, n in enumerate(names)}
    back = {v: k for k, v in case_id.items()}
    rows = [(vid, case_id[name], emb_of[path], q, None) for vid, name, path, q in gallery_rows]
    matcher = TemplateMatcher.from_rows(rows, strategy=strategy)
    matcher.max_photos = max_photos
    tmp = tempfile.mkdtemp(prefix="locatenet-mt-")
    try:
        svc = SearchService(index_path=tmp)
        svc.rebuild_face_index([(r[0], r[2]) for r in rows])
        out = []
        for p, ident, mated in probes.itertuples(index=False):
            emb = emb_of.get(p)
            if emb is None:
                out.append((ident, mated, -np.inf, 0))
                continue
            scores, ids = rank_cases(emb, svc, matcher, top_k=5)
            found = [back[int(i)] for i in ids]
            rank = found.index(ident) + 1 if ident in found else 0
            out.append((ident, mated, float(scores[0]) if len(scores) else -np.inf, rank))
        return pd.DataFrame(out, columns=["identity_id", "mated", "top_score", "rank"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(tune_recs, test_recs, production_threshold, n_boot, seed):
    t = M.threshold_at_rate(tune_recs[~tune_recs["mated"]]["top_score"].to_numpy(float), 0.01)
    units = openset.to_units(test_recs)
    fnir = M.bootstrap_ci(lambda us: M.fnir_fpir_at_threshold(*openset.unit_arrays(us), t)[0], units, n=n_boot, seed=seed)
    fpir = M.fnir_fpir_at_threshold(*openset.unit_arrays(units), t)[1]
    fnir_p, fpir_p = M.fnir_fpir_at_threshold(*openset.unit_arrays(units), production_threshold)
    mated = test_recs[test_recs["mated"]]
    return {"threshold_fpir1_tune": t, "test_fnir": fnir[0], "test_fnir_ci": [fnir[1], fnir[2]], "test_fpir": fpir,
            "test_rank1": float((mated["rank"] == 1).mean()),
            "fnir_at_production_threshold": fnir_p, "fpir_at_production_threshold": fpir_p}


def lfw_part(cfg, n_boot, seed):
    manifest, out, emb_of, proto = openset.load_lfw_protocol(cfg)
    det = dict(zip(manifest["image_path"], out["det_score"]))
    box = dict(zip(manifest["image_path"], out["bbox"]))
    by_id = manifest.groupby("identity_id")["image_path"].apply(lambda s: sorted(p for p in s if p in emb_of)).to_dict()
    rng = np.random.default_rng(seed + 31)
    per_split = {}
    for split in ("tune", "test"):
        p = proto[split]
        ids = sorted(i for i in set(p["gallery_ids"]) | set(p["probes"]["identity_id"]) if len(by_id.get(i, [])) >= 6)
        ids = list(rng.permutation(ids))
        n_non = int(round(0.3 * len(ids)))
        non_ids, gal_ids = ids[:n_non], ids[n_non:]
        order = {i: list(rng.permutation(by_id[i])) for i in gal_ids}
        probes = [(q, i, True) for i in gal_ids for q in order[i][5:15]]
        probes += [(q, i, False) for i in non_ids for q in list(rng.permutation(by_id[i]))[:10]]
        singles = sorted(p["distractors"])
        half = len(singles) // 2
        distractors = singles[:half]
        probes += [(p["distractors"][i], i, False) for i in singles[half:]]
        pad = max(0, int(cfg.get("gallery_size", 1000)) - len(gal_ids))
        per_split[split] = {"gal": gal_ids, "order": order, "distractors": distractors[:pad],
                            "dist_path": p["distractors"],
                            "probes": pd.DataFrame(probes, columns=["image_path", "identity_id", "mated"])}
    res = {}
    for k in KS:
        res[str(k)] = {}
        recs = {}
        for split, d in per_split.items():
            rows, vid = [], 1
            for i in d["gal"]:
                for path in d["order"][i][:k]:
                    rows.append((vid, i, path, _quality(det[path], box[path])))
                    vid += 1
            for i in d["distractors"]:
                path = d["dist_path"][i]
                rows.append((vid, i, path, _quality(det[path], box[path])))
                vid += 1
            recs[split] = {s: search(rows, d["probes"], emb_of, s, max(KS)) for s in STRATEGIES}
        for s in STRATEGIES:
            res[str(k)][s] = evaluate(recs["tune"][s], recs["test"][s], config.MATCH_THRESHOLD, n_boot, seed)
        print("LFW k=%d " % k + "  ".join(f"{s}: {v['test_fnir']:.4f} (t={v['threshold_fpir1_tune']:.3f}, "
                                            f"fpir@prod {v['fpir_at_production_threshold']:.4f})" for s, v in res[str(k)].items()))
    return res, {s: {"n_gallery_cases": len(d["gal"]), "n_distractors": len(d["distractors"]),
                     "n_mated_probes": int(d["probes"]["mated"].sum()), "n_nonmated_probes": int((~d["probes"]["mated"]).sum())}
                 for s, d in per_split.items()}


def age_diverse(pool, k):
    """k photos spread evenly over the pool's ages (pool sorted by age), always including the latest."""
    if k >= len(pool):
        return list(pool)
    idx = np.unique(np.round(np.linspace(len(pool) - 1, 0, k)).astype(int))
    return [pool[i] for i in sorted(idx)]


def fgnet_part(cfg, n_boot, seed):
    man = fgnet.load_manifest(DATA_DIR)
    out = embed_paths(list(man["image_path"]), cache_name("fgnet"))
    emb_of = {p: e for p, e, st in zip(man["image_path"], out["emb"], out["status"]) if st == STATUS_OK}
    det = dict(zip(man["image_path"], out["det_score"]))
    box = dict(zip(man["image_path"], out["bbox"]))
    age = dict(zip(man["image_path"], fgnet.ages(man)))
    _, _, lfw_emb, proto = openset.load_lfw_protocol({"seed": 0, "nonmated_frac": 0.3, "max_probes_per_identity": 10,
                                                       "identity_aliases": cfg.get("identity_aliases")})
    emb_all = {**lfw_emb, **emb_of}
    rng = np.random.default_rng(seed + 53)
    subjects = list(rng.permutation(sorted(man["identity_id"].unique())))
    splits = {"tune": subjects[: len(subjects) // 2], "test": subjects[len(subjects) // 2:]}
    per_split = {}
    for split, subs in splits.items():
        subs = list(rng.permutation(subs))
        n_non = int(round(0.3 * len(subs)))
        non, gal = subs[:n_non], subs[n_non:]
        pools, probes = {}, []
        for s in gal:
            photos = sorted((p for p in man.loc[man["identity_id"] == s, "image_path"] if p in emb_of), key=lambda p: (age[p], p))
            n_probe = max(1, len(photos) // 3)
            pools[s] = photos[:-n_probe]
            probes += [(p, s, True) for p in photos[-n_probe:]]
        for s in non:
            probes += [(p, s, False) for p in man.loc[man["identity_id"] == s, "image_path"]]
        lfw_non = proto[split]["probes"]
        lfw_non = lfw_non[~lfw_non["mated"]]
        probes += list(lfw_non[["image_path", "identity_id", "mated"]].itertuples(index=False, name=None))
        per_split[split] = {"pools": {s: p for s, p in pools.items() if p}, "probes": pd.DataFrame(probes, columns=["image_path", "identity_id", "mated"]),
                            "distractors": proto[split]["distractors"]}
    res = {}
    for mode in ("random", "age_diverse"):
        res[mode] = {}
        for k in KS:
            recs = {}
            for split, d in per_split.items():
                rows, vid = [], 1
                for s, pool in d["pools"].items():
                    pick = age_diverse(pool, k) if mode == "age_diverse" else list(rng.permutation(pool))[:k]
                    for path in pick:
                        rows.append((vid, s, path, _quality(det[path], box[path])))
                        vid += 1
                for i, path in d["distractors"].items():
                    rows.append((vid, i, path, 0.5))
                    vid += 1
                recs[split] = {s: search(rows, d["probes"], emb_all, s, max(KS)) for s in STRATEGIES}
            res[mode][str(k)] = {s: evaluate(recs["tune"][s], recs["test"][s], config.MATCH_THRESHOLD, n_boot, seed)
                                 for s in STRATEGIES}
            print(f"FG-NET {mode} k={k} " + "  ".join(f"{s}: {v['test_fnir']:.3f}" for s, v in res[mode][str(k)].items()))
    return res, {s: {"n_gallery_subjects": len(d["pools"]), "n_mated_probes": int(d["probes"]["mated"].sum()),
                     "n_nonmated_probes": int((~d["probes"]["mated"]).sum())} for s, d in per_split.items()}


def choose(lfw_res, fg_res):
    """Lowest mean test FNIR over k >= 2 on LFW and FG-NET (age-diverse); ties -> max (simplest)."""
    score = {}
    for s in STRATEGIES:
        vals = [lfw_res[str(k)][s]["test_fnir"] for k in KS if k > 1]
        vals += [fg_res["age_diverse"][str(k)][s]["test_fnir"] for k in KS if k > 1]
        score[s] = float(np.mean(vals))
    best = min(score.values())
    winners = [s for s in STRATEGIES if score[s] <= best + 1e-9]
    return ("max" if "max" in winners else winners[0]), score


def run(cfg):
    n_boot, seed = int(cfg.get("n_boot", 500)), int(cfg["seed"])
    lfw_res, lfw_sizes = lfw_part(cfg, n_boot, seed)
    fg_res, fg_sizes = fgnet_part(cfg, n_boot, seed)
    chosen, score = choose(lfw_res, fg_res)
    result = {"meta": metadata(cfg, f"LFW (Week 2 protocol) + {fgnet.DATASET_VERSION}"),
              "protocol": __doc__.strip().split("\n\n", 1)[1], "ks": KS, "strategies": list(STRATEGIES),
              "production_threshold": config.MATCH_THRESHOLD, "lfw": lfw_res, "lfw_sizes": lfw_sizes,
              "fgnet": fg_res, "fgnet_sizes": fg_sizes, "mean_fnir_k_ge_2": score, "chosen_strategy": chosen,
              "plot": "eval/plots/multi_template.png"}
    plot(result)
    save_result("multi_template", result)
    print("chosen strategy:", chosen, score)


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=False)
    panels = (("LFW (>= 6 photos)", r["lfw"]), ("FG-NET random pick", r["fgnet"]["random"]),
              ("FG-NET age-diverse pick", r["fgnet"]["age_diverse"]))
    for ax, (title, d) in zip(axes, panels):
        for s in r["strategies"]:
            ys = [d[str(k)][s]["test_fnir"] for k in r["ks"]]
            ax.plot(r["ks"], ys, marker="o", label=s)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("reference photos per case (k)")
        ax.set_xticks(r["ks"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test FNIR @ FPIR 1% (threshold per strategy)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path("multi_template.png"), dpi=130)
    plt.close(fig)
