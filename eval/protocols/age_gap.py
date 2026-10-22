"""Experiment `age_gap`: how matching degrades when the reference photo is years old.

1:1 on FG-NET (ages 0-69, 82 subjects):
  * genuine pairs: every within-subject pair; gap = |age_a - age_b| bucketed {0-2, 3-5, 6-10, 11-20, 21+}
  * impostor pairs: for each genuine pair, `impostors_per_genuine` pairs of two different subjects
    whose ages match the genuine pair (each within +-1 year when possible), so impostors have the
    same age mix as the genuine pairs of the bucket
  * FNMR at the production MATCH_THRESHOLD and TAR at FAR=1e-3 per bucket; pairs with an image
    that failed to enrol count as non-matches (production behaviour)
  * reported separately for child enrolment (younger photo < 18 years) and adult enrolment
1:1 on CALFW (cross-age LFW, 6000 official pairs; no per-image ages): 10-fold accuracy, TAR@FAR=1e-3,
  FNMR / FMR at the production threshold, next to the LFW numbers.
Open-set on FG-NET: subjects are split (seeded) into gallery subjects and held-out non-mated subjects;
  the gallery is the earliest enrolled photo of each gallery subject plus LFW single-image test
  distractors; probes are the later photos. The FPIR = 1% threshold comes from the non-mated
  probes (held-out FG-NET subjects + LFW non-mated test probes). Rank-1 and FNIR per age-gap bucket.
CIs: 95% bootstrap over subjects.
"""
import itertools

import numpy as np
import pandas as pd

from eval import metrics as M
from eval.common import DATA_DIR, load_result, metadata, plot_path, save_result
from eval.datasets import calfw, fgnet
from eval.embed import STATUS_OK, cache_name, embed_paths
from eval.protocols import openset

BUCKETS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 200)]


def bucket_name(lo, hi):
    return f"{lo}-{hi}" if hi < 200 else f"{lo}+"


def bucket_of(gap):
    for lo, hi in BUCKETS:
        if lo <= gap <= hi:
            return bucket_name(lo, hi)
    return None


def production_threshold(cfg):
    if "threshold" in cfg:
        return float(cfg["threshold"])
    return float(load_result("operating_point")["recommended"]["MATCH_THRESHOLD"])


def fgnet_pairs(manifest, ages, emb_of, rng, impostors_per_genuine):
    """Genuine pairs of every subject and age-matched impostor pairs. Score -1 if an image failed."""
    paths = manifest["image_path"].tolist()
    subj = manifest["identity_id"].tolist()
    by_subject = manifest.groupby("identity_id").indices
    by_age = {}
    for i, a in enumerate(ages):
        by_age.setdefault(int(a), []).append(i)

    def score(i, j):
        a, b = emb_of.get(paths[i]), emb_of.get(paths[j])
        return float(np.dot(a, b)) if a is not None and b is not None else -1.0

    def pick(age, exclude):
        for tol in (0, 1, 2, 3, 5, 8, 100):
            cands = [k for a in range(age - tol, age + tol + 1) for k in by_age.get(a, []) if subj[k] not in exclude]
            if cands:
                return cands[int(rng.integers(len(cands)))]
        return None

    rows = []
    for s, idx in by_subject.items():
        for i, j in itertools.combinations(sorted(idx), 2):
            ai, aj = int(ages[i]), int(ages[j])
            young, old = min(ai, aj), max(ai, aj)
            rows.append((s, True, ai, aj, old - young, young, score(i, j)))
            for _ in range(impostors_per_genuine):
                x = pick(ai, {s})
                y = pick(aj, {s, subj[x]}) if x is not None else None
                if x is None or y is None:
                    continue
                ax, ay = int(ages[x]), int(ages[y])
                rows.append((s, False, ax, ay, abs(ax - ay), min(ax, ay), score(x, y)))
    df = pd.DataFrame(rows, columns=["subject", "same", "age_a", "age_b", "gap", "younger_age", "score"])
    df["bucket"] = df["gap"].map(bucket_of)
    df["child"] = df["younger_age"] < 18
    return df


def bucket_stats(df, threshold, n_boot, seed):
    units = [(g.loc[g["same"], "score"].to_numpy(), g.loc[~g["same"], "score"].to_numpy()) for _, g in df.groupby("subject")]

    def fnmr(us):
        gen = np.concatenate([u[0] for u in us])
        return float(np.mean(gen < threshold)) if gen.size else float("nan")

    def tar(us):
        gen = np.concatenate([u[0] for u in us])
        imp = np.concatenate([u[1] for u in us])
        return M.tar_at_far(gen, imp, 1e-3)[0] if gen.size and imp.size else float("nan")

    def fmr(us):
        imp = np.concatenate([u[1] for u in us])
        return float(np.mean(imp >= threshold)) if imp.size else float("nan")
    a = M.bootstrap_ci(fnmr, units, n=n_boot, seed=seed)
    b = M.bootstrap_ci(tar, units, n=n_boot, seed=seed)
    c = M.bootstrap_ci(fmr, units, n=n_boot, seed=seed)
    gen = df[df["same"]]
    return {"n_subjects": int(df["subject"].nunique()), "n_genuine": int(len(gen)), "n_impostor": int((~df["same"]).sum()),
            "genuine_failed_enrol": float(np.mean(gen["score"] <= -1.0)) if len(gen) else None,
            "fnmr": a[0], "fnmr_ci": [a[1], a[2]], "tar_at_far_1e-3": b[0], "tar_ci": [b[1], b[2]],
            "fmr": c[0], "fmr_ci": [c[1], c[2]], "genuine_mean_score": float(gen["score"].clip(lower=-1).mean()) if len(gen) else None}


def calfw_stats(threshold, cfg):
    pairs = calfw.load_pairs(DATA_DIR)
    paths = sorted(set(pairs["path_a"]) | set(pairs["path_b"]))
    out = embed_paths(paths, cache_name("calfw"))
    lookup = {p: e for p, e, s in zip(paths, out["emb"], out["status"]) if s == STATUS_OK}
    scores = np.array([float(np.dot(lookup[a], lookup[b])) if a in lookup and b in lookup else -1.0
                       for a, b in zip(pairs["path_a"], pairs["path_b"])])
    same, folds = pairs["same"].to_numpy(), pairs["fold"].to_numpy()
    enrolled = np.array([a in lookup and b in lookup for a, b in zip(pairs["path_a"], pairs["path_b"])])
    accs, accs_e = [], []
    for k in range(10):
        tr, te = folds != k, folds == k
        t, _ = M.best_accuracy_threshold(scores[tr & same], scores[tr & ~same])
        accs.append(M.verification_accuracy(scores[te & same], scores[te & ~same], t))
        t, _ = M.best_accuracy_threshold(scores[tr & same & enrolled], scores[tr & ~same & enrolled])
        accs_e.append(M.verification_accuracy(scores[te & same & enrolled], scores[te & ~same & enrolled], t))
    gen, imp = scores[same], scores[~same]
    tar, thr = M.tar_at_far(gen, imp, 1e-3)
    return {"n_pairs": int(len(pairs)), "n_pairs_with_failed_image": int((~enrolled).sum()),
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "accuracy_enrolled_pairs_mean": float(np.mean(accs_e)), "accuracy_enrolled_pairs_std": float(np.std(accs_e)),
            "tar_at_far_1e-3": tar, "tar_threshold": thr,
            "fnmr_at_production_threshold": float(np.mean(gen < threshold)),
            "fmr_at_production_threshold": float(np.mean(imp >= threshold))}


def fgnet_openset(manifest, ages, emb_of, threshold, cfg, n_boot, seed):
    rng = np.random.default_rng(seed + 7)
    subjects = sorted(manifest["identity_id"].unique())
    held = set(rng.choice(subjects, int(round(cfg.get("heldout_frac", 0.25) * len(subjects))), replace=False))
    gallery, probes = {}, []
    age_of = dict(zip(manifest["image_path"], ages))
    for s, g in manifest.groupby("identity_id"):
        rows = sorted(zip(g["image_path"], [age_of[p] for p in g["image_path"]]), key=lambda r: (r[1], r[0]))
        if s in held:
            probes += [(p, s, False, a, None) for p, a in rows]
            continue
        first = next(((p, a) for p, a in rows if p in emb_of), None)
        if first is None:
            continue
        gallery[s] = first[0]
        probes += [(p, s, True, a, a - first[1]) for p, a in rows if p != first[0]]
    # LFW test-split distractors pad the gallery; LFW non-mated test probes join the non-mated pool
    # same LFW protocol (seed, splits, merged names) as the Week 2 open-set experiment
    _, _, lfw_emb, proto = openset.load_lfw_protocol({"seed": 0, "nonmated_frac": 0.3, "max_probes_per_identity": 10,
                                                       "merge_duplicate_identities": True,
                                                       "identity_aliases": cfg.get("identity_aliases")})
    test = proto["test"]
    gallery.update(test["distractors"])
    lfw_non = test["probes"][~test["probes"]["mated"]]
    emb_all = {**lfw_emb, **emb_of}
    probe_df = pd.DataFrame(probes, columns=["image_path", "identity_id", "mated", "age", "gap"])
    lfw_df = pd.DataFrame({"image_path": lfw_non["image_path"], "identity_id": lfw_non["identity_id"], "mated": False,
                           "age": np.nan, "gap": np.nan})
    all_probes = pd.concat([probe_df, lfw_df], ignore_index=True)
    recs = openset.search_probes(gallery, all_probes[["image_path", "identity_id", "mated"]], emb_all, top_k=5)
    recs = recs.merge(all_probes[["image_path", "gap"]], on="image_path", how="left")
    non = recs[~recs["mated"]]["top_score"].to_numpy(float)
    t1 = M.threshold_at_rate(non, 0.01)
    out = {"n_gallery_subjects": len([k for k in gallery if k.startswith("fgnet_")]), "n_distractors": len(test["distractors"]),
           "n_nonmated_probes": int(len(non)), "threshold_fpir1": t1,
           "fpir_at_production_threshold": float(np.mean(non >= threshold)), "by_gap": {}}
    mated = recs[recs["mated"]].copy()
    mated["bucket"] = mated["gap"].map(lambda g: bucket_of(int(g)) if pd.notna(g) else None)
    for lo, hi in BUCKETS:
        name = bucket_name(lo, hi)
        g = mated[mated["bucket"] == name]
        if g.empty:
            continue
        units = [(u["top_score"].to_numpy(float), u["rank"].to_numpy(int)) for _, u in g.groupby("identity_id")]

        def fnir_at(t):
            def f(us):
                s = np.concatenate([u[0] for u in us])
                r = np.concatenate([u[1] for u in us])
                return float(1 - np.mean((r == 1) & (s >= t)))
            return f

        def rank1(us):
            return float(np.mean(np.concatenate([u[1] for u in us]) == 1))
        a = M.bootstrap_ci(fnir_at(t1), units, n=n_boot, seed=seed)
        b = M.bootstrap_ci(rank1, units, n=n_boot, seed=seed)
        c = M.bootstrap_ci(fnir_at(threshold), units, n=n_boot, seed=seed)
        out["by_gap"][name] = {"n_probes": int(len(g)), "n_subjects": int(g["identity_id"].nunique()),
                               "fnir_fpir1": a[0], "fnir_fpir1_ci": [a[1], a[2]], "rank1": b[0], "rank1_ci": [b[1], b[2]],
                               "fnir_production": c[0], "fnir_production_ci": [c[1], c[2]]}
    return out


def run(cfg):
    threshold = production_threshold(cfg)
    n_boot, seed = int(cfg.get("n_boot", 1000)), int(cfg["seed"])
    rng = np.random.default_rng(seed)
    man = fgnet.load_manifest(DATA_DIR)
    out = embed_paths(list(man["image_path"]), cache_name("fgnet"))
    emb_of = {p: e for p, e, s in zip(man["image_path"], out["emb"], out["status"]) if s == STATUS_OK}
    ages = fgnet.ages(man).to_numpy()
    pairs = fgnet_pairs(man, ages, emb_of, rng, int(cfg.get("impostors_per_genuine", 10)))

    one_to_one = {"all": {}, "child_enrolment": {}, "adult_enrolment": {}}
    for lo, hi in BUCKETS:
        name = bucket_name(lo, hi)
        b = pairs[pairs["bucket"] == name]
        one_to_one["all"][name] = bucket_stats(b, threshold, n_boot, seed)
        for key, mask in (("child_enrolment", b["child"]), ("adult_enrolment", ~b["child"])):
            sub = b[mask]
            if sub["same"].sum() >= int(cfg.get("min_genuine", 20)):
                one_to_one[key][name] = bucket_stats(sub, threshold, n_boot, seed)

    lfw_v = load_result("verification_lfw")
    result = {
        "meta": metadata(cfg, f"{fgnet.DATASET_VERSION}; {calfw.DATASET_VERSION}"),
        "protocol": __doc__.strip().split("\n\n", 1)[1], "threshold": threshold,
        "fgnet_fte": float(np.mean(out["status"] != STATUS_OK)), "fgnet_n_images": int(len(man)),
        "fgnet_1to1": one_to_one,
        "fgnet_openset": fgnet_openset(man, ages, emb_of, threshold, cfg, n_boot, seed),
        "calfw_1to1": calfw_stats(threshold, cfg) if calfw.available(DATA_DIR) else None,
        "lfw_reference": {"accuracy_enrolled_pairs_mean": lfw_v["accuracy_enrolled_pairs_mean"],
                          "tar_at_far_1e-3": lfw_v["tar_at_far"]["0.001"]["tar"]},
        "agedb": "not available (licence request needed); loader in eval/datasets/restricted.py",
        "plot": "eval/plots/age_gap_fnmr.png",
    }
    a = one_to_one["all"]
    first, last = bucket_name(*BUCKETS[0]), bucket_name(*BUCKETS[-1])
    child, adult = one_to_one["child_enrolment"], one_to_one["adult_enrolment"]
    result["headline"] = (f"FNMR at the production threshold rises from {a[first]['fnmr']:.1%} (gap {first} years) "
                          f"to {a[last]['fnmr']:.1%} (gap {last} years); child enrolments degrade fastest "
                          f"({child[first]['fnmr']:.1%} -> {child[last]['fnmr']:.1%}) while adult enrolments stay at "
                          f"{adult[last]['fnmr']:.1%} even at {last} years")
    for k, v in a.items():
        print(f"gap {k:6s}: genuine {v['n_genuine']:5d}  FNMR {v['fnmr']:.3f} {np.round(v['fnmr_ci'], 3)}  TAR@1e-3 {v['tar_at_far_1e-3']:.3f}")
    plot(one_to_one)
    save_result("age_gap", result)
    print(result["headline"])


def plot(one_to_one):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [bucket_name(*b) for b in BUCKETS]
    for key, label, style in (("all", "all pairs", "-"), ("adult_enrolment", "adult enrolment (younger photo >= 18)", "--"),
                              ("child_enrolment", "child enrolment (younger photo < 18)", ":")):
        d = one_to_one[key]
        xs = [i for i, n in enumerate(names) if n in d]
        if not xs:
            continue
        v = np.array([d[names[i]]["fnmr"] for i in xs])
        lo = np.array([d[names[i]]["fnmr_ci"][0] for i in xs])
        hi = np.array([d[names[i]]["fnmr_ci"][1] for i in xs])
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], marker="o", ls=style, capsize=3, label=label)
    ax.set_xticks(range(len(names)), [f"{n} y" for n in names])
    ax.set_xlabel("age gap between the two photos")
    ax.set_ylabel("FNMR at production threshold")
    ax.set_ylim(0, 1)
    ax.set_title("FG-NET: misses grow with the age gap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path("age_gap_fnmr.png"), dpi=130)
    plt.close(fig)

