"""Experiment `bias`: demographic error rates at the single global operating threshold.

Preferred data: RFW (official 1:1 pairs per group). If RFW is not present, the FALLBACK uses LFW
with the Kumar et al. attribute scores (see eval.datasets.lfw.identity_groups for the labelling
rule). The fallback is weaker evidence: LFW is mostly white and male, the labels come from a
classifier and are noisy, and some groups have very few identities.

Views, all on the LFW **test** split only (the threshold came from the tune split):
  * 1:1 within-group: FNMR on genuine pairs (up to 10 per person) and FMR on ALL impostor pairs of
    two different people of the same group (first enrolled image of each), at the global
    MATCH_THRESHOLD. Because FMR at that threshold is close to zero everywhere, we also report a
    diagnostic FMR at the stricter-tail threshold where the pooled FMR is 1e-4, and impostor
    score tails (99th / 99.9th percentile, max), which show differences even when errors are rare.
  * open-set (Task 3 protocol, gallery 1000): per group FPIR and FNIR at the global threshold.
CIs: 95% bootstrap over identities. Disparity = max/min across groups; groups with zero errors get
a rule-of-three 95% upper bound (3/n) instead of a point ratio.
"""
import itertools

import numpy as np
import pandas as pd

from eval import metrics as M
from eval.common import DATA_DIR, load_result, metadata, plot_path, save_result
from eval.datasets import lfw, rfw
from eval.embed import STATUS_OK, cache_name, embed_paths
from eval.protocols import openset

GROUPS = ["white", "black", "asian", "indian", "african", "caucasian"]
DIAGNOSTIC_FMR = 1e-4


def global_threshold(cfg):
    if "threshold" in cfg:
        return float(cfg["threshold"]), "config"
    op = load_result("operating_point")
    return float(op["recommended"]["MATCH_THRESHOLD"]), "eval/results/operating_point.json"


def units_from_embeddings(ids, imgs_by_id, emb_of, rng, max_genuine_per_id=10):
    """One unit per identity: its genuine pair scores and its impostor scores against every later
    identity of the same group (so each impostor pair is counted once)."""
    gen, firsts = [], []
    for ident in ids:
        imgs = [p for p in imgs_by_id[ident] if p in emb_of]
        pairs = list(itertools.combinations(imgs, 2))
        if len(pairs) > max_genuine_per_id:
            pairs = [pairs[i] for i in rng.choice(len(pairs), max_genuine_per_id, replace=False)]
        gen.append(np.array([float(np.dot(emb_of[a], emb_of[b])) for a, b in pairs]))
        firsts.append(imgs[0] if imgs else None)
    keep = [k for k, f in enumerate(firsts) if f is not None]
    emb = np.stack([emb_of[firsts[k]] for k in keep]) if keep else np.zeros((0, 512), "float32")
    sims = emb @ emb.T
    units = []
    for row, k in enumerate(keep):
        units.append({"id": ids[k], "gen": gen[k], "imp": sims[row, row + 1:].astype("float64")})
    return units


def units_from_pairs(pairs):
    return [{"id": i, "gen": g.loc[g["same"], "score"].to_numpy(float), "imp": g.loc[~g["same"], "score"].to_numpy(float)}
            for i, g in pairs.groupby("id_a")]


def _cat(us, key):
    return np.concatenate([u[key] for u in us]) if us else np.zeros(0)


def fmr_at(t):
    def f(us):
        imp = _cat(us, "imp")
        return float(np.mean(imp >= t)) if imp.size else float("nan")
    return f


def fnmr_at(t):
    def f(us):
        gen = _cat(us, "gen")
        return float(np.mean(gen < t)) if gen.size else float("nan")
    return f


def group_stats(units, threshold, diag_threshold, n_boot, seed):
    imp, gen = _cat(units, "imp"), _cat(units, "gen")
    a = M.bootstrap_ci(fmr_at(threshold), units, n=n_boot, seed=seed)
    b = M.bootstrap_ci(fnmr_at(threshold), units, n=n_boot, seed=seed)
    c = M.bootstrap_ci(fmr_at(diag_threshold), units, n=n_boot, seed=seed)
    out = {"n_identities": len(units), "n_genuine": int(gen.size), "n_impostor": int(imp.size),
           "fmr": a[0], "fmr_ci": [a[1], a[2]], "fnmr": b[0], "fnmr_ci": [b[1], b[2]],
           "fmr_diagnostic": c[0], "fmr_diagnostic_ci": [c[1], c[2]],
           "fmr_errors": int(np.sum(imp >= threshold)), "fnmr_errors": int(np.sum(gen < threshold)),
           "impostor_p99": float(np.quantile(imp, 0.99)) if imp.size else None,
           "impostor_p999": float(np.quantile(imp, 0.999)) if imp.size else None,
           "impostor_max": float(imp.max()) if imp.size else None,
           "genuine_p01": float(np.quantile(gen, 0.01)) if gen.size else None}
    # rule of three: with zero observed errors in n trials, the 95% upper bound is about 3/n
    out["fmr_upper_if_zero"] = 3.0 / imp.size if imp.size and out["fmr_errors"] == 0 else None
    out["fnmr_upper_if_zero"] = 3.0 / gen.size if gen.size and out["fnmr_errors"] == 0 else None
    return out


def disparity(stats, key):
    vals = {g: s[key] for g, s in stats.items() if s[key] is not None and np.isfinite(s[key])}
    if len(vals) < 2:
        return {"ratio": None, "note": "fewer than two groups with data"}
    hi_g, lo_g = max(vals, key=vals.get), min(vals, key=vals.get)
    ratio = vals[hi_g] / vals[lo_g] if vals[lo_g] > 0 else None
    note = None
    if ratio is None:
        note = (f"{lo_g} has zero observed errors; ratio unbounded. "
                f"Using the rule-of-three upper bound for {lo_g} gives a ratio of at least "
                f"{vals[hi_g] / stats[lo_g][key + '_upper_if_zero']:.2f}x" if stats[lo_g].get(key + "_upper_if_zero")
                and vals[hi_g] > 0 else f"{lo_g} has zero observed errors; ratio unbounded")
    nz = {g: v for g, v in vals.items() if v > 0}
    nz_lo = min(nz, key=nz.get) if len(nz) >= 2 else None
    return {"ratio": ratio, "max_group": hi_g, "min_group": lo_g, "max": vals[hi_g], "min": vals[lo_g], "note": note,
            "ratio_nonzero_groups": vals[hi_g] / nz[nz_lo] if nz_lo else None, "min_nonzero_group": nz_lo,
            "nonzero_groups": sorted(nz)}


def ratio_ci(units_by_group, fn, n_boot, seed):
    """Bootstrap CI of max/min across groups, resampling identities within each group."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        vals = []
        for units in units_by_group.values():
            if not units:
                continue
            pick = [units[i] for i in rng.integers(0, len(units), len(units))]
            vals.append(fn(pick))
        vals = np.array([v for v in vals if np.isfinite(v)])
        if vals.size >= 2 and vals.min() > 0:
            ratios.append(vals.max() / vals.min())
    if len(ratios) < n_boot * 0.5:
        return [None, None]
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def analyse(units_by_group, threshold, n_boot, seed):
    pooled_imp = np.concatenate([_cat(u, "imp") for u in units_by_group.values()])
    diag = M.threshold_at_rate(pooled_imp, DIAGNOSTIC_FMR)
    stats = {g: group_stats(u, threshold, diag, n_boot, seed) for g, u in units_by_group.items() if u}
    disp = {}
    for key, fn in (("fmr", fmr_at(threshold)), ("fnmr", fnmr_at(threshold)), ("fmr_diagnostic", fmr_at(diag))):
        d = disparity(stats, key)
        d["ci"] = ratio_ci(units_by_group, fn, n_boot, seed)
        nz = d.get("nonzero_groups") or []
        d["ci_nonzero_groups"] = ratio_ci({g: units_by_group[g] for g in nz}, fn, n_boot, seed) if len(nz) >= 2 else [None, None]
        disp[key] = d
    return stats, disp, diag


def openset_by_group(race, threshold, n_boot, seed, gallery=1000):
    recs = pd.read_csv(openset.records_path("test", gallery))
    recs["group"] = recs["identity_id"].map(race).fillna("unknown")
    out = {}
    for group in GROUPS + ["unknown"]:
        g = recs[recs["group"] == group]
        if g.empty:
            continue
        units = openset.to_units(g)
        a = M.bootstrap_ci(lambda us: M.fnir_fpir_at_threshold(*openset.unit_arrays(us), threshold)[0], units, n=n_boot, seed=seed)
        b = M.bootstrap_ci(lambda us: M.fnir_fpir_at_threshold(*openset.unit_arrays(us), threshold)[1], units, n=n_boot, seed=seed)
        out[group] = {"n_identities": len(units), "n_mated": int(g["mated"].sum()), "n_nonmated": int((~g["mated"]).sum()),
                      "fnir": a[0], "fnir_ci": [a[1], a[2]], "fpir": b[0], "fpir_ci": [b[1], b[2]]}
    return out


def run(cfg):
    threshold, source = global_threshold(cfg)
    n_boot, seed = int(cfg.get("n_boot", 1000)), int(cfg["seed"])
    if rfw.available(DATA_DIR) and not cfg.get("force_fallback"):
        return run_rfw(cfg, threshold, source, n_boot, seed)
    print("RFW not available -> LFW attribute FALLBACK")

    manifest, out, emb_of, proto = openset.load_lfw_protocol(cfg)
    groups = lfw.identity_groups(lfw.load_attributes(DATA_DIR), min_margin=float(cfg.get("min_margin", 0.0)))
    test = proto["test"]
    test_ids = set(test["gallery_ids"]) | set(test["distractors"]) | set(test["probes"]["identity_id"])
    imgs_by_id = manifest.groupby("identity_id")["image_path"].apply(list).to_dict()
    rng = np.random.default_rng(seed)
    race_units = {g: units_from_embeddings(sorted(i for i in test_ids if groups["race"].get(i) == g), imgs_by_id, emb_of, rng)
                  for g in ("white", "black", "asian", "indian")}
    gender_units = {g: units_from_embeddings(sorted(i for i in test_ids if groups["gender"].get(i) == g), imgs_by_id, emb_of, rng)
                    for g in ("male", "female")}
    stats, disp, diag = analyse(race_units, threshold, n_boot, seed)
    gender_stats, gender_disp, _ = analyse(gender_units, threshold, n_boot, seed)
    open_stats = openset_by_group(groups["race"], threshold, n_boot, seed)

    result = {
        "meta": metadata(cfg, lfw.DATASET_VERSION + " + Kumar et al. LFW attributes v1.2"),
        "data": "FALLBACK: LFW attributes (RFW not available)",
        "protocol": __doc__.strip().split("\n\n", 1)[1],
        "group_rule": "race = argmax of mean Asian/White/Black/Indian attribute score per identity "
                      "(unknown if that max <= min_margin); gender = sign of mean Male score",
        "threshold": threshold, "threshold_source": source,
        "diagnostic_threshold": diag, "diagnostic_fmr_target": DIAGNOSTIC_FMR,
        "race_1to1": stats, "gender_1to1": gender_stats, "gender_disparity": gender_disp,
        "race_openset_gallery1000": open_stats, "disparity": disp,
        "plot": "eval/plots/bias_fmr_fnmr.png",
    }
    for g, s in stats.items():
        print(f"{g:7s} ids {s['n_identities']:4d} imp {s['n_impostor']:8d}  FMR {s['fmr']:.2e} (diag {s['fmr_diagnostic']:.2e})  "
              f"FNMR {s['fnmr']:.4f}  imp p99.9 {s['impostor_p999']:.3f}")
    print("disparity:", {k: (v.get("ratio"), v.get("ratio_nonzero_groups"), v.get("ci_nonzero_groups")) for k, v in disp.items()})
    plot(stats, diag)
    save_result("bias", result)


def run_rfw(cfg, threshold, source, n_boot, seed):
    """Official RFW 1:1 pairs per group: accuracy (10-fold, like LFW) and FMR/FNMR at the global threshold."""
    pairs = rfw.load_pairs(DATA_DIR)
    paths = sorted(set(pairs["path_a"]) | set(pairs["path_b"]))
    out = embed_paths(paths, cache_name("rfw"))
    emb_of = {p: e for p, e, s in zip(paths, out["emb"], out["status"]) if s == STATUS_OK}
    pairs["score"] = [float(np.dot(emb_of[a], emb_of[b])) if a in emb_of and b in emb_of else -1.0
                      for a, b in zip(pairs["path_a"], pairs["path_b"])]
    pairs["id_a"] = [p.replace("\\", "/").split("/")[-2] for p in pairs["path_a"]]
    pairs["group"] = pairs["group"].str.lower()
    units = {g: units_from_pairs(p) for g, p in pairs.groupby("group")}
    stats, disp, diag = analyse(units, threshold, n_boot, seed)
    for group, g in pairs.groupby("group"):
        folds = np.arange(len(g)) * 10 // len(g)
        s, same = g["score"].to_numpy(), g["same"].to_numpy()
        accs = []
        for k in range(10):
            t, _ = M.best_accuracy_threshold(s[(folds != k) & same], s[(folds != k) & ~same])
            accs.append(M.verification_accuracy(s[(folds == k) & same], s[(folds == k) & ~same], t))
        stats[group]["accuracy_mean"], stats[group]["accuracy_std"] = float(np.mean(accs)), float(np.std(accs))
    save_result("bias", {
        "meta": metadata(cfg, rfw.DATASET_VERSION), "data": "RFW (official pairs)",
        "threshold": threshold, "threshold_source": source, "diagnostic_threshold": diag,
        "diagnostic_fmr_target": DIAGNOSTIC_FMR, "race_1to1": stats, "disparity": disp,
        "plot": "eval/plots/bias_fmr_fnmr.png",
    })
    plot(stats, diag, suptitle="RFW official pairs (95% CI)")


def plot(stats, diag, suptitle="LFW attribute FALLBACK, test split (95% CI)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if g in stats]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    panels = (("fmr_diagnostic", f"FMR at diagnostic threshold {diag:.3f}\n(pooled FMR = 1e-4)"),
              ("fmr", "FMR at operating threshold"),
              ("fnmr", "FNMR at operating threshold"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (key, title) in zip(axes, panels):
        vals = np.array([stats[g][key] for g in groups])
        lo = np.array([stats[g][f"{key}_ci"][0] for g in groups])
        hi = np.array([stats[g][f"{key}_ci"][1] for g in groups])
        ax.bar(groups, vals, yerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)], capsize=4,
               color=colors[: len(groups)])
        upper_key = {"fmr": "fmr_upper_if_zero", "fnmr": "fnmr_upper_if_zero"}.get(key)
        for i, g in enumerate(groups):
            label = f"n={stats[g]['n_identities']}"
            if upper_key and stats[g].get(upper_key):
                label += f"\n0 errors\n(<= {stats[g][upper_key]:.1e})"
            ax.text(i, max(hi[i], vals[i]), label, ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.35)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_path("bias_fmr_fnmr.png"), dpi=130)
    plt.close(fig)
