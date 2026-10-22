"""Experiment `calibration`: map the rank-1 cosine score to P(correct mated match).

Data: the open-set search records (eval/cache/openset_records_<split>_<gallery>.csv), so run the
`openset` experiment first. Fit on the tune split, evaluate on the test split (disjoint identities).

Label: 1 when the search is mated AND the rank-1 candidate is the right identity, else 0.
Prior: the share of mated searches is fixed explicitly (default 10%) through sample weights, both
when fitting and when scoring the test split. The fitted probability is only valid at that prior;
services/calibration.py documents the Bayes re-targeting formula.
"""
import json
import os

import numpy as np
import pandas as pd

from eval import metrics as M
from eval.common import ROOT, metadata, plot_path, save_result
from eval.datasets import lfw
from eval.protocols.openset import records_path


def prior_weights(mated, prior):
    mated = np.asarray(mated, dtype=bool)
    w = np.empty(mated.size)
    n_m, n_n = mated.sum(), (~mated).sum()
    w[mated] = prior / max(n_m, 1)
    w[~mated] = (1 - prior) / max(n_n, 1)
    return w * mated.size  # mean weight 1


def load_split(split, gallery):
    df = pd.read_csv(records_path(split, gallery))
    df = df[np.isfinite(df["top_score"])]  # failed templates never reach the officer
    y = (df["mated"] & (df["rank"] == 1)).to_numpy(int)
    return df["top_score"].to_numpy(float), y, df["mated"].to_numpy(bool)


def fit_platt(s, y, w):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(s.reshape(-1, 1), y, sample_weight=w)
    return {"type": "platt", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def fit_isotonic(s, y, w):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(s, y, sample_weight=w)
    return {"type": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}, iso


def run(cfg):
    from services.calibration import Calibrator

    prior = float(cfg.get("prior", 0.1))
    gallery = cfg.get("gallery_size", 1000)
    n_bins = int(cfg.get("n_bins", 15))
    s_tr, y_tr, m_tr = load_split("tune", gallery)
    s_te, y_te, m_te = load_split("test", gallery)
    w_tr, w_te = prior_weights(m_tr, prior), prior_weights(m_te, prior)

    lo, hi = float(s_tr.min()), float(s_tr.max())
    raw_te = np.clip((s_te - lo) / (hi - lo), 0, 1)
    platt = fit_platt(s_tr, y_tr, w_tr)
    iso_spec, _ = fit_isotonic(s_tr, y_tr, w_tr)
    preds = {
        "raw_minmax": raw_te,
        "platt": Calibrator({**platt, "prior": prior}).predict_many(s_te),
        "isotonic": Calibrator({**iso_spec, "prior": prior}).predict_many(s_te),
    }
    scores = {k: {"ece": M.ece(p, y_te, n_bins, w_te), "brier": M.brier(p, y_te, w_te)} for k, p in preds.items()}
    for k, v in scores.items():
        print(f"{k:10s} ECE {v['ece']:.4f}  Brier {v['brier']:.4f}")

    # Prefer the 2-parameter Platt curve (smooth, hard to overfit) unless isotonic is clearly better;
    # isotonic fits with few positives become coarse step functions that jump in the officer UI.
    margin = float(cfg.get("isotonic_ece_margin", 0.005))
    chosen = "isotonic" if scores["isotonic"]["ece"] + margin < scores["platt"]["ece"] else "platt"
    spec = {"platt": platt, "isotonic": iso_spec}[chosen]
    meta = metadata(cfg, lfw.DATASET_VERSION)
    export = {**spec, "prior": prior, "input": "rank-1 cosine similarity (L2-normalised ArcFace embeddings)",
              "output": "estimated P(correct mated match) at the stated prior",
              "fitted_on": f"LFW open-set tune split, gallery {gallery}",
              "test_metrics": scores[chosen], "model": meta["model"], "git_commit": meta["git_commit"],
              "timestamp": meta["timestamp"]}
    out_path = os.path.join(ROOT, cfg.get("export_path", "models/calibrator.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
        f.write("\n")
    print(f"Chosen calibrator: {chosen} -> {os.path.relpath(out_path, ROOT)}")

    plot_reliability(preds, y_te, w_te, n_bins, scores, prior)
    p_ch = preds[chosen]
    mid = (p_ch > 0.05) & (p_ch < 0.95)
    middle = {"share_of_searches": float(w_te[mid].sum() / w_te.sum()),
              "mean_predicted": float(np.average(p_ch[mid], weights=w_te[mid])) if mid.any() else None,
              "observed_correct": float(np.average(y_te[mid], weights=w_te[mid])) if mid.any() else None,
              "n": int(mid.sum())}
    examples = {f"{s:.2f}": float(Calibrator({**spec, "prior": prior}).predict(s)) for s in (0.3, 0.4, 0.5, 0.6, 0.7)}
    save_result("calibration", {
        "meta": meta, "prior": prior, "gallery_size": gallery,
        "n_tune": int(s_tr.size), "n_test": int(s_te.size),
        "n_test_mated": int(m_te.sum()), "n_test_nonmated": int((~m_te).sum()),
        "metrics_test": scores, "chosen": chosen,
        "selection_rule": f"platt unless isotonic ECE is lower by more than {margin}", "export": os.path.relpath(out_path, ROOT).replace("\\", "/"),
        "example_predictions": examples,
        "middle_range_0.05_0.95": middle,
        "bayes_adjustment": "odds' = odds * (pi'/(1-pi')) / (pi/(1-pi)), with pi the fitting prior",
        "plot": "eval/plots/calibration_reliability.png",
    })


def plot_reliability(preds, y, w, n_bins, scores, prior):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.6, 6.4), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    total = float(np.sum(w))
    for name, p in preds.items():
        bins = M.reliability_bins(p, y, n_bins, w)
        sizes = [max(8.0, 400.0 * b[2] / total) for b in bins]
        line = ax.plot([b[0] for b in bins], [b[1] for b in bins], lw=1, alpha=0.7)[0]
        ax.scatter([b[0] for b in bins], [b[1] for b in bins], s=sizes, color=line.get_color(), alpha=0.8,
                   label=f"{name} (ECE {scores[name]['ece']:.3f})")
        axh.hist(p, bins=np.linspace(0, 1, n_bins + 1), weights=w, histtype="step", lw=1.5, color=line.get_color())
    ax.set_ylabel("Observed fraction correct")
    ax.set_title(f"Reliability, LFW test split, prior {prior:.0%} mated (marker area = share of searches)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    axh.set_yscale("log")
    axh.set_xlabel("Predicted probability")
    axh.set_ylabel("weighted count")
    axh.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path("calibration_reliability.png"), dpi=130)
    plt.close(fig)
