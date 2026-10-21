"""Experiment `quality`: is det_score a good quality gate, and what should DET_SCORE_MIN be?

Data: the per-probe records of the robustness experiment (original + degraded LFW test probes),
so the gate sees a realistic spread of quality. Every probe with a detected face is included.

Error-versus-reject curve: for each cut-off, probes below it are rejected (production: no alert);
we plot FNIR on the RETAINED mated probes against the fraction of probes rejected. A good quality
score removes the probes that would have been misses first, so the curve drops quickly.

Combined score: average percentile rank of det_score, log face width and log Laplacian variance
(equal weights, nothing fitted, so no tuning on this data).
"""
import os

import numpy as np
import pandas as pd

from eval.common import CACHE_DIR, load_result, metadata, plot_path, save_result
from eval.datasets import lfw


def erc(df, quality, threshold, fractions):
    """FNIR on retained mated probes at each rejection fraction (lowest quality rejected first)."""
    mated = df[df["mated"]].copy()
    mated["q"] = quality[df["mated"].to_numpy()]
    order = np.argsort(mated["q"].to_numpy(), kind="stable")
    hit = ((mated["rank"] == 1) & (mated["top_score"] >= threshold)).to_numpy()[order]
    n = hit.size
    out = []
    for f in fractions:
        k = int(round(f * n))
        kept = hit[k:]
        out.append(float(1 - kept.mean()) if kept.size else float("nan"))
    return out


def run(cfg):
    threshold = float(cfg.get("threshold") or load_result("operating_point")["recommended"]["MATCH_THRESHOLD"])
    df = pd.read_csv(os.path.join(CACHE_DIR, "robustness_records.csv"))
    df = df[df["status"] != "no_face"].reset_index(drop=True)
    df["top_score"] = df["top_score"].astype(float)

    def pct(x):
        return pd.Series(x).rank(pct=True).to_numpy()
    det = df["det_score"].to_numpy(float)
    width = np.log(np.clip(df["face_width"].to_numpy(float), 1, None))
    blur = np.log1p(np.nan_to_num(df["blur_var"].to_numpy(float), nan=0.0))
    combined = (pct(det) + pct(width) + pct(blur)) / 3.0

    fractions = np.round(np.linspace(0, 0.5, 26), 3).tolist()
    curves = {"det_score": erc(df, det, threshold, fractions), "combined": erc(df, combined, threshold, fractions)}
    auc = {k: float(np.trapz(v, fractions) / (fractions[-1] - fractions[0])) for k, v in curves.items()}

    cutoffs = np.round(np.arange(0.3, 0.901, 0.05), 2).tolist()
    sweep = []
    undegraded = (df["kind"] == "blur") & (df["level"] == 0)
    for c in cutoffs:
        kept = df["det_score"] >= c
        m = df["mated"]
        hits = (df["rank"] == 1) & (df["top_score"] >= threshold) & kept
        false_alert = (~m) & kept & (df["top_score"] >= threshold)
        sweep.append({"cutoff": c, "rejected_fraction": float(1 - kept.mean()),
                      "rejected_fraction_undegraded": float(1 - kept[undegraded].mean()),
                      "fnir_retained": float(1 - hits[m & kept].sum() / max((m & kept).sum(), 1)),
                      "fnir_overall": float(1 - hits[m].sum() / max(m.sum(), 1)),
                      "fpir_overall": float(false_alert[~m].sum() / max((~m).sum(), 1))})
    rec = recommend(sweep, cfg)
    plot(fractions, curves, auc, sweep)
    save_result("quality", {
        "meta": metadata(cfg, lfw.DATASET_VERSION), "threshold": threshold, "n_probes": int(len(df)),
        "n_mated": int(df["mated"].sum()), "protocol": __doc__.strip().split("\n\n", 1)[1],
        "erc": {"rejected_fraction": fractions, **curves}, "erc_mean_fnir": auc,
        "combined_beats_det_score": bool(auc["combined"] < auc["det_score"]),
        "det_score_sweep": sweep, "recommended_DET_SCORE_MIN": rec,
        "note": "Cut-offs at or below the detector threshold (det_thresh 0.5) reject nothing, because the detector "
                "never returns weaker faces. On LFW the gate barely changes FPIR; its intended benefit, suppressing "
                "false detections (non-faces) in crowd photos, is not measured by this LFW-based experiment.",
        "plot": "eval/plots/quality_erc.png",
    })
    print(f"ERC mean FNIR: det_score {auc['det_score']:.4f}, combined {auc['combined']:.4f}; "
          f"recommended DET_SCORE_MIN {rec['value']}")


def recommend(sweep, cfg):
    """Highest cut-off that rejects at most `max_reject_undegraded` of undegraded probes and does
    not raise overall FNIR by more than `max_fnir_increase` over the lowest cut-off."""
    max_rej = float(cfg.get("max_reject_undegraded", 0.01))
    max_inc = float(cfg.get("max_fnir_increase", 0.01))
    base = sweep[0]["fnir_overall"]
    ok = [s for s in sweep if s["rejected_fraction_undegraded"] <= max_rej and s["fnir_overall"] <= base + max_inc]
    best = max(ok, key=lambda s: s["cutoff"]) if ok else sweep[0]
    return {"value": best["cutoff"], "rule": f"highest cut-off rejecting <= {max_rej:.0%} of undegraded probes "
            f"and raising overall FNIR by <= {max_inc:.0%} (absolute)", "at_value": best}


def plot(fractions, curves, auc, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for name, ys in curves.items():
        axes[0].plot(fractions, ys, marker=".", label=f"{name} (mean FNIR {auc[name]:.3f})")
    axes[0].set_xlabel("fraction of probes rejected")
    axes[0].set_ylabel("FNIR on retained mated probes")
    axes[0].set_title("Error-versus-reject")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    c = [s["cutoff"] for s in sweep]
    axes[1].plot(c, [s["fnir_overall"] for s in sweep], marker="o", label="FNIR (rejects count as misses)")
    axes[1].plot(c, [s["rejected_fraction"] for s in sweep], marker="s", label="rejected (all probes)")
    axes[1].plot(c, [s["fpir_overall"] for s in sweep], marker="^", label="FPIR")
    axes[1].set_xlabel("DET_SCORE_MIN cut-off")
    axes[1].set_title("det_score sweep")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path("quality_erc.png"), dpi=130)
    plt.close(fig)


