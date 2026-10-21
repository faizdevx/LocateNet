"""Experiment `verification`: LFW 1:1 standard protocol (View 2, 10-fold pairs.txt).

For each fold the accuracy threshold is chosen on the other 9 folds (unrestricted, labeled outside
data protocol). A pair whose image failed to enrol gets score -1 (always rejected), so failures
lower the accuracy instead of being silently dropped.
"""
import numpy as np

from eval import metrics as M
from eval.common import DATA_DIR, metadata, plot_path, save_result
from eval.datasets import lfw
from eval.embed import STATUS_OK, cache_name, embed_paths


def pair_scores(pairs, emb_lookup):
    scores = []
    for a, b in zip(pairs["path_a"], pairs["path_b"]):
        ea, eb = emb_lookup.get(a), emb_lookup.get(b)
        scores.append(float(np.dot(ea, eb)) if ea is not None and eb is not None else -1.0)
    return np.array(scores)


def run(cfg):
    pairs = lfw.load_pairs(DATA_DIR)
    paths = sorted(set(pairs["path_a"]) | set(pairs["path_b"]))
    out = embed_paths(paths, cache_name("lfw"))
    lookup = {p: e for p, e, s in zip(paths, out["emb"], out["status"]) if s == STATUS_OK}
    failed = int(sum(1 for s in out["status"] if s != STATUS_OK))

    scores = pair_scores(pairs, lookup)
    same = pairs["same"].to_numpy()
    folds = pairs["fold"].to_numpy()

    fold_acc, fold_thr = [], []
    for k in range(int(folds.max()) + 1):
        train, test = folds != k, folds == k
        t, _ = M.best_accuracy_threshold(scores[train & same], scores[train & ~same])
        fold_thr.append(t)
        fold_acc.append(M.verification_accuracy(scores[test & same], scores[test & ~same], t))

    # Published LFW numbers use pre-aligned crops, where enrolment cannot fail. For a like-for-like
    # comparison we also report accuracy on the pairs whose two images both enrolled.
    enrolled = np.array([a in lookup and b in lookup for a, b in zip(pairs["path_a"], pairs["path_b"])])
    enr_acc = []
    for k in range(int(folds.max()) + 1):
        train, test = (folds != k) & enrolled, (folds == k) & enrolled
        t, _ = M.best_accuracy_threshold(scores[train & same], scores[train & ~same])
        enr_acc.append(M.verification_accuracy(scores[test & same], scores[test & ~same], t))

    gen, imp = scores[same], scores[~same]
    far_targets = [float(f) for f in cfg.get("far_targets", [1e-3, 1e-2])]
    tar = {str(f): M.tar_at_far(gen, imp, f) for f in far_targets}

    far, tpr, _ = M.roc(gen, imp)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    floor = 1.0 / max(imp.size, 1)  # smallest non-zero FAR measurable with this many impostor pairs
    ax.step(np.clip(far, floor, 1), tpr, where="post", lw=2, label="all pairs (failed enrolment = reject)")
    far_e, tpr_e, _ = M.roc(scores[same & enrolled], scores[~same & enrolled])
    ax.step(np.clip(far_e, floor, 1), tpr_e, where="post", lw=2, ls="--", label="pairs where both images enrolled")
    ax.set_xscale("log")
    ax.set_xlim(floor, 1)
    ax.set_ylim(0.97, 1.0)
    ax.set_xlabel("False accept rate (log)")
    ax.set_ylabel("True accept rate")
    ax.set_title(f"LFW 1:1 ROC (buffalo_l): acc {np.mean(enr_acc):.4f} enrolled / {np.mean(fold_acc):.4f} all")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_path("lfw_roc.png"), dpi=130)
    plt.close(fig)

    result = {
        "meta": metadata(cfg, lfw.DATASET_VERSION),
        "protocol": "LFW View 2, 10 folds x 600 pairs; threshold chosen on 9 folds, tested on the 10th",
        "n_pairs": int(len(pairs)),
        "n_pair_images_failed_enrol": failed,
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "fold_accuracy": fold_acc,
        "fold_threshold": fold_thr,
        "n_pairs_with_failed_image": int((~enrolled).sum()),
        "accuracy_enrolled_pairs_mean": float(np.mean(enr_acc)),
        "accuracy_enrolled_pairs_std": float(np.std(enr_acc)),
        "note_accuracy": "accuracy_mean counts pairs with a failed enrolment as rejected (end-to-end, production "
                         "detector thresholds); accuracy_enrolled_pairs_* excludes them and is comparable to "
                         "published LFW numbers, which use pre-aligned crops.",
        "tar_at_far": {k: {"tar": v[0], "threshold": v[1]} for k, v in tar.items()},
        "note_far": "Only 3000 impostor pairs: FAR=1e-3 rests on 3 impostor scores, so TAR@1e-3 is noisy.",
        "plot": "eval/plots/lfw_roc.png",
    }
    print(f"LFW accuracy end-to-end {result['accuracy_mean']:.4f} +/- {result['accuracy_std']:.4f}; enrolled pairs only "
          f"{result['accuracy_enrolled_pairs_mean']:.4f} +/- {result['accuracy_enrolled_pairs_std']:.4f} "
          f"({result['n_pairs_with_failed_image']} pairs with a failed image); "
          f"TAR@FAR=1e-3 {tar[str(1e-3)][0]:.4f} (thr {tar[str(1e-3)][1]:.3f})")
    save_result("verification_lfw", result)
