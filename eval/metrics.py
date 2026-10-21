"""Pure metric functions. Convention everywhere: a comparison is accepted when score >= threshold."""
import numpy as np


def _as_array(x):
    return np.asarray(x, dtype="float64").ravel()


def threshold_at_rate(negative_scores, rate):
    """
    Smallest threshold t such that the fraction of negative scores with score >= t is <= rate.
    Used for FAR (1:1) and FPIR (1:N). With n negatives, rates below 1/n are capped at 0 errors.
    """
    neg = np.sort(_as_array(negative_scores))[::-1]
    if neg.size == 0:
        return -np.inf
    k = int(np.floor(rate * neg.size))
    if k >= neg.size:
        return -np.inf
    # Accept only scores strictly above the (k+1)-th highest negative.
    return float(np.nextafter(neg[k], np.inf))


def roc(genuine, impostor):
    """Return (far, tar, thresholds) over every distinct score, sorted by decreasing threshold."""
    gen, imp = _as_array(genuine), _as_array(impostor)
    thresholds = np.unique(np.concatenate([gen, imp]))[::-1]
    gen_sorted, imp_sorted = np.sort(gen), np.sort(imp)
    tar = 1.0 - np.searchsorted(gen_sorted, thresholds, side="left") / max(gen.size, 1)
    far = 1.0 - np.searchsorted(imp_sorted, thresholds, side="left") / max(imp.size, 1)
    return far, tar, thresholds


def tar_at_far(genuine, impostor, far):
    """True accept rate at the threshold that gives FAR <= far. Returns (tar, threshold)."""
    t = threshold_at_rate(impostor, far)
    gen = _as_array(genuine)
    return float(np.mean(gen >= t)) if gen.size else float("nan"), t


def verification_accuracy(genuine, impostor, threshold):
    gen, imp = _as_array(genuine), _as_array(impostor)
    correct = np.sum(gen >= threshold) + np.sum(imp < threshold)
    return float(correct / (gen.size + imp.size))


def best_accuracy_threshold(genuine, impostor):
    """Threshold maximising verification accuracy (used on the 9 training folds of LFW)."""
    gen, imp = _as_array(genuine), _as_array(impostor)
    candidates = np.unique(np.concatenate([gen, imp]))
    gen_sorted, imp_sorted = np.sort(gen), np.sort(imp)
    accepted_gen = gen.size - np.searchsorted(gen_sorted, candidates, side="left")
    rejected_imp = np.searchsorted(imp_sorted, candidates, side="left")
    acc = (accepted_gen + rejected_imp) / (gen.size + imp.size)
    i = int(np.argmax(acc))
    best_t, best_acc = candidates[i], acc[i]
    return float(best_t), float(best_acc)


def fnir_at_fpir(mated_top_scores, mated_rank1_correct, nonmated_top_scores, fpir):
    """
    NIST FRVT 1:N definitions (rank-1, threshold T):
      FPIR(T) = fraction of non-mated searches whose top candidate scores >= T
      FNIR(T) = fraction of mated searches where the mate is NOT returned at rank 1 with score >= T
    Mated searches that failed to produce a template should be passed with score -inf / correct False,
    so they count as misses. Returns (fnir, threshold).
    """
    t = threshold_at_rate(nonmated_top_scores, fpir)
    scores = _as_array(mated_top_scores)
    correct = np.asarray(mated_rank1_correct, dtype=bool).ravel()
    if scores.size == 0:
        return float("nan"), t
    hit = correct & (scores >= t)
    return float(1.0 - hit.mean()), t


def fnir_fpir_at_threshold(mated_top_scores, mated_rank1_correct, nonmated_top_scores, threshold):
    scores = _as_array(mated_top_scores)
    correct = np.asarray(mated_rank1_correct, dtype=bool).ravel()
    non = _as_array(nonmated_top_scores)
    fnir = float(1.0 - np.mean(correct & (scores >= threshold))) if scores.size else float("nan")
    fpir = float(np.mean(non >= threshold)) if non.size else float("nan")
    return fnir, fpir


def det_curve(mated_top_scores, mated_rank1_correct, nonmated_top_scores):
    """FNIR vs FPIR over all thresholds (finite scores only). Returns (fpir, fnir, thresholds)."""
    scores = _as_array(mated_top_scores)
    correct = np.asarray(mated_rank1_correct, dtype=bool).ravel()
    non = _as_array(nonmated_top_scores)
    finite = np.concatenate([scores[np.isfinite(scores)], non[np.isfinite(non)]])
    thresholds = np.unique(finite)[::-1]
    hit_scores = np.sort(scores[correct & np.isfinite(scores)])
    non_sorted = np.sort(non)
    hits = hit_scores.size - np.searchsorted(hit_scores, thresholds, side="left")
    fnir = 1.0 - hits / max(scores.size, 1)
    fpir = 1.0 - np.searchsorted(non_sorted, thresholds, side="left") / max(non.size, 1)
    return fpir, fnir, thresholds


def _weights(w, n):
    return np.ones(n) if w is None else _as_array(w)


def ece(probs, labels, n_bins=15, weights=None):
    """Expected calibration error with equal-width bins on [0, 1]. Optional sample weights
    (used to evaluate at a fixed mated/non-mated prior)."""
    p, y = _as_array(probs), _as_array(labels)
    if p.size == 0:
        return float("nan")
    w = _weights(weights, p.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any() and w[mask].sum() > 0:
            wb = w[mask]
            acc = np.average(y[mask], weights=wb)
            conf = np.average(p[mask], weights=wb)
            total += wb.sum() / w.sum() * abs(acc - conf)
    return float(total)


def reliability_bins(probs, labels, n_bins=15, weights=None):
    """(mean confidence, observed accuracy, total weight) per non-empty bin, for reliability diagrams."""
    p, y = _as_array(probs), _as_array(labels)
    w = _weights(weights, p.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any() and w[mask].sum() > 0:
            out.append((float(np.average(p[mask], weights=w[mask])),
                        float(np.average(y[mask], weights=w[mask])), float(w[mask].sum())))
    return out


def brier(probs, labels, weights=None):
    p, y = _as_array(probs), _as_array(labels)
    if p.size == 0:
        return float("nan")
    return float(np.average((p - y) ** 2, weights=_weights(weights, p.size)))


def bootstrap_ci(fn, units, n=1000, seed=0, alpha=0.05):
    """
    Percentile bootstrap that resamples *identities*. `units` is a sequence with one entry per
    identity (for example the list of that identity's comparison records); `fn(list_of_units)`
    returns a scalar. Returns (point_estimate, low, high).
    """
    units = list(units)
    point = fn(units)
    if not units:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(units), size=len(units))
        value = fn([units[i] for i in pick])
        if value is not None and np.isfinite(value):
            stats.append(value)
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
