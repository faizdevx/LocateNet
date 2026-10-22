"""Field metrics from officer reviews (pure functions; the admin page feeds them database rows).

Every alert-producing face that an officer reviewed is a labelled example: confirm_match = true
match, false_positive = wrong person. `ignore` is not counted either way. Rates are only shown
with Wilson 95% intervals and only when n >= MIN_N; below that the page says "insufficient data".

Selection bias: only faces above the threshold are alerted and reviewed, so misses never appear
here. The random QA sample of below-threshold faces (AUDIT_SAMPLE_RATE) estimates the miss rate.
"""
import math
from collections import defaultdict
from datetime import datetime

MIN_N = 30
Z = 1.959963984540054


def wilson_ci(k, n, z=Z):
    """Wilson score interval for k successes in n trials; (nan, nan) for n = 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k, n, min_n=MIN_N):
    """{'n','k','rate','ci','enough'}; rate and ci are None when n < min_n (never shown from tiny n)."""
    enough = n >= min_n
    return {"n": int(n), "k": int(k), "enough": enough,
            "rate": (k / n) if enough else None, "ci": list(wilson_ci(k, n)) if enough else None}


def _label(row):
    d = row.get("review_decision")
    return 1 if d == "confirm_match" else 0 if d == "false_positive" else None


def _week(ts):
    dt = datetime.strptime(str(ts)[:10], "%Y-%m-%d")
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def precision_by(rows, key_fn, min_n=MIN_N):
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        y = _label(r)
        if y is None:
            continue
        k = key_fn(r)
        if k is None:
            continue
        groups[k][0] += y
        groups[k][1] += 1
    return {k: rate(v[0], v[1], min_n) for k, v in sorted(groups.items())}


def weekly_precision(rows, min_n=MIN_N):
    return precision_by(rows, lambda r: _week(r["created_at"]) if r.get("created_at") else None, min_n)


def similarity_bucket(sim, edges):
    if sim is None:
        return None
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= sim < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def calibration_by_bucket(rows, edges, min_n=MIN_N):
    """Per similarity bucket: observed confirm rate (with CI) vs mean predicted likelihood."""
    out = {}
    buckets = defaultdict(list)
    for r in rows:
        y = _label(r)
        b = similarity_bucket(r.get("similarity"), edges)
        if y is not None and b is not None:
            buckets[b].append((y, r.get("calibrated_likelihood")))
    for b, items in sorted(buckets.items()):
        preds = [p for _, p in items if p is not None]
        out[b] = {**rate(sum(y for y, _ in items), len(items), min_n),
                  "mean_predicted": (sum(preds) / len(preds)) if preds and len(items) >= min_n else None}
    return out


def face_size_bucket(px):
    if px is None:
        return None
    return "< 32 px" if px < 32 else "32-63 px" if px < 64 else "64-111 px" if px < 112 else ">= 112 px"


def review_latency_hours(rows):
    """Hours from alert creation to review, for reviewed rows; returns (median, p90, n)."""
    vals = []
    for r in rows:
        if r.get("review_timestamp") and r.get("created_at"):
            a = datetime.strptime(str(r["created_at"])[:19], "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(str(r["review_timestamp"])[:19], "%Y-%m-%d %H:%M:%S")
            vals.append((b - a).total_seconds() / 3600.0)
    if not vals:
        return None, None, 0
    vals.sort()

    def q(p):
        return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]
    return q(0.5), q(0.9), len(vals)
