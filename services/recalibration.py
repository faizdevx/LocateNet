"""Calibrator re-fit proposals from reviewed field alerts (never applied automatically).

A proposal is a JSON file in models/proposals/. An Admin approves or rejects it on
/admin/model-health; approving swaps models/calibrator.json atomically and keeps the previous
file for rollback.

Selection bias: only faces above MATCH_THRESHOLD become alerts and get reviewed, so the field
data says nothing about scores below the threshold and nothing about misses. The re-fitted curve
is P(confirmed | alert, score) at the field's own mix of true and false alerts; it is documented as
such in the proposal ("basis").
"""
import json
import os
from datetime import datetime

import numpy as np

MIN_REVIEWED = 200
MIN_PER_CLASS = 30


def fit_platt(scores, labels, iters=100):
    """Logistic regression p = 1/(1+exp(-(a*s+b))) by Newton-Raphson (IRLS), tiny ridge for stability."""
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    X = np.column_stack([s, np.ones_like(s)])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p) - 1e-6 * w)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"type": "platt", "a": float(w[0]), "b": float(w[1])}


def _predict(spec, s):
    s = np.asarray(s, dtype="float64")
    if spec["type"] == "platt":
        return 1.0 / (1.0 + np.exp(-(spec["a"] * s + spec["b"])))
    return np.interp(s, spec["x"], spec["y"])


def _metrics(p, y, n_bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(n_bins) if (idx == b).any())
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}


def check_minimums(labels, min_reviewed=MIN_REVIEWED, min_per_class=MIN_PER_CLASS):
    y = np.asarray(labels)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if len(y) < min_reviewed:
        return False, f"only {len(y)} reviewed alerts (need {min_reviewed})"
    if n_pos < min_per_class or n_neg < min_per_class:
        return False, f"confirmed={n_pos}, false_positive={n_neg} (need at least {min_per_class} of each)"
    return True, None


def build_proposal(scores, labels, current_spec, seed=0, min_reviewed=MIN_REVIEWED, min_per_class=MIN_PER_CLASS):
    """Return (proposal dict, None) or (None, reason)."""
    ok, reason = check_minimums(labels, min_reviewed, min_per_class)
    if not ok:
        return None, reason
    s, y = np.asarray(scores, dtype="float64"), np.asarray(labels, dtype="float64")
    new = fit_platt(s, y)
    # 5-fold cross-validated estimate for the new curve (the old curve was not fitted on this data)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(s)) % 5
    cv_pred = np.empty(len(s))
    for k in range(5):
        spec = fit_platt(s[folds != k], y[folds != k])
        cv_pred[folds == k] = _predict(spec, s[folds == k])
    prior = float(y.mean())
    return {
        "status": "pending", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_reviewed": int(len(y)), "n_confirmed": int(y.sum()), "n_false_positive": int(len(y) - y.sum()),
        "calibrator": {**new, "prior": prior, "basis": "field alerts reviewed by officers (selection-biased: "
                       "only scores above MATCH_THRESHOLD), P(confirmed | alert, score)",
                       "fitted_on": f"{len(y)} reviewed alerts", "created_at": datetime.now().isoformat(timespec="seconds")},
        "metrics": {"current_on_field_data": _metrics(_predict(current_spec, s), y) if current_spec else None,
                    "proposed_cross_validated": _metrics(cv_pred, y)},
        "current": {k: current_spec.get(k) for k in ("type", "a", "b", "prior") if current_spec and k in current_spec},
    }, None


def proposal_dir(base_dir):
    return os.path.join(base_dir, "models", "proposals")


def write_proposal(base_dir, proposal):
    os.makedirs(proposal_dir(base_dir), exist_ok=True)
    stem = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path, n = os.path.join(proposal_dir(base_dir), stem + ".json"), 1
    while os.path.exists(path):  # never overwrite an earlier proposal
        n += 1
        path = os.path.join(proposal_dir(base_dir), f"{stem}_{n}.json")
    with open(path, "w") as f:
        json.dump(proposal, f, indent=2)
    return path


def list_proposals(base_dir):
    d = proposal_dir(base_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d), reverse=True):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as f:
                out.append({"name": name, **json.load(f)})
    return out


def decide(base_dir, name, approve, admin, calibrator_path):
    """Approve (atomic swap, keep previous for rollback) or reject a pending proposal."""
    if os.path.basename(name) != name or not name.endswith(".json"):
        raise ValueError("bad proposal name")
    path = os.path.join(proposal_dir(base_dir), name)
    with open(path) as f:
        proposal = json.load(f)
    if proposal.get("status") != "pending":
        raise ValueError("proposal is not pending")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if approve:
        if os.path.exists(calibrator_path):
            backup = calibrator_path.replace(".json", f".previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(calibrator_path) as src, open(backup, "w") as dst:
                dst.write(src.read())
            proposal["previous_calibrator_backup"] = os.path.basename(backup)
        tmp = calibrator_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(proposal["calibrator"], f, indent=2)
        os.replace(tmp, calibrator_path)
    proposal.update({"status": "approved" if approve else "rejected", "decided_by": admin, "decided_at": now})
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(proposal, f, indent=2)
    os.replace(tmp, path)
    return proposal
