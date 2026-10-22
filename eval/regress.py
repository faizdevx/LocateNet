"""Regression gate: `python -m eval regress [--smoke]` (also `make eval-regress`).

Full mode re-runs the Week 2 headline suite (LFW verification, LFW open-set, bias audit) into a
scratch results folder with the CURRENT code and config, then compares three headline metrics
with eval/results/baseline/headline.json. A metric fails when it is worse than the baseline by
more than the baseline's 95% CI half-width.

Smoke mode (used in CI) runs the stub-model smoke experiment and compares its deterministic
metrics with eval/results/baseline/smoke.json (tolerance 1e-6), so wiring changes that alter
results are caught without downloading any model.
"""
import json
import os
import subprocess
import sys
import tempfile

from eval.common import RESULTS_DIR, ROOT

BASELINE_DIR = os.path.join(ROOT, "eval", "results", "baseline")
HEADLINE = os.path.join(BASELINE_DIR, "headline.json")
SMOKE = os.path.join(BASELINE_DIR, "smoke.json")


def _load(path):
    with open(path) as f:
        return json.load(f)


def headline_metrics(results_dir):
    """Extract the three gated metrics and their CI half-widths from a results folder."""
    v = _load(os.path.join(results_dir, "verification_lfw.json"))
    o = _load(os.path.join(results_dir, "openset_lfw.json"))
    b = _load(os.path.join(results_dir, "bias.json"))
    f = o["splits"]["test"]["by_gallery_size"]["1000"]["fnir_at_fpir"]["0.01"]
    d = b["disparity"]["fmr_diagnostic"]
    ratio = d.get("ratio") or d.get("ratio_nonzero_groups")
    rci = d["ci"] if d.get("ratio") else d.get("ci_nonzero_groups")
    return {
        "lfw_accuracy_enrolled": {"value": v["accuracy_enrolled_pairs_mean"], "higher_is_better": True,
                                  "half_width": 1.96 * v["accuracy_enrolled_pairs_std"] / len(v["fold_accuracy"]) ** 0.5},
        "openset_fnir_fpir1_gallery1000": {"value": f["fnir"], "higher_is_better": False,
                                           "half_width": (f["fnir_ci"][1] - f["fnir_ci"][0]) / 2},
        "bias_fmr_ratio_diagnostic": {"value": ratio, "higher_is_better": False,
                                      "half_width": (rci[1] - rci[0]) / 2 if rci and rci[0] is not None else None},
    }


def compare(base, new):
    rows, failed = [], False
    for name, b in base.items():
        n = new[name]["value"]
        hw = b["half_width"] or 0.0
        worse = (b["value"] - n) if b["higher_is_better"] else (n - b["value"])
        bad = n is None or worse > hw + 1e-12
        failed |= bad
        rows.append((name, b["value"], n, hw, "FAIL" if bad else "ok"))
    return rows, failed


def print_rows(rows):
    print(f"{'metric':34s} {'baseline':>10s} {'now':>10s} {'tolerance':>10s}  status")
    for name, b, n, hw, st in rows:
        print(f"{name:34s} {b:10.5f} {n if n is not None else float('nan'):10.5f} {hw:10.5f}  {st}")


def write_baseline():
    """Snapshot the committed results as the new baseline (run deliberately, never automatically)."""
    os.makedirs(BASELINE_DIR, exist_ok=True)
    with open(HEADLINE, "w") as f:
        json.dump(headline_metrics(RESULTS_DIR), f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(HEADLINE, ROOT)}")


def run_full():
    scratch = tempfile.mkdtemp(prefix="locatenet-regress-")
    env = dict(os.environ, LOCATENET_EVAL_RESULTS=scratch, LOCATENET_EVAL_PLOTS=os.path.join(scratch, "plots"),
               PYTHONWARNINGS="ignore")
    for exp in ("verification", "openset", "operating_point", "bias"):
        print(f"[regress] running {exp}")
        # operating_point is needed because bias reads the threshold from it
        subprocess.run([sys.executable, "-m", "eval", "run", f"eval/config/{exp}.yaml"],
                       cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    rows, failed = compare(_load(HEADLINE), headline_metrics(scratch))
    print_rows(rows)
    return failed


def run_smoke():
    out = os.path.join(tempfile.mkdtemp(prefix="locatenet-smoke-"), "smoke.json")
    subprocess.run([sys.executable, "-m", "eval", "run", "eval/config/smoke.yaml"],
                   cwd=ROOT, check=True, env=dict(os.environ, LOCATENET_SMOKE_METRICS=out))
    now, base = _load(out), _load(SMOKE)
    failed = False
    for k, v in base.items():
        ok = abs(now.get(k, float("nan")) - v) <= 1e-6
        failed |= not ok
        print(f"{k:28s} baseline {v:.6f} now {now.get(k, float('nan')):.6f}  {'ok' if ok else 'FAIL'}")
    return failed


def main(smoke=False, update_baseline=False):
    if update_baseline:
        write_baseline()
        return 0
    failed = run_smoke() if smoke else run_full()
    print("REGRESSION GATE:", "FAILED" if failed else "passed")
    return 1 if failed else 0
