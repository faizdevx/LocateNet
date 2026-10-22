"""Experiment `neg_suppression`: offline simulation of false-positive suppression.

Setup: the Week 2 LFW open-set protocol at gallery 1000, searched at the production
MATCH_THRESHOLD. Every probe of the split is replayed as a stream in a random order (20 seeded
orders, results averaged). Each alert is "reviewed" with the ground truth: a non-mated alert is a
false positive and its embedding becomes a negative exemplar of the case it hit (what an officer's
false_positive click does). Before raising each new alert we apply
services.feedback.suppression_decision with that case's exemplars so far.
Benefit: share of false-positive alerts that are downgraded (repeat look-alikes).
Cost: share of true alerts (mated, correct case) that are downgraded.
NEG_SUPPRESS_THRESHOLD is chosen on the tune split as the strictest value with the largest benefit at zero
cost; the test split reports both with Wilson 95% CIs. Ship only if the test cost is not larger
than the CI half-width of the deployed FNIR (a downgraded true alert still reaches the officer,
so this is a conservative bound).
"""
import shutil
import tempfile

import numpy as np

import config
from eval.common import load_result, metadata, save_result
from eval.datasets import lfw
from eval.protocols import openset
from services.feedback import suppression_decision
from services.field_metrics import wilson_ci


def alerts_for_split(proto_split, emb_of, threshold, seed):
    from vector_db.search_service import SearchService
    gallery = openset.gallery_for_size(proto_split, 1000, seed=seed)
    ids = {n: k + 1 for k, n in enumerate(sorted(gallery))}
    names = {v: k for k, v in ids.items()}
    tmp = tempfile.mkdtemp(prefix="locatenet-neg-")
    try:
        svc = SearchService(index_path=tmp)
        svc.rebuild_face_index([(ids[i], emb_of[p]) for i, p in gallery.items()])
        alerts = []
        for p, ident, mated in proto_split["probes"].itertuples(index=False):
            e = emb_of.get(p)
            if e is None or (mated and ident not in gallery):
                continue
            scores, found = svc.search_face(e, top_k=1)
            if len(scores) and scores[0] >= threshold:
                case = names[int(found[0])]
                alerts.append({"emb": e, "case": case, "sim": float(scores[0]), "true": bool(mated and case == ident)})
        return alerts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def simulate(alerts, neg_threshold, n_orders, seed):
    rng = np.random.default_rng(seed)
    fp_down = fp_n = tp_down = tp_n = 0
    for _ in range(n_orders):
        exemplars = {}
        for i in rng.permutation(len(alerts)):
            a = alerts[i]
            down, _ = suppression_decision(a["emb"], a["sim"], exemplars.get(a["case"], []), neg_threshold)
            if a["true"]:
                tp_n += 1
                tp_down += int(down)
            else:
                fp_n += 1
                fp_down += int(down)
                exemplars.setdefault(a["case"], []).append(a["emb"])
    return {"fp_alerts": fp_n / n_orders, "fp_downgraded": fp_down / n_orders, "tp_alerts": tp_n / n_orders,
            "tp_downgraded": tp_down / n_orders,
            "benefit": fp_down / fp_n if fp_n else 0.0, "cost": tp_down / tp_n if tp_n else 0.0}


def run(cfg):
    seed, n_orders = int(cfg["seed"]), int(cfg.get("n_orders", 20))
    threshold = float(config.MATCH_THRESHOLD)
    _, _, emb_of, proto = openset.load_lfw_protocol(cfg)
    alerts = {s: alerts_for_split(proto[s], emb_of, threshold, seed) for s in ("tune", "test")}
    grid = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
    tune = {str(t): simulate(alerts["tune"], t, n_orders, seed) for t in grid}
    zero_cost = [t for t in grid if tune[str(t)]["cost"] == 0.0]
    # among zero-cost values with the best benefit, take the strictest (least likely to touch true alerts)
    chosen = max(zero_cost, key=lambda t: (tune[str(t)]["benefit"], t)) if zero_cost else max(grid)
    test = simulate(alerts["test"], chosen, n_orders, seed)
    n_fp, n_tp = round(test["fp_alerts"]), round(test["tp_alerts"])
    test["benefit_ci"] = list(wilson_ci(round(test["fp_downgraded"]), n_fp)) if n_fp else [None, None]
    test["cost_ci"] = list(wilson_ci(round(test["tp_downgraded"]), n_tp)) if n_tp else [None, None]
    op = load_result("openset_lfw")["splits"]["test"]["by_gallery_size"]["1000"]["fnir_at_fpir"]["0.01"]
    fnir_half = (op["fnir_ci"][1] - op["fnir_ci"][0]) / 2
    ship = test["cost"] <= fnir_half and test["benefit"] > 0
    result = {"meta": metadata(cfg, lfw.DATASET_VERSION), "protocol": __doc__.strip().split("\n\n", 1)[1],
              "match_threshold": threshold, "tune_grid": tune, "chosen_threshold": chosen, "test": test,
              "deployed_fnir_ci_half_width": fnir_half, "ship": bool(ship),
              "ship_rule": "test cost (true alerts downgraded) <= CI half-width of deployed FNIR, and benefit > 0"}
    print(f"NEG_SUPPRESS_THRESHOLD {chosen}: test benefit {test['benefit']:.3f} of {test['fp_alerts']:.0f} FP alerts, "
          f"cost {test['cost']:.4f} of {test['tp_alerts']:.0f} true alerts; ship={ship}")
    save_result("neg_suppression", result)
