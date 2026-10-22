"""Propose a calibrator re-fit from reviewed field alerts. Writes models/proposals/<date>.json and
prints old vs new metrics. Nothing changes until an Admin approves it on /admin/model-health.

    python scripts/propose_recalibration.py [--min-reviewed 200] [--min-per-class 30]

Needs at least 200 reviewed alerts (confirm_match / false_positive) and 30 of each class.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from models import create_db, reviewed_alert_rows  # noqa: E402
from services import recalibration  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reviewed", type=int, default=recalibration.MIN_REVIEWED)
    ap.add_argument("--min-per-class", type=int, default=recalibration.MIN_PER_CLASS)
    args = ap.parse_args(argv)
    create_db()
    rows = [r for r in reviewed_alert_rows() if r["similarity"] is not None]
    scores = [r["similarity"] for r in rows]
    labels = [1 if r["review_decision"] == "confirm_match" else 0 for r in rows]
    current = None
    if os.path.exists(config.CALIBRATOR_PATH):
        with open(config.CALIBRATOR_PATH) as f:
            current = json.load(f)
    proposal, reason = recalibration.build_proposal(scores, labels, current, min_reviewed=args.min_reviewed,
                                                     min_per_class=args.min_per_class)
    if proposal is None:
        print(f"No proposal: {reason}.")
        return 1
    path = recalibration.write_proposal(ROOT, proposal)
    m = proposal["metrics"]
    print(f"Proposal written to {os.path.relpath(path, ROOT)} ({proposal['n_reviewed']} reviewed alerts).")
    if m["current_on_field_data"]:
        print(f"  current  : Brier {m['current_on_field_data']['brier']:.4f}  ECE {m['current_on_field_data']['ece']:.4f}")
    print(f"  proposed : Brier {m['proposed_cross_validated']['brier']:.4f}  ECE {m['proposed_cross_validated']['ece']:.4f} (5-fold CV)")
    old = proposal["current"]
    new = proposal["calibrator"]
    print(f"  a: {old.get('a')} -> {new['a']:.4f}   b: {old.get('b')} -> {new['b']:.4f}   prior: {old.get('prior')} -> {new['prior']:.3f}")
    print("An Admin must approve it on /admin/model-health before it is used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
