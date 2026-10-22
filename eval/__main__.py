"""CLI: python -m eval run <experiment.yaml>   |   python -m eval report"""
import argparse
import importlib
import os
import sys

# Allow `python -m eval` from the repo root to import the app packages (services, vector_db, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.common import load_config  # noqa: E402

EXPERIMENTS = {
    "embed": "eval.protocols.embed_dataset",
    "verification": "eval.protocols.verification",
    "openset": "eval.protocols.openset",
    "operating_point": "eval.protocols.operating_point",
    "calibration": "eval.calibrate",
    "bias": "eval.protocols.bias",
    "robustness": "eval.protocols.robustness",
    "quality": "eval.protocols.quality",
    "detection": "eval.protocols.detection",
    "latency": "eval.protocols.latency",
    "detsize_check": "eval.protocols.detsize_check",
    "label_noise": "eval.protocols.label_noise",
    "age_gap": "eval.protocols.age_gap",
    "multi_template": "eval.protocols.multi_template",
    "age_mitigation": "eval.protocols.age_mitigation",
    "video": "eval.protocols.video",
    "neg_suppression": "eval.protocols.neg_suppression",
    "smoke": "eval.protocols.smoke",
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one experiment from a YAML config")
    run.add_argument("config")
    sub.add_parser("report", help="build docs/EVALUATION.md from eval/results/*.json")
    reg = sub.add_parser("regress", help="re-run the headline suite and compare with the baseline")
    reg.add_argument("--smoke", action="store_true", help="stub-model smoke comparison (CI)")
    reg.add_argument("--update-baseline", action="store_true", help="snapshot current results as the baseline")
    args = parser.parse_args(argv)

    if args.cmd == "regress":
        from eval.regress import main as regress_main
        return regress_main(smoke=args.smoke, update_baseline=args.update_baseline)

    if args.cmd == "report":
        from eval.report import build_report
        build_report()
        return 0

    cfg = load_config(args.config)
    name = cfg.get("experiment")
    if name not in EXPERIMENTS:
        parser.error(f"unknown experiment '{name}'. Known: {', '.join(EXPERIMENTS)}")
    module = importlib.import_module(EXPERIMENTS[name])
    module.run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
