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
    "smoke": "eval.protocols.smoke",
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one experiment from a YAML config")
    run.add_argument("config")
    sub.add_parser("report", help="build docs/EVALUATION.md from eval/results/*.json")
    args = parser.parse_args(argv)

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
