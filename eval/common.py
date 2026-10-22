"""Shared helpers: paths, experiment config, result metadata."""
import datetime
import json
import os
import platform
import subprocess

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "eval", "cache")
# The regression gate re-runs experiments into a scratch folder instead of the committed results.
RESULTS_DIR = os.environ.get("LOCATENET_EVAL_RESULTS", os.path.join(ROOT, "eval", "results"))
PLOTS_DIR = os.environ.get("LOCATENET_EVAL_PLOTS", os.path.join(ROOT, "eval", "plots"))
MODEL_NAME = "buffalo_l"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", 0)
    try:
        cfg["_config_path"] = os.path.relpath(os.path.abspath(path), ROOT).replace("\\", "/")
    except ValueError:  # config on another drive
        cfg["_config_path"] = os.path.abspath(path).replace("\\", "/")
    return cfg


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _version(module):
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30)
            if out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def metadata(cfg, dataset_version):
    return {
        "experiment": cfg.get("experiment"),
        "config": cfg.get("_config_path"),
        "model": MODEL_NAME,
        "insightface_version": _version("insightface"),
        "onnxruntime_version": _version("onnxruntime"),
        "faiss_version": _version("faiss"),
        "dataset_version": dataset_version,
        "git_commit": git_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.get("seed"),
        "python": platform.python_version(),
    }


def _clean(obj):
    """Make numpy values JSON-safe and round floats for stable diffs."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if v != v:
            return None
        if v in (float("inf"), float("-inf")):
            return str(v)
        return round(v, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_result(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(_clean(payload), f, indent=2)
        f.write("\n")
    try:
        shown = os.path.relpath(path, ROOT)
    except ValueError:  # results folder on another drive (regression gate scratch dir)
        shown = path
    print(f"Saved {shown}")
    return path


def load_result(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def plot_path(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return os.path.join(PLOTS_DIR, name)
