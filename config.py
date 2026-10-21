"""Central configuration. Every value can be overridden with an environment variable."""
import os

try:  # optional: load a local .env file before anything is read
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Matching thresholds (cosine similarity), measured on LFW; see docs/OPERATING_POINT.md.
# Source: eval/results/operating_point.json (generated at commit 090fe12, gallery 1000, tune split).
MATCH_THRESHOLD = _env_float("MATCH_THRESHOLD", 0.312)        # FPIR 1%: alert only at or above this
HIGH_CONF_THRESHOLD = _env_float("HIGH_CONF_THRESHOLD", 0.634)  # FPIR 0.1%: "High Similarity" label only
DET_SCORE_MIN = _env_float("DET_SCORE_MIN", 0.6)              # minimum face detector score
TOP_K = _env_int("TOP_K", 5)                                  # FAISS neighbours inspected per face

# Score calibration (eval/calibrate.py). The likelihood shown to officers is only valid at this
# assumed share of searched faces that belong to an enrolled person; see services/calibration.py.
CALIBRATOR_PATH = os.environ.get("LOCATENET_CALIBRATOR", os.path.join(BASE_DIR, "models", "calibrator.json"))
CALIBRATION_PRIOR = _env_float("CALIBRATION_PRIOR", 0.1)

# Body ReID is a placeholder (untrained head), so it is off by default.
ENABLE_REID = _env_bool("ENABLE_REID", False)

# Flask debug mode. Off by default; enable with FLASK_DEBUG=1 for local development only.
DEBUG = _env_bool("FLASK_DEBUG", False)

# SQLite database location (override to keep it on a mounted volume).
DB_PATH = os.environ.get("LOCATENET_DB", os.path.join(BASE_DIR, "database.db"))
