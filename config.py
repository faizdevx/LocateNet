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
# Minimum face-detector score (quality gate). Kept at 0.6: eval/results/quality.json (recommended_DET_SCORE_MIN).
DET_SCORE_MIN = _env_float("DET_SCORE_MIN", 0.6)
TOP_K = _env_int("TOP_K", 5)                                  # FAISS neighbours inspected per face
# Detector input size (square). Kept at 640: 1024 gains crowd (WIDER Hard) AP but misses large close-up
# faces. Sources: eval/results/detection_widerface.json, eval/results/detsize_check.json.
DET_SIZE = (_env_int("DET_SIZE", 640),) * 2

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
