"""Central configuration. Every value can be overridden with an environment variable."""
import os


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


# NOTE: these thresholds are provisional. Week 2 evaluation will choose them from data.
MATCH_THRESHOLD = _env_float("MATCH_THRESHOLD", 0.45)         # cosine; alert only at or above this
HIGH_CONF_THRESHOLD = _env_float("HIGH_CONF_THRESHOLD", 0.60)
DET_SCORE_MIN = _env_float("DET_SCORE_MIN", 0.6)              # minimum face detector score
TOP_K = _env_int("TOP_K", 5)                                  # FAISS neighbours inspected per face

# Body ReID is a placeholder (untrained head), so it is off by default.
ENABLE_REID = _env_bool("ENABLE_REID", False)

# Flask debug mode. Off by default; enable with FLASK_DEBUG=1 for local development only.
DEBUG = _env_bool("FLASK_DEBUG", False)
