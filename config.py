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
# Retry detection on a 2x padded canvas when no face passes DET_SCORE_MIN (close-up portraits are
# otherwise missed). Source: eval/results/age_gap.json and embed_*.json (failure to enrol).
DETECT_PAD_RETRY = _env_bool("DETECT_PAD_RETRY", True)
# Detector input size (square). Kept at 640: 1024 gains crowd (WIDER Hard) AP but misses large close-up
# faces. Sources: eval/results/detection_widerface.json, eval/results/detsize_check.json.
DET_SIZE = (_env_int("DET_SIZE", 640),) * 2

# Multi-photo cases (Week 3). FAISS holds one vector per reference photo; TEMPLATE_STRATEGY turns the
# per-photo similarities of a case into one score. Source: eval/results/multi_template.json.
TEMPLATE_STRATEGY = os.environ.get("TEMPLATE_STRATEGY", "max")
MAX_PHOTOS_PER_CASE = _env_int("MAX_PHOTOS_PER_CASE", 5)
# Reference photos whose face is narrower than this are rejected (eval/results/robustness.json: 32 px).
MIN_ENROL_FACE_PX = _env_float("MIN_ENROL_FACE_PX", 32)
# Recency weighting of reference photos. Off: no variant beat the baseline beyond its CI on FG-NET
# (eval/results/age_mitigation.json, negative result for max, inconclusive for quality_weighted_mean).
ENABLE_RECENCY_WEIGHTING = _env_bool("ENABLE_RECENCY_WEIGHTING", False)
RECENCY_TAU = _env_float("RECENCY_TAU", 5.0)
# Warn officers when the newest reference photo is older than this (years). The expected miss rate
# shown with the warning is read from AGE_GAP_RESULTS (eval/results/age_gap.json), not hardcoded.
AGE_GAP_WARNING_YEARS = _env_float("AGE_GAP_WARNING_YEARS", 5)
AGE_GAP_RESULTS = os.environ.get("LOCATENET_AGE_GAP_RESULTS", os.path.join(BASE_DIR, "eval", "results", "age_gap.json"))

# Video (Week 3). Source: eval/results/video.json (SYNTHETIC pseudo-tracks; YouTube Faces not available).
# Aggregating a track into one embedding gave no accuracy gain over searching every frame, so the
# baseline stays (best_frame); the other strategies are available. Splitting tracks at an ID switch
# is on: it found 97.6% of glued tracks on the tune split and keeps each sub-track's crops one person.
TRACK_AGG_STRATEGY = os.environ.get("TRACK_AGG_STRATEGY", "best_frame")
ENABLE_TRACK_SPLIT = _env_bool("ENABLE_TRACK_SPLIT", True)
TRACK_SPLIT_THRESHOLD = _env_float("TRACK_SPLIT_THRESHOLD", 0.1)
# Every 5th frame: FNIR within noise of stride 3 at 40% less CPU; stride 10 loses clearly.
VIDEO_FRAME_STRIDE = _env_int("VIDEO_FRAME_STRIDE", 5)

# Human-feedback loop (Week 3), see docs/FEEDBACK_LOOP.md.
# False-positive suppression: downgrade (never drop) an alert that looks more like a face the officer
# already rejected for this case. Source: eval/results/neg_suppression.json (test: 11% of repeat
# false alerts downgraded, 0 true alerts downgraded).
ENABLE_NEG_SUPPRESSION = _env_bool("ENABLE_NEG_SUPPRESSION", True)
NEG_SUPPRESS_THRESHOLD = _env_float("NEG_SUPPRESS_THRESHOLD", 0.6)
# Confirmed sighting faces that may be added as extra reference photos per case (anti-poisoning cap).
MAX_SIGHTING_TEMPLATES = _env_int("MAX_SIGHTING_TEMPLATES", 3)
# Share of below-threshold faces sent to a "Random QA sample - not a match alert" review queue, so
# the miss rate can be estimated from field data without selection bias.
AUDIT_SAMPLE_RATE = _env_float("AUDIT_SAMPLE_RATE", 0.02)
# Negative exemplars, QA samples and stored face embeddings are deleted after this many days
# (scripts/purge_expired.py).
RETENTION_DAYS = _env_int("RETENTION_DAYS", 90)
# Field metrics are shown only with at least this many reviews per row.
FIELD_MIN_N = _env_int("FIELD_MIN_N", 30)

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
