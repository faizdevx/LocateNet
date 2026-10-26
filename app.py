import os
import sys
import time
import json
import html
import secrets
import bcrypt
import folium
import uuid
import logging
from functools import wraps, lru_cache
from flask import (Flask, render_template, request, redirect, session, flash, jsonify,
                   url_for, send_from_directory, abort)
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESOURCES_FOLDER = os.path.join(BASE_DIR, "resources")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Logging is configured before the services load so their startup messages are captured.
# Never log mobile numbers or emails.
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "system.log"),
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Internal project imports
import config
from models import *
from services.alert_service import AlertService, normalize_location
import random

from services import age_risk, ages, field_metrics, photo_quality, recalibration
from services.feedback import DOWNGRADED_CATEGORY, suppression_decision, template_growth_check
from services.calibration import Calibrator
from services.face_service import FaceService
from services.template_matcher import TemplateMatcher
from vector_db.rebuild import ensure_index_in_sync, rebuild_face_index
from vector_db.search_service import SearchService
from pipeline.detection_pipeline import DetectionPipeline
from pipeline.video_pipeline import VideoPipeline

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def _load_secret_key():
    key = os.environ.get("LOCATENET_SECRET_KEY")
    if key:
        return key
    if config.DEBUG:
        message = "LOCATENET_SECRET_KEY is not set; using a random key (sessions reset on restart)."
        logging.warning(message)
        print(f"WARNING: {message}", file=sys.stderr)
        return secrets.token_hex(32)
    raise RuntimeError(
        "LOCATENET_SECRET_KEY is not set. Set it to a long random value "
        "(or set FLASK_DEBUG=1 for local development)."
    )


app = Flask(__name__)
app.secret_key = _load_secret_key()
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    RESOURCES_FOLDER=RESOURCES_FOLDER,
    MAX_CONTENT_LENGTH=MAX_VIDEO_BYTES,  # global cap; per-type limits are checked in validate_upload
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
csrf = CSRFProtect(app)

# ZIPNET public-record sync: admin status/controls only. The worker is a separate process
# (see zipnet_worker.py); importing the app never starts it.
from blueprints.zipnet_sync import bp as zipnet_sync_bp  # noqa: E402
app.register_blueprint(zipnet_sync_bp)


def load_calibrator(path):
    """Optional score calibrator. Missing or unreadable file -> None (the UI then shows raw similarity only)."""
    try:
        return Calibrator.from_json(path)
    except FileNotFoundError:
        logging.info("No calibrator at %s; showing raw similarity only.", path)
    except (ValueError, KeyError, TypeError):
        logging.exception("Calibrator at %s could not be loaded; showing raw similarity only.", path)
    return None


calibrator = load_calibrator(config.CALIBRATOR_PATH)


def estimate_likelihood(similarity):
    """Estimated P(correct match) at config.CALIBRATION_PRIOR, or None without a calibrator."""
    if calibrator is None or similarity is None:
        return None
    return calibrator.predict(float(similarity), prior=config.CALIBRATION_PRIOR)


# --- AI PIPELINE INITIALIZATION ---
# These must be global so all routes can access them
face_svc = FaceService(det_size=config.DET_SIZE)
if config.ENABLE_REID:
    from services.reid_service import ReIDService
    reid_svc = ReIDService()
else:
    reid_svc = None
search_svc = SearchService()
if create_db():  # legacy single vectors were moved into case_embeddings: FAISS ids changed meaning
    rebuild_face_index(search_svc)
else:
    ensure_index_in_sync(search_svc)

# Current multi-photo templates (rebuilt whenever reference photos or case status change).
template_state = {"matcher": None}


def reload_matcher():
    rows = []
    for r in active_template_rows():
        now_age = ages.current_age(r["dob"], r["age_reported"], r["age_reported_on"])
        years_old = ages.photo_years_old(r["photo_date"], r["age_at_photo"], now_age)
        rows.append((r["id"], r["case_id"], json.loads(r["embedding"]), r["quality_score"], years_old))
    template_state["matcher"] = TemplateMatcher.from_rows(
        rows, strategy=config.TEMPLATE_STRATEGY, tau=config.RECENCY_TAU if config.ENABLE_RECENCY_WEIGHTING else None)
    return template_state["matcher"]


AGE_RESULTS = age_risk.load_age_results(config.AGE_GAP_RESULTS)


def case_age_warning(case):
    """Banner text when the newest reference photo is older than AGE_GAP_WARNING_YEARS, else None."""
    if not case:
        return None
    now_age = ages.current_age(case["dob"], case["age_reported"], case["age_reported_on"])
    years_old, age_at = age_risk.newest_photo_age(get_case_embeddings(case["id"], active_only=True), now_age,
                                                  ages.photo_years_old)
    estimate = age_risk.expected_miss(AGE_RESULTS, years_old, age_at)
    return age_risk.warning_text(years_old, estimate, config.AGE_GAP_WARNING_YEARS)


reload_matcher()
pipeline = DetectionPipeline(face_svc, reid_svc, search_svc, matcher_provider=lambda: template_state["matcher"])
video_pipeline = VideoPipeline(face_svc, reid_svc, search_svc, frame_stride=config.VIDEO_FRAME_STRIDE,
                               matcher_provider=lambda: template_state["matcher"])

CITY_COORDS_FALLBACK = {
    "Delhi": (28.6139, 77.2090),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777)
}

# Initialization
for folder in [UPLOAD_FOLDER, RESOURCES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

geolocator = Nominatim(user_agent="locate_net_app")

# --- DECORATORS ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# --- UTILITY ROUTES & FUNCTIONS ---

def validate_upload(file, kind):
    """Check extension and size. Returns (extension, error_message)."""
    ext = os.path.splitext(secure_filename(file.filename or ""))[1].lower().lstrip(".")
    allowed = ALLOWED_VIDEO_EXT if kind == "video" else ALLOWED_IMAGE_EXT
    if ext not in allowed:
        return None, f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}."
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    limit = MAX_VIDEO_BYTES if kind == "video" else MAX_IMAGE_BYTES
    if size > limit:
        return None, f"File is too large (limit {limit // (1024 * 1024)} MB)."
    return ext, None


@app.errorhandler(413)
def file_too_large(_error):
    flash("The upload is too large (images 10 MB, videos 100 MB).")
    return redirect(request.referrer or url_for("index"))


@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/resources/<path:filename>')
@login_required
def resource_file(filename):
    return send_from_directory(app.config['RESOURCES_FOLDER'], filename)


@app.route('/case_photo/<int:case_id>')
def case_photo(case_id):
    """Public photo for the registry. Only cases that are still missing (NF) are exposed."""
    case = db_queries.get_case_by_id(case_id)
    if not case or case["status"] != "NF" or not case["image_path"]:
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(case["image_path"]))

@app.route("/api/get_cities")
def get_cities_api():
    try:
        json_path = os.path.join(BASE_DIR, 'static', 'json', 'cities.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return jsonify(json.load(f))
    except (OSError, ValueError):
        logging.exception("Could not read cities.json")
    return jsonify(["Delhi", "Mumbai", "Lucknow", "Bangalore", "Chennai"])

@lru_cache(maxsize=512)
def _geocode_city(city_name):
    # Raises on failure so lru_cache never stores a failed lookup.
    location = geolocator.geocode(f"{city_name}, India", timeout=10)
    if not location:
        raise LookupError(city_name)
    return location.latitude, location.longitude


def get_coords(city_name):
    """Geocode a city (cached). Only used when a case is registered, never on page loads."""
    try:
        return _geocode_city((city_name or "").strip())
    except Exception:
        logging.warning("Geocoding failed for a city; using fallback coordinates.")
    return CITY_COORDS_FALLBACK.get((city_name or "").strip().title(), (None, None))


def can_manage_case(case):
    return bool(case) and (session.get("role") == "Admin" or case["officer"] == session.get("user"))


def process_reference_photo(file, photo_date=None):
    """Quality-gate one uploaded reference photo and save it. Returns (record, None) or (None, reason)."""
    ext, upload_error = validate_upload(file, "image")
    if upload_error:
        return None, upload_error
    image = image_obj_to_numpy(file)
    if image is None:
        return None, "could not be read as an image"
    face, reason = photo_quality.gate_reference_face(face_svc.detect_faces(image), config.DET_SCORE_MIN,
                                                     config.MIN_ENROL_FACE_PX)
    if reason:
        return None, reason
    width = float(face.bbox[2] - face.bbox[0])
    blur = photo_quality.blur_variance(image, face.bbox)
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.seek(0)
    file.save(filepath)
    taken = ages.parse_date(photo_date)
    return {"image_path": f"uploads/{filename}", "filepath": filepath,
            "embedding": [float(x) for x in face.embedding], "det_score": float(face.det_score),
            "quality_score": photo_quality.quality_score(face.det_score, width, blur), "face_px": width,
            "photo_date": taken.isoformat() if taken else None}, None


def enrol_reference_photos(case, records, source="enrolment"):
    """Store photos in case_embeddings and FAISS (one vector per photo id). Undo everything on failure."""
    added = []
    try:
        for rec in records:
            age_at = ages.age_at_photo(rec["photo_date"], case["dob"], case["age_reported"], case["age_reported_on"])
            emb_id = add_case_embedding(case["id"], rec["embedding"], rec["image_path"], rec["det_score"],
                                        rec["quality_score"], rec["face_px"], rec["photo_date"], age_at, source,
                                        session.get("user"))
            added.append(emb_id)
            search_svc.add_face_embedding(rec["embedding"], emb_id)
    except Exception:
        for emb_id in added:
            set_case_embedding_active(emb_id, False)
            try:
                search_svc.remove_face(emb_id)
            except Exception:
                logging.exception("Could not remove photo %s from the face index", emb_id)
        raise
    finally:
        reload_matcher()
    return added


def remove_case_from_index(case_id):
    for row in get_case_embeddings(case_id, active_only=True):
        try:
            search_svc.remove_face(row["id"])
        except Exception:
            logging.exception("Could not remove photo %s of case %s from the face index", row["id"], case_id)
    reload_matcher()


def is_case_active(case_id):
    case = db_queries.get_case_by_id(case_id)
    return bool(case and case["status"] == "NF")


def add_match_to_db(case_id, sighting_id, confidence, similarity=None, likelihood=None):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO matches (case_id, sighting_id, confidence, date_detected,
                                 similarity, calibrated_likelihood, likelihood_prior)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (case_id, sighting_id, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              similarity, likelihood, config.CALIBRATION_PRIOR if likelihood is not None else None))
        conn.commit()

def create_map():
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    data = get_case_counts_by_city()
    for city, counts in data.items():
        # Coordinates are stored on the case at registration; no geocoding on dashboard loads.
        lat, lon = counts["lat"], counts["lon"]
        if lat is None or lon is None:
            lat, lon = CITY_COORDS_FALLBACK.get((city or "").strip().title(), (None, None))
        if lat and lon:
            total = counts["not_found"] + counts["found"]
            color = "red" if counts["not_found"] > 0 else "green"
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + min(total * 2, 30),
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"<b>{html.escape(str(city))}</b><br>Missing: {counts['not_found']}<br>Found: {counts['found']}"
            ).add_to(m)
    return m._repr_html_()

# --- PUBLIC ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search")
def search_cases():
    cases = get_public_cases()
    return render_template("search_cases.html", cases=cases)

@app.route("/report_sighting", methods=["GET", "POST"])
def report_sighting():
    if request.method == "POST":
        start_time = time.time()
        name = request.form.get("name")
        mobile = request.form.get("mobile")
        location = request.form.get("location")
        email = request.form.get("email")
        birth_marks = request.form.get("birth_marks")
        upload_type = request.form.get("upload_type", "image")
        image_file = request.files.get("image")
        video_file = request.files.get("video")
        file = video_file if upload_type == "video" else image_file

        # 1. Validation
        if not all([name, mobile, location]) or len(mobile) != 10:
            flash("Please provide your Name, 10-digit Mobile, and Location.")
            return redirect(url_for("report_sighting"))

        if not file or not file.filename:
            flash("Please upload an image or video before submitting.")
            return redirect(url_for("report_sighting"))

        location = normalize_location(location)

        if upload_type == "video" and not getattr(video_pipeline, "available", False):
            flash("Video upload is not ready because video AI dependencies are not installed yet.")
            return redirect(url_for("report_sighting"))

        # 2. Save Sighting Media
        ext, upload_error = validate_upload(file, "video" if upload_type == "video" else "image")
        if upload_error:
            flash(upload_error)
            return redirect(url_for("report_sighting"))

        unique_id = str(uuid.uuid4())
        filename = f"sighting_{unique_id}.{ext}"
        filepath = os.path.join(app.config["RESOURCES_FOLDER"], filename)
        file.save(filepath)
        
        try:
            # 3. Process via AI Pipeline
            if upload_type == "video":
                video_result = video_pipeline.process_video(filepath, is_active=is_case_active)
                results = []
                for track in video_result.get("tracks", []):
                    results.append({
                        "id": track.get("match_id", -1),
                        "similarity": track.get("similarity", 0.0),
                        "percentage": track.get("percentage", 0.0),
                        "category": track.get("category", "No Match"),
                        "bbox": track.get("face_bbox") or track.get("person_bbox"),
                        "face_crop_path": track.get("face_crop_path"),
                        "embedding": track.get("face_embedding", []),
                        "extra_crops": track.get("extra_crops"),
                        "track_start_s": track.get("start_s"),
                        "track_end_s": track.get("end_s"),
                        "n_frames": track.get("n_frames"),
                    })
            else:
                image_np = cv2.imread(filepath)
                if image_np is None:
                    os.remove(filepath)
                    flash("The uploaded file could not be read as an image. Please try another photo.")
                    return redirect(url_for("report_sighting"))
                results = pipeline.process_frame(image_np, is_active=is_case_active)
            inference_time = time.time() - start_time
            
            # Log the performance
            logging.info(f"Inference: {inference_time:.2f}s | ID: {unique_id}")

            # 4. Save Public Submission to SQL
            # We use the highest-similarity face for the primary record, or an empty list if none
            best_result = max(results, key=lambda r: r["similarity"]) if results else None
            primary_vector = best_result["embedding"] if best_result else []
            
            details = PublicSubmissions(
                id=unique_id, submitted_by=name, location=location,
                email=email, mobile=mobile, face_vector=json.dumps(primary_vector),
                birth_marks=birth_marks, status="NF", image_path=f"resources/{filename}"
            )
            db_queries.new_public_case(details)

            # 5. Filter Matches (Hide Resolved Cases)
            faces_detected_count = 0
            active_matches = []

            for face_data in results:
                face_data["calibrated_likelihood"] = estimate_likelihood(face_data.get("similarity"))
                # False-positive suppression: downgrade (never drop) alerts that look like a face the
                # officer already rejected for this case.
                face_data["suppressed"] = False
                if config.ENABLE_NEG_SUPPRESSION and face_data["id"] != -1 and face_data.get("embedding") is not None:
                    down, _ = suppression_decision(face_data["embedding"], face_data["similarity"],
                                                   get_negative_exemplars(face_data["id"]), config.NEG_SUPPRESS_THRESHOLD)
                    if down:
                        face_data["category"], face_data["suppressed"] = DOWNGRADED_CATEGORY, True
                # Random QA sample of below-threshold faces (unbiased miss-rate estimate over time)
                face_data["qa_sample"] = (face_data["id"] == -1 and face_data.get("candidate_id", -1) != -1
                                          and random.random() < config.AUDIT_SAMPLE_RATE)
                bbox = face_data.get("bbox")
                db_queries.save_sighting_face({
                    "sighting_id": unique_id,
                    "face_crop_path": face_data.get("face_crop_path") or f"resources/{filename}",
                    "match_id": face_data["id"],
                    "percentage": face_data["percentage"],
                    "similarity": face_data.get("similarity"),
                    "calibrated_likelihood": face_data["calibrated_likelihood"],
                    "likelihood_prior": config.CALIBRATION_PRIOR if face_data["calibrated_likelihood"] is not None else None,
                    "category": face_data["category"],
                    "bbox": face_data["bbox"],
                    "extra_crops": face_data.get("extra_crops"),
                    "track_start_s": face_data.get("track_start_s"),
                    "track_end_s": face_data.get("track_end_s"),
                    "n_frames": face_data.get("n_frames"),
                    "embedding": face_data.get("embedding"),
                    "det_score": face_data.get("det_score"),
                    "face_px": float(bbox[2] - bbox[0]) if bbox is not None else None,
                    "candidate_id": face_data.get("candidate_id"),
                    "qa_sample": face_data["qa_sample"],
                    "suppressed": face_data["suppressed"],
                })
                faces_detected_count += 1

                # The pipeline already applied the threshold and the active-case check.
                if face_data["id"] == -1:
                    continue

                if is_case_active(face_data["id"]):
                    active_matches.append({
                        "case_id": face_data["id"],
                        "similarity": face_data["similarity"],
                        "percentage": face_data["percentage"],
                        "likelihood": face_data["calibrated_likelihood"],
                    })

            if active_matches:
                top_match = max(active_matches, key=lambda x: x["similarity"])
                add_match_to_db(top_match["case_id"], unique_id, top_match["percentage"],
                                top_match["similarity"], top_match["likelihood"])

                # 6. Hotspot alert: the same case matched repeatedly in one city within 48h
                is_hotspot, hotspot_count = AlertService.check_repeated_sightings(top_match["case_id"], location)
                if is_hotspot:
                    hotspot_case = db_queries.get_case_by_id(top_match["case_id"])
                    flash(f"HOTSPOT: {hotspot_case['person_name']} has been matched {hotspot_count} times in this area in the last 48 hours!")

            # 7. Final Response to User
            if active_matches:
                flash(f"Success: {faces_detected_count} faces detected in group. {len(active_matches)} matches identified! Officer notified.")
            else:
                flash(f"Sighting recorded. {faces_detected_count} faces detected, but no immediate active matches found above threshold.")

        except Exception:
            logging.exception("Sighting processing failed")
            flash("Something went wrong while processing the sighting. Please try again.")

        return redirect(url_for("report_sighting"))
        
    return render_template("report_sighting.html")


@app.route("/api/process_video", methods=["POST"])
@login_required
def process_video_api():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Video file is required."}), 400

    _, upload_error = validate_upload(file, "video")
    if upload_error:
        return jsonify({"ok": False, "error": upload_error}), 400

    if not getattr(video_pipeline, "available", False):
        return jsonify({
            "ok": False,
            "error": "Video pipeline dependencies are unavailable.",
        }), 503

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4()}_{filename}"
    video_dir = os.path.join(app.config["RESOURCES_FOLDER"], "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, unique_name)
    file.save(video_path)

    try:
        max_frames = request.form.get("max_frames", type=int)
        result = video_pipeline.process_video(video_path, max_frames=max_frames)
        return jsonify({"ok": True, "result": result})
    except Exception:
        logging.exception("Video pipeline failed")
        return jsonify({"ok": False, "error": "Video processing failed."}), 500

# --- AUTH & DASHBOARD ROUTES ---

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username, password = request.form.get("username"), request.form.get("password")
        user = get_user_by_username(username) if (username and password) else None
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session.clear()  # prevent session fixation
            session.update({"user": username, "role": user['role'], "name": user['name']})
            return redirect(url_for("dashboard"))
        flash("Invalid Credentials")
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    user, role = session["user"], session.get("role")
    map_html = create_map()
    
    # 1. Fetch unread match alerts for this specific officer
    with get_db_connection() as conn:
        alerts = conn.execute("""
            SELECT m.*, c.person_name, p.location, p.submitted_by 
            FROM matches m
            JOIN cases c ON m.case_id = c.id
            JOIN public_submissions p ON m.sighting_id = p.id
            JOIN (
                SELECT MAX(id) AS latest_match_id
                FROM matches
                WHERE is_read = 0
                GROUP BY sighting_id
            ) latest ON latest.latest_match_id = m.id
            WHERE c.officer = ? AND m.is_read = 0
            ORDER BY m.date_detected DESC
        """, (user,)).fetchall()

    alerts = [dict(a, age_warning=case_age_warning(get_case_by_id(a["case_id"]))) for a in alerts]
    with get_db_connection() as conn:
        qa_queue = [dict(r) for r in conn.execute("""
            SELECT sf.id, sf.sighting_id, sf.similarity, c.person_name
            FROM sighting_faces sf JOIN cases c ON c.id = sf.candidate_id
            WHERE sf.qa_sample = 1 AND sf.review_decision IS NULL AND (c.officer = ? OR ? = 'Admin')
            ORDER BY sf.id DESC LIMIT 20
        """, (user, role)).fetchall()]

    if role == "Admin":
        cases = get_all_cases_admin()
        return render_template("admin_dashboard.html", alerts=alerts, qa_queue=qa_queue, 
                               found=get_total_count("F"), not_found=get_total_count("NF"), 
                               user=user, cases=cases, map=map_html)
    
    cases = get_all_cases(user)
    found_count = len([c for c in cases if c['status'] == 'F'])
    not_found_count = len([c for c in cases if c['status'] == 'NF'])
    
    return render_template("officer_dashboard.html", alerts=alerts, qa_queue=qa_queue,
                           found=found_count, not_found=not_found_count, 
                           user=user, cases=cases, map=map_html)

@app.route("/mark_read/<int:alert_id>", methods=["POST"])
@login_required
def mark_read(alert_id):
    with get_db_connection() as conn:
        match = conn.execute("""
            SELECT m.sighting_id, c.officer
            FROM matches m JOIN cases c ON m.case_id = c.id
            WHERE m.id = ?
        """, (alert_id,)).fetchone()
        if match:
            if session.get("role") != "Admin" and match["officer"] != session["user"]:
                abort(403)
            conn.execute("UPDATE matches SET is_read = 1 WHERE sighting_id = ?", (match["sighting_id"],))
        conn.commit()
    return redirect(url_for("dashboard"))


@app.route("/sighting/<string:sighting_id>")
@login_required
def sighting_detail(sighting_id):
    sighting = db_queries.get_public_submission_by_id(sighting_id)
    if not sighting:
        flash("Sighting not found.")
        return redirect(url_for("dashboard"))

    faces = db_queries.get_sighting_faces(sighting_id)
    for f in faces:
        f["age_warning"] = case_age_warning(get_case_by_id(f["match_id"])) if f.get("match_id") not in (None, -1) else None
    reviews = db_queries.get_sighting_face_reviews(sighting_id)
    return render_template("officer_sighting_detail.html", sighting=sighting, faces=faces, reviews=reviews)


@app.route("/sighting/<string:sighting_id>/review/<int:face_id>", methods=["POST"])
@login_required
def review_sighting_face(sighting_id, face_id):
    decision = request.form.get("decision")
    allowed_decisions = {"confirm_match", "false_positive", "ignore"}

    if decision not in allowed_decisions:
        flash("Invalid review decision.")
        return redirect(url_for("sighting_detail", sighting_id=sighting_id))

    owner = db_queries.get_sighting_face_owner(face_id)
    if owner is None:
        flash("Face review target not found.")
        return redirect(url_for("dashboard"))
    # Only the officer who owns the matched case (or an Admin) may review. Faces without a
    # matched case can only be reviewed by an Admin.
    if session.get("role") != "Admin" and owner["officer"] != session["user"]:
        abort(403)

    reviewed_sighting_id = db_queries.review_sighting_face(
        sighting_face_id=face_id,
        decision=decision,
        reviewed_by=session["user"],
    )

    if reviewed_sighting_id is None:
        flash("Face review target not found.")
        return redirect(url_for("dashboard"))

    flash(f"Review saved: {decision.replace('_', ' ').title()}.")
    face = db_queries.get_sighting_face(face_id)
    matched_case = face["match_id"] if face and face["match_id"] not in (None, -1) else None
    if decision == "false_positive" and matched_case and face.get("embedding"):
        add_negative_exemplar(matched_case, face_id, json.loads(face["embedding"]), session["user"])
        log_audit(session["user"], "negative_exemplar_added", matched_case, face_id)
    if decision == "confirm_match" and matched_case and request.form.get("add_as_template") == "1":
        add_confirmed_face_as_template(face, get_case_by_id(matched_case))
    return redirect(url_for("sighting_detail", sighting_id=reviewed_sighting_id))


def add_confirmed_face_as_template(face, case):
    """Template growth with the anti-poisoning guards of services/feedback.py."""
    enrolment = [json.loads(r["embedding"]) for r in get_case_embeddings(case["id"], active_only=True)
                 if r["source"] == "enrolment"]
    ok, reason = template_growth_check(
        is_owner_or_admin=can_manage_case(case), n_existing_sighting_templates=count_sighting_templates(case["id"]),
        max_sighting_templates=config.MAX_SIGHTING_TEMPLATES, face_px=face.get("face_px"),
        det_score=face.get("det_score"), det_score_min=config.DET_SCORE_MIN, min_face_px=config.MIN_ENROL_FACE_PX,
        embedding=json.loads(face["embedding"]) if face.get("embedding") else None, enrolment_embeddings=enrolment,
        match_threshold=config.MATCH_THRESHOLD)
    if not ok:
        flash(f"Not added as a reference photo: {reason}.")
        return False
    record = {"image_path": face["face_crop_path"], "embedding": json.loads(face["embedding"]),
              "det_score": face.get("det_score"), "face_px": face.get("face_px"),
              "quality_score": photo_quality.quality_score(face.get("det_score") or 0, face.get("face_px") or 0,
                                                           photo_quality.REFERENCE_SHARPNESS),
              "photo_date": (face.get("created_at") or "")[:10] or None}
    [emb_id] = enrol_reference_photos(case, [record], source="confirmed_sighting")
    log_audit(session["user"], "photo_added", case["id"], emb_id, {"source": "confirmed_sighting", "sighting_face": face["id"]})
    flash("Face added as a reference photo (source: confirmed sighting). You can deactivate it on the case page.")
    return True

@app.route("/register_case", methods=["GET", "POST"])
@login_required
def register_case():
    if request.method == "POST":
        person_name = request.form.get("person_name")
        city = request.form.get("city")
        files = [f for f in request.files.getlist("images") if f and f.filename]
        legacy = request.files.get("image")
        if legacy and legacy.filename:
            files.insert(0, legacy)
        dates = request.form.getlist("photo_dates")

        if not (person_name and city and files):
            flash("Name, city and at least one photo are required.")
            return redirect(url_for("register_case"))
        if len(files) > config.MAX_PHOTOS_PER_CASE:
            flash(f"At most {config.MAX_PHOTOS_PER_CASE} reference photos per case.")
            return redirect(url_for("register_case"))

        dob = ages.parse_date(request.form.get("dob"))
        try:
            age_reported = float(request.form.get("age")) if request.form.get("age") else None
        except ValueError:
            age_reported = None

        accepted, rejected = [], []
        for i, file in enumerate(files):
            record, reason = process_reference_photo(file, dates[i] if i < len(dates) else None)
            if record:
                accepted.append(record)
            else:
                rejected.append(f"{secure_filename(file.filename) or 'photo ' + str(i + 1)}: {reason}")

        if not accepted:
            flash("No usable reference photo. " + "; ".join(rejected)
                  + ". Please provide a clear, front-facing photo.")
            return redirect(url_for("register_case"))

        lat, lon = get_coords(city)
        case_id = add_case(officer=session["user"], person_name=person_name, city=city,
                           image_path=accepted[0]["image_path"], lat=lat, lon=lon,
                           face_vector=accepted[0]["embedding"], dob=dob.isoformat() if dob else None,
                           age_reported=age_reported)
        try:
            enrol_reference_photos(get_case_by_id(case_id), accepted)
        except Exception:
            logging.exception("FAISS enrollment failed for case %s", case_id)
            delete_case(case_id)
            for rec in accepted:
                try:
                    os.remove(rec["filepath"])
                except OSError:
                    pass
            flash("Could not enroll the face. The case was not saved; please try again.")
            return redirect(url_for("register_case"))

        log_audit(session["user"], "case_registered", case_id, details={"photos": len(accepted), "rejected": len(rejected)})
        flash(f"Success: Case for {person_name} registered with {len(accepted)} reference photo(s).")
        for msg in rejected:
            flash(f"Photo rejected: {msg}")
        logging.info(f"New Case Registered: {person_name} by {session['user']}")
        return redirect(url_for("case_detail", case_id=case_id))

    return render_template("register_case.html", user=session.get("name"), max_photos=config.MAX_PHOTOS_PER_CASE)


@app.route("/case/<int:case_id>")
@login_required
def case_detail(case_id):
    case = get_case_by_id(case_id)
    if not can_manage_case(case):
        abort(403)
    photos = get_case_embeddings(case_id)
    return render_template("case_detail.html", case=case, photos=photos, audit=get_audit_log(case_id),
                           age_warning=case_age_warning(case),
                           max_photos=config.MAX_PHOTOS_PER_CASE,
                           n_active=sum(1 for p in photos if p["active"]))


@app.route("/case/<int:case_id>/photos", methods=["POST"])
@login_required
def add_reference_photo(case_id):
    case = get_case_by_id(case_id)
    if not can_manage_case(case):
        abort(403)
    if case["status"] != "NF":
        flash("This case is resolved; reference photos can no longer be added.")
        return redirect(url_for("case_detail", case_id=case_id))
    if len(get_case_embeddings(case_id, active_only=True)) >= config.MAX_PHOTOS_PER_CASE:
        flash(f"This case already has {config.MAX_PHOTOS_PER_CASE} active photos. Deactivate one first.")
        return redirect(url_for("case_detail", case_id=case_id))
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Choose a photo to add.")
        return redirect(url_for("case_detail", case_id=case_id))
    record, reason = process_reference_photo(file, request.form.get("photo_date"))
    if not record:
        flash(f"Photo rejected: {reason}")
        return redirect(url_for("case_detail", case_id=case_id))
    try:
        [emb_id] = enrol_reference_photos(case, [record])
    except Exception:
        logging.exception("Could not add a reference photo to case %s", case_id)
        flash("Could not add the photo; please try again.")
        return redirect(url_for("case_detail", case_id=case_id))
    log_audit(session["user"], "photo_added", case_id, emb_id, {"source": "enrolment", "photo_date": record["photo_date"]})
    flash("Reference photo added.")
    return redirect(url_for("case_detail", case_id=case_id))


@app.route("/case/<int:case_id>/photo/<int:photo_id>/deactivate", methods=["POST"])
@login_required
def deactivate_reference_photo(case_id, photo_id):
    case = get_case_by_id(case_id)
    if not can_manage_case(case):
        abort(403)
    photo = get_case_embedding(photo_id)
    if not photo or photo["case_id"] != case_id:
        abort(404)
    if photo["active"] and len(get_case_embeddings(case_id, active_only=True)) <= 1:
        flash("This is the only active reference photo. Add another photo before deactivating it.")
        return redirect(url_for("case_detail", case_id=case_id))
    set_case_embedding_active(photo_id, False)
    try:
        search_svc.remove_face(photo_id)
    except Exception:
        logging.exception("Could not remove photo %s from the face index", photo_id)
    reload_matcher()
    log_audit(session["user"], "photo_deactivated", case_id, photo_id, {"source": photo["source"]})
    flash("Reference photo deactivated.")
    return redirect(url_for("case_detail", case_id=case_id))


@app.route("/resolve/<int:case_id>", methods=["POST"])
@login_required
def resolve(case_id):
    case = get_case_by_id(case_id)
    if case and (session["role"] == "Admin" or case['officer'] == session["user"]):
        resolve_case(case_id)
        remove_case_from_index(case_id)
        log_audit(session["user"], "case_resolved", case_id)
        flash("Case resolved.")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index")) # This looks for a function named 'index'


@app.route("/create_officer", methods=["GET", "POST"])
@login_required
def create_officer():
    # Only Admins should be able to create new officers
    if session.get("role") != "Admin":
        flash("Unauthorized access!", "danger")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        name = request.form.get("name")
        
        if not all([username, password, name]):
            flash("All fields are required.", "warning")
            return redirect(url_for("create_officer"))

        # Hash the password for security
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Save to database using the function in models.py
        if add_new_user(username, hashed_pw, name, role="Officer"):
            flash(f"Officer account for {name} created successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Username already exists. Choose another.", "danger")

    return render_template("create_officer.html")


def _model_health_context():
    rows = alert_face_rows()
    alerts = [r for r in rows if not r["qa_sample"]]
    qa = [r for r in rows if r["qa_sample"]]
    min_n = config.FIELD_MIN_N
    reviewed = [r for r in alerts if r["review_decision"] in ("confirm_match", "false_positive")]
    edges = sorted({config.MATCH_THRESHOLD, 0.4, 0.5, config.HIGH_CONF_THRESHOLD, 1.01})
    volume = daily_volume(30)
    plan_fp, test_fpir = None, None
    try:
        with open(os.path.join(BASE_DIR, "eval", "results", "operating_point.json")) as f:
            op = json.load(f)
        test_fpir = op["previous_test_stats"]["MATCH_THRESHOLD"]["fpir"]
        plan_fp = op["uploads_per_day"] * op["faces_per_upload"] * test_fpir
    except (OSError, KeyError, ValueError):
        pass
    volume["predicted_false_alerts_per_day"] = (volume["uploads_per_day"] * volume["faces_per_upload"] * test_fpir
                                                if test_fpir is not None else None)
    volume["plan_false_alerts_per_day"] = plan_fp
    med, p90, n = field_metrics.review_latency_hours(reviewed)
    qa_reviewed = [r for r in qa if r["review_decision"] in ("confirm_match", "false_positive")]
    return {
        "min_n": min_n,
        "overall": field_metrics.rate(sum(r["review_decision"] == "confirm_match" for r in reviewed), len(reviewed), min_n),
        "weekly": field_metrics.weekly_precision(reviewed, min_n),
        "by_bucket": field_metrics.calibration_by_bucket(reviewed, edges, min_n),
        "by_source": field_metrics.precision_by(reviewed, lambda r: r["source_type"], min_n),
        "by_size": field_metrics.precision_by(reviewed, lambda r: field_metrics.face_size_bucket(r["face_px"]), min_n),
        "volume": volume, "latency": {"median": med, "p90": p90, "n": n},
        "backlog": sum(1 for r in alerts if not r["review_decision"]),
        "qa": field_metrics.rate(sum(r["review_decision"] == "confirm_match" for r in qa_reviewed), len(qa_reviewed), min_n),
        "audit_rate": config.AUDIT_SAMPLE_RATE, "proposals": recalibration.list_proposals(BASE_DIR),
    }


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "Admin":
            abort(403)
        return f(*args, **kwargs)
    return login_required(decorated)


@app.route("/admin/model-health")
@admin_required
def model_health():
    return render_template("model_health.html", **_model_health_context())


@app.route("/admin/model-health/chart/<name>.png")
@admin_required
def model_health_chart(name):
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctx = _model_health_context()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    if name == "precision":
        items = [(k, v) for k, v in ctx["weekly"].items() if v["enough"]]
        if items:
            xs = range(len(items))
            vals = [v["rate"] for _, v in items]
            ax.errorbar(xs, vals, yerr=[[v - r["ci"][0] for v, (_, r) in zip(vals, items)],
                                        [r["ci"][1] - v for v, (_, r) in zip(vals, items)]], marker="o", capsize=3)
            ax.set_xticks(list(xs), [k for k, _ in items], rotation=45, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("precision")
        ax.set_title("Weekly alert precision (weeks with n >= %d)" % ctx["min_n"], fontsize=9)
    elif name == "calibration":
        items = [(k, v) for k, v in ctx["by_bucket"].items() if v["enough"] and v["mean_predicted"] is not None]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for k, v in items:
            ax.errorbar([v["mean_predicted"]], [v["rate"]], yerr=[[v["rate"] - v["ci"][0]], [v["ci"][1] - v["rate"]]],
                        marker="o", capsize=3, label=k)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted likelihood")
        ax.set_ylabel("observed confirm rate")
        ax.set_title("Field calibration by similarity bucket", fontsize=9)
        if items:
            ax.legend(fontsize=7)
    else:
        abort(404)
    if not ax.has_data():
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes, color="grey")
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return app.response_class(buf.getvalue(), mimetype="image/png")


@app.route("/admin/model-health/proposal/<name>", methods=["POST"])
@admin_required
def decide_proposal(name):
    global calibrator
    decision = request.form.get("decision")
    if decision not in ("approve", "reject"):
        abort(400)
    try:
        recalibration.decide(BASE_DIR, name, decision == "approve", session["user"], config.CALIBRATOR_PATH)
    except (ValueError, OSError):
        logging.exception("Proposal %s could not be decided", name)
        flash("This proposal cannot be decided (not found or not pending).")
        return redirect(url_for("model_health"))
    log_audit(session["user"], f"calibration_proposal_{decision}d", details={"proposal": name})
    if decision == "approve":
        calibrator = load_calibrator(config.CALIBRATOR_PATH)
        flash("Proposal approved: the new calibrator is live; the previous one was kept for rollback.")
    else:
        flash("Proposal rejected.")
    return redirect(url_for("model_health"))


@app.route("/admin/logs")
@login_required
def view_logs():
    if session.get("role") != "Admin":
        return redirect(url_for("dashboard"))
    
    log_path = os.path.join(BASE_DIR, "logs", "system.log")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()[-50:]  # Get last 50 entries
        return render_template("logs.html", logs=lines)
    return "No logs found."



if __name__ == "__main__":
    # Development entry point only: Gunicorn imports `app` and never reaches this block, so a web
    # worker can never spawn a duplicate sync worker. Docker runs the worker as its own service.
    from services.zipnet_supervisor import autostart_if_enabled
    autostart_if_enabled()
    app.run(debug=config.DEBUG, use_reloader=False)
