import os
import time
import json
import bcrypt
import folium
import uuid
from functools import wraps
from flask import Flask, render_template, request, redirect, session, flash, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import logging

# Internal project imports
from models import *
from services.alert_service import AlertService
from services.face_service import FaceService
from services.reid_service import ReIDService
from vector_db.search_service import SearchService
from pipeline.detection_pipeline import DetectionPipeline
app = Flask(__name__)
app.secret_key = "super_secret_locate_net_key"

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESOURCES_FOLDER = os.path.join(BASE_DIR, "resources")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESOURCES_FOLDER"] = RESOURCES_FOLDER


# --- AI PIPELINE INITIALIZATION (ADD THIS HERE) ---
# These must be global so all routes can access them
face_svc = FaceService()
reid_svc = ReIDService()
search_svc = SearchService()
pipeline = DetectionPipeline(face_svc, reid_svc, search_svc)

CITY_COORDS_FALLBACK = {
    "Delhi": [28.6139, 77.2090],
    "Lucknow": [26.8467, 80.9462],
    "Mumbai": [19.0760, 72.8777]
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

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/resources/<path:filename>')
def resource_file(filename):
    return send_from_directory(app.config['RESOURCES_FOLDER'], filename)

@app.route("/api/get_cities")
def get_cities_api():
    try:
        json_path = os.path.join(BASE_DIR, 'static', 'json', 'cities.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return jsonify(json.load(f))
    except:
        pass
    return jsonify(["Delhi", "Mumbai", "Lucknow", "Bangalore", "Chennai"])

def get_coords(city_name):
    try:
        location = geolocator.geocode(f"{city_name}, India", timeout=10)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, Exception):
        pass
    return CITY_COORDS_FALLBACK.get(city_name, (None, None))


def add_match_to_db(case_id, sighting_id, confidence):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO matches (case_id, sighting_id, confidence, date_detected)
            VALUES (?, ?, ?, ?)
        """, (case_id, sighting_id, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

def create_map():
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    data = get_case_counts_by_city()
    for city, counts in data.items():
        lat, lon = get_coords(city)
        if lat and lon:
            total = counts["not_found"] + counts["found"]
            color = "red" if counts["not_found"] > 0 else "green"
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + min(total * 2, 30),
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"<b>{city}</b><br>Missing: {counts['not_found']}<br>Found: {counts['found']}"
            ).add_to(m)
    return m._repr_html_()

# --- PUBLIC ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search")
def search_cases():
    cases = get_all_cases_admin()
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
        file = request.files.get("image")

        # 1. Validation
        if not all([name, mobile, location, file]) or len(mobile) != 10:
            flash("Please provide your Name, 10-digit Mobile, Location, and an Image.")
            return redirect(url_for("report_sighting"))

        # 2. Save Sighting Image
        unique_id = str(uuid.uuid4())
        filename = f"sighting_{unique_id}.jpg"
        filepath = os.path.join(app.config["RESOURCES_FOLDER"], filename)
        file.save(filepath)
        
        try:
            # 3. Process via AI Pipeline (Handles Groups & Higher Accuracy)
            image_np = cv2.imread(filepath)
            results = pipeline.process_frame(image_np)
            inference_time = time.time() - start_time
            
            # Log the performance
            logging.info(f"Inference: {inference_time:.2f}s | ID: {unique_id}")

            # 4. Save Public Submission to SQL
            # We use the first face detected for the primary record, or an empty list if none
            primary_vector = results[0]['embedding'] if (results and 'embedding' in results[0]) else []
            
            details = PublicSubmissions(
                id=unique_id, submitted_by=name, location=location,
                email=email, mobile=mobile, face_vector=json.dumps(primary_vector),
                birth_marks=birth_marks, status="NF", image_path=f"resources/{filename}"
            )
            db_queries.new_public_case(details)

            # 5. Filter Matches (Hide Resolved Cases)
            faces_detected_count = 0
            matches_found_count = 0

            for face_data in results:
                db_queries.save_sighting_face({
                    "sighting_id": unique_id,
                    "face_crop_path": face_data["face_crop_path"],
                    "match_id": face_data["id"],
                    "percentage": face_data["confidence"],
                    "category": face_data["category"],
                    "bbox": face_data["bbox"],
                })
                faces_detected_count += 1

                if face_data["id"] == -1:
                    continue

                case = db_queries.get_case_by_id(face_data["id"])
                if case and case["status"] == "NF":
                    matches_found_count += 1
                    add_match_to_db(face_data["id"], unique_id, face_data["confidence"])
                elif case and case["status"] == "F":
                    logging.info(f"AI spotted resolved person ID {face_data['id']}. Ignoring alert.")

            # 6. Step 11: Hotspot Alert Logic
            is_hotspot, count = AlertService.check_repeated_sightings(location, location, "face")
            if is_hotspot:
                flash(f"🔥 HOTSPOT: This person has been seen {count} times in this area recently!")

            # 7. Final Response to User
            if matches_found_count > 0:
                flash(f"Success: {faces_detected_count} faces detected in group. {matches_found_count} matches identified! Officer notified.")
            else:
                flash(f"Sighting recorded. {faces_detected_count} faces detected, but no immediate active matches found above threshold.")

        except Exception as e:
                 logging.error(f"Sighting Error: {str(e)}")
                 flash(f"Error: {str(e)}") # This will show the real error on the screen

        return redirect(url_for("report_sighting"))
        
    return render_template("report_sighting.html")
# --- AUTH & DASHBOARD ROUTES ---

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username, password = request.form.get("username"), request.form.get("password")
        user = get_user_by_username(username)
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
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
            WHERE c.officer = ? AND m.is_read = 0
            ORDER BY m.date_detected DESC
        """, (user,)).fetchall()

    if role == "Admin":
        cases = get_all_cases_admin()
        return render_template("admin_dashboard.html", alerts=alerts, 
                               found=get_total_count("F"), not_found=get_total_count("NF"), 
                               user=user, cases=cases, map=map_html)
    
    cases = get_all_cases(user)
    found_count = len([c for c in cases if c['status'] == 'F'])
    not_found_count = len([c for c in cases if c['status'] == 'NF'])
    
    return render_template("officer_dashboard.html", alerts=alerts,
                           found=found_count, not_found=not_found_count, 
                           user=user, cases=cases, map=map_html)

@app.route("/mark_read/<int:alert_id>")
@login_required
def mark_read(alert_id):
    with get_db_connection() as conn:
        conn.execute("UPDATE matches SET is_read = 1 WHERE id = ?", (alert_id,))
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
    return render_template("officer_sighting_detail.html", sighting=sighting, faces=faces)

@app.route("/register_case", methods=["GET", "POST"])
@login_required
def register_case():
    if request.method == "POST":
        person_name = request.form.get("person_name")
        city = request.form.get("city")
        file = request.files.get('image')
        
        if not (person_name and city and file):
            flash("All fields are required.")
            return redirect(url_for("register_case"))

        # 1. Convert file to numpy for AI processing
        image_numpy = image_obj_to_numpy(file)
        
        # 2. Extract high-accuracy embedding (using FaceService, not MediaPipe)
        # This fixes the accuracy issues with Ali vs Atharv
        face_vector = face_svc.get_single_embedding(image_numpy)
        
        if face_vector is None:
            flash("❌ Face not detected. Please provide a clear, front-facing photo.")
            return redirect(url_for("register_case"))

        # 3. Save the physical file
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.seek(0) 
        file.save(filepath)
        
        # 4. Add to SQLite Database first so FAISS can store the real case id
        lat, lon = get_coords(city)
        # Convert numpy vector to list so it can be JSON serialized
        vector_list = face_vector.tolist() if hasattr(face_vector, 'tolist') else face_vector
        
        case_id = add_case(
            officer=session["user"], 
            person_name=person_name, 
            city=city, 
            image_path=f"uploads/{filename}", 
            lat=lat, 
            lon=lon, 
            face_vector=vector_list
        )

        # 5. Enroll into AI Pipeline & persist Vector DB
        enrolled = pipeline.enroll_new_person(image_numpy, person_id=case_id, name=person_name)
        search_svc.save_index()
        
        if enrolled:
            flash(f"Success: Case for {person_name} registered and added to the vector database.")
        else:
            flash(f"Success: Case for {person_name} registered, but no face embedding was enrolled.")
        logging.info(f"New Case Registered: {person_name} by {session['user']}")
        
        return redirect(url_for("dashboard"))

    return render_template("register_case.html", user=session.get("name"))

@app.route("/resolve/<int:case_id>")
@login_required
def resolve(case_id):
    case = get_case_by_id(case_id)
    if case and (session["role"] == "Admin" or case['officer'] == session["user"]):
        resolve_case(case_id)
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



# --- STEP 12: LOGGING CONFIGURATION ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/system.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


if __name__ == "__main__":
    create_db()
    app.run(debug=True)


