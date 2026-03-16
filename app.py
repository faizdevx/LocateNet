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

# Internal project imports
from models import *

app = Flask(__name__)
app.secret_key = "super_secret_locate_net_key"

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESOURCES_FOLDER = os.path.join(BASE_DIR, "resources")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESOURCES_FOLDER"] = RESOURCES_FOLDER

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
        name, mobile, location = request.form.get("name"), request.form.get("mobile"), request.form.get("location")
        email, birth_marks = request.form.get("email"), request.form.get("birth_marks")
        file = request.files.get("image")

        if not all([name, mobile, location, file]) or len(mobile) != 10:
            flash("Provide Name, 10-digit Mobile, Location, and Image.")
            return redirect(url_for("report_sighting"))

        unique_id = str(uuid.uuid4())
        filepath = os.path.join(app.config["RESOURCES_FOLDER"], f"{unique_id}.jpg")
        file.save(filepath)
        
        try:
            image_numpy = image_obj_to_numpy(open(filepath, "rb"))
            sighting_vector = extract_face_vector(image_numpy)
            
            if not sighting_vector:
                os.remove(filepath)
                flash("No face detected. Use a clearer photo.")
                return redirect(url_for("report_sighting"))

            # 1. Save submission to DB first so find_matches can link it
            details = PublicSubmissions(
                id=unique_id, submitted_by=name, location=location,
                email=email, mobile=mobile, face_vector=json.dumps(sighting_vector),
                birth_marks=birth_marks, status="NF"
            )
            db_queries.new_public_case(details)

            # 2. MATCHING ENGINE (Now passes unique_id to create matches)
            matches = MatchingEngine.find_matches(unique_id, sighting_vector, threshold=90.0)

            if matches:
                flash(f"Match found! ({matches[0]['confidence']}% confidence). Officer notified.")
            else:
                flash("Sighting reported. No matches found.")

        except Exception as e:
            flash(f"Error: {str(e)}")

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

@app.route("/register_case", methods=["GET", "POST"])
@login_required
def register_case():
    if request.method == "POST":
        person_name, city = request.form.get("person_name"), request.form.get("city")
        file = request.files.get('image')
        if person_name and city and file:
            image_numpy = image_obj_to_numpy(file)
            vector = extract_face_vector(image_numpy)
            if not vector:
                flash("Face not detected. Try again.")
                return redirect(url_for("register_case"))

            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.seek(0) 
            file.save(filepath)
            
            lat, lon = get_coords(city)
            add_case(session["user"], person_name, city, f"uploads/{filename}", lat, lon, vector)
            flash(f"Case for {person_name} registered.")
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

if __name__ == "__main__":
    create_db()
    app.run(debug=True)


