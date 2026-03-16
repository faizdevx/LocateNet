import os
import time
import json
import bcrypt
import folium
import requests
from flask import Flask, render_template, request, redirect, session, flash, jsonify
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Internal project imports
from models import *

app = Flask(__name__)
app.secret_key = "super_secret_locate_net_key"

# Configuration
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CITY_COORDS_FALLBACK = {
    "Delhi": [28.6139, 77.2090],
    "Lucknow": [26.8467, 80.9462],
    "Mumbai": [19.0760, 72.8777]
}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Geocoder
geolocator = Nominatim(user_agent="locate_net_app")

# --- UTILITY FUNCTIONS ---

def get_coords(city_name):
    """Fetches lat/lon for a city name. Returns (lat, lon) or (None, None)."""
    try:
        location = geolocator.geocode(f"{city_name}, India", timeout=10)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, Exception):
        return None, None
    return None, None

def create_map():
    """Generates the Folium map with city markers."""
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    data = get_case_counts_by_city()
    
    for city, counts in data.items():
        api_lat, api_lon = get_coords(city)
        coords = [api_lat, api_lon] if api_lat else CITY_COORDS_FALLBACK.get(city)
        
        if coords:
            total = counts["found"] + counts["not_found"]
            color = "red" if counts["not_found"] > 0 else "green"
            folium.CircleMarker(
                location=coords,
                radius=10 + (total * 2),
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"City: {city}<br>Missing: {counts['not_found']}<br>Found: {counts['found']}"
            ).add_to(m)
    return m._repr_html_()

# --- API ROUTES ---

@app.route("/api/get_cities")
def get_cities_sync():
    """Updated to look in the static/js folder based on your file structure."""
    try:
        # Change this path to match your folder structure
        file_path = os.path.join(os.path.dirname(__file__), 'static', 'json', 'cities.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # If it's the list of objects format
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return jsonify([item['name'] for item in data if 'name' in item])
                return jsonify(data)
        
        # Log to terminal so you can see if the file was missed
        print(f"File not found at: {file_path}")
        return jsonify(["Mumbai", "Delhi", "Lucknow"]) # Fallback
    except Exception as e:
        print(f"Sync Error: {e}")
        return jsonify([])
# --- AUTH ROUTES ---

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password_entered = request.form["password"]
        user = get_user_by_username(username)

        if user and bcrypt.checkpw(password_entered.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session.update({
                "user": username,
                "role": user['role'],
                "name": user['name']
            })
            return redirect("/dashboard")
        
        flash("Invalid Username or Password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# --- DASHBOARD & CASE ROUTES ---

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    user, role = session["user"], session.get("role")
    map_html = create_map()

    if role == "Admin":
        cases = get_all_cases_admin() 
        return render_template("admin_dashboard.html", 
                               found=get_total_count("F"), 
                               not_found=get_total_count("NF"), 
                               user=user, cases=cases, map=map_html)
    
    found_cases = get_registered_cases_count(user, "F")
    not_found_cases = get_registered_cases_count(user, "NF")
    return render_template("officer_dashboard.html", 
                           found=len(found_cases), 
                           not_found=len(not_found_cases), 
                           user=user, cases=get_all_cases(user), map=map_html)

@app.route("/register_case", methods=["GET", "POST"])
def register_case():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        person_name = request.form["person_name"]
        city = request.form["city"]
        lat, lon = get_coords(city)
        file = request.files.get('image')
        
        if file and file.filename:
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            add_case(session["user"], person_name, city, filepath, lat, lon)
            return redirect("/dashboard")

    return render_template("register_case.html", user=session.get("name"))

@app.route("/resolve/<int:case_id>")
def resolve(case_id):
    if "user" not in session:
        return redirect("/")
    
    case = get_case_by_id(case_id)
    if session["role"] == "Admin" or case['officer'] == session["user"]:
        resolve_case(case_id)
    return redirect("/dashboard")

@app.route("/create_officer", methods=["GET", "POST"])
def create_officer():
    if session.get("role") != "Admin":
        return redirect("/")

    if request.method == "POST":
        new_user = request.form["username"]
        hashed = bcrypt.hashpw(request.form["password"].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        if add_new_user(new_user, hashed, request.form["name"]):
            flash(f"Officer {new_user} created successfully!")
            return redirect("/dashboard")
        flash("Username already exists!")

    return render_template("create_officer.html")

if __name__ == "__main__":
    create_db()
    app.run(debug=True)