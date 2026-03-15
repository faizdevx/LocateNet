from flask import Flask, render_template, request, redirect, session, flash
import bcrypt 
from models import *
import os
import time
from werkzeug.utils import secure_filename
import folium

app = Flask(__name__)
app.secret_key = "super_secret_locate_net_key" # Change this for production

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

CITY_COORDS = {
    "Delhi": [28.6139, 77.2090],
    "Lucknow": [26.8467, 80.9462],
    "Mumbai": [19.0760, 72.8777]
}

def create_map():
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    data = get_case_counts_by_city()
    for city, counts in data.items():
        coords = CITY_COORDS.get(city)
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

# --- ROUTES ---

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password_entered = request.form["password"]

        # Fetch user from Database instead of YAML
        user = get_user_by_username(username)

        if user:
            # checkpw requires bytes
            if bcrypt.checkpw(password_entered.encode('utf-8'), user['password_hash'].encode('utf-8')):
                session["user"] = username
                session["role"] = user['role']
                session["name"] = user['name']
                return redirect("/dashboard")
            else:
                flash("Invalid Password")
        else:
            flash("User not found")
            
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    user = session["user"]
    role = session.get("role")
    map_html = create_map()

    if role == "Admin":
        cases = get_all_cases_admin() 
        found = get_total_count("F")
        not_found = get_total_count("NF")
        return render_template("admin_dashboard.html", found=found, not_found=not_found, user=user, cases=cases, map=map_html)
    
    else:
        found_cases = get_registered_cases_count(user, "F")
        non_found_cases = get_registered_cases_count(user, "NF")
        cases = get_all_cases(user) 
        return render_template("officer_dashboard.html", found=len(found_cases), not_found=len(non_found_cases), user=user, cases=cases, map=map_html)

@app.route("/register_case", methods=["GET", "POST"])
def register_case():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        person_name = request.form["person_name"]
        city = request.form["city"]
        file = request.files['image']
        
        if file and file.filename != '':
            # Unique filename using timestamp to prevent overwriting
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            add_case(session["user"], person_name, city, filepath)
            return redirect("/dashboard")

    return render_template("register_case.html")

@app.route("/resolve/<int:case_id>")
def resolve(case_id):
    if "user" not in session:
        return redirect("/")
    
    case = get_case_by_id(case_id)
    # Security: Admins can resolve any, Officers only their own
    if session["role"] == "Admin" or case['officer'] == session["user"]:
        resolve_case(case_id)
    
    return redirect("/dashboard")

@app.route("/create_officer", methods=["GET", "POST"])
def create_officer():
    if session.get("role") != "Admin":
        return redirect("/")

    if request.method == "POST":
        new_user = request.form["username"]
        new_pass = request.form["password"]
        full_name = request.form["name"]

        hashed = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        if add_new_user(new_user, hashed, full_name):
            flash(f"Officer {new_user} created successfully!")
            return redirect("/dashboard")
        else:
            flash("Username already exists!")

    return render_template("create_officer.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    create_db()
    app.run(debug=True)