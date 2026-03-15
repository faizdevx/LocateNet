from flask import Flask, render_template, request, redirect, session
import yaml
import bcrypt 
from models import *
import os
from werkzeug.utils import secure_filename
import folium

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define this before the routes so they can use it
CITY_COORDS = {
    "Delhi": [28.6139, 77.2090],
    "Lucknow": [26.8467, 80.9462],
    "Mumbai": [19.0760, 72.8777]
}

# Helper function to generate the map
def create_map():
    # Center map on India
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    
    data = get_case_counts_by_city() # Ensure this is defined in models.py
    
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

# Load YAML config
with open("login_config.yml") as f:
    config = yaml.safe_load(f)

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password_entered = request.form["password"]
        users = config["credentials"]["usernames"]

        if username in users:
            stored_hash = users[username]["password"].encode('utf-8')
            if bcrypt.checkpw(password_entered.encode('utf-8'), stored_hash):
                session["user"] = username
                session["role"] = users[username]["role"]
                return redirect("/dashboard")
            else:
                print("Wrong password")
        else:
            print("User not found")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    user = session["user"]
    role = session.get("role") # Retrieve role stored during login
    map_html = create_map()

    if role == "Admin":
        # Admins see GLOBAL data
        all_cases = get_all_cases_admin() 
        found_count = get_total_count("F")
        not_found_count = get_total_count("NF")
        
        return render_template(
            "admin_dashboard.html",
            found=found_count,
            not_found=not_found_count,
            user=user,
            cases=all_cases,
            map=map_html
        )
    
    else:
        # Officers see PERSONAL data
        found_cases = get_registered_cases_count(user, "F")
        non_found_cases = get_registered_cases_count(user, "NF")
        cases_list = get_all_cases(user) 
        
        return render_template(
            "officer_dashboard.html",
            found=len(found_cases),
            not_found=len(non_found_cases),
            user=user,
            cases=cases_list,
            map=map_html
        )

@app.route("/register_case", methods=["GET", "POST"])
def register_case():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        person_name = request.form["person_name"]
        city = request.form["city"]
        file = request.files['image']
        
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            add_case(session["user"], person_name, city, filepath)
            return redirect("/dashboard")

    return render_template("register_case.html")


@app.route("/resolve/<int:case_id>")
def resolve(case_id):
    if "user" not in session:
        return redirect("/")
    
    # Optional: Only let Admins resolve cases
    # if session.get("role") != "Admin":
    #     return "Unauthorized", 403

    resolve_case(case_id)
    return redirect("/dashboard")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    create_db()
    app.run(debug=True)