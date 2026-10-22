import sqlite3
import config
import cv2
import numpy as np
import json
from datetime import datetime
from contextlib import contextmanager

# --- DB CONFIG ---
DB_NAME = config.DB_PATH

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# --- DATA MODELS --- #

class PublicSubmissions:
    def __init__(self, id, submitted_by, location, mobile, face_vector, email=None, birth_marks=None, status="NF", image_path=None):
        self.id = id
        self.submitted_by = submitted_by
        self.location = location
        self.email = email
        self.mobile = mobile
        self.face_vector = face_vector 
        self.birth_marks = birth_marks
        self.status = status
        self.image_path = image_path

# --- REFINED DATABASE CORE ---

def create_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Original Tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT, city TEXT, officer TEXT, image_path TEXT,
                status TEXT, date_reported TEXT, latitude REAL, longitude REAL,
                face_vector TEXT
            )""")
        
        cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT, name TEXT)")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public_submissions (
                id TEXT PRIMARY KEY, submitted_by TEXT, location TEXT, email TEXT, 
                mobile TEXT, face_vector TEXT, birth_marks TEXT, status TEXT, date_submitted TEXT,
                image_path TEXT
            )""")
        columns = [row[1] for row in cursor.execute("PRAGMA table_info(public_submissions)").fetchall()]
        if "image_path" not in columns:
            cursor.execute("ALTER TABLE public_submissions ADD COLUMN image_path TEXT")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                sighting_id TEXT,
                confidence REAL,
                is_read INTEGER DEFAULT 0,
                date_detected TEXT,
                FOREIGN KEY(case_id) REFERENCES cases(id)
            )""")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sighting_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sighting_id TEXT NOT NULL,
                face_crop_path TEXT NOT NULL,
                match_id INTEGER,
                percentage REAL,
                category TEXT,
                bbox TEXT,
                review_decision TEXT,
                reviewed_by TEXT,
                review_timestamp TEXT,
                created_at TEXT,
                FOREIGN KEY(sighting_id) REFERENCES public_submissions(id),
                FOREIGN KEY(match_id) REFERENCES cases(id)
            )""")
        sighting_face_columns = [row[1] for row in cursor.execute("PRAGMA table_info(sighting_faces)").fetchall()]
        if "review_decision" not in sighting_face_columns:
            cursor.execute("ALTER TABLE sighting_faces ADD COLUMN review_decision TEXT")
        if "reviewed_by" not in sighting_face_columns:
            cursor.execute("ALTER TABLE sighting_faces ADD COLUMN reviewed_by TEXT")
        if "review_timestamp" not in sighting_face_columns:
            cursor.execute("ALTER TABLE sighting_faces ADD COLUMN review_timestamp TEXT")
        # Week 2: raw cosine similarity and calibrated likelihood (additive migration)
        for col in ("similarity", "calibrated_likelihood", "likelihood_prior"):
            if col not in sighting_face_columns:
                cursor.execute(f"ALTER TABLE sighting_faces ADD COLUMN {col} REAL")
        match_columns = [row[1] for row in cursor.execute("PRAGMA table_info(matches)").fetchall()]
        for col in ("similarity", "calibrated_likelihood", "likelihood_prior"):
            if col not in match_columns:
                cursor.execute(f"ALTER TABLE matches ADD COLUMN {col} REAL")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sighting_face_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sighting_face_id INTEGER NOT NULL,
                sighting_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewed_by TEXT NOT NULL,
                review_timestamp TEXT NOT NULL,
                FOREIGN KEY(sighting_face_id) REFERENCES sighting_faces(id),
                FOREIGN KEY(sighting_id) REFERENCES public_submissions(id)
            )""")

        conn.commit()

# --- AUTH & DASHBOARD FUNCTIONS ---

def add_new_user(username, password_hash, name, role="Officer"):
    with get_db_connection() as conn:
        try:
            conn.execute("INSERT INTO users (username, password_hash, role, name) VALUES (?, ?, ?, ?)", 
                         (username, password_hash, role, name))
            conn.commit()
            return True
        except sqlite3.IntegrityError: return False 

def get_user_by_username(username):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

def get_total_count(status):
    try:
        with get_db_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM cases WHERE status = ?", (status,)).fetchone()
            return result[0] if result else 0
    except sqlite3.OperationalError: return 0

def get_all_cases(officer):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM cases WHERE officer = ? ORDER BY date_reported DESC", (officer,)).fetchall()

# --- CASE MANAGEMENT ---

def add_case(officer, person_name, city, image_path, lat, lon, face_vector):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO cases (person_name, city, officer, image_path, status, date_reported, latitude, longitude, face_vector) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (person_name, city, officer, image_path, "NF", current_time, lat, lon, json.dumps(face_vector)))
        conn.commit()
        return cursor.lastrowid

class db_queries:
    @staticmethod
    def new_public_case(details: PublicSubmissions):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO public_submissions (id, submitted_by, location, email, mobile, face_vector, birth_marks, status, date_submitted, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (details.id, details.submitted_by, details.location, details.email, 
                  details.mobile, details.face_vector, details.birth_marks, details.status, current_time, details.image_path))
            conn.commit()

    @staticmethod
    def save_sighting_face(face_data):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bbox = face_data.get("bbox")
        bbox_json = None
        if bbox is not None:
            bbox_json = json.dumps([int(v) for v in bbox])

        match_id = face_data.get("match_id")
        percentage = face_data.get("percentage")

        def opt_float(key):
            value = face_data.get(key)
            return float(value) if value is not None else None
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO sighting_faces (sighting_id, face_crop_path, match_id, percentage, category, bbox, created_at,
                                            similarity, calibrated_likelihood, likelihood_prior)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                face_data["sighting_id"],
                face_data["face_crop_path"],
                int(match_id) if match_id is not None else None,
                float(percentage) if percentage is not None else None,
                face_data.get("category"),
                bbox_json,
                current_time,
                opt_float("similarity"),
                opt_float("calibrated_likelihood"),
                opt_float("likelihood_prior"),
            ))
            conn.commit()

    @staticmethod
    def get_case_by_id(case_id):
        with get_db_connection() as conn:
            return conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

    @staticmethod
    def get_public_submission_by_id(sighting_id):
        with get_db_connection() as conn:
            return conn.execute(
                "SELECT * FROM public_submissions WHERE id = ?",
                (sighting_id,),
            ).fetchone()

    @staticmethod
    def get_sighting_faces(sighting_id):
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT sf.*, c.person_name AS matched_person_name
                FROM sighting_faces sf
                LEFT JOIN cases c ON sf.match_id = c.id
                WHERE sf.sighting_id = ?
                ORDER BY sf.id ASC
            """, (sighting_id,)).fetchall()

        faces = []
        for row in rows:
            face = dict(row)
            face["bbox"] = json.loads(face["bbox"]) if face.get("bbox") else None
            faces.append(face)
        return faces

    @staticmethod
    def review_sighting_face(sighting_face_id, decision, reviewed_by):
        review_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db_connection() as conn:
            sighting_face = conn.execute(
                "SELECT id, sighting_id FROM sighting_faces WHERE id = ?",
                (sighting_face_id,),
            ).fetchone()
            if not sighting_face:
                return None

            conn.execute("""
                INSERT INTO sighting_face_reviews (sighting_face_id, sighting_id, decision, reviewed_by, review_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                sighting_face["id"],
                sighting_face["sighting_id"],
                decision,
                reviewed_by,
                review_timestamp,
            ))
            conn.execute("""
                UPDATE sighting_faces
                SET review_decision = ?, reviewed_by = ?, review_timestamp = ?
                WHERE id = ?
            """, (decision, reviewed_by, review_timestamp, sighting_face["id"]))
            conn.commit()
            return sighting_face["sighting_id"]

    @staticmethod
    def get_sighting_face_owner(sighting_face_id):
        """Return the face row plus the officer who owns the matched case (None if unmatched)."""
        with get_db_connection() as conn:
            return conn.execute("""
                SELECT sf.id, sf.sighting_id, sf.match_id, c.officer
                FROM sighting_faces sf
                LEFT JOIN cases c ON sf.match_id = c.id
                WHERE sf.id = ?
            """, (sighting_face_id,)).fetchone()

    @staticmethod
    def get_sighting_face_reviews(sighting_id):
        with get_db_connection() as conn:
            return conn.execute("""
                SELECT *
                FROM sighting_face_reviews
                WHERE sighting_id = ?
                ORDER BY review_timestamp DESC, id DESC
            """, (sighting_id,)).fetchall()

def get_case_by_id(case_id):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

def delete_case(case_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM cases WHERE id=?", (case_id,))
        conn.commit()

def resolve_case(case_id):
    with get_db_connection() as conn:
        conn.execute("UPDATE cases SET status='F' WHERE id=?", (case_id,))
        conn.commit()

# --- UTILITIES ---

def get_all_cases_admin():
    try:
        with get_db_connection() as conn:
            return conn.execute("SELECT * FROM cases ORDER BY date_reported DESC").fetchall()
    except sqlite3.OperationalError: return []

def get_public_cases():
    """Public registry view: only cases still missing, and only the fields safe to show."""
    try:
        with get_db_connection() as conn:
            return conn.execute("""
                SELECT id, person_name, city, date_reported, status
                FROM cases WHERE status = 'NF' ORDER BY date_reported DESC
            """).fetchall()
    except sqlite3.OperationalError: return []

def get_case_counts_by_city():
    try:
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT city, SUM(CASE WHEN status = 'NF' THEN 1 ELSE 0 END) as not_found,
                SUM(CASE WHEN status = 'F' THEN 1 ELSE 0 END) as found,
                AVG(latitude) as lat, AVG(longitude) as lon FROM cases GROUP BY city
            """).fetchall()
            return {row['city']: {"not_found": row['not_found'], "found": row['found'],
                                  "lat": row['lat'], "lon": row['lon']} for row in rows}
    except sqlite3.OperationalError: return {}

def image_obj_to_numpy(image_file):
    try:
        if hasattr(image_file, 'read'):
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            image_file.seek(0) 
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            return cv2.imread(image_file)
    except Exception as e:
        print(f"Error converting image to numpy: {e}")
        return None
