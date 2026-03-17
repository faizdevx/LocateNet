import sqlite3
import os 
import cv2
import numpy as np
import json
import mediapipe as mp
from datetime import datetime
from contextlib import contextmanager

# --- DB CONFIG ---
DB_NAME = "database.db"

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# --- BIOMETRIC INITIALIZATION ---
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except (ImportError, AttributeError):
    import mediapipe.solutions.face_mesh as mp_face_mesh

face_mesh_processor = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)



# --- SIGHTING & EMBEDDING MANAGEMENT ---

def add_sighting(embedding_type, vector, location, image_path):
    """
    Logs a new detection (face or body) into the sightings table.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sightings (embedding_type, embedding_vector, location, timestamp, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (embedding_type, json.dumps(vector), location, current_time, image_path))
        conn.commit()
        return cursor.lastrowid

def get_recent_sightings(limit=10):
    """
    Fetches the latest sightings for the dashboard.
    """
    with get_db_connection() as conn:
        return conn.execute("""
            SELECT * FROM sightings 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,)).fetchall()
    
    
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

# --- CORE BIOMETRIC ENGINE ---

def extract_face_vector(image_numpy):
    if image_numpy is None: return None
    rgb_image = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    results = face_mesh_processor.process(rgb_image)
    if not results.multi_face_landmarks: return None
    
    landmarks = results.multi_face_landmarks[0].landmark
    vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return vector.tolist()

class MatchingEngine:
    @staticmethod
    def calculate_similarity(v1, v2):
        vec1, vec2 = np.array(v1), np.array(v2)
        dot_product = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        return (dot_product / (norm_v1 * norm_v2)) * 100

    @staticmethod
    def find_matches(sighting_id, sighting_vector, threshold=90.0):
        matches = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db_connection() as conn:
            cases = conn.execute("SELECT id, person_name, officer, face_vector FROM cases WHERE status='NF'").fetchall()
            for case in cases:
                if case['face_vector']:
                    stored_vector = json.loads(case['face_vector'])
                    score = MatchingEngine.calculate_similarity(sighting_vector, stored_vector)
                    
                    if score >= threshold:
                        conn.execute("""
                            INSERT INTO matches (case_id, sighting_id, confidence, date_detected)
                            VALUES (?, ?, ?, ?)
                        """, (case['id'], sighting_id, round(score, 2), current_time))
                        conn.commit()
                        
                        matches.append({
                            "name": case['person_name'],
                            "confidence": round(score, 2)
                        })
        return matches

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
                created_at TEXT,
                FOREIGN KEY(sighting_id) REFERENCES public_submissions(id),
                FOREIGN KEY(match_id) REFERENCES cases(id)
            )""")

        # NEW: Sightings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_type TEXT, -- 'face' or 'body'
                embedding_vector TEXT, -- JSON string of the vector
                location TEXT,
                timestamp TEXT,
                image_path TEXT
            )""")

        # NEW: Global Embeddings Table (For master vector storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT, -- case_id or user_id
                entity_type TEXT,
                vector TEXT,
                created_at TEXT
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
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO sighting_faces (sighting_id, face_crop_path, match_id, percentage, category, bbox, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                face_data["sighting_id"],
                face_data["face_crop_path"],
                int(match_id) if match_id is not None else None,
                float(percentage) if percentage is not None else None,
                face_data.get("category"),
                bbox_json,
                current_time,
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

def get_case_by_id(case_id):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

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

def get_case_counts_by_city():
    try:
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT city, SUM(CASE WHEN status = 'NF' THEN 1 ELSE 0 END) as not_found,
                SUM(CASE WHEN status = 'F' THEN 1 ELSE 0 END) as found FROM cases GROUP BY city
            """).fetchall()
            return {row['city']: {"not_found": row['not_found'], "found": row['found']} for row in rows}
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
