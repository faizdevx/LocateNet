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

        # Week 3: several reference photos per case. FAISS stores one vector per case_embeddings.id.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                image_path TEXT,
                embedding TEXT NOT NULL,
                det_score REAL,
                quality_score REAL,
                face_px REAL,
                photo_date TEXT,
                age_at_photo REAL,
                source TEXT NOT NULL DEFAULT 'enrolment',
                added_by TEXT,
                created_at TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(case_id) REFERENCES cases(id)
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                actor TEXT,
                action TEXT NOT NULL,
                case_id INTEGER,
                target_id INTEGER,
                details TEXT
            )""")
        case_columns = [row[1] for row in cursor.execute("PRAGMA table_info(cases)").fetchall()]
        for col, kind in (("dob", "TEXT"), ("age_reported", "REAL"), ("age_reported_on", "TEXT")):
            if col not in case_columns:
                cursor.execute(f"ALTER TABLE cases ADD COLUMN {col} {kind}")
        # Additive migration: every legacy single vector becomes the case's first enrolment photo.
        migrated = cursor.execute("""
            INSERT INTO case_embeddings (case_id, image_path, embedding, source, added_by, created_at, active)
            SELECT c.id, c.image_path, c.face_vector, 'enrolment', c.officer, c.date_reported, 1
            FROM cases c
            WHERE c.face_vector IS NOT NULL AND c.face_vector NOT IN ('', '[]', 'null')
              AND NOT EXISTS (SELECT 1 FROM case_embeddings e WHERE e.case_id = c.id)
        """).rowcount

        conn.commit()
    return migrated  # number of legacy vectors moved into case_embeddings (the FAISS index must be rebuilt)

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

def add_case(officer, person_name, city, image_path, lat, lon, face_vector, dob=None, age_reported=None):
    # DEPRECATED: cases.face_vector is kept for one release; case_embeddings is the source of truth.
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO cases (person_name, city, officer, image_path, status, date_reported, latitude, longitude, face_vector,
                               dob, age_reported, age_reported_on)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (person_name, city, officer, image_path, "NF", current_time, lat, lon, json.dumps(face_vector),
              dob, age_reported, current_time[:10] if age_reported is not None else None))
        conn.commit()
        return cursor.lastrowid


# --- REFERENCE PHOTOS (case_embeddings) & AUDIT LOG ---

def add_case_embedding(case_id, embedding, image_path=None, det_score=None, quality_score=None, face_px=None,
                       photo_date=None, age_at_photo=None, source="enrolment", added_by=None):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO case_embeddings (case_id, image_path, embedding, det_score, quality_score, face_px, photo_date,
                                         age_at_photo, source, added_by, created_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (case_id, image_path, json.dumps([float(x) for x in embedding]), det_score, quality_score, face_px,
              photo_date, age_at_photo, source, added_by, current_time))
        conn.commit()
        return cursor.lastrowid


def get_case_embeddings(case_id, active_only=False):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM case_embeddings WHERE case_id = ?" + (" AND active = 1" if active_only else "") + " ORDER BY id",
            (case_id,)).fetchall()
    return [dict(r) for r in rows]


def get_case_embedding(embedding_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM case_embeddings WHERE id = ?", (embedding_id,)).fetchone()
    return dict(row) if row else None


def set_case_embedding_active(embedding_id, active):
    with get_db_connection() as conn:
        conn.execute("UPDATE case_embeddings SET active = ? WHERE id = ?", (1 if active else 0, embedding_id))
        conn.commit()


def active_template_rows():
    """Active reference photos of cases that are still missing: the rows the FAISS index is built from."""
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT e.id, e.case_id, e.embedding, e.quality_score, e.photo_date, e.age_at_photo, e.source,
                   c.dob, c.age_reported, c.age_reported_on
            FROM case_embeddings e JOIN cases c ON c.id = e.case_id
            WHERE e.active = 1 AND c.status = 'NF'
            ORDER BY e.id
        """).fetchall()
    return [dict(r) for r in rows]


def log_audit(actor, action, case_id=None, target_id=None, details=None):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO audit_log (ts, actor, action, case_id, target_id, details) VALUES (?, ?, ?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), actor, action, case_id, target_id,
                      json.dumps(details) if details is not None else None))
        conn.commit()


def get_audit_log(case_id, limit=100):
    with get_db_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM audit_log WHERE case_id = ? ORDER BY id DESC LIMIT ?", (case_id, limit)).fetchall()]


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
        conn.execute("DELETE FROM case_embeddings WHERE case_id=?", (case_id,))
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
