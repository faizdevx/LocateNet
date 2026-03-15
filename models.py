import sqlite3
import os 
from datetime import datetime

DB_NAME = "database.db"

def create_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Cases Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            city TEXT,
            officer TEXT,
            image_path TEXT,
            status TEXT,
            date_reported TEXT
        )
    """)
    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            role TEXT,
            name TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- AUTHENTICATION FUNCTIONS ---

def add_new_user(username, password_hash, name, role="Officer"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash, role, name) VALUES (?, ?, ?, ?)",
                       (username, password_hash, role, name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False 
    finally:
        conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# --- CASE MANAGEMENT ---

def add_case(officer, person_name, city, image_path):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO cases (person_name, city, officer, image_path, status, date_reported) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (person_name, city, officer, image_path, "NF", current_time))
    conn.commit()
    conn.close()

def get_case_by_id(case_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
    case = cursor.fetchone()
    conn.close()
    return case

# --- DASHBOARD DATA FUNCTIONS ---

# ADDED THIS BACK TO FIX YOUR NameError
def get_registered_cases_count(user, status):
    """Fetches cases for a specific officer and status to count them."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM cases WHERE officer=? AND status=?",
        (user, status)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_cases(officer):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE officer = ? ORDER BY date_reported DESC", (officer,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_cases_admin():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases ORDER BY date_reported DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_total_count(status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cases WHERE status = ?", (status,))
    result = cursor.fetchone()
    count = result[0] if result else 0
    conn.close()
    return count

def resolve_case(case_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE cases SET status='F' WHERE id=?", (case_id,))
    conn.commit()
    conn.close()

def get_case_counts_by_city():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT city, 
               SUM(CASE WHEN status = 'NF' THEN 1 ELSE 0 END) as not_found,
               SUM(CASE WHEN status = 'F' THEN 1 ELSE 0 END) as found
        FROM cases 
        GROUP BY city
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: {"not_found": row[1], "found": row[2]} for row in rows}