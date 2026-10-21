import sqlite3
import bcrypt
from models import create_db, DB_NAME

from vector_db.rebuild import rebuild_face_index
from vector_db.search_service import SearchService

def bootstrap_admin():
    # 1. Create the tables
    create_db()
    
    # 2. Add the initial Admin
    username = "admin"
    password = "admin123"  # Change this later!
    full_name = "System Administrator"
    role = "Admin"
    
    # Hash the password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash, role, name) VALUES (?, ?, ?, ?)",
                       (username, hashed, role, full_name))
        conn.commit()
        print("Success: Admin account created!")
        print(f"Username: {username} | Password: {password}")
    except sqlite3.IntegrityError:
        print("Note: Admin user already exists.")
    finally:
        conn.close()

def init_vector_index():
    """Create (or rebuild) the FAISS index from the active cases. Loads no AI models."""
    count = rebuild_face_index(SearchService())
    print(f"Vector index ready with {count} active case vector(s).")

if __name__ == "__main__":
    bootstrap_admin()
    init_vector_index()
