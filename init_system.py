import os
import secrets
import sqlite3
import bcrypt
from models import create_db, DB_NAME

from vector_db.rebuild import rebuild_face_index
from vector_db.search_service import SearchService

def bootstrap_admin():
    # 1. Create the tables
    create_db()

    # 2. Add the initial Admin (password from LOCATENET_ADMIN_PASSWORD, or randomly generated)
    username = "admin"
    full_name = "System Administrator"
    role = "Admin"

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        if cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            print("Note: Admin user already exists. Its password was not changed.")
            return

        password = os.environ.get("LOCATENET_ADMIN_PASSWORD")
        generated = not password
        if generated:
            password = secrets.token_urlsafe(12)

        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("INSERT INTO users (username, password_hash, role, name) VALUES (?, ?, ?, ?)",
                       (username, hashed, role, full_name))
        conn.commit()
        print("Success: Admin account created!")
        if generated:
            print(f"Username: {username} | Generated password: {password}")
            print("This password is shown only once. Store it now and change it after the first login.")
        else:
            print(f"Username: {username} | Password: taken from LOCATENET_ADMIN_PASSWORD")
    finally:
        conn.close()

def init_zipnet_store():
    """Create or upgrade the separate ZIPNET database (safe to repeat; never touches the main tables)."""
    from services import zipnet_store
    version = zipnet_store.migrate()
    print(f"ZIPNET database ready (schema v{version}) at {zipnet_store.db_path()}")


def init_vector_index():
    """Create (or rebuild) the FAISS index from the active cases. Loads no AI models."""
    count = rebuild_face_index(SearchService())
    print(f"Vector index ready with {count} active case vector(s).")

if __name__ == "__main__":
    bootstrap_admin()
    init_zipnet_store()
    init_vector_index()
