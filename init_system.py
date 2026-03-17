import sqlite3
import bcrypt
from models import create_db, DB_NAME

# New imports for AI services
from services.face_service import FaceService
from services.reid_service import ReIDService
from vector_db.search_service import SearchService
from pipeline.detection_pipeline import DetectionPipeline

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
        print("✅ Success: Admin account created!")
        print(f"Username: {username} | Password: {password}")
    except sqlite3.IntegrityError:
        print("⚠️ Note: Admin user already exists.")
    finally:
        conn.close()

def init_ai_systems():
    """
    Initializes AI models once at startup to save memory and processing time.
    """
    print("🤖 Initializing AI Systems...")
    
    # Instantiate services
    face_service = FaceService()
    reid_service = ReIDService()
    search_service = SearchService()

    # Initialize the pipeline with the services
    pipeline = DetectionPipeline(
        face_service,
        reid_service,
        search_service
    )
    
    print("✅ AI Systems Ready.")
    return pipeline

if __name__ == "__main__":
    # Run Database Setup
    bootstrap_admin()
    
    # Run AI System Initialization
    pipeline = init_ai_systems()