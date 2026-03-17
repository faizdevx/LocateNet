import os
import shutil
import sqlite3

def reset_system():
    print("🧹 Starting system cleanup...")

    # 1. Delete SQLite Database
    if os.path.exists("database.db"):
        os.remove("database.db")
        print("✅ Database file removed.")

    # 2. Clear Upload Folders (but keep the folders themselves)
    folders_to_clear = ['uploads', 'resources', 'logs', 'vector_db/storage']
    
    for folder in folders_to_clear:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.is_link(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"✅ Folder '{folder}' cleared.")

    print("\n✨ System is clean. Ready for re-initialization.")

if __name__ == "__main__":
    reset_system()