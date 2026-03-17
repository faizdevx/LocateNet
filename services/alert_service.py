import sqlite3
from datetime import datetime, timedelta
from models import DB_NAME, get_db_connection

class AlertService:
    @staticmethod
    def check_repeated_sightings(location, city, embedding_type, threshold=3):
        """
        Rule: If similar reports >= 3 within 48h in same city -> Trigger High Priority Alert.
        """
        forty_eight_hours_ago = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")
        
        with get_db_connection() as conn:
            # Count sightings in the same city/location recently
            query = """
                SELECT COUNT(*) FROM sightings 
                WHERE location LIKE ? 
                AND timestamp > ?
                AND embedding_type = ?
            """
            count = conn.execute(query, (f"%{city}%", forty_eight_hours_ago, embedding_type)).fetchone()[0]
            
            if count >= threshold:
                return True, count
        return False, count