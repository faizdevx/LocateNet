from datetime import datetime, timedelta

from models import get_db_connection


def normalize_location(location):
    return (location or "").strip().lower()


class AlertService:
    @staticmethod
    def check_repeated_sightings(case_id, location, window_hours=48, threshold=3):
        """
        Hotspot rule: the same case matched >= threshold times in the same city within window_hours.
        Returns (is_hotspot, count).
        """
        cutoff = (datetime.now() - timedelta(hours=window_hours)).strftime("%Y-%m-%d %H:%M:%S")
        city = normalize_location(location)

        with get_db_connection() as conn:
            count = conn.execute("""
                SELECT COUNT(*)
                FROM matches m
                JOIN public_submissions p ON m.sighting_id = p.id
                WHERE m.case_id = ?
                  AND LOWER(TRIM(p.location)) = ?
                  AND m.date_detected > ?
            """, (case_id, city, cutoff)).fetchone()[0]

        return count >= threshold, count
