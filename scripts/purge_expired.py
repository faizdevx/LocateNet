"""Delete feedback data older than RETENTION_DAYS (default 90):
  * negative exemplars (embeddings of faces marked false_positive)
  * random QA samples (below-threshold faces sent for review): row, review and face crop
  * stored face embeddings of other sighting faces (the crop and the decision stay)

Run daily, e.g. cron:  15 3 * * *  cd /srv/locatenet && .venv/bin/python scripts/purge_expired.py
Use --dry-run to see the counts without deleting.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from models import create_db, get_db_connection  # noqa: E402


def purge(days, dry_run=False, root=ROOT):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        neg = conn.execute("SELECT COUNT(*) FROM negative_exemplars WHERE created_at < ?", (cutoff,)).fetchone()[0]
        qa = conn.execute("SELECT id, face_crop_path FROM sighting_faces WHERE qa_sample = 1 AND created_at < ?",
                          (cutoff,)).fetchall()
        emb = conn.execute("SELECT COUNT(*) FROM sighting_faces WHERE embedding IS NOT NULL AND created_at < ?",
                           (cutoff,)).fetchone()[0]
        if not dry_run:
            conn.execute("DELETE FROM negative_exemplars WHERE created_at < ?", (cutoff,))
            for row in qa:
                conn.execute("DELETE FROM sighting_face_reviews WHERE sighting_face_id = ?", (row["id"],))
                conn.execute("DELETE FROM sighting_faces WHERE id = ?", (row["id"],))
                path = os.path.join(root, row["face_crop_path"]) if row["face_crop_path"] else None
                if path and os.path.isfile(path):
                    os.remove(path)
            conn.execute("UPDATE sighting_faces SET embedding = NULL WHERE embedding IS NOT NULL AND created_at < ?", (cutoff,))
            conn.commit()
    return {"cutoff": cutoff, "negative_exemplars": int(neg), "qa_samples": len(qa), "face_embeddings_cleared": int(emb)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=config.RETENTION_DAYS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    create_db()
    out = purge(args.days, args.dry_run)
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} (older than {out['cutoff']}): {out['negative_exemplars']} negative exemplars, "
          f"{out['qa_samples']} QA samples; cleared {out['face_embeddings_cleared']} face embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
