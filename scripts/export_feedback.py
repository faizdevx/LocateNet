"""Admin-only export of review decisions for offline evaluation. Exports embeddings, decisions and
metadata as JSON lines; never face crops or images.

    python scripts/export_feedback.py --admin <username> --out feedback.jsonl

The username must be an Admin account; its password is asked for and checked.
"""
import argparse
import getpass
import json
import os
import sys

import bcrypt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import create_db, get_user_by_username, reviewed_alert_rows  # noqa: E402

FIELDS = ("id", "case_id", "similarity", "calibrated_likelihood", "category", "review_decision", "reviewed_by",
          "review_timestamp", "created_at", "source_type", "face_px", "qa_sample", "embedding")


def check_admin(username, password):
    user = get_user_by_username(username)
    return bool(user and user["role"] == "Admin"
                and bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")))


def export(path):
    n = 0
    with open(path, "w") as f:
        for r in reviewed_alert_rows(include_qa=True):
            row = {k: r[k] for k in FIELDS if k in r}
            if row.get("embedding"):
                row["embedding"] = json.loads(row["embedding"])
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    create_db()
    if not check_admin(args.admin, getpass.getpass("Admin password: ")):
        print("Refused: not an Admin account or wrong password.")
        return 1
    print(f"Exported {export(args.out)} reviewed faces (no images) to {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
