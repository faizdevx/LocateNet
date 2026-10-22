"""Create the two demo accounts used by scripts/make_walkthrough.py (run against an EMPTY demo database).

    LOCATENET_DB=demo.db python scripts/seed_walkthrough.py
"""
import os
import sys

import bcrypt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import add_new_user, create_db  # noqa: E402

ACCOUNTS = [
    ("officer_meera", "demo-officer-pass", "Inspector Meera Rao", "Officer"),
    ("admin_demo", "demo-admin-pass", "Demo Admin", "Admin"),
]


def main():
    create_db()
    for username, password, name, role in ACCOUNTS:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        print(f"{username}: {'created' if add_new_user(username, hashed, name, role) else 'already exists'}")


if __name__ == "__main__":
    main()
