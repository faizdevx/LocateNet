"""SQLite store for imported ZIPNET records, checkpoints, jobs, the worker lease and control flags.

Imported records live in their own database file (config.ZIPNET_DB_PATH), never in the native
case tables, so a multi-million-row import cannot contend with case registration or alter any
existing LocateNet table. The schema is versioned with ``PRAGMA user_version`` and every
migration is additive and safe to repeat.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any

import config

from .zipnet_adapters import NormalizedRecord

# --- schema -----------------------------------------------------------------------------------

MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            """CREATE TABLE IF NOT EXISTS zipnet_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_id_num INTEGER,
                title TEXT NOT NULL,
                status TEXT,
                name TEXT,
                name_search TEXT,
                sex TEXT,
                age_min INTEGER,
                age_max INTEGER,
                reported_date TEXT,
                reported_date_kind TEXT,
                source_created_date TEXT,
                state TEXT,
                district TEXT,
                police_station TEXT,
                published_location TEXT,
                case_reference TEXT,
                vehicle_registration_raw TEXT,
                vehicle_registration_norm TEXT,
                vehicle_type TEXT,
                vehicle_make TEXT,
                imei TEXT,
                image_paths TEXT NOT NULL DEFAULT '[]',
                attributes TEXT NOT NULL DEFAULT '{}',
                extraction_warnings TEXT NOT NULL DEFAULT '[]',
                fingerprint TEXT NOT NULL,
                source_listing_url TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_changed_at TEXT NOT NULL,
                UNIQUE (section, source_id)
            )""",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_records_section_created "
            "ON zipnet_records (section, source_created_date)",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_records_area ON zipnet_records (state, district)",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_records_reg ON zipnet_records (vehicle_registration_norm)",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_records_imei ON zipnet_records (imei)",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_records_name ON zipnet_records (name_search)",
            """CREATE TABLE IF NOT EXISTS zipnet_record_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL REFERENCES zipnet_records(id) ON DELETE CASCADE,
                job_id INTEGER,
                changed_at TEXT NOT NULL,
                change_type TEXT NOT NULL,
                changed_fields TEXT NOT NULL DEFAULT '{}'
            )""",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_changes_record ON zipnet_record_changes (record_id)",
            """CREATE TABLE IF NOT EXISTS zipnet_checkpoints (
                section TEXT PRIMARY KEY,
                backfill_cursor INTEGER NOT NULL DEFAULT 0,
                backfill_pages INTEGER NOT NULL DEFAULT 0,
                backfill_complete INTEGER NOT NULL DEFAULT 0,
                backfill_completed_at TEXT,
                backfill_limit_reached INTEGER NOT NULL DEFAULT 0,
                high_water_id INTEGER,
                incr_cursor INTEGER NOT NULL DEFAULT 0,
                incr_pending_hw INTEGER,
                incr_in_progress INTEGER NOT NULL DEFAULT 0,
                last_incremental_at TEXT,
                caught_up INTEGER NOT NULL DEFAULT 0,
                indexed_count INTEGER NOT NULL DEFAULT 0,
                source_total INTEGER,
                last_status TEXT,
                last_message TEXT,
                last_success_at TEXT,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL,
                updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS zipnet_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                pages INTEGER NOT NULL DEFAULT 0,
                requests INTEGER NOT NULL DEFAULT 0,
                retries INTEGER NOT NULL DEFAULT 0,
                scanned INTEGER NOT NULL DEFAULT 0,
                new_records INTEGER NOT NULL DEFAULT 0,
                updated_records INTEGER NOT NULL DEFAULT 0,
                unchanged_records INTEGER NOT NULL DEFAULT 0,
                failed_records INTEGER NOT NULL DEFAULT 0,
                source_total INTEGER,
                stop_reason TEXT,
                message TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS ix_zipnet_jobs_started ON zipnet_jobs (id DESC)",
            """CREATE TABLE IF NOT EXISTS zipnet_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                section TEXT NOT NULL,
                source_id TEXT,
                page_start INTEGER,
                stage TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS zipnet_lease (
                name TEXT PRIMARY KEY,
                holder TEXT NOT NULL,
                pid INTEGER,
                acquired_at REAL NOT NULL,
                heartbeat_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS zipnet_control (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL,
                updated_by TEXT
            )""",
            # Explicit, human-made mapping from an external record to a native case. Creating a link
            # changes nothing else: no biometric enrolment, no status change.
            """CREATE TABLE IF NOT EXISTS zipnet_case_links (
                record_id INTEGER NOT NULL REFERENCES zipnet_records(id) ON DELETE CASCADE,
                case_id INTEGER NOT NULL,
                linked_by TEXT NOT NULL,
                linked_at TEXT NOT NULL,
                note TEXT,
                PRIMARY KEY (record_id, case_id)
            )""",
        ],
    ),
]

MIGRATIONS.append((
    2,
    [
        # Public missing-person panel: filter by section + status, newest first.
        "CREATE INDEX IF NOT EXISTS ix_zipnet_records_panel ON zipnet_records (section, status, source_created_date DESC)",
    ],
))

SCHEMA_VERSION = MIGRATIONS[-1][0]

SECTIONS = (
    "missing_persons",
    "unidentified_persons",
    "unidentified_dead_bodies",
    "stolen_vehicles",
    "unclaimed_seized_vehicles",
    "missing_mobiles",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def db_path() -> str:
    return config.ZIPNET_DB_PATH


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # explicit transactions below
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, ROLLBACK on error. One writer at a time, readers never block."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def migrate(conn: sqlite3.Connection | None = None) -> int:
    """Bring the ZIPNET database to the latest schema. Safe to call repeatedly and concurrently."""
    own = conn is None
    conn = conn or connect()
    try:
        if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return SCHEMA_VERSION
        for version, statements in MIGRATIONS:
            with transaction(conn):
                current = conn.execute("PRAGMA user_version").fetchone()[0]  # re-read under the write lock
                if current >= version:
                    continue
                for stmt in statements:
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version={int(version)}")
        return SCHEMA_VERSION
    finally:
        if own:
            conn.close()


# --- helpers ----------------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_COLUMN_FIELDS = (
    "source_id_num", "title", "status", "name", "sex", "age_min", "age_max", "reported_date",
    "reported_date_kind", "source_created_date", "state", "district", "police_station",
    "published_location", "case_reference", "vehicle_registration_raw", "vehicle_registration_norm",
    "vehicle_type", "vehicle_make", "imei",
)


def _content_of(nr: NormalizedRecord) -> dict[str, Any]:
    c = jsonable(nr.content())
    out = {k: c.get(k) for k in _COLUMN_FIELDS}
    out["image_paths"] = list(nr.image_urls)
    out["attributes"] = c.get("attributes") or {}
    return out


def _row_content(row: sqlite3.Row) -> dict[str, Any]:
    out = {k: row[k] for k in _COLUMN_FIELDS}
    out["image_paths"] = json.loads(row["image_paths"])
    out["attributes"] = json.loads(row["attributes"])
    return out


# --- records ----------------------------------------------------------------------------------


def upsert_records(conn: sqlite3.Connection, section: str, listing_url: str, records: list[NormalizedRecord],
                   job_id: int | None) -> tuple[int, int, int]:
    """Idempotent upsert of one page. Caller owns the transaction. Returns (new, updated, unchanged)."""
    now = now_iso()
    by_id: dict[str, NormalizedRecord] = {}
    for nr in records:
        by_id[nr.source_id] = nr  # last occurrence wins for in-page duplicates
    if not by_id:
        return 0, 0, 0
    marks = ",".join("?" * len(by_id))
    existing = {
        r["source_id"]: r
        for r in conn.execute(
            f"SELECT * FROM zipnet_records WHERE section=? AND source_id IN ({marks})",  # noqa: S608 - placeholders only
            [section, *by_id],
        )
    }
    new = updated = unchanged = 0
    for sid, nr in by_id.items():
        content = _content_of(nr)
        row = existing.get(sid)
        if row is None:
            cur = conn.execute(
                """INSERT INTO zipnet_records (section, source_id, source_id_num, title, status, name, name_search, sex,
                   age_min, age_max, reported_date, reported_date_kind, source_created_date, state, district,
                   police_station, published_location, case_reference, vehicle_registration_raw,
                   vehicle_registration_norm, vehicle_type, vehicle_make, imei, image_paths, attributes,
                   extraction_warnings, fingerprint, source_listing_url, first_seen_at, last_seen_at, last_changed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (section, sid, content["source_id_num"], content["title"], content["status"], content["name"],
                 content["name"].lower() if content["name"] else None, content["sex"], content["age_min"],
                 content["age_max"], content["reported_date"], content["reported_date_kind"],
                 content["source_created_date"], content["state"], content["district"], content["police_station"],
                 content["published_location"], content["case_reference"], content["vehicle_registration_raw"],
                 content["vehicle_registration_norm"], content["vehicle_type"], content["vehicle_make"],
                 content["imei"], dumps(content["image_paths"]), dumps(content["attributes"]),
                 dumps(nr.warnings), nr.fingerprint, listing_url, now, now, now),
            )
            conn.execute(
                "INSERT INTO zipnet_record_changes (record_id, job_id, changed_at, change_type, changed_fields) "
                "VALUES (?,?,?,?,?)", (cur.lastrowid, job_id, now, "created", "{}"))
            new += 1
        elif row["fingerprint"] != nr.fingerprint:
            old = _row_content(row)
            diff = {k: {"old": old.get(k), "new": content.get(k)} for k in content if old.get(k) != content.get(k)}
            if diff:
                conn.execute(
                    """UPDATE zipnet_records SET source_id_num=?, title=?, status=?, name=?, name_search=?, sex=?,
                       age_min=?, age_max=?, reported_date=?, reported_date_kind=?, source_created_date=?, state=?,
                       district=?, police_station=?, published_location=?, case_reference=?,
                       vehicle_registration_raw=?, vehicle_registration_norm=?, vehicle_type=?, vehicle_make=?,
                       imei=?, image_paths=?, attributes=?, extraction_warnings=?, fingerprint=?,
                       source_listing_url=?, last_seen_at=?, last_changed_at=? WHERE id=?""",
                    (content["source_id_num"], content["title"], content["status"], content["name"],
                     content["name"].lower() if content["name"] else None, content["sex"], content["age_min"],
                     content["age_max"], content["reported_date"], content["reported_date_kind"],
                     content["source_created_date"], content["state"], content["district"],
                     content["police_station"], content["published_location"], content["case_reference"],
                     content["vehicle_registration_raw"], content["vehicle_registration_norm"],
                     content["vehicle_type"], content["vehicle_make"], content["imei"],
                     dumps(content["image_paths"]), dumps(content["attributes"]), dumps(nr.warnings),
                     nr.fingerprint, listing_url, now, now, row["id"]),
                )
                conn.execute(
                    "INSERT INTO zipnet_record_changes (record_id, job_id, changed_at, change_type, changed_fields) "
                    "VALUES (?,?,?,?,?)", (row["id"], job_id, now, "updated", dumps(diff)))
                updated += 1
            else:  # fingerprint scheme changed but content is identical
                conn.execute("UPDATE zipnet_records SET fingerprint=?, last_seen_at=? WHERE id=?",
                             (nr.fingerprint, now, row["id"]))
                unchanged += 1
        else:
            conn.execute("UPDATE zipnet_records SET last_seen_at=? WHERE id=?", (now, row["id"]))
            unchanged += 1
    return new, updated, unchanged


# --- checkpoints, jobs, failures -----------------------------------------------------------------

CHECKPOINT_FIELDS = (
    "backfill_cursor", "backfill_pages", "backfill_complete", "backfill_completed_at", "backfill_limit_reached",
    "high_water_id", "incr_cursor", "incr_pending_hw", "incr_in_progress", "last_incremental_at", "caught_up",
    "indexed_count", "source_total", "last_status", "last_message", "last_success_at", "consecutive_failures",
    "next_attempt_at",
)


def get_checkpoint(conn: sqlite3.Connection, section: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM zipnet_checkpoints WHERE section=?", (section,)).fetchone()
    if row is None:
        conn.execute("INSERT OR IGNORE INTO zipnet_checkpoints (section, updated_at) VALUES (?,?)",
                     (section, now_iso()))
        row = conn.execute("SELECT * FROM zipnet_checkpoints WHERE section=?", (section,)).fetchone()
    return dict(row)


def update_checkpoint(conn: sqlite3.Connection, section: str, **fields: Any) -> None:
    """Caller owns the transaction (so data and checkpoint commit together)."""
    unknown = set(fields) - set(CHECKPOINT_FIELDS)
    if unknown:
        raise ValueError(f"unknown checkpoint fields: {sorted(unknown)}")
    conn.execute("INSERT OR IGNORE INTO zipnet_checkpoints (section, updated_at) VALUES (?,?)", (section, now_iso()))
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE zipnet_checkpoints SET {sets}, updated_at=? WHERE section=?",  # noqa: S608 - allow-listed names
                 [*fields.values(), now_iso(), section])


def start_job(conn: sqlite3.Connection, section: str, mode: str) -> int:
    cur = conn.execute("INSERT INTO zipnet_jobs (section, mode, status, started_at) VALUES (?,?, 'running', ?)",
                       (section, mode, now_iso()))
    return int(cur.lastrowid)


def update_job(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    allowed = {"status", "finished_at", "pages", "requests", "retries", "scanned", "new_records", "updated_records",
               "unchanged_records", "failed_records", "source_total", "stop_reason", "message"}
    if set(fields) - allowed:
        raise ValueError("unknown job fields")
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE zipnet_jobs SET {sets} WHERE id=?", [*fields.values(), job_id])  # noqa: S608


def add_failure(conn: sqlite3.Connection, job_id: int | None, section: str, stage: str, error: str,
                source_id: str | None = None, page_start: int | None = None) -> None:
    """Dead-letter entry. Stores an error message only, never any part of the source payload."""
    conn.execute(
        "INSERT INTO zipnet_failures (job_id, section, source_id, page_start, stage, error, created_at) "
        "VALUES (?,?,?,?,?,?,?)", (job_id, section, source_id, page_start, stage, error[:500], now_iso()))


def recover_stale_jobs(conn: sqlite3.Connection) -> int:
    """Mark jobs left 'running' by a crashed worker as interrupted (checkpoints are untouched)."""
    with transaction(conn):
        cur = conn.execute(
            "UPDATE zipnet_jobs SET status='interrupted', finished_at=?, stop_reason='worker_restart', "
            "message='worker stopped before this job finished; resumed from the last committed checkpoint' "
            "WHERE status='running'", (now_iso(),))
    return cur.rowcount


def prune_jobs(conn: sqlite3.Connection, keep: int = 500) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM zipnet_jobs WHERE id <= (SELECT MAX(id) FROM zipnet_jobs) - ?", (keep,))
        conn.execute("DELETE FROM zipnet_failures WHERE id <= (SELECT COALESCE(MAX(id),0) FROM zipnet_failures) - ?",
                     (keep * 4,))


# --- worker lease -----------------------------------------------------------------------------------

LEASE = "sync-worker"


def acquire_lease(conn: sqlite3.Connection, holder: str, ttl: float, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    with transaction(conn):
        row = conn.execute("SELECT holder, expires_at FROM zipnet_lease WHERE name=?", (LEASE,)).fetchone()
        if row is not None and row["holder"] != holder and row["expires_at"] > now:
            return False
        conn.execute(
            "INSERT INTO zipnet_lease (name, holder, pid, acquired_at, heartbeat_at, expires_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET holder=excluded.holder, pid=excluded.pid, acquired_at=excluded.acquired_at, "
            "heartbeat_at=excluded.heartbeat_at, expires_at=excluded.expires_at",
            (LEASE, holder, os.getpid(), now, now, now + ttl))
    return True


def heartbeat_lease(conn: sqlite3.Connection, holder: str, ttl: float, now: float | None = None) -> bool:
    """Extend the lease. Returns False when it was lost to another holder."""
    now = time.time() if now is None else now
    with transaction(conn):
        cur = conn.execute("UPDATE zipnet_lease SET heartbeat_at=?, expires_at=? WHERE name=? AND holder=?",
                           (now, now + ttl, LEASE, holder))
    return cur.rowcount == 1


def release_lease(conn: sqlite3.Connection, holder: str) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM zipnet_lease WHERE name=? AND holder=?", (LEASE, holder))


def live_lease(conn: sqlite3.Connection, now: float | None = None) -> dict[str, Any] | None:
    now = time.time() if now is None else now
    row = conn.execute("SELECT * FROM zipnet_lease WHERE name=?", (LEASE,)).fetchone()
    if row is None or row["expires_at"] <= now:
        return None
    return dict(row)


# --- control flags ------------------------------------------------------------------------------------


def get_control(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM zipnet_control WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_control(conn: sqlite3.Connection, key: str, value: str | None, actor: str | None = None) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO zipnet_control (key, value, updated_at, updated_by) VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at, "
            "updated_by=excluded.updated_by", (key, value, now_iso(), actor))


def get_worker_state(conn: sqlite3.Connection) -> dict[str, Any]:
    raw = get_control(conn, "worker_state")
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


def set_worker_state(conn: sqlite3.Connection, state: str, **extra: Any) -> None:
    set_control(conn, "worker_state",
                json.dumps({"state": state, "pid": os.getpid(), "updated_at": now_iso(), **extra}), "worker")


# --- queries used by the admin API -----------------------------------------------------------------------

RECORD_COLUMNS = (
    "id", "section", "source_id", "title", "status", "name", "sex", "age_min", "age_max", "reported_date",
    "reported_date_kind", "source_created_date", "state", "district", "police_station", "published_location",
    "case_reference", "vehicle_registration_raw", "vehicle_type", "vehicle_make", "imei", "image_paths",
    "source_listing_url", "first_seen_at", "last_seen_at", "last_changed_at",
)


def search_records(conn: sqlite3.Connection, *, section: str | None = None, q: str | None = None,
                   state: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    where, params = [], []
    if section:
        where.append("section=?")
        params.append(section)
    if state:
        where.append("state=? COLLATE NOCASE")
        params.append(state)
    if q:
        like = "%" + q.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        reg = "".join(ch for ch in q.upper() if ch.isalnum())
        clause = "(name_search LIKE ? ESCAPE '\\' OR LOWER(case_reference) LIKE ? ESCAPE '\\' OR LOWER(title) LIKE ? ESCAPE '\\'"
        params += [like, like, like]
        if reg:
            clause += " OR vehicle_registration_norm LIKE ? ESCAPE '\\'"
            params.append(f"%{reg}%")
        if q.isdigit():
            clause += " OR imei LIKE ? ESCAPE '\\'"
            params.append(like)
        where.append(clause + ")")
    sql = f"SELECT {', '.join(RECORD_COLUMNS)} FROM zipnet_records"  # noqa: S608 - fixed column list
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY source_created_date DESC, id DESC LIMIT ? OFFSET ?"
    rows = conn.execute(sql, [*params, limit, offset]).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["image_paths"] = json.loads(d["image_paths"])
        out.append(d)
    return out


PANEL_STATUS = {"open": "Untraced", "traced": "Traced"}
COUNT_CAP = 10_000


def _like(value: str) -> str:
    return "%" + value.lower().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%"


def panel_query(conn: sqlite3.Connection, *, section: str = "missing_persons", q: str | None = None,
                state: str | None = None, status: str = "open", photo_only: bool = False, sex: str | None = None,
                limit: int = 24, offset: int = 0) -> dict[str, Any]:
    """Records for the public missing-person panel. Allow-listed columns only; the total is capped for speed."""
    where, params = ["section=?"], [section]
    if status in PANEL_STATUS:
        where.append("status=? COLLATE NOCASE")
        params.append(PANEL_STATUS[status])
    if state:
        where.append("state=? COLLATE NOCASE")
        params.append(state)
    if sex:
        where.append("sex=? COLLATE NOCASE")
        params.append(sex)
    if photo_only:
        where.append("image_paths != '[]'")
    if q:
        pat = _like(q)
        where.append("(name_search LIKE ? ESCAPE '\\' OR LOWER(published_location) LIKE ? ESCAPE '\\' "
                     "OR LOWER(police_station) LIKE ? ESCAPE '\\' OR LOWER(district) LIKE ? ESCAPE '\\')")
        params += [pat, pat, pat, pat]
    clause = " AND ".join(where)
    cols = ("id, source_id, title, name, sex, status, reported_date, source_created_date, state, district, "
            "police_station, published_location, image_paths, source_listing_url, first_seen_at, "
            "json_extract(attributes, '$.birth_year_reported') AS birth_year")
    rows = conn.execute(
        f"SELECT {cols} FROM zipnet_records WHERE {clause} "  # noqa: S608 - fixed fragments, bound parameters
        "ORDER BY source_created_date DESC, source_id_num DESC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM (SELECT 1 FROM zipnet_records WHERE {clause} LIMIT ?)",  # noqa: S608
        [*params, COUNT_CAP + 1]).fetchone()[0]
    items = []
    for r in rows:
        d = dict(r)
        d["image_paths"] = json.loads(d["image_paths"])
        items.append(d)
    return {"items": items, "total": min(total, COUNT_CAP), "total_capped": total > COUNT_CAP}


def panel_facts(conn: sqlite3.Connection, section: str = "missing_persons") -> dict[str, Any]:
    cp = get_checkpoint(conn, section)
    newest = conn.execute("SELECT COALESCE(MAX(id), 0) FROM zipnet_records WHERE section=?", (section,)).fetchone()[0]
    return {"indexed": cp["indexed_count"], "source_total": cp["source_total"],
            "backfill_complete": bool(cp["backfill_complete"]), "last_success_at": cp["last_success_at"],
            "newest_id": newest}


def states_for_panel(conn: sqlite3.Connection, section: str = "missing_persons") -> list[str]:
    rows = conn.execute("SELECT state FROM zipnet_records WHERE section=? AND state IS NOT NULL "
                        "GROUP BY state ORDER BY COUNT(*) DESC LIMIT 60", (section,)).fetchall()
    return [r[0] for r in rows]


def link_record_to_case(conn: sqlite3.Connection, record_id: int, case_id: int, linked_by: str,
                        note: str | None = None) -> bool:
    """Record an explicit, human-made mapping. Returns False if it already existed (idempotent)."""
    with transaction(conn):
        if conn.execute("SELECT 1 FROM zipnet_records WHERE id=?", (record_id,)).fetchone() is None:
            raise LookupError("record not found")
        cur = conn.execute(
            "INSERT OR IGNORE INTO zipnet_case_links (record_id, case_id, linked_by, linked_at, note) "
            "VALUES (?,?,?,?,?)", (record_id, case_id, linked_by, now_iso(), (note or "")[:300] or None))
    return cur.rowcount == 1
