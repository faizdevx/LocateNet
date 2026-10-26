"""Admin endpoints for the ZIPNET sync: status, start, stop, jobs, records, export, case links.

Every route needs an authenticated Admin. State-changing routes are POST and protected by the
app-wide CSRF protection (send the token in the ``X-CSRFToken`` header). Responses contain only
allow-listed record fields, counts and short status messages: never source payloads, file paths,
secrets or stack traces.
"""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
import time
from functools import wraps

from flask import Blueprint, Response, abort, jsonify, redirect, render_template, request, session, url_for

from models import get_db_connection, log_audit
from services import zipnet_store as store
from services.zipnet_adapters import ADAPTERS
from services.zipnet_settings import ZipnetSettings

log = logging.getLogger("locatenet.zipnet.admin")

bp = Blueprint("zipnet_sync", __name__, url_prefix="/admin/zipnet-sync")

SECTION_LABELS = {k: v.label for k, v in ADAPTERS.items()}
EXPORT_MAX_ROWS = 5000


def admin_api(f):
    """401 for anonymous callers, 403 for non-admins, JSON errors instead of redirects."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return jsonify(error="authentication required"), 401
        if session.get("role") != "Admin":
            return jsonify(error="administrator role required"), 403
        return f(*args, **kwargs)

    return wrapper


def admin_page(f):
    """Same rules as ``admin_required`` in app.py: login redirect, 403 for non-admins."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "Admin":
            abort(403)
        return f(*args, **kwargs)

    return wrapper


def _conn() -> sqlite3.Connection:
    conn = store.connect()
    store.migrate(conn)
    return conn


def _settings() -> ZipnetSettings:
    return ZipnetSettings.from_config()


def _db_error(exc: Exception):
    log.error("zipnet admin database error: %s", type(exc).__name__)
    return jsonify(error="the sync database is unavailable"), 503


def _section_view(st: ZipnetSettings, section: str, cp: dict) -> dict:
    cap = st.backfill_max_pages
    limit_hit = bool(cap and cp["backfill_pages"] >= cap and not cp["backfill_complete"])
    total = cp["source_total"]
    cursor = cp["backfill_cursor"]
    if cp["backfill_complete"]:
        phase = "incremental"
    elif limit_hit:
        phase = "backfill_incomplete_page_limit"
    else:
        phase = "backfill"
    progress = None
    if cp["backfill_complete"]:
        progress = 1.0
    elif total:
        progress = round(min(cursor / total, 1.0), 4)
    failing = cp["last_status"] in ("incomplete", "failed", "blocked", "selector_changed")
    return {
        "section": section,
        "label": SECTION_LABELS.get(section, section),
        "phase": phase,
        "backfill_complete": bool(cp["backfill_complete"]),
        "backfill_progress": progress,
        "backfill_offset": cursor,
        "backfill_pages": cp["backfill_pages"],
        "backfill_page_limit_reached": limit_hit,
        "indexed_records": cp["indexed_count"],
        "source_total": total,
        "incremental_caught_up": bool(cp["caught_up"]),
        "incremental_in_progress": bool(cp["incr_in_progress"]),
        "last_incremental_at": cp["last_incremental_at"],
        "last_success_at": cp["last_success_at"],
        "last_status": cp["last_status"],
        "last_message": cp["last_message"],
        "consecutive_failures": cp["consecutive_failures"],
        "retry_after_seconds": max(0, round(cp["next_attempt_at"] - time.time())) if cp["next_attempt_at"] else 0,
        "warning": cp["last_message"] if failing and cp["last_message"] else None,
    }


def build_status(conn: sqlite3.Connection, st: ZipnetSettings) -> dict:
    lease = store.live_lease(conn)
    state = store.get_worker_state(conn)
    paused = store.get_control(conn, "paused") == "1"
    worker_alive = lease is not None
    worker_state = state.get("state") if worker_alive else "not_running"
    sections = [_section_view(st, s, store.get_checkpoint(conn, s)) for s in st.sections]
    last = conn.execute("SELECT MAX(last_success_at) FROM zipnet_checkpoints").fetchone()[0]
    warnings = [f"{s['label']}: {s['warning']}" for s in sections if s["warning"]]
    warnings += [f"{s['label']}: historical backfill stopped at the configured page limit "
                 f"({s['backfill_pages']} pages) and is NOT complete; raise ZIPNET_BACKFILL_MAX_PAGES to continue."
                 for s in sections if s["backfill_page_limit_reached"]]
    if st.enabled and not worker_alive and not paused:
        warnings.append("No sync worker is running. Start it with `python zipnet_worker.py` "
                        "(or restart `python app.py`).")
    if not st.enabled:
        warnings.append("Synchronization is disabled (ZIPNET_SYNC_ENABLED=0).")
    return {
        "enabled": st.enabled,
        "source": st.base_url,
        "worker": {
            "alive": worker_alive,
            "state": worker_state,
            "current_section": state.get("section") if worker_alive else None,
            "current_mode": state.get("mode") if worker_alive else None,
            "message": state.get("message") if worker_alive else None,
            "next_incremental_at": state.get("next_incremental_at") if worker_alive else None,
        },
        "stop_requested": paused,
        # "stopped" is claimed only after the worker itself acknowledged the request.
        "stop_acknowledged": bool(paused and worker_alive and worker_state == "paused"),
        "last_successful_sync_at": last,
        "interval_minutes": st.incremental_interval_minutes,
        "sections": sections,
        "warnings": warnings,
    }


@bp.get("")
@bp.get("/")
@admin_page
def page():
    return render_template("zipnet_sync.html", user=session.get("user"))


@bp.get("/status")
@admin_api
def status():
    try:
        conn = _conn()
        try:
            return jsonify(build_status(conn, _settings()))
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _db_error(e)


@bp.post("/start")
@admin_api
def start():
    """Clear a stop request and ask for a sync turn now. Idempotent."""
    st = _settings()
    if not st.enabled:
        return jsonify(error="synchronization is disabled (ZIPNET_SYNC_ENABLED=0)"), 409
    try:
        conn = _conn()
        try:
            was_paused = store.get_control(conn, "paused") == "1"
            store.set_control(conn, "paused", "0", session.get("user"))
            store.set_control(conn, "run_now_at", store.now_iso(), session.get("user"))
            spawned = False
            if not store.live_lease(conn):
                from services.zipnet_supervisor import ensure_worker_started

                spawned = ensure_worker_started(st) is not None
            log_audit(session.get("user"), "zipnet_sync_start", details=f"resumed={was_paused} spawned={spawned}")
            body = build_status(conn, st)
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _db_error(e)
    body["accepted"] = True
    body["worker_spawned"] = spawned
    return jsonify(body), 202


@bp.post("/stop")
@admin_api
def stop():
    """Request a stop at the next page boundary. Idempotent; 'acknowledged' only after the worker confirms."""
    try:
        conn = _conn()
        try:
            already = store.get_control(conn, "paused") == "1"
            if not already:
                store.set_control(conn, "paused", "1", session.get("user"))
                log_audit(session.get("user"), "zipnet_sync_stop", details="requested")
            body = build_status(conn, _settings())
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _db_error(e)
    body["accepted"] = True
    return jsonify(body), (200 if body["stop_acknowledged"] or not body["worker"]["alive"] else 202)


@bp.get("/jobs")
@admin_api
def jobs():
    limit = max(1, min(request.args.get("limit", 25, type=int), 200))
    try:
        conn = _conn()
        try:
            rows = conn.execute("SELECT * FROM zipnet_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            fails = conn.execute(
                "SELECT job_id, section, source_id, page_start, stage, error, created_at FROM zipnet_failures "
                "ORDER BY id DESC LIMIT 50").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _db_error(e)
    return jsonify(jobs=[dict(r) for r in rows], recent_failures=[dict(f) for f in fails])


def _search_args():
    section = request.args.get("section")
    if section and section not in SECTION_LABELS:
        return None
    return {"section": section, "q": (request.args.get("q") or "")[:100] or None,
            "state": (request.args.get("state") or "")[:100] or None}


@bp.get("/records")
@admin_api
def records():
    args = _search_args()
    if args is None:
        return jsonify(error="unknown section"), 422
    limit = max(1, min(request.args.get("limit", 50, type=int), 200))
    offset = max(0, request.args.get("offset", 0, type=int))
    try:
        conn = _conn()
        try:
            items = store.search_records(conn, limit=limit, offset=offset, **args)
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _db_error(e)
    return jsonify(items=items, limit=limit, offset=offset,
                   note="External public records. A record is not a LocateNet case and never establishes identity.")


@bp.get("/export.csv")
@admin_api
def export_csv():
    args = _search_args()
    if args is None:
        return jsonify(error="unknown section"), 422
    limit = max(1, min(request.args.get("limit", 1000, type=int), EXPORT_MAX_ROWS))
    try:
        conn = _conn()
        try:
            items = store.search_records(conn, limit=limit, **args)
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _db_error(e)
    cols = [c for c in store.RECORD_COLUMNS if c != "image_paths"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for it in items:
        # Neutralise spreadsheet formula injection.
        w.writerow([("'" + str(it[c])) if str(it[c] or "")[:1] in ("=", "+", "-", "@") else it[c] for c in cols])
    log_audit(session.get("user"), "zipnet_export", details=f"rows={len(items)} section={args['section'] or 'all'}")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="zipnet-records.csv"'})


@bp.post("/records/<int:record_id>/link")
@admin_api
def link(record_id: int):
    """Explicit human mapping of an external record to a native case. Changes nothing else."""
    data = request.get_json(silent=True) or {}
    case_id = data.get("case_id")
    if not isinstance(case_id, int) or isinstance(case_id, bool) or case_id <= 0:
        return jsonify(error="case_id (positive integer) is required"), 422
    with get_db_connection() as main:
        if main.execute("SELECT 1 FROM cases WHERE id=?", (case_id,)).fetchone() is None:
            return jsonify(error="case not found"), 404
    try:
        conn = _conn()
        try:
            created = store.link_record_to_case(conn, record_id, case_id, session.get("user"),
                                                str(data.get("note") or "")[:300] or None)
        finally:
            conn.close()
    except LookupError:
        return jsonify(error="record not found"), 404
    except sqlite3.Error as e:
        return _db_error(e)
    if created:
        log_audit(session.get("user"), "zipnet_link_case", case_id=case_id, target_id=record_id,
                  details="external record linked by an administrator; no enrolment or status change")
    return jsonify(linked=True, created=created), (201 if created else 200)
