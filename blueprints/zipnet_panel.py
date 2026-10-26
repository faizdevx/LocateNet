"""Missing-person panel for imported ZIPNET records: list with photos, filters and live-update hint.

These are external public records, shown read-only and labelled as such. They are not LocateNet cases,
are never face-matched, and a photograph here is never a verified current photograph.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from flask import Blueprint, Response, abort, jsonify, redirect, render_template, request, session, url_for

import config
from services import zipnet_photos
from services import zipnet_store as store
from services.zipnet_settings import ZipnetSettings

log = logging.getLogger("locatenet.zipnet.panel")

bp = Blueprint("zipnet_panel", __name__, url_prefix="/zipnet-records")
PAGE_SIZE = 24
SECTION = "missing_persons"


@bp.before_request
def _access():
    if not config.ZIPNET_PANEL_PUBLIC and "user" not in session:
        if request.path.endswith(("/updates",)) or "/photo/" in request.path:
            abort(401)
        return redirect(url_for("login"))
    return None


def _conn() -> sqlite3.Connection:
    conn = store.connect()
    store.migrate(conn)
    return conn


def _age_text(birth_year) -> str | None:
    try:
        by = int(birth_year)
    except (TypeError, ValueError):
        return None
    now = datetime.now().year
    if not 1900 <= by <= now:
        return None
    return f"about {now - by} (born {by}, as reported)"


def _card(r: dict) -> dict:
    return {
        "id": r["id"],
        "name": r["name"] or "Name not published",
        "sex": r["sex"],
        "age": _age_text(r["birth_year"]),
        "status": r["status"],
        "missing_since": r["reported_date"],
        "entered_on": r["source_created_date"],
        "place": r["published_location"],
        "district": r["district"],
        "state": r["state"],
        "police_station": r["police_station"],
        "has_photo": bool(r["image_paths"]),
        "photo_count": len(r["image_paths"]),
        "source_url": r["source_listing_url"],
        "source_id": r["source_id"],
    }


@bp.get("")
@bp.get("/")
def panel():
    status = request.args.get("status", "open")
    status = status if status in ("open", "traced", "all") else "open"
    q = (request.args.get("q") or "").strip()[:100] or None
    state = (request.args.get("state") or "").strip()[:100] or None
    sex = (request.args.get("sex") or "").strip()[:10] or None
    photo_only = request.args.get("photo") == "1"
    page = max(1, min(request.args.get("page", 1, type=int), 500))
    error = None
    result = {"items": [], "total": 0, "total_capped": False}
    facts = {"indexed": 0, "source_total": None, "backfill_complete": False, "last_success_at": None, "newest_id": 0}
    states: list[str] = []
    try:
        conn = _conn()
        try:
            result = store.panel_query(conn, section=SECTION, q=q, state=state, status=status, photo_only=photo_only,
                                       sex=sex, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
            facts = store.panel_facts(conn, SECTION)
            states = store.states_for_panel(conn, SECTION)
        finally:
            conn.close()
    except sqlite3.Error as e:
        log.error("panel database error: %s", type(e).__name__)
        error = "The imported records are temporarily unavailable."
    st = ZipnetSettings.from_config()
    pages = max(1, -(-result["total"] // PAGE_SIZE))
    return render_template(
        "zipnet_missing.html", cards=[_card(r) for r in result["items"]], total=result["total"],
        total_capped=result["total_capped"], page=page, pages=pages, q=q or "", state=state or "", status=status,
        sex=sex or "", photo_only=photo_only, states=states, facts=facts, sync_enabled=st.enabled, error=error,
        shown_newest_id=facts["newest_id"])


@bp.get("/photo/<int:record_id>")
def photo(record_id: int):
    """First published photograph of a record, streamed for display only."""
    try:
        conn = _conn()
        try:
            row = conn.execute("SELECT image_paths FROM zipnet_records WHERE id=? AND section=?",
                               (record_id, SECTION)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        abort(503)
    if row is None:
        abort(404)
    import json

    paths = json.loads(row["image_paths"])
    if not paths:
        abort(404)
    got = zipnet_photos.fetch_photo(ZipnetSettings.from_config(), paths[0])
    if got is None:
        abort(404)
    ctype, data = got
    return Response(data, mimetype=ctype, headers={"Cache-Control": "public, max-age=3600",
                                                    "X-Content-Type-Options": "nosniff"})


@bp.get("/updates")
def updates():
    """Cheap poll used by the panel to tell the viewer that new records have arrived."""
    since = max(0, request.args.get("newest", 0, type=int))
    try:
        conn = _conn()
        try:
            facts = store.panel_facts(conn, SECTION)
            new = conn.execute("SELECT COUNT(*) FROM zipnet_records WHERE section=? AND id>? AND status=? COLLATE NOCASE",
                               (SECTION, since, "Untraced")).fetchone()[0] if since else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return jsonify(error="unavailable"), 503
    return jsonify(newest_id=facts["newest_id"], new_records=new, indexed=facts["indexed"],
                   last_success_at=facts["last_success_at"], backfill_complete=facts["backfill_complete"])
