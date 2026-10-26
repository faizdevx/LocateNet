"""Admin API: authorization, CSRF, idempotent controls, honest status, canary (prohibited field) tests."""

from __future__ import annotations

import logging
import re
import sqlite3

import pytest

import config
import models
from services import zipnet_store as store
from services import zipnet_supervisor as sup
from services.zipnet_adapters import ADAPTERS
from conftest import add_case, csrf_token, db_rows, login, make_user
from tests.zipnet_fakes import Env, FakeTransport, RunningWorker, all_sections, json_response, rows_for, wait_for
from tests.zipnet_helpers import CANARIES, make_row

BASE = "/admin/zipnet-sync"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(config, "ZIPNET_SYNC_ENABLED", True)


def admin_client(flask_app):
    make_user("boss", role="Admin")
    c = flask_app.test_client()
    login(c, "boss")
    return c


def ajax_token(client):
    html = client.get(BASE).get_data(as_text=True)
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


@pytest.fixture
def admin(flask_app, zdb):
    c = admin_client(flask_app)
    c.csrf = ajax_token(c)
    return c


def post(c, path, **kw):
    return c.post(BASE + path, headers={"X-CSRFToken": c.csrf}, **kw)


# --- authorization and CSRF -------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/status", "/jobs", "/records", "/export.csv"])
def test_anonymous_visitors_cannot_read_anything(client, zdb, path):
    assert client.get(BASE + path).status_code == 401


def test_anonymous_visitors_cannot_start_or_stop_the_worker(client, zdb):
    for path in ("/start", "/stop"):
        assert client.post(BASE + path).status_code in (400, 401)  # CSRF or auth rejects first
        token = csrf_token(client)
        assert client.post(BASE + path, headers={"X-CSRFToken": token}).status_code == 401
    assert store.get_control(store.connect(), "paused") is None  # nothing changed


def test_the_html_page_redirects_anonymous_users_to_login(client, zdb):
    assert client.get(BASE).status_code == 302


def test_officers_are_forbidden(flask_app, zdb):
    make_user("off1", role="Officer")
    c = flask_app.test_client()
    login(c, "off1")
    assert c.get(BASE + "/status").status_code == 403
    assert c.get(BASE).status_code == 403
    token = csrf_token(c)
    for path in ("/start", "/stop"):
        assert c.post(BASE + path, headers={"X-CSRFToken": token}).status_code == 403


def test_state_changing_requests_need_a_csrf_token(admin):
    assert admin.post(BASE + "/stop").status_code == 400
    assert admin.post(BASE + "/start").status_code == 400
    assert admin.post(BASE + "/records/1/link", json={"case_id": 1}).status_code == 400
    assert store.get_control(store.connect(), "paused") is None
    assert post(admin, "/stop").status_code in (200, 202)


# --- status, start, stop -----------------------------------------------------------------------------------------


def test_status_reports_an_empty_installation_honestly(admin):
    body = admin.get(BASE + "/status").get_json()
    assert body["worker"]["alive"] is False and body["worker"]["state"] == "not_running"
    assert body["last_successful_sync_at"] is None
    assert len(body["sections"]) == 6
    s = body["sections"][0]
    assert s["phase"] == "backfill" and s["backfill_complete"] is False and s["indexed_records"] == 0
    assert any("No sync worker is running" in w for w in body["warnings"])


def test_stop_is_idempotent_and_not_acknowledged_without_a_worker_ack(admin, zdb):
    w = RunningWorker(zdb, request_delay=0.5)
    w.svc.exit_token = None
    w.start()
    try:
        assert wait_for(lambda: w.state() == "idle")
        r1 = post(admin, "/stop")
        assert r1.status_code in (200, 202) and r1.get_json()["stop_requested"] is True
        assert wait_for(lambda: w.state() == "paused", 15)
        r2 = post(admin, "/stop")
        assert r2.status_code == 200 and r2.get_json()["stop_acknowledged"] is True
        assert models.get_db_connection  # audit trail: only the first request was logged
        with models.get_db_connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='zipnet_sync_stop'").fetchone()[0] == 1
    finally:
        w.stop()


def test_stop_does_not_claim_acknowledgement_before_the_worker_confirms(admin, zdb):
    conn = store.connect(zdb)
    store.acquire_lease(conn, "busy-worker", 60)  # alive, but never acknowledges
    store.set_worker_state(conn, "running", section="missing_persons", mode="backfill")
    body = post(admin, "/stop").get_json()
    assert body["stop_requested"] is True and body["stop_acknowledged"] is False


def test_start_clears_the_stop_request_and_asks_for_a_sync_now(admin, zdb, monkeypatch):
    store.set_control(store.connect(zdb), "paused", "1", "admin")
    monkeypatch.setattr(sup, "ensure_worker_started", lambda st=None, **k: object())
    r = post(admin, "/start")
    assert r.status_code == 202 and r.get_json()["stop_requested"] is False and r.get_json()["worker_spawned"] is True
    conn = store.connect(zdb)
    assert store.get_control(conn, "paused") == "0" and store.get_control(conn, "run_now_at")
    assert post(admin, "/start").status_code == 202  # idempotent


def test_start_does_not_spawn_a_second_worker_when_one_is_alive(admin, zdb, monkeypatch):
    store.acquire_lease(store.connect(zdb), "alive", 60)
    called = []
    monkeypatch.setattr(sup, "ensure_worker_started", lambda *a, **k: called.append(1))
    assert post(admin, "/start").get_json()["worker_spawned"] is False and not called


def test_start_is_refused_while_sync_is_disabled(admin, monkeypatch):
    monkeypatch.setattr(config, "ZIPNET_SYNC_ENABLED", False)
    r = post(admin, "/start")
    assert r.status_code == 409 and "disabled" in r.get_json()["error"]
    assert any("disabled" in w for w in admin.get("/admin/zipnet-sync/status").get_json()["warnings"])


def test_status_surfaces_progress_incomplete_backfills_and_failures(admin, zdb):
    e = Env(zdb, FakeTransport({"missing_persons": {0: {"recordsTotal": 40, "data": rows_for("missing_persons", 10)}}}),
            settings={"sections": ("missing_persons", "stolen_vehicles"), "pages_per_turn": 5,
                      "backfill_max_pages": 0})
    e.transport.hook = lambda url, data: json_response({}, 503) if data and "GetMissingVehicles" in url else None
    e.drain()
    body = admin.get(BASE + "/status").get_json()
    mp = next(s for s in body["sections"] if s["section"] == "missing_persons")
    assert mp["backfill_complete"] is False and mp["last_status"] == "incomplete" and mp["backfill_offset"] == 10
    assert mp["backfill_progress"] == 0.25 and mp["source_total"] == 40 and mp["indexed_records"] == 10
    sv = next(s for s in body["sections"] if s["section"] == "stolen_vehicles")
    assert sv["last_status"] == "failed" and sv["consecutive_failures"] == 1 and sv["retry_after_seconds"] > 0
    assert any("Stolen Vehicles" in w for w in body["warnings"])
    jobs = admin.get(BASE + "/jobs").get_json()
    assert {j["status"] for j in jobs["jobs"]} >= {"incomplete", "failed"}


def test_page_limit_is_visible_as_an_incomplete_backfill(admin, zdb, monkeypatch):
    monkeypatch.setattr(config, "ZIPNET_BACKFILL_MAX_PAGES", 1)
    e = Env(zdb, FakeTransport({"missing_persons": {0: {"recordsTotal": 40, "data": rows_for("missing_persons", 10)}}}),
            settings={"sections": ("missing_persons",), "backfill_max_pages": 1})
    e.drain()
    sec = admin.get(BASE + "/status").get_json()["sections"][0]
    assert sec["phase"] == "backfill_incomplete_page_limit" and sec["backfill_page_limit_reached"] is True
    assert sec["backfill_complete"] is False
    assert any("NOT complete" in w for w in admin.get(BASE + "/status").get_json()["warnings"])


def test_database_errors_do_not_leak_paths_or_stack_traces(admin, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("unable to open database file C:/secret/dir/zipnet.db")

    monkeypatch.setattr(store, "connect", boom)
    r = admin.get(BASE + "/status")
    assert r.status_code == 503
    text = r.get_data(as_text=True)
    assert "secret" not in text and "Traceback" not in text and "zipnet.db" not in text


# --- records, export, link ----------------------------------------------------------------------------------------------


def test_record_search_is_validated_and_bounded(admin, zdb):
    Env(zdb, FakeTransport(all_sections(12))).drain()
    assert admin.get(BASE + "/records?section=nope").status_code == 422
    body = admin.get(BASE + "/records?section=missing_persons&limit=5").get_json()
    assert len(body["items"]) == 5 and "never establishes identity" in body["note"]
    assert len(admin.get(BASE + "/records?limit=9999").get_json()["items"]) <= 200
    hits = admin.get(BASE + "/records?section=missing_persons&q=test person 95").get_json()["items"]
    assert [h["name"] for h in hits] == ["TEST PERSON 95"]
    assert admin.get(BASE + "/records?q=%25").get_json()["items"] == []  # LIKE wildcards are escaped
    assert admin.get(BASE + "/records?q=' OR 1=1 --").get_json()["items"] == []


def test_export_is_audited_capped_and_formula_safe(admin, zdb):
    Env(zdb, FakeTransport({"missing_persons": {0: {"recordsTotal": 3, "data": [
        make_row("missing_persons", 3, Name="=HYPERLINK(1)"), make_row("missing_persons", 2), make_row("missing_persons", 1)]}}}),
        settings={"sections": ("missing_persons",)}).drain()
    r = admin.get(BASE + "/export.csv?section=missing_persons&limit=2")
    assert r.mimetype == "text/csv" and len(r.get_data(as_text=True).strip().splitlines()) == 3
    full = admin.get(BASE + "/export.csv").get_data(as_text=True)
    assert "'=HYPERLINK" in full and "\n=HYPERLINK" not in full and ",=HYPERLINK" not in full
    with models.get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='zipnet_export'").fetchone()[0] == 2


def test_linking_an_external_record_to_a_case_is_explicit_and_changes_nothing_else(admin, zdb):
    Env(zdb, FakeTransport({"missing_persons": {0: {"recordsTotal": 1, "data": [make_row("missing_persons", 1)]}}}),
        settings={"sections": ("missing_persons",)}).drain()
    case_id = add_case()
    before = (db_rows("SELECT * FROM cases"), db_rows("SELECT COUNT(*) FROM case_embeddings"),
              db_rows("SELECT COUNT(*) FROM matches"))
    rid = admin.get(BASE + "/records").get_json()["items"][0]["id"]
    assert post(admin, f"/records/{rid}/link", json={"case_id": case_id, "note": "same family reported"}).status_code == 201
    assert post(admin, f"/records/{rid}/link", json={"case_id": case_id}).status_code == 200  # idempotent
    assert post(admin, f"/records/{rid}/link", json={"case_id": 99999}).status_code == 404
    assert post(admin, "/records/99999/link", json={"case_id": case_id}).status_code == 404
    assert post(admin, f"/records/{rid}/link", json={"case_id": "x"}).status_code == 422
    after = (db_rows("SELECT * FROM cases"), db_rows("SELECT COUNT(*) FROM case_embeddings"),
             db_rows("SELECT COUNT(*) FROM matches"))
    assert before == after  # no status change, no enrolment, no alert
    assert store.connect(zdb).execute("SELECT COUNT(*) FROM zipnet_case_links").fetchone()[0] == 1
    with models.get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='zipnet_link_case'").fetchone()[0] == 1


# --- prohibited-field canaries -------------------------------------------------------------------------------------------------


def dump_all_tables(path):
    conn = sqlite3.connect(path)
    try:
        return "\n".join(conn.iterdump())
    finally:
        conn.close()


def test_prohibited_fields_never_reach_storage_logs_api_responses_or_exports(admin, zdb, caplog):
    caplog.set_level(logging.DEBUG)
    pages = all_sections(12)
    # a row the pipeline must dead-letter, still full of canaries
    bad = make_row("missing_persons", 7)
    bad["MissingPersonId"] = None
    pages["missing_persons"][10]["data"].append(bad)
    for page in pages["missing_persons"].values():
        page["recordsTotal"] = 13
    # free text containing a phone number must be redacted, not stored
    pages["missing_persons"][0]["data"][1]["MissingFrom"] = "near the market, call 9123456789 for details"
    e = Env(zdb, FakeTransport(pages))
    e.transport.hook = lambda url, data: None
    e.drain()
    assert e.count() > 60

    outputs = {"database": dump_all_tables(zdb), "logs": caplog.text}
    for path in ("/status", "/jobs", "/records", "/records?section=missing_persons", "/export.csv"):
        outputs[path] = admin.get(BASE + path).get_data(as_text=True)
    outputs["page"] = admin.get(BASE).get_data(as_text=True)
    for where, text in outputs.items():
        for canary in CANARIES + ["9123456789", "SECRET LANE", "CANARYRELIGION", "CANARYENGINE"]:
            assert canary not in text, f"{canary!r} leaked into {where}"
    assert "[redacted]" in outputs["database"]  # the phone number in free text was masked
    assert e.conn.execute("SELECT COUNT(*) FROM zipnet_failures").fetchone()[0] >= 1  # dead letter kept, payload-free


def test_a_failed_request_never_logs_the_source_payload(zdb, caplog):
    caplog.set_level(logging.DEBUG)
    leaky = make_row("missing_persons", 1)
    e = Env(zdb, FakeTransport(hook=lambda url, data: json_response({"data": [leaky], "recordsTotal": "x"}, 500)),
            settings={"sections": ("missing_persons",)})
    e.drain()
    assert "CANARY" not in caplog.text and "9876500001" not in caplog.text


def test_sync_leaves_native_tables_and_the_face_index_untouched(zdb):
    case_id = add_case()
    tables = [r[0] for r in db_rows("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    counts = {t: db_rows(f"SELECT COUNT(*) FROM {t}")[0][0] for t in tables}  # noqa: S608
    Env(zdb, FakeTransport(all_sections(15))).drain()
    after_tables = [r[0] for r in db_rows("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    assert after_tables == tables and not any(t.startswith("zipnet_") for t in after_tables)
    assert {t: db_rows(f"SELECT COUNT(*) FROM {t}")[0][0] for t in tables} == counts  # noqa: S608
    assert db_rows("SELECT status FROM cases WHERE id=?", (case_id,)) == [("NF",)]  # never marks a case found


def test_init_is_safe_on_an_existing_populated_installation(zdb, capsys):
    import init_system

    add_case()
    Env(zdb, FakeTransport(all_sections(5))).drain()
    before = store.connect(zdb).execute("SELECT COUNT(*) FROM zipnet_records").fetchone()[0]
    init_system.init_zipnet_store()
    init_system.init_zipnet_store()
    assert "schema v" in capsys.readouterr().out
    assert store.connect(zdb).execute("SELECT COUNT(*) FROM zipnet_records").fetchone()[0] == before
    assert db_rows("SELECT COUNT(*) FROM cases")[0][0] == 1


def test_adapters_declare_every_section_and_nothing_is_stored_outside_the_allow_list(zdb):
    assert set(ADAPTERS) == {"missing_persons", "unidentified_persons", "unidentified_dead_bodies",
                             "stolen_vehicles", "unclaimed_seized_vehicles", "missing_mobiles"}
    cols = {r[1] for r in store.connect(zdb).execute("PRAGMA table_info(zipnet_records)")}
    forbidden = ("address", "phone", "mobile", "email", "religion", "guardian", "parent", "payload", "complainant")
    assert not [c for c in cols if any(f in c for f in forbidden) or c in ("raw", "json", "body")]
