"""ZIPNET sync: backfill, resume, idempotency, change detection, retries, stop, catch-up, page limits."""

from __future__ import annotations

import sqlite3

import pytest

from pipeline import zipnet_pipeline
from pipeline.zipnet_pipeline import run_section
from services import zipnet_store as store
from services.zipnet_adapters import ADAPTERS, get_adapter
from services.zipnet_scraper import RawResponse, SourceClient, TransportError
from tests.zipnet_fakes import Env, FakeTransport, all_sections, json_response, make_settings, pages_of, rows_for
from tests.zipnet_helpers import make_row

MP = "missing_persons"
BASE = "https://zipnet.delhipolice.gov.in"


@pytest.fixture
def env(zdb):
    def make(pages=None, **kw):
        over = kw.pop("settings", {})
        return Env(zdb, FakeTransport(pages or {}, **kw), **over)

    return make


# --- backfill -------------------------------------------------------------------------------------------


def test_initial_backfill_imports_every_section_in_bounded_turns(env):
    e = env(all_sections(25))
    results = e.drain()
    backfills = [r for r in results if r.mode == "backfill"]
    assert all(r.pages <= 2 for r in backfills)  # pages_per_turn bound
    assert len({r.section for r in backfills}) == 6
    for s in ADAPTERS:
        cp = e.cp(s)
        assert cp["backfill_complete"] == 1 and cp["indexed_count"] == 25 == e.count(s)
        assert cp["backfill_cursor"] == 25 and cp["source_total"] == 25 and cp["last_status"] == "completed"


def test_backfill_rotates_sections_so_none_starves(env):
    e = env(all_sections(60))
    order = []
    for _ in range(6):
        order.append(e.svc.step(e.conn).section)
    assert order == list(ADAPTERS)  # one turn each before any section gets a second turn


def test_resume_after_interruption_continues_from_last_committed_page(env, monkeypatch):
    pages = {MP: pages_of(rows_for(MP, 50))}
    crash_at = {"n": 0}

    def hook(url, data):
        if data and data["start"] == "30":
            crash_at["n"] += 1
            if crash_at["n"] == 1:
                return RuntimeError("simulated crash")
        return None

    e = env(pages, hook=hook, settings={"sections": (MP,), "pages_per_turn": 10})
    first = e.svc.step(e.conn)
    assert first.status == "failed" and first.stop_reason == "internal_error"
    assert e.cp(MP)["backfill_cursor"] == 30 and e.count() == 30 and not e.cp(MP)["backfill_complete"]
    e.conn.execute("UPDATE zipnet_checkpoints SET next_attempt_at=NULL")
    calls_before = len(e.transport.calls)
    second = e.svc.step(e.conn)
    assert second.status == "completed" and e.count() == 50
    first_post = next(c for c in e.transport.calls[calls_before:] if c[0] == "POST")
    assert first_post[2]["start"] == "30"  # did not start over


def test_failed_page_write_rolls_back_the_page_and_keeps_the_checkpoint(env, monkeypatch):
    e = env({MP: pages_of(rows_for(MP, 30))}, settings={"sections": (MP,), "pages_per_turn": 10})
    real = store.upsert_records
    calls = {"n": 0}

    def flaky(conn, *a, **k):
        calls["n"] += 1
        out = real(conn, *a, **k)
        if calls["n"] == 2:
            raise sqlite3.OperationalError("disk I/O error")
        return out

    monkeypatch.setattr(store, "upsert_records", flaky)
    r = e.svc.step(e.conn)
    assert r.status == "failed" and r.stop_reason == "persist_error"
    assert e.count() == 10  # page 2 was rolled back as a whole, page 1 survived
    assert e.cp(MP)["backfill_cursor"] == 10 and e.cp(MP)["indexed_count"] == 10
    assert not e.cp(MP)["backfill_complete"]


def test_idempotent_reimport_creates_no_duplicates(env):
    e = env({MP: pages_of(rows_for(MP, 25))}, settings={"sections": (MP,)})
    e.drain()
    first = e.count()
    e.conn.execute("UPDATE zipnet_checkpoints SET backfill_complete=0, backfill_cursor=0")
    e.drain()
    assert e.count() == first == 25
    assert e.conn.execute("SELECT COUNT(*) FROM zipnet_record_changes").fetchone()[0] == 25  # only the 'created' rows
    assert e.cp(MP)["indexed_count"] == 25  # not double counted


def test_changed_source_record_is_detected_with_a_field_diff(env):
    pages = {MP: pages_of(rows_for(MP, 10))}
    e = env(pages, settings={"sections": (MP,)})
    e.drain()
    pages[MP][0]["data"][0]["TracingStatus"] = "Traced"
    e.conn.execute("UPDATE zipnet_checkpoints SET backfill_complete=0, backfill_cursor=0")
    results = [r for r in e.drain() if r.mode == "backfill"]
    assert sum(r.updated for r in results) == 1
    row = e.conn.execute("SELECT * FROM zipnet_record_changes WHERE change_type='updated'").fetchone()
    assert '"status":{"new":"Traced","old":"Untraced"}' in row["changed_fields"]
    rec = e.conn.execute("SELECT status, last_changed_at, first_seen_at FROM zipnet_records WHERE id=?",
                         (row["record_id"],)).fetchone()
    assert rec["status"] == "Traced" and rec["last_changed_at"] >= rec["first_seen_at"]


def test_sections_keep_independent_checkpoints(env):
    pages = all_sections(25)

    def hook(url, data):
        if data and ADAPTERS["stolen_vehicles"].data_path in url:
            return json_response({}, status=503)
        return None

    e = env(pages, hook=hook)
    e.drain()
    bad = e.cp("stolen_vehicles")
    assert bad["last_status"] == "failed" and bad["backfill_cursor"] == 0 and not bad["backfill_complete"]
    assert bad["consecutive_failures"] == 1 and bad["next_attempt_at"] is not None  # cooling down, not hammered
    for s in ADAPTERS:
        if s != "stolen_vehicles":
            assert e.cp(s)["backfill_complete"] == 1 and e.count(s) == 25


def test_empty_page_before_the_reported_total_is_never_marked_complete(env):
    pages = {MP: {0: pages_of(rows_for(MP, 10), total=40)[0]}}  # source claims 40 but serves one page
    e = env(pages, settings={"sections": (MP,), "pages_per_turn": 10})
    r = e.svc.step(e.conn)
    assert r.status == "incomplete" and r.stop_reason == "empty_page_before_end"
    cp = e.cp(MP)
    assert cp["backfill_complete"] == 0 and cp["backfill_cursor"] == 10 and cp["source_total"] == 40
    assert cp["next_attempt_at"] is not None  # no progress: cool down instead of hammering the source
    calls = len(e.transport.calls)
    assert e.svc.step(e.conn) is None and len(e.transport.calls) == calls


def test_backfill_page_limit_is_reported_incomplete_and_resumes_when_raised(env):
    e = env({MP: pages_of(rows_for(MP, 50))}, settings={"sections": (MP,), "backfill_max_pages": 2, "pages_per_turn": 10})
    results = e.drain()
    cp = e.cp(MP)
    assert cp["backfill_complete"] == 0 and cp["backfill_pages"] == 2 and cp["backfill_limit_reached"] == 1
    assert any(r.mode == "backfill" and r.status == "incomplete" and r.stop_reason == "page_limit" for r in results)
    e.st.backfill_max_pages = 0  # administrator lifts the cap
    e.drain()
    cp = e.cp(MP)
    assert cp["backfill_complete"] == 1 and cp["backfill_limit_reached"] == 0 and e.count() == 50


# --- network behaviour ------------------------------------------------------------------------------------


def test_transient_errors_are_retried_with_bounded_backoff(env):
    attempts = {"n": 0}
    good = pages_of(rows_for(MP, 5))

    def hook(url, data):
        if data is None:
            return None
        attempts["n"] += 1
        if attempts["n"] == 1:
            return TransportError("timeout")
        if attempts["n"] == 2:
            return json_response({}, status=503)
        return json_response(good[0])

    e = env(hook=hook, settings={"sections": (MP,)})
    r = e.svc.step(e.conn)
    assert r.status == "completed" and e.count() == 5
    backoffs = [s for s in e.sleeps if s < 0.1]
    assert len(backoffs) == 2 and all(s <= 0.05 for s in backoffs)  # capped by backoff_max
    assert e.conn.execute("SELECT retries FROM zipnet_jobs").fetchone()[0] == 2


def test_retry_limit_marks_the_section_failed_and_keeps_its_checkpoint(env):
    e = env(hook=lambda url, data: TransportError("down") if data else None, settings={"sections": (MP,)})
    r = e.svc.step(e.conn)
    assert r.status == "failed" and "after 3 attempts" in r.message
    assert e.cp(MP)["backfill_cursor"] == 0 and e.count() == 0


def test_blocked_source_is_not_retried_or_evaded(env):
    e = env(hook=lambda url, data: json_response({}, status=403) if data else None, settings={"sections": (MP,)})
    r = e.svc.step(e.conn)
    assert r.status == "blocked"
    assert len([c for c in e.transport.calls if c[0] == "POST"]) == 1
    assert e.cp(MP)["next_attempt_at"] > e.svc.clock() + 3000  # long cool-down


def test_html_login_page_instead_of_json_is_treated_as_blocked(env):
    html = RawResponse(200, {"Content-Type": "text/html"}, b"<form action='/Login/Login'>login</form>")
    e = env(hook=lambda url, data: html if data else None, settings={"sections": (MP,)})
    assert e.svc.step(e.conn).status == "blocked"


def test_changed_listing_markup_stops_the_section(env):
    e = env({MP: pages_of(rows_for(MP, 5))}, listing_override={MP: "<html><table id='other'></table></html>"},
            settings={"sections": (MP,)})
    r = e.svc.step(e.conn)
    assert r.status == "selector_changed" and "missingPersonGrid" in r.message
    assert not [c for c in e.transport.calls if c[0] == "POST"] and e.count() == 0


def test_changed_json_schema_is_detected(env):
    bad = [{"MissingPersonId": i, "Foo": 1} for i in range(5)]
    e = env({MP: {0: {"recordsTotal": 5, "data": bad}}}, settings={"sections": (MP,)})
    assert e.svc.step(e.conn).status == "selector_changed"


def test_pagination_loop_is_detected(env):
    same = pages_of(rows_for(MP, 10), total=100)[0]
    e = env({MP: {0: same, 10: same}}, settings={"sections": (MP,), "pages_per_turn": 10})
    r = e.svc.step(e.conn)
    assert r.stop_reason == "pagination_loop" and r.status == "incomplete" and e.count() == 10


def test_requests_are_throttled(env):
    e = env({MP: pages_of(rows_for(MP, 30))}, settings={"sections": (MP,), "request_delay": 2.0, "pages_per_turn": 10})
    e.svc.step(e.conn)
    assert e.sleeps and all(s <= 2.0 for s in e.sleeps) and sum(e.sleeps) >= 4.0


# --- stop requests ------------------------------------------------------------------------------------------


def test_stop_request_halts_at_a_page_boundary_and_progress_is_kept(env):
    holder = {}

    def hook(url, data):
        if data and data["start"] == "20":
            holder["env"].svc.stop_event.set()
        return None

    e = env({MP: pages_of(rows_for(MP, 50))}, hook=hook, settings={"sections": (MP,), "pages_per_turn": 10})
    holder["env"] = e
    r = e.svc.step(e.conn)
    assert r.status == "stopped" and r.stop_reason == "stop_requested"
    # the request in flight at start=20 finishes and commits; the next page is never requested
    assert e.count() == 30 and e.cp(MP)["backfill_cursor"] == 30 and not e.cp(MP)["backfill_complete"]


def test_pause_flag_stops_a_turn_before_any_request(env):
    e = env({MP: pages_of(rows_for(MP, 20))}, settings={"sections": (MP,)})
    store.set_control(e.conn, "paused", "1", "test")
    r = e.svc.step(e.conn)
    assert r.status == "stopped" and not e.transport.calls and e.cp(MP)["backfill_cursor"] == 0


# --- incremental ------------------------------------------------------------------------------------------------


def settle_backfill(e):
    e.drain()
    assert all(e.cp(s)["backfill_complete"] for s in e.st.sections)


def test_incremental_runs_only_after_backfill_and_catches_up_without_a_full_download(env):
    old = rows_for(MP, 40, newest=100, CreatedOn="01/01/2026")
    pages = {MP: pages_of(old)}
    e = env(pages, settings={"sections": (MP,), "pages_per_turn": 10})
    results = e.drain()
    assert [r.mode for r in results[:1]] == ["backfill"] and e.cp(MP)["backfill_complete"]
    # Two new records appear on top; everything else is old and already indexed.
    fresh = [make_row(MP, 102), make_row(MP, 101)] + old
    e.transport.pages[MP] = pages_of(fresh)
    e.conn.execute("UPDATE zipnet_checkpoints SET last_incremental_at='2000-01-01T00:00:00.000+00:00'")
    e.transport.calls.clear()
    r = e.svc.step(e.conn)
    assert r.mode == "incremental" and r.status == "completed" and r.stop_reason == "caught_up"
    assert r.new == 2 and e.count() == 42
    assert len([c for c in e.transport.calls if c[0] == "POST"]) == 2  # stopped early, not 5 pages
    cp = e.cp(MP)
    assert cp["caught_up"] == 1 and cp["high_water_id"] == 1102 - 1000 + 1000 and cp["incr_cursor"] == 0


def test_incremental_that_cannot_catch_up_warns_keeps_its_cursor_and_continues(env):
    old = rows_for(MP, 10, newest=100, CreatedOn="01/01/2026")
    e = env({MP: pages_of(old)}, settings={"sections": (MP,), "incremental_max_pages": 2})
    e.drain()
    fresh = [make_row(MP, 200 - i) for i in range(50)] + old  # 5 pages of brand-new records
    e.transport.pages[MP] = pages_of(fresh)
    e.conn.execute("UPDATE zipnet_checkpoints SET last_incremental_at='2000-01-01T00:00:00.000+00:00'")
    first = e.svc.step(e.conn)
    assert first.status == "incomplete" and first.stop_reason == "max_pages"
    cp = e.cp(MP)
    assert cp["caught_up"] == 0 and cp["incr_in_progress"] == 1 and cp["incr_cursor"] == 20
    assert "has not yet reached" in cp["last_message"] and first.warnings
    old_hw = cp["high_water_id"]
    assert old_hw == 1100  # not promoted: the gap is not closed yet
    results = e.drain()
    assert results[-1].status == "completed" and e.count() == 60
    assert e.cp(MP)["incr_in_progress"] == 0 and e.cp(MP)["high_water_id"] == 1200


def test_incremental_waits_for_its_interval(env):
    e = env({MP: pages_of(rows_for(MP, 10, CreatedOn="01/01/2026"))}, settings={"sections": (MP,)})
    e.drain()
    assert e.svc.step(e.conn) is None  # just synchronised: nothing due


def test_start_request_makes_incremental_due_immediately(env):
    e = env({MP: pages_of(rows_for(MP, 10, CreatedOn="01/01/2026"))}, settings={"sections": (MP,)})
    e.drain()
    e.conn.execute("UPDATE zipnet_checkpoints SET last_incremental_at=?", (store.now_iso(),))
    assert e.svc.step(e.conn) is None  # synchronised a moment ago
    store.set_control(e.conn, "run_now_at", store.now_iso(), "admin")  # administrator presses "sync now"
    assert e.svc.step(e.conn).mode == "incremental"


# --- store / schema ------------------------------------------------------------------------------------------------


def test_migration_is_idempotent_and_versioned(zdb):
    conn = store.connect(zdb)
    assert store.migrate(conn) == store.SCHEMA_VERSION == store.migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
    conn.execute("INSERT INTO zipnet_checkpoints (section, updated_at, indexed_count) VALUES ('x', 'now', 7)")
    store.migrate(conn)  # existing data survives a repeat
    assert conn.execute("SELECT indexed_count FROM zipnet_checkpoints WHERE section='x'").fetchone()[0] == 7


def test_unique_key_and_foreign_keys_are_enforced(zdb):
    conn = store.connect(zdb)
    nr = get_adapter(MP, BASE).normalize(make_row(MP, 1))
    with store.transaction(conn):
        store.upsert_records(conn, MP, "u", [nr], None)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO zipnet_records (section, source_id, title, fingerprint, source_listing_url, "
                     "first_seen_at, last_seen_at, last_changed_at) VALUES (?, ?, 't', 'f', 'u', 'a', 'a', 'a')",
                     (MP, nr.source_id))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO zipnet_record_changes (record_id, changed_at, change_type) VALUES (999, 'a', 'x')")


def test_unknown_checkpoint_field_is_rejected(zdb):
    with pytest.raises(ValueError):
        store.update_checkpoint(store.connect(zdb), MP, bogus=1)


def test_crashed_jobs_are_marked_interrupted_on_restart(zdb):
    conn = store.connect(zdb)
    with store.transaction(conn):
        jid = store.start_job(conn, MP, "backfill")
    assert store.recover_stale_jobs(conn) == 1
    row = conn.execute("SELECT status, stop_reason FROM zipnet_jobs WHERE id=?", (jid,)).fetchone()
    assert (row["status"], row["stop_reason"]) == ("interrupted", "worker_restart")


def test_pipeline_module_has_no_network_or_face_dependencies():
    import inspect

    src = inspect.getsource(zipnet_pipeline)
    assert "face_service" not in src and "vector_db" not in src and "insightface" not in src


def test_run_section_direct_call_with_custom_client(zdb):
    conn = store.connect(zdb)
    st = make_settings(zdb, sections=(MP,))
    client = SourceClient(FakeTransport({MP: pages_of(rows_for(MP, 8))}), delay_seconds=0.5, max_retries=2,
                          backoff_base=0.01, backoff_max=0.02, sleep=lambda s: None)
    r = run_section(conn, get_adapter(MP, BASE), client, st, "backfill")
    assert r.status == "completed" and r.new == 8 and r.job_id
