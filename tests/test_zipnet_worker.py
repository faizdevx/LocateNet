"""Worker lifecycle: auto-start, duplicate prevention, graceful shutdown, pause acknowledgement, real process."""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import pytest

import config
from services import zipnet_store as store
from services import zipnet_supervisor as sup
from services.zipnet_adapters import ADAPTERS
from services.zipnet_sync_service import EXIT_DUPLICATE, EXIT_OK, ZipnetSyncService
from tests.zipnet_fakes import FakeTransport, RunningWorker, make_settings, wait_for
from tests.zipnet_helpers import listing_html, make_row

ROOT = Path(__file__).resolve().parent.parent
MP = "missing_persons"


@pytest.fixture
def worker(zdb):
    w = RunningWorker(zdb)
    yield w
    if w.thread.is_alive():
        w.stop()


# --- in-process lifecycle -----------------------------------------------------------------------------


def test_worker_takes_the_lease_syncs_then_idles(worker):
    worker.start()
    assert wait_for(lambda: worker.state() == "idle")
    cp = store.get_checkpoint(worker.conn, MP)
    assert cp["backfill_complete"] == 1  # empty source: backfill finished, nothing to import
    assert wait_for(lambda: store.get_checkpoint(worker.conn, MP)["last_incremental_at"] is not None)


def test_graceful_stop_releases_the_lease_and_reports_stopped(worker):
    worker.start()
    worker.stop()
    assert worker.code == EXIT_OK
    assert store.live_lease(worker.conn) is None
    assert worker.state() == "stopped"
    assert worker.svc._client is None or worker.svc._client.transport.closed


def test_exit_request_from_the_supervisor_stops_only_the_matching_worker(worker):
    worker.start()
    store.set_control(worker.conn, "exit_request", "someone-elses-token", "supervisor")
    time.sleep(1.5)
    assert worker.thread.is_alive()
    store.set_control(worker.conn, "exit_request", "test-token", "supervisor")
    worker.thread.join(15)
    assert not worker.thread.is_alive() and worker.code == EXIT_OK
    assert store.get_control(worker.conn, "exit_request") is None  # cleared on exit


def test_second_worker_cannot_run_while_the_first_holds_the_lease(zdb, worker):
    worker.start()
    second = ZipnetSyncService(make_settings(zdb, sections=(MP,)), transport_factory=lambda s: FakeTransport({}))
    assert second.run() == EXIT_DUPLICATE
    assert store.live_lease(worker.conn)["holder"] == worker.svc.holder  # the first worker is undisturbed


def test_expired_lease_of_a_crashed_worker_can_be_taken_over(zdb):
    conn = store.connect(zdb)
    assert store.acquire_lease(conn, "dead-worker", 30, now=time.time() - 1000)
    assert store.live_lease(conn) is None  # expired
    assert store.acquire_lease(conn, "new-worker", 30)
    assert not store.acquire_lease(conn, "third-worker", 30)
    assert not store.heartbeat_lease(conn, "dead-worker", 30)  # a stale holder learns it lost the lease


def test_lost_lease_makes_the_worker_exit(zdb):
    w = RunningWorker(zdb, lock_ttl=15.0)
    w.start()
    w.svc._lease_lost.set()  # what the heartbeat thread does when another holder took over
    w.thread.join(15)
    assert not w.thread.is_alive()


def test_pause_is_acknowledged_by_the_worker_and_resume_continues(worker):
    worker.start()
    assert wait_for(lambda: worker.state() == "idle")
    store.set_control(worker.conn, "paused", "1", "admin")
    assert wait_for(lambda: worker.state() == "paused"), "worker never acknowledged the stop request"
    store.set_control(worker.conn, "paused", "0", "admin")
    store.set_control(worker.conn, "run_now_at", store.now_iso(), "admin")
    assert wait_for(lambda: worker.state() in ("running", "idle"))


def test_worker_exits_when_its_parent_disappears(zdb):
    w = RunningWorker(zdb)
    w.svc.parent_pid = 999999999  # no such process
    w.thread.start()
    w.thread.join(15)
    assert not w.thread.is_alive() and w.code == EXIT_OK


# --- supervisor: automatic start ------------------------------------------------------------------------


class FakePopen:
    instances: list[FakePopen] = []

    def __init__(self, cmd, **kwargs):
        self.cmd, self.kwargs, self.pid, self._code = cmd, kwargs, 424242, None
        FakePopen.instances.append(self)

    def poll(self):
        return self._code

    def terminate(self):
        self._code = -15

    def kill(self):
        self._code = -9

    def wait(self, timeout=None):
        return self._code


@pytest.fixture
def popen(monkeypatch, zdb):
    FakePopen.instances = []
    monkeypatch.setattr(sup.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(sup, "_child", None)
    monkeypatch.setattr(config, "ZIPNET_AUTOSTART_WORKER", True)
    monkeypatch.setattr(sup.atexit, "register", lambda *a, **k: None)
    return FakePopen


def test_worker_autostarts_when_sync_is_enabled(popen, zdb):
    child = sup.ensure_worker_started(make_settings(zdb))
    assert child is not None and len(popen.instances) == 1
    cmd = popen.instances[0].cmd
    assert cmd[0] == sys.executable and cmd[2].endswith("zipnet_worker.py") and "--parent-pid" in cmd


def test_no_worker_starts_when_sync_is_disabled(popen, zdb):
    assert sup.ensure_worker_started(make_settings(zdb, enabled=False)) is None
    assert popen.instances == []


def test_no_worker_starts_when_autostart_is_off_for_external_workers(popen, zdb, monkeypatch):
    monkeypatch.setattr(config, "ZIPNET_AUTOSTART_WORKER", False)
    assert sup.ensure_worker_started(make_settings(zdb)) is None and popen.instances == []


def test_repeated_startup_callbacks_never_spawn_a_second_worker(popen, zdb):
    st = make_settings(zdb)
    first = sup.ensure_worker_started(st)
    again = sup.ensure_worker_started(st)
    assert again is first and len(popen.instances) == 1


def test_no_spawn_when_another_process_already_holds_the_lease(popen, zdb):
    conn = store.connect(zdb)
    store.acquire_lease(conn, "external-worker", 60)
    assert sup.ensure_worker_started(make_settings(zdb)) is None and popen.instances == []


def test_startup_failure_never_breaks_the_web_app(monkeypatch):
    def boom(*a, **k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(sup, "ensure_worker_started", boom)
    assert sup.autostart_if_enabled() is None


def test_stop_worker_asks_politely_first(popen, zdb, monkeypatch):
    child = sup.ensure_worker_started(make_settings(zdb))
    monkeypatch.setattr(sup.time, "sleep", lambda s: setattr(child, "_code", 0))  # child exits after the request
    sup.stop_worker(grace=5)
    token = child.cmd[child.cmd.index("--exit-token") + 1]
    assert store.get_control(store.connect(zdb), "exit_request") == token
    assert child._code == 0  # exited by itself: never terminated or killed


def test_stop_worker_escalates_only_when_the_child_ignores_the_request(popen, zdb, monkeypatch):
    child = sup.ensure_worker_started(make_settings(zdb))
    monkeypatch.setattr(sup, "_terminate", lambda proc: proc.terminate())
    sup.stop_worker(grace=0.1)
    assert child._code == -15


def test_importing_the_app_does_not_start_a_worker():
    import app  # noqa: F401  (already imported by conftest)

    assert sup._child is None
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert source.count("autostart_if_enabled") == 2  # import + the single call
    assert source.index("autostart_if_enabled()") > source.index('if __name__ == "__main__":')


def test_disabled_worker_entry_point_makes_no_requests(monkeypatch, capsys, zdb):
    import zipnet_worker

    monkeypatch.setattr(config, "ZIPNET_SYNC_ENABLED", False)
    assert zipnet_worker.main(["--once"]) == 0
    assert "ZIPNET_SYNC_ENABLED" in capsys.readouterr().out
    assert zipnet_worker.main(["--healthcheck"]) == 0  # idling on purpose is healthy
    assert store.live_lease(store.connect(zdb)) is None


def test_disabled_worker_process_idles_instead_of_exiting_so_restart_policies_do_not_loop(tmp_path):
    env = {**os.environ, "ZIPNET_SYNC_ENABLED": "0", "ZIPNET_DB": str(tmp_path / "z.db"),
           "LOCATENET_DB": str(tmp_path / "m.db"), "PYTHONUTF8": "1"}
    proc = subprocess.Popen([sys.executable, str(ROOT / "zipnet_worker.py")], cwd=ROOT, env=env,  # noqa: S603
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        time.sleep(3)
        assert proc.poll() is None  # still up, doing nothing
        assert not (tmp_path / "z.db").exists()  # no database, lease or request was created
    finally:
        proc.kill()


# --- Docker / deployment configuration ---------------------------------------------------------------------


def test_compose_runs_the_worker_as_its_own_single_service_and_not_inside_gunicorn():
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    web, worker = compose["services"]["locatenet"], compose["services"]["zipnet-worker"]
    assert str(web["environment"]["ZIPNET_AUTOSTART_WORKER"]) == "0"  # gunicorn never spawns one
    assert worker["command"] == ["python", "zipnet_worker.py"]
    assert worker["build"] == web["build"] or worker.get("image") == web.get("image")
    assert any(v.startswith("locatenet_data:") for v in worker["volumes"])
    assert any(v.startswith("locatenet_data:") for v in web["volumes"])  # shared database volume
    assert "healthcheck" in worker and "--healthcheck" in " ".join(worker["healthcheck"]["test"])
    assert worker.get("deploy", {}).get("replicas", 1) == 1 and "scale" not in worker
    assert worker["restart"] == "unless-stopped"
    assert worker["stop_grace_period"]  # time to finish the current page


def test_gunicorn_is_not_asked_to_start_the_worker_and_stays_single_process():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "gunicorn" in dockerfile and '"-w", "1"' in dockerfile
    assert "zipnet_worker" not in dockerfile.split("CMD")[-1]


def test_worker_dependencies_are_pinned_and_optional_in_lightweight_ci():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scrapling[fetchers]==" in reqs
    assert "scrapling" not in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").lower()


def test_env_example_documents_every_zipnet_setting():
    import re

    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    names = set(re.findall(r'"(ZIPNET_\w+)"', (ROOT / "config.py").read_text(encoding="utf-8")))
    assert len(names) >= 20
    for name in names:
        assert name in env, f"{name} is not documented in .env.example"


# --- real process against a local stub source ----------------------------------------------------------------------


class StubZipnet(http.server.BaseHTTPRequestHandler):
    rows: list[dict] = []
    hits = 0

    def log_message(self, *a):  # keep test output quiet
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        a = ADAPTERS[MP]
        self._send(listing_html(a.grid_id, a.script_path).encode(), "text/html; charset=utf-8")

    def do_POST(self):
        StubZipnet.hits += 1
        form = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        start, length = int(form["start"][0]), int(form["length"][0])
        payload = {"draw": "1", "recordsTotal": len(self.rows), "recordsFiltered": len(self.rows),
                   "data": self.rows[start:start + length]}
        self._send(json.dumps(payload).encode(), "application/json; charset=utf-8")


@pytest.fixture
def stub_source():
    pytest.importorskip("scrapling")
    StubZipnet.rows = []
    StubZipnet.hits = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubZipnet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_real_worker_process_imports_resumes_and_shuts_down_cleanly(stub_source, tmp_path):
    """Spawns `python zipnet_worker.py` (real Scrapling transport) against a local stub of ZIPNET."""
    StubZipnet.rows = [make_row(MP, 100 - i) for i in range(45)]
    db = str(tmp_path / "proc.db")
    env = {**os.environ, "ZIPNET_SYNC_ENABLED": "1", "ZIPNET_BASE_URL": stub_source, "ZIPNET_DB": db,
           "ZIPNET_SECTIONS": MP, "ZIPNET_PAGE_SIZE": "10", "ZIPNET_REQUEST_DELAY_SECONDS": "0.5",
           "ZIPNET_PAGES_PER_TURN": "1", "LOCATENET_DB": str(tmp_path / "main.db"), "PYTHONUTF8": "1"}
    cmd = [sys.executable, "-u", str(ROOT / "zipnet_worker.py"), "--exit-token", "proc-token"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # noqa: S603
    conn = None
    try:
        assert wait_for(lambda: os.path.exists(db), 30)
        conn = store.connect(db)
        assert wait_for(lambda: store.get_checkpoint(conn, MP)["backfill_cursor"] >= 10, 30)
        # a second worker process is refused while the first is alive
        dup = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=60)  # noqa: S603
        assert dup.returncode == EXIT_DUPLICATE
        # stop the way the supervisor does: by token, after the current page
        store.set_control(conn, "exit_request", "proc-token", "test")
        assert proc.wait(timeout=30) == EXIT_OK
        assert store.live_lease(conn) is None
        assert store.get_worker_state(conn)["state"] == "stopped"
        cp = store.get_checkpoint(conn, MP)
        assert 10 <= cp["backfill_cursor"] < 45 and not cp["backfill_complete"]  # interrupted mid-backfill
        stopped_at = cp["backfill_cursor"]
        assert conn.execute("SELECT COUNT(*) FROM zipnet_records").fetchone()[0] == stopped_at
    finally:
        if proc.poll() is None:
            proc.kill()

    # restart: resumes from the checkpoint (does not start over) and finishes without duplicates
    hits_before = StubZipnet.hits
    proc2 = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # noqa: S603
    try:
        assert wait_for(lambda: store.get_checkpoint(conn, MP)["backfill_complete"] == 1, 60)
        assert conn.execute("SELECT COUNT(*) FROM zipnet_records").fetchone()[0] == 45
        # 45 rows = 5 pages; the first run fetched stopped_at/10 of them, the rest were fetched now (+ incremental)
        assert StubZipnet.hits - hits_before >= (45 - stopped_at) // 10
        store.set_control(conn, "exit_request", "proc-token", "test")
        assert proc2.wait(timeout=30) == EXIT_OK
    finally:
        if proc2.poll() is None:
            proc2.kill()
        conn.close()
