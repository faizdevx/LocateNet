"""Starts and stops the ZIPNET sync worker as a child process of ``python app.py``.

Only the development entry point (``python app.py``) uses this. Gunicorn never calls it, and Docker
Compose runs the worker as its own service, so a web worker can never spawn a duplicate. Even if
two launchers race, the second worker fails to take the database lease and exits at once.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import time
import uuid

import config

from . import zipnet_store as store
from .zipnet_settings import ZipnetSettings

log = logging.getLogger("locatenet.zipnet.supervisor")

_child: subprocess.Popen | None = None
_exit_token: str | None = None
_registered = False


def worker_command(token: str) -> list[str]:
    return [sys.executable, "-u", os.path.join(config.BASE_DIR, "zipnet_worker.py"), "--parent-pid", str(os.getpid()),
            "--exit-token", token]


def worker_alive(st: ZipnetSettings | None = None) -> bool:
    st = st or ZipnetSettings.from_config()
    try:
        conn = store.connect(st.db_path)
    except Exception:
        return False
    try:
        store.migrate(conn)
        return store.live_lease(conn) is not None
    finally:
        conn.close()


def ensure_worker_started(st: ZipnetSettings | None = None, *, force: bool = False) -> subprocess.Popen | None:
    """Spawn the worker when sync is enabled and none is running. Returns the child, or None."""
    global _child, _registered, _exit_token
    st = st or ZipnetSettings.from_config()
    if not st.enabled:
        log.info("ZIPNET sync is disabled (ZIPNET_SYNC_ENABLED=0); worker not started")
        return None
    if not (config.ZIPNET_AUTOSTART_WORKER or force):
        log.info("ZIPNET_AUTOSTART_WORKER=0; expecting an external worker")
        return None
    if _child is not None and _child.poll() is None:
        return _child  # we already own a live child
    if worker_alive(st):
        log.info("a ZIPNET sync worker is already running; not starting another")
        return None
    log_dir = os.path.join(config.BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out = open(os.path.join(log_dir, "zipnet_worker.log"), "ab")  # noqa: SIM115 - handed to the child
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    _exit_token = uuid.uuid4().hex
    try:
        _child = subprocess.Popen(worker_command(_exit_token), cwd=config.BASE_DIR, stdout=out, stderr=subprocess.STDOUT,  # noqa: S603
                                  **kwargs)
    finally:
        out.close()
    log.info("started ZIPNET sync worker (pid %s); log: logs/zipnet_worker.log", _child.pid)
    if not _registered:
        atexit.register(stop_worker)
        _registered = True
    return _child


def stop_worker(grace: float | None = None) -> None:
    """Ask the child worker to exit after its current page; escalate only if it does not."""
    global _child, _exit_token
    proc, _child, token, _exit_token = _child, None, _exit_token, None
    if proc is None or proc.poll() is not None:
        return
    grace = config.ZIPNET_SHUTDOWN_GRACE_SECONDS if grace is None else grace
    try:
        conn = store.connect()
        try:
            store.set_control(conn, "exit_request", token, "supervisor")
        finally:
            conn.close()
    except Exception:
        log.exception("could not write the exit request")
    deadline = time.monotonic() + grace
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        log.warning("worker did not exit within %.0fs; terminating", grace)
        _terminate(proc)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _terminate(proc: subprocess.Popen) -> None:
    if os.name == "nt":  # the venv launcher spawns the real interpreter; kill the whole tree
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)  # noqa: S603, S607
    else:
        proc.terminate()


def autostart_if_enabled() -> subprocess.Popen | None:
    """Called by ``python app.py`` after the app is configured. Never raises."""
    try:
        return ensure_worker_started()
    except Exception:
        log.exception("could not start the ZIPNET sync worker; the web app continues without it")
        return None
