"""Scheduling for the ZIPNET sync worker: lease, control flags, backfill vs incremental, graceful stop.

Lifecycle of one worker process (``python zipnet_worker.py``):
  1. migrate the ZIPNET database, take the single-worker lease (exit if another worker holds it);
  2. mark jobs left "running" by a previous crash as interrupted (checkpoints are never reset);
  3. loop: honour pause/exit requests at page boundaries; run incremental turns that are due,
     otherwise one bounded backfill turn of the next unfinished section (round-robin);
  4. on SIGINT/SIGTERM/exit request: finish the current page, release the lease, exit.
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pipeline.zipnet_pipeline import SectionResult, run_section

from . import zipnet_store as store
from .zipnet_adapters import get_adapter
from .zipnet_scraper import ScraplingTransport, SourceClient, Transport
from .zipnet_settings import ZipnetSettings

log = logging.getLogger("locatenet.zipnet.sync")

EXIT_OK = 0
EXIT_DUPLICATE = 3

TransportFactory = Callable[[ZipnetSettings], Transport]


def default_transport(st: ZipnetSettings) -> Transport:
    return ScraplingTransport(st.request_timeout, st.user_agent)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class ZipnetSyncService:
    def __init__(self, st: ZipnetSettings, transport_factory: TransportFactory | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
                 parent_pid: int | None = None, exit_token: str | None = None):
        self.st = st
        self.transport_factory = transport_factory or default_transport
        self._sleep = sleep
        self.clock = clock
        self.parent_pid = parent_pid
        # Compared with the "exit_request" control value. A token (not a pid) is used because on Windows
        # the venv python.exe is a launcher, so the pid the supervisor sees is not the worker's own pid.
        self.exit_token = exit_token
        self.holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.stop_event = threading.Event()
        self._lease_lost = threading.Event()
        self._hb_stop = threading.Event()
        self._ctl_at = 0.0
        self._ctl_cache: dict[str, Any] = {}
        self._rr = 0
        self._turns = 0
        self._client: SourceClient | None = None

    # -- connections ---------------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        conn = store.connect(self.st.db_path)
        return conn

    # -- control flags (cached for 1 s so the per-request check is cheap) ----------------------------

    def _control(self, conn: sqlite3.Connection, fresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if fresh or now - self._ctl_at >= 1.0:
            self._ctl_cache = {
                "paused": store.get_control(conn, "paused") == "1",
                "exit_request": store.get_control(conn, "exit_request"),
                "run_now_at": store.get_control(conn, "run_now_at"),
            }
            self._ctl_at = now
        return self._ctl_cache

    def should_exit(self, conn: sqlite3.Connection) -> bool:
        if self.stop_event.is_set() or self._lease_lost.is_set():
            return True
        if self.parent_pid and not pid_alive(self.parent_pid):
            log.info("parent process %s is gone; shutting down", self.parent_pid)
            return True
        return self.exit_token is not None and self._control(conn)["exit_request"] == self.exit_token

    def should_stop_turn(self, conn: sqlite3.Connection) -> bool:
        return self.should_exit(conn) or bool(self._control(conn)["paused"])

    # -- scheduling ---------------------------------------------------------------------------------

    def _backfill_open(self, cp: dict[str, Any]) -> bool:
        """True when the section still has historical pages to import and may import them."""
        if cp["backfill_complete"]:
            return False
        cap = self.st.backfill_max_pages
        return not (cap and cp["backfill_pages"] >= cap)

    def _incremental_due(self, cp: dict[str, Any], run_now_at: str | None, now: datetime) -> bool:
        if self._backfill_open(cp):
            return False  # historical import first; incremental then picks up the newest records
        if cp["incr_in_progress"] or not cp["last_incremental_at"]:
            return True
        last = parse_iso(cp["last_incremental_at"])
        if last is None or now - last >= timedelta(minutes=self.st.incremental_interval_minutes):
            return True
        return bool(run_now_at and cp["last_incremental_at"] <= run_now_at)

    def next_turn(self, conn: sqlite3.Connection) -> tuple[str, str] | None:
        """Pick (section, mode). Due incremental turns come before backfill turns."""
        ctl = self._control(conn, fresh=True)
        now = datetime.fromtimestamp(self.clock(), UTC)
        sections = list(self.st.sections)
        order = sections[self._rr % len(sections):] + sections[:self._rr % len(sections)] if sections else []
        incremental: list[str] = []
        backfill: list[str] = []
        for section in order:
            with store.transaction(conn):
                cp = store.get_checkpoint(conn, section)
            cooling = cp["next_attempt_at"] and self.clock() < cp["next_attempt_at"]
            if cooling and not (ctl["run_now_at"] and ctl["run_now_at"] > cp["updated_at"]):
                continue
            if self._incremental_due(cp, ctl["run_now_at"], now):
                incremental.append(section)
            elif self._backfill_open(cp):
                backfill.append(section)
        pick = (incremental[0], "incremental") if incremental else (backfill[0], "backfill") if backfill else None
        if pick:
            self._rr = (sections.index(pick[0]) + 1) % len(sections)
        return pick

    def seconds_until_work(self, conn: sqlite3.Connection) -> float:
        """How long the worker can sleep before something becomes due (capped at 30 s)."""
        soonest = 30.0
        for section in self.st.sections:
            cp = store.get_checkpoint(conn, section)
            if cp["next_attempt_at"]:
                soonest = min(soonest, max(1.0, cp["next_attempt_at"] - self.clock()))
            last = parse_iso(cp["last_incremental_at"])
            if last and not self._backfill_open(cp):
                due = (last + timedelta(minutes=self.st.incremental_interval_minutes)).timestamp()
                soonest = min(soonest, max(1.0, due - self.clock()))
        return soonest

    def next_incremental_at(self, conn: sqlite3.Connection) -> str | None:
        times = []
        for section in self.st.sections:
            cp = store.get_checkpoint(conn, section)
            last = parse_iso(cp["last_incremental_at"])
            if last and not self._backfill_open(cp):
                times.append(last + timedelta(minutes=self.st.incremental_interval_minutes))
        return min(times).isoformat(timespec="seconds") if times else None

    # -- one turn -----------------------------------------------------------------------------------------

    def client(self) -> SourceClient:
        if self._client is None:
            self._client = SourceClient(
                self.transport_factory(self.st),
                delay_seconds=self.st.request_delay, max_retries=self.st.max_retries,
                backoff_base=self.st.backoff_base, backoff_max=self.st.backoff_max,
                should_halt=lambda: self._halt_check(), sleep=self._sleep)
        return self._client

    def _halt_check(self) -> bool:
        conn = self._halt_conn
        return self.should_stop_turn(conn) if conn is not None else self.stop_event.is_set()

    _halt_conn: sqlite3.Connection | None = None

    def step(self, conn: sqlite3.Connection) -> SectionResult | None:
        """Run at most one turn. Returns its result, or None when nothing is due."""
        pick = self.next_turn(conn)
        if pick is None:
            return None
        section, mode = pick
        store.set_worker_state(conn, "running", section=section, mode=mode)
        adapter = get_adapter(section, self.st.base_url)
        self._halt_conn = conn
        try:
            result = run_section(conn, adapter, self.client(), self.st, mode,
                                 should_stop=lambda: self.should_stop_turn(conn), clock=self.clock)
        finally:
            self._halt_conn = None
        self._turns += 1
        if self._turns % 50 == 0:
            store.prune_jobs(conn)
        return result

    # -- worker lifecycle --------------------------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        conn = self.connect()
        try:
            interval = max(1.0, self.st.lock_ttl / 4)
            while not self._hb_stop.wait(interval):
                try:
                    if not store.heartbeat_lease(conn, self.holder, self.st.lock_ttl):
                        log.error("worker lease lost to another process; shutting down")
                        self._lease_lost.set()
                        return
                except sqlite3.Error:
                    log.exception("heartbeat failed")
        finally:
            conn.close()

    def _wait(self, conn: sqlite3.Connection, seconds: float) -> None:
        """Sleep in short slices; wake early on exit, un-pause or a manual run request."""
        ctl0 = self._control(conn, fresh=True)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop_event.wait(min(1.0, max(0.0, end - time.monotonic()))):
                return
            if self.should_exit(conn):
                return
            ctl = self._control(conn, fresh=True)
            if ctl["paused"] != ctl0["paused"] or ctl["run_now_at"] != ctl0["run_now_at"]:
                return

    def run(self) -> int:
        conn = self.connect()
        store.migrate(conn)
        if not store.acquire_lease(conn, self.holder, self.st.lock_ttl):
            log.warning("another ZIPNET sync worker holds the lease; exiting")
            conn.close()
            return EXIT_DUPLICATE
        hb = threading.Thread(target=self._heartbeat_loop, name="zipnet-heartbeat", daemon=True)
        hb.start()
        log.info("zipnet sync worker started (pid %s, sections %s)", os.getpid(), ",".join(self.st.sections))
        try:
            recovered = store.recover_stale_jobs(conn)
            if recovered:
                log.warning("%d interrupted job(s) from a previous run; resuming from checkpoints", recovered)
            store.set_worker_state(conn, "starting")
            while not self.should_exit(conn):
                if self._control(conn, fresh=True)["paused"]:
                    store.set_worker_state(conn, "paused", message="stop acknowledged; waiting to be started")
                    self._wait(conn, 5.0)
                    continue
                result = self.step(conn)
                if result is None:
                    store.set_worker_state(conn, "idle", next_incremental_at=self.next_incremental_at(conn))
                    self._wait(conn, self.seconds_until_work(conn))
                elif result.status in ("failed", "blocked", "selector_changed"):
                    log.warning("zipnet %s/%s %s: %s", result.section, result.mode, result.status, result.message)
            return EXIT_OK
        finally:
            self._hb_stop.set()
            hb.join(timeout=5)
            try:
                if self._client is not None:
                    self._client.close()
                if self.exit_token and store.get_control(conn, "exit_request") == self.exit_token:
                    store.set_control(conn, "exit_request", None, "worker")
                store.set_worker_state(conn, "stopped", message="worker exited cleanly")
                store.release_lease(conn, self.holder)
            except sqlite3.Error:
                log.exception("error during shutdown")
            conn.close()
            log.info("zipnet sync worker stopped")
