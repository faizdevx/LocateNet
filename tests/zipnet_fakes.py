"""Fake transport and shared helpers for the ZIPNET tests. Nothing here touches the network."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from services.zipnet_adapters import ADAPTERS
from services.zipnet_scraper import RawResponse
from services.zipnet_settings import ZipnetSettings
import threading
import time

from services import zipnet_store as store
from services.zipnet_sync_service import ZipnetSyncService
from tests.zipnet_helpers import listing_html, make_row

Handler = Callable[[str, dict[str, str] | None], RawResponse | Exception | None]


def make_settings(db_path: str, **over: Any) -> ZipnetSettings:
    base = dict(enabled=True, sections=tuple(ADAPTERS), page_size=10, request_delay=0.5, max_retries=3,
                backoff_base=0.01, backoff_max=0.05, pages_per_turn=2, incremental_max_pages=3,
                incremental_interval_minutes=60, incremental_overlap_days=1, lock_ttl=30.0,
                failure_cooldown=60.0, blocked_cooldown=3600.0, db_path=db_path)
    base.update(over)
    return ZipnetSettings(**base)


def _html(body: str) -> RawResponse:
    def has_selector(css: str) -> bool:
        m = re.fullmatch(r"table#(\w+)", css)
        if m:
            return f'<table id="{m.group(1)}"' in body
        m = re.fullmatch(r'script\[src\*="([^"]+)"\]', css)
        return bool(m and m.group(1) in body)

    return RawResponse(200, {"Content-Type": "text/html; charset=utf-8"}, body.encode(), has_selector=has_selector)


def json_response(payload: Any, status: int = 200) -> RawResponse:
    return RawResponse(status, {"Content-Type": "application/json; charset=utf-8"}, json.dumps(payload).encode())


class FakeTransport:
    """``pages[section]`` maps the requested ``start`` offset to a DataTables payload."""

    def __init__(self, pages: dict[str, dict[int, dict[str, Any]]] | None = None,
                 listing_override: dict[str, str] | None = None, hook: Handler | None = None):
        self.pages = pages or {}
        self.listing_override = listing_override or {}
        self.hook = hook
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []
        self.closed = False

    def _adapter(self, url: str):
        for a in ADAPTERS.values():
            if url.endswith(a.listing_path) or url.endswith(a.data_path):
                return a
        raise AssertionError(f"unexpected url {url}")

    def _hooked(self, url: str, data: dict[str, str] | None) -> RawResponse | None:
        if self.hook is None:
            return None
        out = self.hook(url, data)
        if isinstance(out, Exception):
            raise out
        return out

    def get(self, url: str, headers: dict[str, str]) -> RawResponse:
        self.calls.append(("GET", url, None))
        hooked = self._hooked(url, None)
        if hooked is not None:
            return hooked
        a = self._adapter(url)
        return _html(self.listing_override.get(a.category, listing_html(a.grid_id, a.script_path)))

    def post(self, url: str, data: dict[str, str], headers: dict[str, str]) -> RawResponse:
        self.calls.append(("POST", url, data))
        hooked = self._hooked(url, data)
        if hooked is not None:
            return hooked
        a = self._adapter(url)
        pages = self.pages.get(a.category, {})
        start = int(data["start"])
        if start in pages:
            return json_response(pages[start])
        total = max([p.get("recordsTotal", 0) for p in pages.values()] or [0])
        return json_response({"draw": "1", "recordsTotal": total, "recordsFiltered": total, "data": []})

    def close(self) -> None:
        self.closed = True


MP = "missing_persons"


def rows_for(section, n, newest=100, **over):
    return [make_row(section, newest - i, **over) for i in range(n)]


def pages_of(rows, size=10, total=None):
    t = len(rows) if total is None else total
    return {i: {"draw": "1", "recordsTotal": t, "recordsFiltered": t, "data": rows[i:i + size]}
            for i in range(0, len(rows), size)}


def all_sections(n=25, size=10, newest=100, **over):
    return {s: pages_of(rows_for(s, n, newest, **over), size) for s in ADAPTERS}


class Env:
    """A service plus its own connection, wired to a FakeTransport and a no-op sleep."""

    def __init__(self, zdb, transport, settings=None, **over):
        self.sleeps: list[float] = []
        self.st = make_settings(zdb, **{**(settings or {}), **over})
        self.transport = transport
        self.svc = ZipnetSyncService(self.st, transport_factory=lambda s: self.transport, sleep=self.sleeps.append)
        self.conn = store.connect(zdb)
        store.migrate(self.conn)

    def drain(self, limit=200):
        results = []
        for _ in range(limit):
            r = self.svc.step(self.conn)
            if r is None:
                return results
            results.append(r)
        raise AssertionError("sync did not settle")

    def count(self, section=None):
        if section:
            return self.conn.execute("SELECT COUNT(*) FROM zipnet_records WHERE section=?", (section,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM zipnet_records").fetchone()[0]

    def cp(self, section):
        return store.get_checkpoint(self.conn, section)


def wait_for(cond, timeout=15.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


class RunningWorker:
    """Runs ZipnetSyncService.run() in a thread against an empty fake source."""

    def __init__(self, zdb, **over):
        self.st = make_settings(zdb, sections=(MP,), **over)
        self.svc = ZipnetSyncService(self.st, transport_factory=lambda s: FakeTransport({}), sleep=lambda s: None)
        self.code = None
        self.svc.exit_token = "test-token"
        self.thread = threading.Thread(target=lambda: setattr(self, "code", self.svc.run()), daemon=True)
        self.conn = store.connect(zdb)

    def start(self):
        self.thread.start()
        assert wait_for(lambda: store.live_lease(self.conn) is not None), "worker never took the lease"
        return self

    def state(self):
        return store.get_worker_state(self.conn).get("state")

    def stop(self):
        self.svc.stop_event.set()
        self.thread.join(15)
        assert not self.thread.is_alive(), "worker did not shut down"
