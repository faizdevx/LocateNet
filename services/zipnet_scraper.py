"""HTTP access to the ZIPNET public endpoints, built on Scrapling's FetcherSession.

The client is deliberately honest and conservative:
- no browser impersonation, no stealth headers, an identifying User-Agent;
- a minimum delay between requests and exponential backoff with jitter;
- 401/403/429 or an HTML login/block page stops the section instead of retrying (no evasion).
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("locatenet.zipnet.http")


class SourceError(Exception):
    """Base class for source access failures."""


class SourceBlocked(SourceError):
    """The source refused access (auth, rate limit, block page). Never retried or evaded."""


class SourceUnavailable(SourceError):
    """Transient failure that persisted after all retries."""


class SchemaChanged(SourceError):
    """The page or JSON no longer has the structure the adapter expects."""


class SyncHalted(SourceError):
    """A stop, pause or shutdown request arrived; checkpointed work is kept."""


class TransportError(Exception):
    """Network-level failure (connection, TLS, timeout). Retried with backoff."""


@dataclass
class RawResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    # Present only for HTML pages: callable that checks a CSS selector exists.
    has_selector: Callable[[str], bool] | None = None

    @property
    def content_type(self) -> str:
        for k, v in self.headers.items():
            if k.lower() == "content-type":
                return v.lower()
        return ""


class Transport(Protocol):
    def get(self, url: str, headers: dict[str, str]) -> RawResponse: ...
    def post(self, url: str, data: dict[str, str], headers: dict[str, str]) -> RawResponse: ...
    def close(self) -> None: ...


class ScraplingTransport:
    """Transport backed by ``scrapling.fetchers.FetcherSession`` (curl_cffi under the hood)."""

    def __init__(self, timeout: float, user_agent: str):
        from scrapling.fetchers import FetcherSession  # imported lazily: tests never load Scrapling

        # Scrapling logs every request at INFO through its own handler; the pipeline logs summaries.
        logging.getLogger("scrapling").setLevel(logging.WARNING)
        self._session = FetcherSession(
            impersonate=None,  # no browser impersonation
            stealthy_headers=False,  # no stealth
            timeout=timeout,
            retries=1,  # retries are handled by SourceClient so they are counted and backed off
            follow_redirects="safe",
            headers={"User-Agent": user_agent, "Accept-Language": "en-IN,en;q=0.8"},
        )
        self._client = self._session.__enter__()

    @staticmethod
    def _wrap(resp: Any) -> RawResponse:
        headers = {str(k): str(v) for k, v in dict(resp.headers or {}).items()}

        def has_selector(sel: str) -> bool:
            try:
                return bool(resp.css(sel))
            except Exception:
                return False

        return RawResponse(status=int(resp.status), headers=headers, body=bytes(resp.body),
                           has_selector=has_selector)

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> RawResponse:
        from curl_cffi import CurlError
        from curl_cffi.requests.exceptions import RequestException

        try:
            return self._wrap(fn(*args, **kwargs))
        except (CurlError, RequestException, OSError) as e:
            raise TransportError(f"{type(e).__name__}: {e}") from e

    def get(self, url: str, headers: dict[str, str]) -> RawResponse:
        return self._call(self._client.get, url, headers=headers)

    def post(self, url: str, data: dict[str, str], headers: dict[str, str]) -> RawResponse:
        return self._call(self._client.post, url, data=data, headers=headers)

    def close(self) -> None:
        try:
            self._session.__exit__(None, None, None)
        except Exception:  # pragma: no cover
            log.debug("session close failed", exc_info=True)


@dataclass
class RequestStats:
    requests: int = 0
    retries: int = 0
    errors: list[str] = field(default_factory=list)


class SourceClient:
    def __init__(
        self,
        transport: Transport,
        *,
        delay_seconds: float,
        max_retries: int,
        backoff_base: float,
        backoff_max: float,
        should_halt: Callable[[], bool] = lambda: False,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.transport = transport
        self.delay = delay_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.should_halt = should_halt
        self._sleep = sleep
        self._mono = monotonic
        self._last: float | None = None
        self.stats = RequestStats()

    def _throttle(self) -> None:
        if self._last is not None:
            wait = self.delay - (self._mono() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._mono()

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        return base * (0.5 + random.random() / 2)  # noqa: S311 - jitter, not security

    def _request(self, method: str, url: str, data: dict[str, str] | None, referer: str | None) -> RawResponse:
        headers = {"Referer": referer} if referer else {}
        if method == "POST":
            headers["X-Requested-With"] = "XMLHttpRequest"
            headers["Accept"] = "application/json, text/javascript, */*; q=0.01"
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            if self.should_halt():
                raise SyncHalted("stop requested")
            self._throttle()
            self.stats.requests += 1
            try:
                resp = (self.transport.post(url, data or {}, headers) if method == "POST"
                        else self.transport.get(url, headers))
            except TransportError as e:
                last_err = str(e)[:300]
            else:
                if resp.status in (401, 403, 407, 429, 451):
                    raise SourceBlocked(f"HTTP {resp.status} from the source; section stopped (no evasion attempted)")
                if 200 <= resp.status < 300:
                    return resp
                if resp.status in (404, 405, 410):
                    raise SchemaChanged(f"HTTP {resp.status} from the source; endpoint moved or removed")
                last_err = f"HTTP {resp.status}"
            self.stats.errors.append(last_err)
            if attempt < self.max_retries:
                self.stats.retries += 1
                delay = self._backoff(attempt)
                log.warning("request failed (%s), retry %d/%d in %.1fs", last_err, attempt, self.max_retries - 1,
                            delay)
                self._sleep(delay)
        raise SourceUnavailable(f"request failed after {self.max_retries} attempts: {last_err}")

    def get_page(self, url: str) -> RawResponse:
        return self._request("GET", url, None, None)

    def post_json(self, url: str, data: dict[str, str], referer: str) -> dict[str, Any]:
        resp = self._request("POST", url, data, referer)
        ctype = resp.content_type
        if "json" not in ctype:
            text = resp.body[:2000].decode("utf-8", "replace").lower()
            if "login" in text or "captcha" in text or "access denied" in text:
                raise SourceBlocked("the source returned an HTML login/block page instead of JSON")
            raise SchemaChanged(f"the source returned {ctype or 'unknown content-type'} instead of JSON")
        try:
            payload = json.loads(resp.body.decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError) as e:
            raise SchemaChanged("the source returned invalid JSON") from e
        if not isinstance(payload, dict):
            raise SchemaChanged("the source JSON is not an object")
        return payload

    def close(self) -> None:
        self.transport.close()
