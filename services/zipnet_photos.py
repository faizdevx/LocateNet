"""Display-only proxy for the record photographs published by ZIPNET.

Photos are streamed from the source for display and kept only in a small in-memory LRU cache. Nothing
is written to disk, and nothing is ever passed to face detection or the FAISS index.

SSRF protection: the host comes from configuration, and the path comes from a database row that was
validated against a strict pattern at import time and is validated again here.
"""

from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request
from collections import OrderedDict

from .zipnet_normalize import IMAGE_PATH_RE
from .zipnet_settings import ZipnetSettings

log = logging.getLogger("locatenet.zipnet.photos")

MAX_BYTES = 2_000_000
CACHE_ITEMS = 200
ALLOWED_TYPES = {"image/jpeg", "image/png"}

_cache: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def fetch_photo(st: ZipnetSettings, image_path: str) -> tuple[str, bytes] | None:
    """Return (content_type, bytes) or None when unavailable. Never raises."""
    if not st.enabled:  # sync disabled means no outbound requests at all
        return None
    path = image_path.strip()
    if not IMAGE_PATH_RE.match(path.lstrip("/")) or ".." in path or "//" in path.lstrip("/"):
        return None
    if not st.base_url.startswith(("https://", "http://")):
        return None
    with _lock:
        hit = _cache.get(path)
        if hit is not None:
            _cache.move_to_end(path)
            return hit
    url = st.base_url.rstrip("/") + "/" + path.lstrip("/")
    req = urllib.request.Request(url, headers={"User-Agent": st.user_agent})  # noqa: S310 - scheme checked above
    try:
        with urllib.request.urlopen(req, timeout=min(st.request_timeout, 15)) as r:  # noqa: S310
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            data = r.read(MAX_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as e:
        log.info("photo unavailable (%s)", type(e).__name__)
        return None
    if ctype not in ALLOWED_TYPES or not data or len(data) > MAX_BYTES:
        return None
    with _lock:
        _cache[path] = (ctype, data)
        while len(_cache) > CACHE_ITEMS:
            _cache.popitem(last=False)
    return ctype, data
