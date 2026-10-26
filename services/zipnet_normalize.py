"""Reusable normalization and validation helpers for scraped values.

Scraped content is untrusted. These helpers strip markup and control characters,
redact phone numbers that leak into free-text fields, and validate identifiers.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any

_TAG = re.compile(r"<[^>]{0,500}>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
# Indian mobile numbers (optionally +91 / 0 prefixed) and e-mail addresses.
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_PLACEHOLDERS = {"", "-", "--", "na", "n/a", "nil", "null", "none", "0"}


def clean_text(value: Any, max_len: int = 300, *, keep_zero: bool = False) -> str | None:
    """Return sanitized single-line text, or None for empty/placeholder values."""
    if value is None or isinstance(value, bool):
        return None
    s = str(value)
    s = html.unescape(s)
    s = _TAG.sub(" ", s)
    s = _CTRL.sub("", s)
    s = _WS.sub(" ", s).strip()
    if not s or (s.lower() in _PLACEHOLDERS and not (keep_zero and s == "0")):
        return None
    return s[:max_len]


def redact_contacts(value: str | None) -> str | None:
    if value is None:
        return None
    out = _PHONE.sub("[redacted]", value)
    out = _EMAIL.sub("[redacted]", out)
    return out


def clean_free_text(value: Any, max_len: int = 300) -> str | None:
    """Free text that may contain contact details (places, descriptions)."""
    return redact_contacts(clean_text(value, max_len))


def to_int(value: Any, lo: int | None = None, hi: int | None = None) -> int | None:
    s = clean_text(value, 20, keep_zero=True)
    if s is None:
        return None
    m = re.match(r"^\s*(\d{1,6})", s)
    if not m:
        return None
    n = int(m.group(1))
    if (lo is not None and n < lo) or (hi is not None and n > hi):
        return None
    return n


# --- Indian vehicle registration -------------------------------------------------

_REG_STD = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")
_REG_BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


def normalize_registration(raw: Any) -> tuple[str | None, str | None, bool]:
    """Return (raw_clean, normalized, is_valid_format).

    Normalized form removes spaces, hyphens, dots and slashes and upper-cases,
    e.g. ``dl-3c ab 1234`` -> ``DL3CAB1234``. The raw value is always retained.
    Delhi registrations sometimes zero-pad the RTO code (``DL03``); both forms are
    left exactly as published because we cannot know which one the source meant.
    """
    raw_clean = clean_text(raw, 40)
    if raw_clean is None:
        return None, None, False
    norm = re.sub(r"[\s\-./]", "", raw_clean).upper()
    if not norm or not re.fullmatch(r"[A-Z0-9]+", norm):
        return raw_clean, None, False
    valid = bool(_REG_STD.match(norm) or _REG_BH.match(norm))
    return raw_clean, norm[:20], valid


def normalize_query_registration(q: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", q).upper()


# --- IMEI -------------------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def normalize_imei(raw: Any) -> tuple[str | None, bool]:
    s = clean_text(raw, 40)
    if s is None:
        return None, False
    digits = re.sub(r"\D", "", s)
    if len(digits) == 15:
        return digits, luhn_ok(digits)
    return (digits or None), False


# --- Images -----------------------------------------------------------------------

IMAGE_PATH_RE = re.compile(
    r"^/?ImageUploads/(?:\d{4}/(?:MissingUploads|UIDBUploads|UIPFUploads)/)?"
    r"[0-9A-Za-z][0-9A-Za-z_\-]{0,80}\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


def validate_image_paths(values: Any) -> tuple[list[str], list[str]]:
    """Keep only same-host image paths in the known upload folders. Returns (paths, warnings)."""
    if not isinstance(values, list):
        return [], ([] if values in (None, "") else ["image_urls_not_a_list"])
    ok: list[str] = []
    warnings: list[str] = []
    for v in values[:10]:
        if not isinstance(v, str):
            warnings.append("image_url_not_string")
            continue
        p = v.strip()
        if IMAGE_PATH_RE.match(p) and ".." not in p:
            ok.append("/" + p.lstrip("/"))
        else:
            warnings.append("image_url_rejected")
    return ok, warnings


# --- Fingerprints -----------------------------------------------------------------


def fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
