"""Time handling. Canonical storage is UTC; display and source interpretation use Asia/Kolkata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone

# India has no daylight saving, so a fixed offset equals Asia/Kolkata and needs no tzdata package.
IST = timezone(timedelta(hours=5, minutes=30), "IST")


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_ist_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(IST).isoformat()


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat them as UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


@dataclass(frozen=True)
class ParsedDate:
    raw: str | None
    date: date | None
    at_utc: datetime | None  # set only when the source supplied a time of day
    ok: bool


_DMY = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?)?$")
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?")


def parse_source_date(value: object) -> ParsedDate:
    """Parse a ZIPNET date string.

    Observed formats: ``dd/mm/yyyy``, ``dd/mm/yyyy HH:MM:SS`` and ``M/D/YYYY h:mm:ss AM``.
    Values carrying AM/PM are month-first (US style); all others are day-first.
    Source times are Indian Standard Time and are converted to UTC.
    """
    if value is None:
        return ParsedDate(None, None, None, True)
    raw = str(value).strip()
    if not raw:
        return ParsedDate(None, None, None, True)

    m = _DMY.match(raw)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ampm = m.group(7)
        day, month = (b, a) if ampm else (a, b)
        hh = int(m.group(4)) if m.group(4) else None
        mm = int(m.group(5)) if m.group(5) else 0
        ss = int(m.group(6)) if m.group(6) else 0
        if ampm and hh is not None:
            hh = hh % 12 + (12 if ampm.lower() == "pm" else 0)
        try:
            d = date(year, month, day)
        except ValueError:
            return ParsedDate(raw, None, None, False)
        at = None
        # "12:00:00 AM" is how the source pads date-only values; do not treat as a real time.
        if hh is not None and not (hh == 0 and mm == 0 and ss == 0):
            try:
                at = datetime(year, month, day, hh, mm, ss, tzinfo=IST).astimezone(UTC)
            except ValueError:
                return ParsedDate(raw, d, None, False)
        return ParsedDate(raw, d, at, True)

    m = _ISO.match(raw)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return ParsedDate(raw, None, None, False)
        at = None
        if m.group(4):
            at = datetime(
                d.year, d.month, d.day, int(m.group(4)), int(m.group(5)), int(m.group(6) or 0), tzinfo=IST
            ).astimezone(UTC)
        return ParsedDate(raw, d, at, True)

    return ParsedDate(raw, None, None, False)
