from datetime import UTC, date, datetime

import pytest

from services.zipnet_normalize import (
    clean_free_text,
    clean_text,
    fingerprint,
    normalize_imei,
    normalize_query_registration,
    normalize_registration,
    validate_image_paths,
)
from services.zipnet_time import parse_source_date, to_ist_iso


class TestDates:
    def test_day_first_plain(self):
        p = parse_source_date("05/10/2026")
        assert p.ok and p.date == date(2026, 10, 5) and p.at_utc is None

    def test_day_first_with_time_is_ist_converted_to_utc(self):
        p = parse_source_date("25/10/2026 16:52:51")
        assert p.date == date(2026, 10, 25)
        assert p.at_utc == datetime(2026, 10, 25, 11, 22, 51, tzinfo=UTC)

    def test_am_pm_values_are_month_first(self):
        p = parse_source_date("10/24/2026 1:03:46 PM")
        assert p.date == date(2026, 10, 24)
        assert p.at_utc == datetime(2026, 10, 24, 7, 33, 46, tzinfo=UTC)

    def test_midnight_padding_is_not_a_time(self):
        p = parse_source_date("10/20/2026 12:00:00 AM")
        assert p.date == date(2026, 10, 20) and p.at_utc is None

    def test_iso(self):
        assert parse_source_date("2026-10-01").date == date(2026, 10, 1)

    @pytest.mark.parametrize("raw", ["31/02/2026", "garbage", "13/13/2026"])
    def test_invalid_flagged_not_guessed(self, raw):
        p = parse_source_date(raw)
        assert not p.ok and p.date is None and p.raw == raw

    @pytest.mark.parametrize("raw", [None, "", "  "])
    def test_empty_is_ok_none(self, raw):
        p = parse_source_date(raw)
        assert p.ok and p.date is None

    def test_ist_display(self):
        assert to_ist_iso(datetime(2026, 10, 25, 20, 0, tzinfo=UTC)) == "2026-10-26T01:30:00+05:30"
        # naive datetimes from SQLite are UTC
        assert to_ist_iso(datetime(2026, 10, 25, 20, 0)) == "2026-10-26T01:30:00+05:30"


class TestText:
    def test_strips_markup_and_control_chars(self):
        assert clean_text("<b>Hello</b>\x00  <script>x</script> world") == "Hello x world"

    def test_placeholders_become_none(self):
        assert clean_text("--") is None and clean_text("NA") is None and clean_text("   ") is None

    def test_redacts_phone_numbers_and_email(self):
        out = clean_free_text("near shop call 98765 43210 or +91-9876543210 / x@y.com")
        assert "98765" not in out and "9876543210" not in out and "x@y.com" not in out
        assert out.count("[redacted]") == 3

    def test_keeps_short_numbers(self):
        assert clean_free_text("H NO 487 sector 4") == "H NO 487 sector 4"


class TestRegistration:
    @pytest.mark.parametrize(
        "raw,norm,valid",
        [
            ("dl-3c ab 1234", "DL3CAB1234", True),
            ("HR26DK8337", "HR26DK8337", True),
            ("22 BH 1234 AA", "22BH1234AA", True),
            ("ZETOR26", "ZETOR26", False),
            ("DL.5S.CY.8370", "DL5SCY8370", True),
        ],
    )
    def test_normalization_keeps_raw(self, raw, norm, valid):
        r, n, v = normalize_registration(raw)
        assert r == raw and n == norm and v is valid

    def test_none(self):
        assert normalize_registration(None) == (None, None, False)

    def test_query_normalization(self):
        assert normalize_query_registration("dl 3c-ab") == "DL3CAB"


class TestImei:
    def test_valid_luhn(self):
        assert normalize_imei("49015420323751 8") == ("490154203237518", True)

    def test_invalid_checksum(self):
        assert normalize_imei("490154203237519") == ("490154203237519", False)

    def test_short(self):
        assert normalize_imei("12345")[1] is False


class TestImages:
    def test_accepts_known_upload_paths(self):
        ok, w = validate_image_paths(
            [
                "/ImageUploads/2026/MissingUploads/00007b74-fdfe-4e41-8712-c07755f5ed0c.jpg",
                "/ImageUploads/374ab200-2cca-4006-8eb5-10e8ea2f3558.jpg",
            ]
        )
        assert len(ok) == 2 and not w

    @pytest.mark.parametrize(
        "bad",
        [
            "https://evil.example/x.jpg",
            "/ImageUploads/../web.config",
            "/etc/passwd",
            "/ImageUploads/2026/Other/x.jpg",
            "/ImageUploads/x.svg",
            "//evil.example/ImageUploads/a.jpg",
        ],
    )
    def test_rejects_foreign_or_traversal(self, bad):
        ok, w = validate_image_paths([bad])
        assert ok == [] and w == ["image_url_rejected"]

    def test_non_list(self):
        assert validate_image_paths(None) == ([], [])
        assert validate_image_paths("x") == ([], ["image_urls_not_a_list"])


def test_fingerprint_is_order_independent():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
