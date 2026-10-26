import json
from datetime import date

import pytest

from services.zipnet_adapters import ADAPTERS, get_adapter
from services.zipnet_adapters import RowRejected
from services.zipnet_store import jsonable
from tests.zipnet_helpers import CANARIES, ID_FIELD, make_row

BASE = "https://zipnet.delhipolice.gov.in"
CATEGORIES = list(ADAPTERS)


@pytest.mark.parametrize("category", CATEGORIES)
def test_every_category_parses_fixture_row(category):
    a = get_adapter(category, BASE)
    nr = a.normalize(make_row(category, 5))
    assert nr.category == category
    assert nr.source_id == "1005" and nr.source_id_num == 1005
    assert nr.state == "DELHI" and nr.district == "NORTH WEST DISTRICT"
    assert nr.source_created_date == date(2026, 10, 6)
    assert nr.title
    assert nr.warnings == []


@pytest.mark.parametrize("category", CATEGORIES)
def test_sensitive_fields_never_survive_normalization(category):
    nr = get_adapter(category, BASE).normalize(make_row(category, 7))
    blob = json.dumps(jsonable(nr.content()), ensure_ascii=False)
    for canary in CANARIES:
        assert canary not in blob, f"{category} leaked {canary}"
    assert "religion" not in blob.lower()


@pytest.mark.parametrize("category", CATEGORIES)
def test_all_fields_missing_except_id_does_not_crash(category):
    a = get_adapter(category, BASE)
    nr = a.normalize({ID_FIELD[category]: 77})
    assert nr.source_id == "77"
    assert nr.state is None and nr.reported_date is None
    assert nr.title  # a neutral placeholder title, never invented data


@pytest.mark.parametrize("category", CATEGORIES)
def test_missing_or_bad_id_is_rejected(category):
    a = get_adapter(category, BASE)
    with pytest.raises(RowRejected):
        a.normalize({"State": "DELHI"})
    with pytest.raises(RowRejected):
        a.normalize({ID_FIELD[category]: "abc"})


@pytest.mark.parametrize("category", CATEGORIES)
def test_required_key_detection(category):
    a = get_adapter(category, BASE)
    row = make_row(category, 1)
    assert a.missing_required_keys(row) == []
    row.pop("State")
    assert "State" in a.missing_required_keys(row)


def test_missing_person_specifics():
    nr = get_adapter("missing_persons", BASE).normalize(make_row("missing_persons", 3))
    assert nr.name == "TEST PERSON 3" and nr.status == "Untraced"
    assert nr.reported_date_kind == "missing_since"
    assert nr.published_location == "near test market"
    assert nr.case_reference == "MP-TEST-0003"
    assert nr.attributes["birth_year_reported"] == 2001
    assert nr.image_urls == ["/ImageUploads/2026/MissingUploads/00000003-aaaa-bbbb-cccc-000000000000.jpg"]


def test_unidentified_body_age_and_found_time():
    nr = get_adapter("unidentified_dead_bodies", BASE).normalize(make_row("unidentified_dead_bodies", 2))
    assert (nr.age_min, nr.age_max) == (30, 35)
    assert nr.reported_date == date(2026, 10, 3) and nr.reported_date_kind == "found"
    assert nr.attributes["found_at_utc"].isoformat() == "2026-10-03T09:00:00+00:00"
    assert nr.status is None  # source publishes no status; we do not invent one


def test_matched_flag_mapping_is_literal():
    a = get_adapter("unidentified_dead_bodies", BASE)
    assert a.normalize(make_row("unidentified_dead_bodies", 1, Matched=True)).status == "Matched"
    assert a.normalize(make_row("unidentified_dead_bodies", 1, Matched="No")).status == "Not matched"
    assert a.normalize(make_row("unidentified_dead_bodies", 1, Matched="Pending DNA")).status == "Pending DNA"


def test_vehicle_registration_raw_and_normalized():
    nr = get_adapter("stolen_vehicles", BASE).normalize(make_row("stolen_vehicles", 4))
    assert nr.vehicle_registration_raw == "DL 3C AB 1004"
    assert nr.vehicle_registration_norm == "DL3CAB1004"
    assert nr.attributes["registration_valid_format"] is True
    assert nr.reported_date_kind == "stolen"
    assert "CANARYENGINE" not in json.dumps(jsonable(nr.content()))


def test_nonstandard_registration_is_flagged_not_rewritten():
    nr = get_adapter("stolen_vehicles", BASE).normalize(make_row("stolen_vehicles", 4, VehicleRegistrationNo="ZETOR26"))
    assert nr.vehicle_registration_raw == "ZETOR26"
    assert "registration_nonstandard_format" in nr.warnings


def test_mobile_imei_and_dropped_numbers():
    nr = get_adapter("missing_mobiles", BASE).normalize(make_row("missing_mobiles", 9))
    assert nr.imei == "490154203237518" and nr.attributes["imei_valid"] is True
    nr2 = get_adapter("missing_mobiles", BASE).normalize(make_row("missing_mobiles", 9, IMEINo="12345"))
    assert "imei_failed_checksum_or_length" in nr2.warnings


def test_unparseable_date_is_warned():
    nr = get_adapter("missing_persons", BASE).normalize(make_row("missing_persons", 1, DateFrom="31/31/2026"))
    assert "unparsed_date:DateFrom" in nr.warnings
    assert nr.extraction_status == "partial"


def test_phone_in_free_text_location_is_redacted():
    nr = get_adapter("missing_persons", BASE).normalize(
        make_row("missing_persons", 1, MissingFrom="from home, call 9123456789")
    )
    assert "9123456789" not in nr.published_location


def test_fingerprint_changes_with_status():
    a = get_adapter("missing_persons", BASE)
    one = a.normalize(make_row("missing_persons", 1))
    two = a.normalize(make_row("missing_persons", 1, TracingStatus="Traced"))
    assert one.fingerprint != two.fingerprint
    assert one.fingerprint == a.normalize(make_row("missing_persons", 1)).fingerprint
