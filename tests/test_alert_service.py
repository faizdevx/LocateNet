from datetime import datetime, timedelta

from services.alert_service import AlertService

from conftest import add_case, add_match


def stamp(hours_ago):
    return (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")


def test_hotspot_when_same_case_matched_three_times_in_city():
    case_id = add_case()
    for i in range(3):
        add_match(case_id, f"s{i}", "aligarh", stamp(i))
    assert AlertService.check_repeated_sightings(case_id, "Aligarh") == (True, 3)


def test_two_matches_are_not_a_hotspot():
    case_id = add_case()
    for i in range(2):
        add_match(case_id, f"s{i}", "aligarh", stamp(i))
    assert AlertService.check_repeated_sightings(case_id, "aligarh") == (False, 2)


def test_different_cases_do_not_add_up():
    cases = [add_case(name=f"Person {i}") for i in range(3)]
    for i, case_id in enumerate(cases):
        add_match(case_id, f"s{i}", "aligarh", stamp(1))
    for case_id in cases:
        is_hotspot, count = AlertService.check_repeated_sightings(case_id, "aligarh")
        assert not is_hotspot and count == 1


def test_matches_older_than_48h_are_ignored():
    case_id = add_case()
    for i in range(3):
        add_match(case_id, f"s{i}", "aligarh", stamp(49 + i))
    assert AlertService.check_repeated_sightings(case_id, "aligarh") == (False, 0)


def test_other_city_is_ignored_and_location_is_normalised():
    case_id = add_case()
    add_match(case_id, "s0", "Aligarh ", stamp(1))
    add_match(case_id, "s1", " ALIGARH", stamp(1))
    add_match(case_id, "s2", "delhi", stamp(1))
    assert AlertService.check_repeated_sightings(case_id, "  aligarh ") == (False, 2)
