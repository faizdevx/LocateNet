import pytest

import config
from pipeline.scoring import (CATEGORY_HIGH, CATEGORY_NONE, CATEGORY_POSSIBLE, categorize,
                              match_fields, select_match)


@pytest.mark.parametrize("similarity,expected", [
    (0.0, CATEGORY_NONE),
    (config.MATCH_THRESHOLD - 0.001, CATEGORY_NONE),
    (config.MATCH_THRESHOLD, CATEGORY_POSSIBLE),
    (config.HIGH_CONF_THRESHOLD - 0.001, CATEGORY_POSSIBLE),
    (config.HIGH_CONF_THRESHOLD, CATEGORY_HIGH),
    (1.0, CATEGORY_HIGH),
])
def test_category_boundaries(similarity, expected):
    assert categorize(similarity) == expected


def test_below_threshold_gives_no_match_id():
    case_id, similarity = select_match([0.2], [7])
    assert case_id == -1
    assert similarity == pytest.approx(0.2)


def test_at_threshold_matches():
    case_id, _ = select_match([config.MATCH_THRESHOLD], [7])
    assert case_id == 7


def test_match_fields_below_threshold():
    fields = match_fields([0.1], [3])
    assert fields["id"] == -1
    assert fields["category"] == CATEGORY_NONE
    assert fields["label"] == "Unknown"
    assert fields["percentage"] == 10.0


def test_empty_results_give_no_match():
    assert select_match([], []) == (-1, 0.0)
