import numpy as np
import pytest

from services.template_matcher import CaseTemplate, TemplateMatcher, case_score, recency_weights


def unit(*xs):
    v = np.array(xs, dtype="float32")
    return v / np.linalg.norm(v)


E1, E2, E3 = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)
PROBE = unit(1, 1, 0)  # cos 0.7071 to E1 and E2, 0 to E3


def template(vectors, qualities=None, years=None, cid=1):
    vecs = np.stack(vectors)
    return CaseTemplate(cid, vecs, np.array(qualities or [1.0] * len(vecs)),
                        np.array(years if years is not None else [np.nan] * len(vecs)), list(range(len(vecs))))


def test_max_takes_best_photo():
    assert case_score(template([E1, E3]), PROBE, "max") == pytest.approx(0.7071, abs=1e-4)


def test_top2_mean_averages_two_best():
    assert case_score(template([E1, E2, E3]), PROBE, "top2_mean") == pytest.approx(0.7071, abs=1e-4)
    assert case_score(template([E1, E3]), PROBE, "top2_mean") == pytest.approx(0.7071 / 2, abs=1e-4)


def test_single_photo_strategies_agree():
    t = template([E1])
    scores = {s: case_score(t, PROBE, s) for s in ("max", "mean_template", "quality_weighted_mean", "top2_mean")}
    assert max(scores.values()) - min(scores.values()) < 1e-6


def test_mean_template_is_cosine_to_normalised_mean():
    # mean of E1 and E2 is (1,1,0)/2 -> normalised equals PROBE -> cosine 1
    assert case_score(template([E1, E2]), PROBE, "mean_template") == pytest.approx(1.0, abs=1e-6)


def test_quality_weighted_mean_follows_weights():
    t = template([E1, E3], qualities=[1.0, 0.0001])
    assert case_score(t, PROBE, "quality_weighted_mean") == pytest.approx(0.7071, abs=1e-3)
    assert case_score(t, PROBE, "mean_template") == pytest.approx(0.5, abs=1e-3)


def test_recency_weights():
    w = recency_weights([0, 5, np.nan], tau=5)
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(np.exp(-1))
    assert w[2] == pytest.approx(np.median([1.0, np.exp(-1)]))
    assert list(recency_weights([1, 2], tau=None)) == [1.0, 1.0]


def test_recency_penalises_old_photos_in_max():
    t = template([E1, unit(1, 0.9, 0)], years=[0, 20])
    plain = case_score(t, PROBE, "max")
    recent = case_score(t, PROBE, "max", tau=5, recency_penalty=0.5)
    assert recent < plain


def test_matcher_groups_vector_hits_by_case_and_reranks():
    rows = [(10, 1, E1, 1, None), (11, 1, E3, 1, None), (20, 2, unit(1, 0.2, 0), 1, None)]
    m = TemplateMatcher.from_rows(rows, strategy="max")
    assert m.search_k(5) == 10 and m.vector_to_case == {10: 1, 11: 1, 20: 2}
    scores, cases = m.rank(PROBE, vector_hits=(np.array([0.8, 0.7]), np.array([20, 10])), top_k=5)
    assert list(cases) == [2, 1]
    assert scores[0] >= scores[1]


def test_mean_strategy_uses_in_memory_template_index():
    rows = [(10, 1, E1, 1, None), (11, 1, E2, 1, None), (20, 2, E3, 1, None)]
    scores, cases = TemplateMatcher.from_rows(rows, strategy="mean_template").rank(PROBE, top_k=2)
    assert list(cases) == [1, 2] and scores[0] == pytest.approx(1.0, abs=1e-6)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        TemplateMatcher([], strategy="median")
