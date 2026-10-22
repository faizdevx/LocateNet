import numpy as np
import pytest

from services.track_aggregator import Observation, aggregate, representative, split_track


def unit(*xs):
    v = np.array(xs, dtype="float32")
    return v / np.linalg.norm(v)


def obs(k, emb, det=0.9, px=100.0, blur=500.0):
    return Observation(k, emb, det, px, blur)


A, B = unit(1, 0, 0), unit(0, 1, 0)


def test_id_switch_is_split_at_the_boundary():
    track = [obs(k, unit(1, 0.05 * k, 0)) for k in range(5)] + [obs(k, unit(0, 1, 0.05 * k)) for k in range(5, 10)]
    segs = split_track(track, split_threshold=0.5)
    assert [len(s) for s in segs] == [5, 5]
    assert [o.frame_index for o in segs[1]] == list(range(5, 10))


def test_clean_track_is_not_split_and_order_does_not_matter():
    track = [obs(k, unit(1, 0.1 * (k % 2), 0)) for k in (3, 1, 2, 0)]
    segs = split_track(track, split_threshold=0.5)
    assert len(segs) == 1 and [o.frame_index for o in segs[0]] == [0, 1, 2, 3]


def test_aggregation_strategies():
    frames = [obs(0, A, det=0.9), obs(1, B, det=0.1)]
    assert aggregate(frames, "best_frame").shape == (2, 3)
    assert np.allclose(aggregate(frames, "mean")[0], unit(1, 1, 0))
    qwm = aggregate(frames, "quality_weighted_mean")[0]
    assert qwm @ A > qwm @ B                      # the high-quality frame dominates
    top1 = aggregate(frames * 1, "top_n_quality_mean", top_n=1)[0]
    assert np.allclose(top1, A)
    with pytest.raises(ValueError):
        aggregate(frames, "median")
    with pytest.raises(ValueError):
        aggregate([], "mean")


def test_representative_picks_best_quality_first():
    frames = [obs(0, A, det=0.6), obs(1, A, det=0.95), obs(2, A, det=0.8), obs(3, A, det=0.7, px=20)]
    assert [o.frame_index for o in representative(frames, 3)] == [1, 2, 0]
