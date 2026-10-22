import numpy as np
import pandas as pd

from eval.datasets.fgnet import parse
from eval.protocols.age_gap import bucket_of, fgnet_pairs


def test_fgnet_filename_parse():
    assert parse("001A02.JPG") == ("fgnet_001", 2)
    assert parse("010A07a.JPG") == ("fgnet_010", 7)
    assert parse("readme.txt") is None


def test_buckets():
    assert [bucket_of(g) for g in (0, 2, 3, 5, 6, 10, 11, 20, 21, 60)] == \
        ["0-2", "0-2", "3-5", "3-5", "6-10", "6-10", "11-20", "11-20", "21+", "21+"]


def test_impostors_are_age_matched_and_failed_images_score_minus_one():
    rows, emb = [], {}
    rng = np.random.default_rng(0)
    for s in range(6):
        for age in (2, 10, 30):
            p = f"s{s}_{age}.jpg"
            rows.append((p, f"subj{s}", "all", "{}"))
            emb[p] = rng.normal(size=8)
    man = pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])
    ages = np.array([2, 10, 30] * 6)
    del emb["s0_2.jpg"]
    pairs = fgnet_pairs(man, ages, emb, np.random.default_rng(1), impostors_per_genuine=2)
    gen, imp = pairs[pairs.same], pairs[~pairs.same]
    assert len(gen) == 6 * 3
    assert sorted(gen.gap.unique()) == sorted(imp.gap.unique())
    assert (gen[gen.subject == "subj0"].score == -1.0).sum() == 2
