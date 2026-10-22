"""CALFW (Cross-Age LFW, Zheng et al. 2017) loader: 6000 1:1 pairs chosen for large age gaps.

Layout (official Google Drive archive, see scripts/get_eval_data.py):
  data/calfw/calfw/images&landmarks/images&landmarks/images/<Name>_<NNNN>.jpg   (250x250, unaligned)
  data/calfw/calfw/pairs_CALFW.txt   two lines per pair: "<file> <label>", label 1 = same person
CALFW publishes no per-image age, so attrs carry age = null; age-gap buckets come from FG-NET.
Folds: 10 consecutive blocks of 600 pairs, as in the LFW protocol.
"""
import os

import pandas as pd

DATASET_VERSION = "CALFW (Zheng, Deng, Hu 2017), official archive, pairs_CALFW.txt"


def calfw_root(data_dir="data"):
    return os.path.join(data_dir, "calfw", "calfw")


def image_dir(data_dir="data"):
    return os.path.join(calfw_root(data_dir), "images&landmarks", "images&landmarks", "images")


def available(data_dir="data"):
    return os.path.exists(os.path.join(calfw_root(data_dir), "pairs_CALFW.txt"))


def _identity(fname):
    return fname.rsplit("_", 1)[0]


def load_pairs(data_dir="data"):
    """Columns: fold, path_a, path_b, same."""
    if not available(data_dir):
        raise FileNotFoundError("CALFW not found. Run: python scripts/get_eval_data.py calfw")
    with open(os.path.join(calfw_root(data_dir), "pairs_CALFW.txt")) as f:
        lines = [line.split() for line in f if line.strip()]
    # The label column is the fold number (1-10) for genuine pairs and 0 for impostor pairs; the
    # 3000 impostor pairs follow in fold order, 300 per fold.
    imgs = image_dir(data_dir)
    rows, n_neg = [], 0
    for k in range(0, len(lines) - 1, 2):
        (a, la), (b, _) = lines[k], lines[k + 1]
        same = la != "0"
        fold = int(la) - 1 if same else n_neg // 300
        n_neg += 0 if same else 1
        rows.append((fold, os.path.join(imgs, a), os.path.join(imgs, b), same))
    return pd.DataFrame(rows, columns=["fold", "path_a", "path_b", "same"])


def load_manifest(data_dir="data"):
    pairs = load_pairs(data_dir)
    paths = sorted(set(pairs["path_a"]) | set(pairs["path_b"]))
    return pd.DataFrame([(p, _identity(os.path.splitext(os.path.basename(p))[0]), "test", '{"age": null}') for p in paths],
                        columns=["image_path", "identity_id", "split", "attrs"])
