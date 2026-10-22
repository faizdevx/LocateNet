"""Labeled Faces in the Wild loader.

Manifest columns: image_path, identity_id, split, attrs (json string).
Images: data/lfw/lfw/<Name>/<Name>_<NNNN>.jpg (original 250x250 release, figshare 5976018).
"""
import json
import os

import pandas as pd

DATASET_VERSION = "LFW original (lfw.tgz, figshare 5976018, sha256 055f7d9c...); pairs.txt View 2"
RACE_COLUMNS = ["Asian", "White", "Black", "Indian"]


def lfw_root(data_dir="data"):
    return os.path.join(data_dir, "lfw", "lfw")


def image_path(root, name, number):
    return os.path.join(root, name, f"{name}_{int(number):04d}.jpg")


def load_manifest(data_dir="data"):
    root = lfw_root(data_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found. Run: python scripts/get_eval_data.py lfw")
    rows = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".jpg"):
                rows.append((os.path.join(folder, fname), name, "all", "{}"))
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])


def _name_tokens(name):
    return frozenset(t for t in name.lower().replace("-", "_").split("_") if t)


def identity_aliases(names, extra_aliases=None):
    """
    LFW files some people under two names. Merge them without looking at any face data:
      * names with the same tokens in a different order (e.g. Yingfan_Wang / Wang_Yingfan)
      * an explicit list of spelling variants (eval/config/openset.yaml: identity_aliases)
    Returns {alias_name: canonical_name}; canonical = alphabetically first name.
    """
    by_tokens = {}
    for n in sorted(names):
        by_tokens.setdefault(_name_tokens(n), []).append(n)
    alias = {}
    for group in by_tokens.values():
        for n in group[1:]:
            alias[n] = group[0]
    for a, b in (extra_aliases or {}).items():
        if a in names and b in names:
            canon = min(a, b)
            alias[max(a, b)] = canon
    return alias


def load_pairs(data_dir="data"):
    """Official View 2 pairs: 10 folds of 300 matched + 300 mismatched pairs.
    Returns a DataFrame with columns fold, path_a, path_b, same."""
    root = lfw_root(data_dir)
    path = os.path.join(data_dir, "lfw", "pairs.txt")
    with open(path) as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]
    n_folds, n_per = int(lines[0][0]), int(lines[0][1])
    rows, i = [], 1
    for fold in range(n_folds):
        for _ in range(n_per):
            name, a, b = lines[i]
            rows.append((fold, image_path(root, name, a), image_path(root, name, b), True))
            i += 1
        for _ in range(n_per):
            name_a, a, name_b, b = lines[i]
            rows.append((fold, image_path(root, name_a, a), image_path(root, name_b, b), False))
            i += 1
    return pd.DataFrame(rows, columns=["fold", "path_a", "path_b", "same"])


def load_attributes(data_dir="data"):
    """
    Kumar et al. LFW attribute scores (classifier outputs; positive means the attribute is present).
    Returns one row per image: identity_id, image_path, Male, Asian, White, Black, Indian.
    """
    path = os.path.join(data_dir, "lfw", "lfw_attributes.txt")
    with open(path) as f:
        f.readline()  # format line
        header = f.readline().lstrip("#").strip().split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    df = pd.DataFrame(rows, columns=header)
    root = lfw_root(data_dir)
    df["identity_id"] = df["person"].str.replace(" ", "_")
    df["image_path"] = [image_path(root, n, k) for n, k in zip(df["identity_id"], df["imagenum"])]
    for col in ["Male"] + RACE_COLUMNS:
        df[col] = df[col].astype(float)
    return df[["identity_id", "image_path", "Male"] + RACE_COLUMNS]


def identity_groups(attrs, min_margin=0.0):
    """
    Assign one demographic group per identity from the noisy attribute scores:
      race   = argmax of the mean Asian/White/Black/Indian score over that identity's images,
               'unknown' when the top mean score is <= min_margin
      gender = 'male' if the mean Male score > 0 else 'female'
    Returns a DataFrame indexed by identity_id with columns race, gender, n_images.
    """
    g = attrs.groupby("identity_id")
    means = g[["Male"] + RACE_COLUMNS].mean()
    race = means[RACE_COLUMNS].idxmax(axis=1)
    race = race.where(means[RACE_COLUMNS].max(axis=1) > min_margin, "unknown")
    out = pd.DataFrame({
        "race": race.str.lower(),
        "gender": (means["Male"] > 0).map({True: "male", False: "female"}),
        "n_images": g.size(),
    })
    return out


def attach_attrs(manifest, groups):
    """Write race/gender into the manifest's attrs json column."""
    lookup = groups.to_dict("index")
    manifest = manifest.copy()
    manifest["attrs"] = [
        json.dumps({k: lookup[i][k] for k in ("race", "gender")} if i in lookup else {})
        for i in manifest["identity_id"]
    ]
    return manifest
