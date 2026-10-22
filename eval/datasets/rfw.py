"""Racial Faces in the Wild (RFW) loader. Needs a manual licence request (see scripts/get_eval_data.py).

Expected layout:
  data/RFW/test/data/<Group>/<identity>/<image>.jpg
  data/RFW/test/txts/<Group>/<Group>_pairs.txt
Pairs file format (one per line): "id img_a img_b" for matched pairs (image numbers),
"id_a img_a id_b img_b" for mismatched pairs, like LFW.
"""
import os

import pandas as pd

GROUPS = ["African", "Asian", "Caucasian", "Indian"]
DATASET_VERSION = "RFW test set (Wang et al., ICCV 2019)"


def rfw_root(data_dir="data"):
    return os.path.join(data_dir, "RFW", "test")


def available(data_dir="data"):
    root = rfw_root(data_dir)
    return all(os.path.exists(os.path.join(root, "txts", g, f"{g}_pairs.txt")) for g in GROUPS)


def _image(root, group, identity, number):
    return os.path.join(root, "data", group, identity, f"{identity}_{int(number):04d}.jpg")


def load_pairs(data_dir="data"):
    """Columns: group, path_a, path_b, same."""
    if not available(data_dir):
        raise FileNotFoundError("RFW not found; see scripts/get_eval_data.py for the licence steps.")
    root, rows = rfw_root(data_dir), []
    for g in GROUPS:
        with open(os.path.join(root, "txts", g, f"{g}_pairs.txt")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    rows.append((g, _image(root, g, parts[0], parts[1]), _image(root, g, parts[0], parts[2]), True))
                elif len(parts) == 4:
                    rows.append((g, _image(root, g, parts[0], parts[1]), _image(root, g, parts[2], parts[3]), False))
    return pd.DataFrame(rows, columns=["group", "path_a", "path_b", "same"])


def load_manifest(data_dir="data"):
    pairs = load_pairs(data_dir)
    rows = {}
    for _, r in pairs.iterrows():
        for p in (r.path_a, r.path_b):
            identity = os.path.basename(os.path.dirname(p))
            rows[p] = (p, identity, "test", f'{{"race": "{r.group.lower()}"}}')
    return pd.DataFrame(list(rows.values()), columns=["image_path", "identity_id", "split", "attrs"])
