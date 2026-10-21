"""WIDER FACE validation loader (images + official easy/medium/hard ground truth).

Layout after scripts/get_eval_data.py wider:
  data/widerface/WIDER_val/images/<event>/<image>.jpg
  data/widerface/eval_tools/ground_truth/wider_face_val.mat (+ wider_{easy,medium,hard}_val.mat)
Manifest: image_path, identity_id (= event), split ('val'), attrs (json, empty).
"""
import glob
import os

import pandas as pd

DATASET_VERSION = "WIDER FACE val (Yang et al., CVPR 2016), official eval_tools ground truth"


def wider_root(data_dir="data"):
    return os.path.join(data_dir, "widerface")


def _find(data_dir, pattern):
    hits = glob.glob(os.path.join(wider_root(data_dir), "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under data/widerface. Run: python scripts/get_eval_data.py wider")
    return hits[0]


def gt_dir(data_dir="data"):
    return os.path.dirname(_find(data_dir, "wider_face_val.mat"))


def images_dir(data_dir="data"):
    # The images folder is the parent of the event folders (0--Parade, 1--Handshaking, ...).
    return os.path.dirname(_find(data_dir, "0--Parade"))


def load_manifest(data_dir="data"):
    root = images_dir(data_dir)
    rows = []
    for event in sorted(os.listdir(root)):
        for img in sorted(glob.glob(os.path.join(root, event, "*.jpg"))):
            rows.append((img, event, "val", "{}"))
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])
