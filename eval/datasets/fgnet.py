"""FG-NET aging database loader: 1002 photos of 82 subjects, ages 0-69, many child photos.

Layout (official mirror yanweifu.github.io/FG_NET_data, see scripts/get_eval_data.py):
  data/fgnet/FGNET/images/<SSS>A<AA>[a|b].JPG   SSS = subject id, AA = age in years at the photo
Manifest attrs: {"age": AA}. Many images are scanned prints (grey, low resolution).
"""
import json
import os
import re

import pandas as pd

DATASET_VERSION = "FG-NET (Lanitis et al.), 1002 images / 82 subjects, mirror yanweifu.github.io/FG_NET_data"
_NAME = re.compile(r"^(\d{3})A(\d{2})([a-z]?)$", re.IGNORECASE)


def image_dir(data_dir="data"):
    return os.path.join(data_dir, "fgnet", "FGNET", "images")


def available(data_dir="data"):
    return os.path.isdir(image_dir(data_dir))


def parse(fname):
    m = _NAME.match(os.path.splitext(fname)[0])
    if not m:
        return None
    return f"fgnet_{m.group(1)}", int(m.group(2))


def load_manifest(data_dir="data"):
    if not available(data_dir):
        raise FileNotFoundError("FG-NET not found. Run: python scripts/get_eval_data.py fgnet")
    rows = []
    for fname in sorted(os.listdir(image_dir(data_dir))):
        parsed = parse(fname)
        if parsed:
            ident, age = parsed
            rows.append((os.path.join(image_dir(data_dir), fname), ident, "all", json.dumps({"age": age})))
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])


def ages(manifest):
    return manifest["attrs"].map(lambda a: json.loads(a)["age"])
