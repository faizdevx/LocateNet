"""Loaders for datasets that need a licence request: AgeDB-30, CACD, YouTube Faces (YTF).

They are implemented so the protocols run unchanged once the data is placed in data/. Until then
`available()` is False and the experiments fall back (FG-NET / CALFW for age, synthetic
pseudo-tracks for video) and say so in their results.

Expected layouts:
  AgeDB  data/agedb/AgeDB/<id>_<name>_<age>_<gender>.jpg       + data/agedb/agedb_30_pairs.txt (optional)
  CACD   data/cacd/CACD2000/<age>_<Name>_<NNNN>.jpg
  YTF    data/ytf/frame_images_DB/<Name>/<video_no>/<frame>.jpg + data/ytf/splits.txt
"""
import json
import os
import re

import pandas as pd

AGEDB_NAME = re.compile(r"^(\d+)_(.+)_(\d+)_([mf])$", re.IGNORECASE)
CACD_NAME = re.compile(r"^(\d+)_(.+)_(\d{4})$")


def agedb_available(data_dir="data"):
    return os.path.isdir(os.path.join(data_dir, "agedb", "AgeDB"))


def load_agedb(data_dir="data"):
    root = os.path.join(data_dir, "agedb", "AgeDB")
    if not agedb_available(data_dir):
        raise FileNotFoundError("AgeDB needs a licence request (ibug.doc.ic.ac.uk/resources/agedb); place it in data/agedb/AgeDB")
    rows = []
    for fname in sorted(os.listdir(root)):
        m = AGEDB_NAME.match(os.path.splitext(fname)[0])
        if m:
            rows.append((os.path.join(root, fname), f"agedb_{m.group(2)}", "all", json.dumps({"age": int(m.group(3))})))
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])


def cacd_available(data_dir="data"):
    return os.path.isdir(os.path.join(data_dir, "cacd", "CACD2000"))


def load_cacd(data_dir="data"):
    root = os.path.join(data_dir, "cacd", "CACD2000")
    if not cacd_available(data_dir):
        raise FileNotFoundError("CACD not found (bcsiriuschen.github.io/CARC); place it in data/cacd/CACD2000")
    rows = []
    for fname in sorted(os.listdir(root)):
        m = CACD_NAME.match(os.path.splitext(fname)[0])
        if m:
            rows.append((os.path.join(root, fname), f"cacd_{m.group(2)}", "all", json.dumps({"age": int(m.group(1))})))
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "split", "attrs"])


def ytf_available(data_dir="data"):
    return os.path.isdir(os.path.join(data_dir, "ytf", "frame_images_DB"))


def load_ytf_tracks(data_dir="data"):
    """One row per video: identity, video_id, sorted frame paths."""
    root = os.path.join(data_dir, "ytf", "frame_images_DB")
    if not ytf_available(data_dir):
        raise FileNotFoundError("YouTube Faces needs a request (cs.tau.ac.il/~wolf/ytfaces); place it in data/ytf")
    rows = []
    for name in sorted(os.listdir(root)):
        person = os.path.join(root, name)
        if not os.path.isdir(person):
            continue
        for vid in sorted(os.listdir(person)):
            frames = sorted(os.path.join(person, vid, f) for f in os.listdir(os.path.join(person, vid)) if f.endswith(".jpg"))
            if frames:
                rows.append((name, f"{name}/{vid}", frames))
    return pd.DataFrame(rows, columns=["identity_id", "video_id", "frames"])
