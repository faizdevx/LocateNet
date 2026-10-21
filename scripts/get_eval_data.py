"""Download the evaluation datasets into data/ (git-ignored) and verify checksums.

Usage:
    python scripts/get_eval_data.py            # everything
    python scripts/get_eval_data.py lfw        # only LFW (+ pairs.txt + attributes)
    python scripts/get_eval_data.py wider      # only WIDER FACE val + official eval ground truth

LFW checksums are the SHA-256 values scikit-learn pins for the same figshare files.
WIDER FACE and LFW-attribute files have no published SHA-256, so their hashes are pinned in
eval/config/checksums.json (recorded on the first download and verified on every later one).

RFW (Racial Faces in the Wild) needs a research licence and cannot be downloaded automatically;
this script prints the manual steps.
"""
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PINNED = os.path.join(ROOT, "eval", "config", "checksums.json")

FILES = {
    "lfw": [
        # Original (not funneled) LFW images, 250x250, official mirror used by scikit-learn.
        ("lfw/lfw.tgz", "https://ndownloader.figshare.com/files/5976018",
         "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"),
        # Official 10-fold View 2 pairs.
        ("lfw/pairs.txt", "https://ndownloader.figshare.com/files/5976006",
         "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"),
        # Kumar et al. attribute scores (gender, race, ...), used for the bias fallback.
        ("lfw/lfw_attributes.txt",
         "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt", None),
    ],
    "wider": [
        ("widerface/WIDER_val.zip",
         "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip", None),
        ("widerface/wider_face_split.zip",
         "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip", None),
        # Official evaluation code + easy/medium/hard ground-truth .mat files.
        ("widerface/eval_tools.zip",
         "http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip", None),
    ],
}

RFW_INSTRUCTIONS = """
RFW (Racial Faces in the Wild) must be requested manually:
  1. Read http://www.whdeng.cn/RFW/index.html and send the signed licence request
     (academic use only) to the dataset authors.
  2. Extract the archive so that these paths exist:
       data/RFW/test/data/{African,Asian,Caucasian,Indian}/<identity>/<image>.jpg
       data/RFW/test/txts/{African,Asian,Caucasian,Indian}/<Group>_pairs.txt
  3. Re-run: python -m eval run eval/config/bias.yaml
Until then the bias audit uses the LFW-attribute fallback.
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pinned():
    if os.path.exists(PINNED):
        with open(PINNED) as f:
            return json.load(f)
    return {}


def save_pinned(pinned):
    os.makedirs(os.path.dirname(PINNED), exist_ok=True)
    with open(PINNED, "w") as f:
        json.dump(pinned, f, indent=2, sort_keys=True)
        f.write("\n")


def download(rel, url):
    dest = os.path.join(DATA, rel)
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "LocateNet-eval/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
        print()
    os.replace(tmp, dest)
    return dest


def verify(rel, path, expected, pinned):
    digest = sha256(path)
    expected = expected or pinned.get(rel)
    if expected is None:
        pinned[rel] = digest
        print(f"  pinned sha256 {digest[:16]}... (first download, recorded in {os.path.relpath(PINNED, ROOT)})")
    elif digest != expected:
        raise SystemExit(f"Checksum mismatch for {rel}: expected {expected}, got {digest}. Delete it and retry.")
    else:
        print(f"  sha256 ok {rel}")


def extract(path):
    out = os.path.dirname(path)
    marker = path + ".extracted"
    if os.path.exists(marker):
        return
    if path.endswith(".tgz"):
        with tarfile.open(path) as t:
            t.extractall(out, filter="data")
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
    else:
        return
    open(marker, "w").close()
    print(f"  extracted {os.path.relpath(path, ROOT)}")


def main(argv):
    groups = argv or list(FILES)
    pinned = load_pinned()
    for group in groups:
        for rel, url, expected in FILES[group]:
            path = download(rel, url)
            verify(rel, path, expected, pinned)
            extract(path)
    save_pinned(pinned)
    print(RFW_INSTRUCTIONS)


if __name__ == "__main__":
    main(sys.argv[1:])
