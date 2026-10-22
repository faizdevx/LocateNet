"""Experiment `robustness`: how image degradation (phone / CCTV realism) changes failure-to-enrol
and misses at the chosen threshold.

Only PROBE images are degraded, at the image level BEFORE detection, so detection failures count.
Probes: a seeded sample of mated and non-mated probes from the LFW open-set TEST split; gallery:
the undegraded test gallery of 1000 identities. Threshold: the recommended MATCH_THRESHOLD from
eval/results/operating_point.json.

Degradations (levels in eval/config/robustness.yaml):
  downscale  face width w px: resize the whole image so the detected face is w px wide (INTER_AREA),
             then upsample back to the original size (INTER_LINEAR)
  blur       Gaussian blur, sigma
  jpeg       JPEG re-encode at quality q
  gamma      brightness gamma: out = 255 * (in/255) ** gamma (gamma < 1 brightens)
  occlusion  grey rectangle over the lower f of the face box height (mask-like)
  cctv       combined: downscale to face width w, Gaussian sensor noise (sigma 4) and JPEG q=40
             applied at the low resolution, then upsampled (closer to a real low-res camera)
Per probe we also store det_score, face width and Laplacian blur variance of the detected face,
which the quality-gate experiment reuses.
"""
import os

import cv2
import numpy as np
import pandas as pd

from eval import metrics as M
from eval.common import CACHE_DIR, load_result, metadata, plot_path, save_result
from eval.datasets import lfw
from eval.embed import STATUS_OK, cache_name, embed_paths
from eval.protocols import openset

DEFAULT_LEVELS = {
    "downscale": [16, 24, 32, 48, 64, 112],
    "blur": [0, 1, 2, 3, 5],
    "jpeg": [95, 50, 30, 15],
    "gamma": [0.4, 0.7, 1.0, 1.5],
    "occlusion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "cctv": [16, 24, 32, 48, 64],
}


def degrade(image, kind, level, face_box):
    h, w = image.shape[:2]
    if kind == "downscale":
        face_w = max(float(face_box[2] - face_box[0]), 1.0)
        scale = min(float(level) / face_w, 1.0)
        if scale >= 1.0:
            return image
        small = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "cctv":
        # combined: downscale to face width w, add sensor noise and JPEG q=40 at low resolution, upsample
        face_w = max(float(face_box[2] - face_box[0]), 1.0)
        scale = min(float(level) / face_w, 1.0)
        small = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
        noise = np.random.default_rng(int(level)).normal(0, 4.0, small.shape)
        small = np.clip(small.astype("float64") + noise, 0, 255).astype("uint8")
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
        small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "blur":
        return image if level <= 0 else cv2.GaussianBlur(image, (0, 0), float(level))
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "gamma":
        lut = (255.0 * (np.arange(256) / 255.0) ** float(level)).clip(0, 255).astype("uint8")
        return cv2.LUT(image, lut)
    if kind == "occlusion":
        if level <= 0:
            return image
        out = image.copy()
        x1, y1, x2, y2 = [int(round(v)) for v in face_box]
        top = int(round(y2 - level * (y2 - y1)))
        out[max(0, top):max(0, y2), max(0, x1):max(0, x2)] = 128
        return out
    raise ValueError(kind)


def laplacian_var(image, box):
    if image is None or not np.all(np.isfinite(box)):
        return np.nan
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return np.nan
    gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_probes(proto, n_mated, n_nonmated, seed, emb_of):
    probes = proto["test"]["probes"]
    probes = probes[probes["image_path"].isin(emb_of)]  # undegraded probe must enrol, so a face box exists
    rng = np.random.default_rng(seed + 101)
    mated = probes[probes["mated"]]
    non = probes[~probes["mated"]]
    pick_m = mated.iloc[sorted(rng.choice(len(mated), min(n_mated, len(mated)), replace=False))]
    pick_n = non.iloc[sorted(rng.choice(len(non), min(n_nonmated, len(non)), replace=False))]
    return pd.concat([pick_m, pick_n], ignore_index=True)


def search_any_face(gallery, probes, emb_any, top_k, gallery_emb):
    """Search every probe that has ANY detected face (ok or low_quality) against the undegraded gallery."""
    return openset.search_probes(gallery, probes, emb_any, top_k, gallery_emb=gallery_emb)


def run(cfg):
    manifest, base, emb_of, proto = openset.load_lfw_protocol(cfg)
    threshold = float(cfg.get("threshold") or load_result("operating_point")["recommended"]["MATCH_THRESHOLD"])
    box_of = dict(zip(manifest["image_path"], base["bbox"]))
    gallery = openset.gallery_for_size(proto["test"], cfg.get("gallery_size", 1000), seed=cfg["seed"])
    probes = sample_probes(proto, int(cfg.get("n_mated", 200)), int(cfg.get("n_nonmated", 200)), cfg["seed"], emb_of)
    levels = cfg.get("levels", DEFAULT_LEVELS)
    top_k = int(cfg.get("top_k", 5))
    n_boot = int(cfg.get("n_boot", 500))

    all_records, summary = [], {}
    for kind, values in levels.items():
        summary[kind] = []
        for level in values:
            variant = f"{kind}{level}"

            def transform(img, path, kind=kind, level=level):
                return degrade(img, kind, level, box_of[path])
            out = embed_paths(list(probes["image_path"]), cache_name("lfw", variant=variant), transform=transform)
            emb_any = {p: e for p, e, s in zip(probes["image_path"], out["emb"], out["status"]) if s != "no_face"}
            recs = search_any_face(gallery, probes, emb_any, top_k, emb_of)
            info = pd.DataFrame({"image_path": probes["image_path"], "status": out["status"],
                                 "det_score": out["det_score"],
                                 "face_width": out["bbox"][:, 2] - out["bbox"][:, 0]})
            recs = recs.merge(info, on="image_path", how="left")
            recs["blur_var"] = [laplacian_var(degrade(cv2.imread(p), kind, level, box_of[p]), b)
                                for p, b in zip(probes["image_path"], out["bbox"])]
            recs["kind"], recs["level"] = kind, level
            all_records.append(recs)

            # production behaviour: only status 'ok' faces are searched
            prod = recs.copy()
            bad = prod["status"] != STATUS_OK
            prod.loc[bad, "top_score"], prod.loc[bad, "rank"] = -np.inf, 0
            mated = prod[prod["mated"]]
            units = openset.to_units(prod)
            fnir = M.bootstrap_ci(lambda us: M.fnir_fpir_at_threshold(*openset.unit_arrays(us), threshold)[0],
                                  units, n=n_boot, seed=cfg["seed"])
            fpir = M.bootstrap_ci(lambda us: M.fnir_fpir_at_threshold(*openset.unit_arrays(us), threshold)[1],
                                  units, n=n_boot, seed=cfg["seed"])
            row = {"level": level, "fte_rate": float(bad.mean()),
                   "fte_rate_mated": float((mated["status"] != STATUS_OK).mean()),
                   "fnir": fnir[0], "fnir_ci": [fnir[1], fnir[2]], "fpir": fpir[0], "fpir_ci": [fpir[1], fpir[2]],
                   "rank1_any_threshold": float((mated["rank"] == 1).mean())}
            summary[kind].append(row)
            print(f"{kind:9s} {level!s:>5}: FTE {row['fte_rate']:.3f}  FNIR {row['fnir']:.3f}  FPIR {row['fpir']:.3f}")

    pd.concat(all_records, ignore_index=True).to_csv(os.path.join(CACHE_DIR, "robustness_records.csv"), index=False)
    min_size = minimum_usable_size(summary, cfg)
    for kind in summary:
        plot_kind(kind, summary[kind], threshold)
    save_result("robustness", {
        "meta": metadata(cfg, lfw.DATASET_VERSION), "threshold": threshold,
        "n_mated_probes": int(probes["mated"].sum()), "n_nonmated_probes": int((~probes["mated"]).sum()),
        "protocol": __doc__.strip().split("\n\n", 1)[1], "results": summary,
        "minimum_usable_face_width": min_size,
        "plots": [f"eval/plots/robust_{k}.png" for k in summary],
    })


def minimum_usable_size(summary, cfg):
    """Smallest face width whose FNIR is within `usable_margin` (absolute) of the undegraded FNIR."""
    margin = float(cfg.get("usable_margin", 0.10))
    base = next((r["fnir"] for r in summary.get("blur", []) if r["level"] == 0), None)

    def smallest(kind):
        rows = sorted(summary.get(kind, []), key=lambda r: r["level"])
        ok = [r["level"] for r in rows if base is not None and r["fnir"] <= base + margin]
        return min(ok) if ok else None
    clean, cctv = smallest("downscale"), smallest("cctv")
    return {"width_px": cctv if cctv is not None else clean, "width_px_clean_downscale": clean,
            "width_px_cctv": cctv, "rule": f"FNIR <= undegraded FNIR + {margin:.0%} (absolute); "
            "headline uses the combined cctv degradation", "baseline_fnir": base}


def plot_kind(kind, rows, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["level"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    f = np.array([r["fnir"] for r in rows])
    lo = np.array([r["fnir_ci"][0] for r in rows])
    hi = np.array([r["fnir_ci"][1] for r in rows])
    ax.plot(x, f, marker="o", lw=2, label=f"FNIR @ {threshold:.3f}")
    ax.fill_between(x, lo, hi, alpha=0.2)
    ax.plot(x, [r["fte_rate"] for r in rows], marker="s", lw=1.5, ls="--", label="failure to enrol")
    labels = {"downscale": "face width (px)", "cctv": "face width (px), with noise + JPEG 40", "blur": "Gaussian sigma", "jpeg": "JPEG quality",
              "gamma": "gamma", "occlusion": "lower-face occlusion (fraction of face height)"}
    ax.set_xlabel(labels[kind])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    if kind == "jpeg":
        ax.invert_xaxis()
    ax.set_title(f"Robustness: {kind} (LFW test probes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path(f"robust_{kind}.png"), dpi=130)
    plt.close(fig)
