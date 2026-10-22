"""Experiment `latency`: CPU timing of the production image pipeline and of FAISS search.

1. DetectionPipeline.process_frame end-to-end (production FaceService: all buffalo_l models
   loaded, as in the app) on synthetic group images with 1, 5, 10, 20, 50 faces, built by tiling
   LFW photos on a grid (each tile 250x250 px).
2. Breakdown of one frame into detection, embedding (recognition model only), the other buffalo_l
   models the app also runs (landmarks, gender/age) and FAISS search.
3. SearchService.search_face latency against galleries of 1k / 10k / 100k random unit vectors.
"""
import glob
import math
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from eval.common import DATA_DIR, cpu_name, metadata, save_result
from eval.datasets import lfw


def group_image(n, paths):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.full((rows * 250, cols * 250, 3), 127, dtype="uint8")
    for k in range(n):
        img = cv2.imread(paths[k])
        r, c = divmod(k, cols)
        canvas[r * 250:(r + 1) * 250, c * 250:(c + 1) * 250] = img
    return canvas


def pct(ts):
    ts = np.asarray(ts) * 1000
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "n": int(ts.size)}


def run(cfg):
    from services.face_service import FaceService
    from vector_db.search_service import SearchService

    repeats = int(cfg.get("repeats", 10))
    rng = np.random.default_rng(cfg["seed"])
    paths = sorted(glob.glob(os.path.join(lfw.lfw_root(DATA_DIR), "*", "*_0001.jpg")))
    paths = [paths[i] for i in rng.choice(len(paths), 60, replace=False)]

    tmp = tempfile.mkdtemp(prefix="locatenet-latency-")
    try:
        search = SearchService(index_path=os.path.join(tmp, "idx"))
        vecs = rng.normal(size=(1000, 512)).astype("float32")
        search.rebuild_face_index([(i + 1, v) for i, v in enumerate(vecs)])
        by_det_size = {}
        for det_size in cfg.get("det_sizes", [640]):
            # production configuration (all buffalo_l models loaded, as in the app)
            face_svc = FaceService(det_size=(int(det_size), int(det_size)))
            by_det_size[str(det_size)] = time_frames(cfg, face_svc, search, paths, tmp, repeats)
        frames = by_det_size[str(cfg.get("det_sizes", [640])[0])]

        faiss_scale = {}
        for size in cfg.get("gallery_sizes", [1000, 10000, 100000]):
            svc = SearchService(index_path=os.path.join(tmp, f"scale{size}"))
            v = rng.normal(size=(size, 512)).astype("float32")
            svc.rebuild_face_index([(i + 1, x) for i, x in enumerate(v)])
            q = rng.normal(size=(int(cfg.get("queries", 300)), 512)).astype("float32")
            ts = []
            for x in q:
                t0 = time.perf_counter()
                svc.search_face(x, top_k=5)
                ts.append(time.perf_counter() - t0)
            faiss_scale[str(size)] = pct(ts)
            print(f"FAISS gallery {size}: p50 {faiss_scale[str(size)]['p50_ms']:.3f} ms")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    import onnxruntime
    save_result("latency", {
        "meta": metadata(cfg, "synthetic grids of LFW photos; random unit vectors for FAISS"),
        "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
        "onnxruntime_threads": "default (one intra-op thread per physical core)",
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "protocol": __doc__.strip().split("\n\n", 1)[1],
        "process_frame": frames, "process_frame_by_det_size": by_det_size, "faiss_search": faiss_scale,
    })


def time_frames(cfg, face_svc, search, paths, tmp, repeats):
    from insightface.app.common import Face
    from pipeline.detection_pipeline import DetectionPipeline
    pipe = DetectionPipeline(face_svc, None, search, storage_path=os.path.join(tmp, "crops"))
    app = face_svc.app
    det_model = app.det_model
    rec_model = app.models["recognition"]
    others = {k: m for k, m in app.models.items() if k not in ("detection", "recognition")}
    frames = {}
    for n in cfg.get("faces", [1, 5, 10, 20, 50]):
        img = group_image(n, paths)
        pipe.process_frame(img)  # warm-up
        total, det_t, emb_t, other_t, search_t, found = [], [], [], [], [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pipe.process_frame(img)
            total.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            bboxes, kpss = det_model.detect(img, max_num=0, metric="default")
            det_t.append(time.perf_counter() - t0)
            faces = [Face(bbox=b[:4], kps=k, det_score=b[4]) for b, k in zip(bboxes, kpss)]
            t0 = time.perf_counter()
            for f in faces:
                for m in others.values():
                    m.get(img, f)
            other_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for f in faces:
                rec_model.get(img, f)
            emb_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for f in faces:
                search.search_face(f.embedding, top_k=5)
            search_t.append(time.perf_counter() - t0)
            found.append(len(faces))
        frames[str(n)] = {"image_size": list(img.shape[:2]), "faces_detected": int(np.median(found)),
                          "process_frame": pct(total), "detection": pct(det_t), "embedding": pct(emb_t),
                          "other_models": pct(other_t), "search": pct(search_t)}
        print(f"{n:3d} faces ({frames[str(n)]['faces_detected']} found): process_frame p50 "
              f"{frames[str(n)]['process_frame']['p50_ms']:.0f} ms, p95 {frames[str(n)]['process_frame']['p95_ms']:.0f} ms")

    return frames
