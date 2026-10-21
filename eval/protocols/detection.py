"""Experiment `detection`: crowd face detection on WIDER FACE val (Easy / Medium / Hard AP).

Detector: the SCRFD-10GF model shipped in buffalo_l (det_10g.onnx), run through the production
FaceService with only the detection model loaded. We compare det_size 640x640 (production) with
1024x1024. Detections are kept down to score 0.02 so the full precision-recall curve exists;
recall at the production threshold (0.5) is reported too.

AP is computed with a Python port of the official WIDER FACE MATLAB evaluation (IoU 0.5, scores
min-max normalised over the dataset, 1000 thresholds, VOC-style AP), following the widely used
re-implementation github.com/wondervictor/WiderFace-Evaluation (MIT licence).
"""
import os
import time

import cv2
import numpy as np
import scipy.io as sio

from eval.common import CACHE_DIR, DATA_DIR, cpu_name, metadata, save_result
from eval.datasets import widerface

THRESH_NUM = 1000


# ---- official evaluation, ported -------------------------------------------------------------
def bbox_overlaps(boxes, query):
    """IoU between N boxes and K query boxes, both x1,y1,x2,y2 (inclusive pixel convention)."""
    n, k = boxes.shape[0], query.shape[0]
    out = np.zeros((n, k))
    for j in range(k):
        q_area = (query[j, 2] - query[j, 0] + 1) * (query[j, 3] - query[j, 1] + 1)
        iw = np.minimum(boxes[:, 2], query[j, 2]) - np.maximum(boxes[:, 0], query[j, 0]) + 1
        ih = np.minimum(boxes[:, 3], query[j, 3]) - np.maximum(boxes[:, 1], query[j, 1]) + 1
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        out[:, j] = np.where((iw > 0) & (ih > 0), inter / (area + q_area - inter), 0)
    return out


def image_eval(pred, gt, ignore, iou_thresh):
    """pred: [x,y,w,h,score] sorted by score desc; gt: [x,y,w,h]."""
    _pred, _gt = pred.copy(), gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])
    _pred[:, 2] += _pred[:, 0]
    _pred[:, 3] += _pred[:, 1]
    _gt[:, 2] += _gt[:, 0]
    _gt[:, 3] += _gt[:, 1]
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        pred_recall[h] = np.sum(recall_list == 1)
    return pred_recall, proposal_list


def img_pr_info(pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((THRESH_NUM, 2))
    for t in range(THRESH_NUM):
        thresh = 1 - (t + 1) / THRESH_NUM
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index):
            r_index = r_index[-1]
            pr_info[t, 0] = np.sum(proposal_list[: r_index + 1] == 1)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(preds, gt_dir, iou_thresh=0.5, normalise=True):
    """preds: {event: {image_name: array[N,5] x,y,w,h,score}}. Returns {'easy','medium','hard'} -> AP, recall."""
    gt = sio.loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    settings = {s: sio.loadmat(os.path.join(gt_dir, f"wider_{s}_val.mat"))["gt_list"] for s in ("easy", "medium", "hard")}
    events, files, boxes = gt["event_list"], gt["file_list"], gt["face_bbx_list"]
    if normalise:
        scores = np.concatenate([p[:, 4] for ev in preds.values() for p in ev.values() if len(p)])
        lo, hi = scores.min(), scores.max()
        preds = {ev: {k: np.column_stack([p[:, :4], (p[:, 4] - lo) / (hi - lo)]) if len(p) else p
                      for k, p in d.items()} for ev, d in preds.items()}
    out = {}
    for name, gt_list in settings.items():
        count_face, pr_curve = 0, np.zeros((THRESH_NUM, 2))
        for i in range(len(events)):
            event = str(events[i][0][0])
            img_list, sub_gt, gt_bbx = files[i][0], gt_list[i][0], boxes[i][0]
            for j in range(len(img_list)):
                pred_info = preds.get(event, {}).get(str(img_list[j][0][0]), np.zeros((0, 5)))
                gt_boxes = gt_bbx[j][0].astype("float")
                keep = sub_gt[j][0]
                count_face += len(keep)
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep):
                    ignore[keep.ravel() - 1] = 1
                pred_recall, proposal = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                pr_curve += img_pr_info(pred_info, proposal, pred_recall)
        precision = np.divide(pr_curve[:, 1], pr_curve[:, 0], out=np.ones(THRESH_NUM), where=pr_curve[:, 0] > 0)
        recall = pr_curve[:, 1] / count_face
        out[name] = {"ap": voc_ap(recall, precision), "max_recall": float(recall.max()), "n_faces": int(count_face)}
    return out


# ---- running the detector ------------------------------------------------------------------
def detect_all(manifest, det_size, limit=None):
    from services.face_service import FaceService
    path = os.path.join(CACHE_DIR, f"widerface_det{det_size}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return _unpack(d), np.array(d["times"])
    svc = FaceService(det_threshold=0.02, det_size=(det_size, det_size), allowed_modules=["detection"])
    rows = manifest if limit is None else manifest.iloc[:limit]
    names, offsets, boxes, times = [], [0], [], []
    for i, p in enumerate(rows["image_path"], 1):
        img = cv2.imread(p)
        t0 = time.perf_counter()
        faces = svc.detect_faces(img)
        times.append(time.perf_counter() - t0)
        arr = np.array([[f.bbox[0], f.bbox[1], f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1], f.det_score]
                        for f in faces], dtype="float32").reshape(-1, 5)
        arr = arr[np.argsort(-arr[:, 4])]
        rel = os.path.relpath(p, widerface.images_dir(DATA_DIR)).replace("\\", "/")
        names.append(rel)
        boxes.append(arr)
        offsets.append(offsets[-1] + len(arr))
        if i % 250 == 0:
            print(f"  det {det_size}: {i}/{len(rows)} images, {np.mean(times):.3f}s/img")
    np.savez(path, names=np.array(names), offsets=np.array(offsets), boxes=np.concatenate(boxes),
             times=np.array(times))
    return _unpack(np.load(path, allow_pickle=False)), np.array(times)


def _unpack(d):
    # read each array once: indexing an NpzFile re-reads the whole array every time
    names, offsets, boxes = d["names"], d["offsets"], d["boxes"]
    preds = {}
    for k, name in enumerate(names):
        event, fname = str(name).split("/")
        preds.setdefault(event, {})[os.path.splitext(fname)[0]] = boxes[offsets[k]:offsets[k + 1]]
    return preds


def run(cfg):
    manifest = widerface.load_manifest(DATA_DIR)
    gt_dir = widerface.gt_dir(DATA_DIR)
    results = {}
    for size in cfg.get("det_sizes", [640, 1024]):
        preds, times = detect_all(manifest, int(size), cfg.get("limit"))
        ap = evaluate(preds, gt_dir)
        at_prod = evaluate({e: {k: v[v[:, 4] >= 0.5] for k, v in d.items()} for e, d in preds.items()},
                           gt_dir, normalise=False)
        results[str(size)] = {
            "ap": {k: v["ap"] for k, v in ap.items()},
            "recall_at_det_thresh_0.5": {k: v["max_recall"] for k, v in at_prod.items()},
            "n_faces": {k: v["n_faces"] for k, v in ap.items()},
            "sec_per_image_mean": float(np.mean(times)), "sec_per_image_p95": float(np.quantile(times, 0.95)),
            "n_images": int(len(times)),
        }
        r = results[str(size)]
        print(f"det_size {size}: AP easy {r['ap']['easy']:.4f} medium {r['ap']['medium']:.4f} hard {r['ap']['hard']:.4f}; "
              f"{r['sec_per_image_mean']:.3f}s/img")
    sizes = sorted(results, key=int)
    best_hard = max(sizes, key=lambda s: results[s]["ap"]["hard"])
    save_result("detection_widerface", {
        "meta": metadata(cfg, widerface.DATASET_VERSION), "cpu": cpu_name(),
        "protocol": __doc__.strip().split("\n\n", 1)[1], "results": results,
        "recommended_det_size": int(cfg.get("recommend", best_hard)),
    })
