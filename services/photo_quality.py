"""Face quality for reference photos and video frames (pure functions).

The score combines the three signals studied in the Week 2 quality-gate experiment
(eval/results/quality.json: a combination beat det_score alone): detector score, face width in
pixels and Laplacian sharpness of the face crop. Each factor is mapped to (0, 1] and multiplied.
"""
import cv2
import numpy as np

REFERENCE_FACE_PX = 112.0     # recognition input size; wider faces carry no extra detail
REFERENCE_SHARPNESS = 500.0   # Laplacian variance of a sharp 112x112 face crop


def blur_variance(image, bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(det_score, face_px, blur_var):
    size = min(1.0, max(float(face_px), 1.0) / REFERENCE_FACE_PX)
    sharp = min(1.0, np.log1p(max(float(blur_var), 0.0)) / np.log1p(REFERENCE_SHARPNESS))
    return float(max(float(det_score), 0.0) * max(size, 0.05) * max(sharp, 0.05))


def gate_reference_face(faces, det_score_min, min_face_px):
    """
    Pick the largest usable face of a reference photo. Returns (face, None) or (None, reason) where
    reason is a sentence the officer can act on.
    """
    if not faces:
        return None, "no face found"
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    best = faces[0]
    if best.det_score < det_score_min:
        return None, f"face too unclear (detector score {best.det_score:.2f} < {det_score_min:.2f})"
    width = float(best.bbox[2] - best.bbox[0])
    if width < min_face_px:
        return None, f"face too small ({width:.0f} px wide, need at least {min_face_px:.0f} px)"
    return best, None
