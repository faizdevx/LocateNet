"""Download / pre-warm the model files the app needs, so the first request is not slow.

Run from anywhere:  python scripts/download_models.py

- InsightFace buffalo_l is downloaded to ~/.insightface/models on first use.
- yolov8n.pt (video pipeline only) is fetched automatically by ultralytics into the repo root.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # ultralytics downloads yolov8n.pt into the current directory


def warm_insightface():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace buffalo_l ready.")


def warm_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed; skipping yolov8n.pt (only needed for video uploads).")
        print("Install it with: pip install -r requirements-video.txt")
        return
    YOLO("yolov8n.pt")
    print("yolov8n.pt ready.")


if __name__ == "__main__":
    warm_insightface()
    warm_yolo()
    sys.exit(0)
