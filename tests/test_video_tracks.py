import io

import app as app_module

from conftest import csrf_token, db_rows, login, make_user


def test_video_sighting_stores_track_span_and_crops(client, monkeypatch):
    def fake_process(path, max_frames=None, is_active=None):
        return {"tracks": [{"match_id": -1, "similarity": 0.1, "percentage": 10.0, "category": "No Match",
                            "face_bbox": [1, 2, 40, 50], "face_crop_path": "resources/video_frames/a.jpg",
                            "extra_crops": ["resources/video_frames/a.jpg", "resources/video_frames/b.jpg"],
                            "start_s": 1.2, "end_s": 4.8, "n_frames": 17, "face_embedding": [0.0] * 512}],
                "num_tracks": 1}
    monkeypatch.setattr(app_module.video_pipeline, "available", True)
    monkeypatch.setattr(app_module.video_pipeline, "process_video", fake_process)
    public = app_module.app.test_client()
    public.post("/report_sighting", data={"name": "R", "mobile": "9876543210", "location": "Delhi", "upload_type": "video",
                                          "video": (io.BytesIO(b"\x00" * 100), "clip.mp4"),
                                          "csrf_token": csrf_token(public, "/report_sighting")},
                content_type="multipart/form-data")
    [(start, end, n, crops, source)] = db_rows("SELECT track_start_s, track_end_s, n_frames, extra_crops, source_type FROM sighting_faces")
    assert (start, end, n, source) == (1.2, 4.8, 17, "video") and "b.jpg" in crops

    make_user("admin", role="Admin")
    login(client, "admin")
    [(sid,)] = db_rows("SELECT id FROM public_submissions")
    page = client.get(f"/sighting/{sid}").get_data(as_text=True)
    assert "17 frames used" in page and "b.jpg" in page
