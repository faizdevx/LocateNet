import os

import pytest
from PIL import Image

import config
from conftest import add_case, add_match, csrf_token, db_rows, login, make_user
from services import geo

PRIOR = (27.8974, 78.0880)  # Aligarh


def write_photo(path, gps=None):
    img = Image.new("RGB", (64, 64), "white")
    exif = Image.Exif()
    if gps:
        lat, lon = gps
        exif.get_ifd(0x8825).update({1: "N" if lat >= 0 else "S", 2: (abs(lat), 0.0, 0.0),
                                      3: "E" if lon >= 0 else "W", 4: (abs(lon), 0.0, 0.0)})
    img.save(path, exif=exif)
    return str(path)


def test_exif_gps_reads_coordinates_and_handles_missing(tmp_path):
    assert geo.exif_gps(write_photo(tmp_path / "a.jpg", (27.5, -78.25))) == pytest.approx((27.5, -78.25))
    assert geo.exif_gps(write_photo(tmp_path / "b.jpg")) is None
    assert geo.exif_gps(str(tmp_path / "missing.jpg")) is None


def test_clean_texts_drops_short_low_confidence_duplicates_and_numbers():
    items = [("Sharma Sweets", 0.9), ("sharma  sweets", 0.8), ("ab", 0.99), ("Gupta Medical", 0.2),
             ("Call 9876543210", 0.95), ("UP81 AB 1234", 0.9), ("Railway Road", 0.7)]
    assert geo.clean_texts(items) == ["Sharma Sweets", "Railway Road"]


def test_sign_hits_ignore_results_outside_the_search_box():
    near, far = (27.90, 78.09, "Sharma Sweets, Aligarh"), (28.61, 77.21, "Sharma Sweets, Delhi")
    hits = geo.sign_hits(["Sharma Sweets"], PRIOR, lambda text, box: [near, far])
    assert [h["name"] for h in hits] == ["Sharma Sweets, Aligarh"]


def test_fuse_merges_agreeing_evidence_and_ranks_gps_first():
    sign_a = {"source": "sign", "text": "Sharma Sweets", "name": "x", "lat": 27.90000, "lon": 78.09000}
    sign_b = {"source": "sign", "text": "Railway Road", "name": "y", "lat": 27.90010, "lon": 78.09010}  # ~15 m away
    lone = {"source": "sign", "text": "Gupta Medical", "name": "z", "lat": 27.95, "lon": 78.05}
    out = geo.fuse((27.88, 78.07), [lone, sign_a, sign_b], PRIOR)
    assert [c["level"] for c in out["candidates"]] == ["strong", "strong", "weak"]
    assert out["candidates"][0]["evidence"][0]["source"] == "exif"
    assert len(out["candidates"][1]["evidence"]) == 2 and not out["abstain"] and out["mismatch_km"] is None


def test_fuse_abstains_without_evidence_and_flags_far_candidates():
    assert geo.fuse(None, [], PRIOR) == {"candidates": [], "abstain": True, "mismatch_km": None}
    far = geo.fuse((28.61, 77.21), [], PRIOR)  # GPS says Delhi, reporter said Aligarh
    assert far["mismatch_km"] > 100


def test_locate_explains_what_was_missing(tmp_path):
    path = write_photo(tmp_path / "c.jpg")
    no_ocr = geo.locate(path, PRIOR, geocode=None, ocr=lambda p: None)
    assert no_ocr["abstain"] and any("not installed" in n for n in no_ocr["notes"])
    no_prior = geo.locate(path, None, geocode=None, ocr=lambda p: ["Sharma Sweets"])
    assert no_prior["abstain"] and any("could not be located" in n for n in no_prior["notes"])


# --- routes ---

@pytest.fixture
def geo_on(monkeypatch, flask_app):
    import app as app_module
    monkeypatch.setattr(config, "GEO_ASSIST", True)
    monkeypatch.setattr(app_module, "get_coords", lambda city: PRIOR)
    monkeypatch.setattr(app_module, "geocode_sign", lambda text, box: [(27.9001, 78.0901, f"{text}, Aligarh")])
    monkeypatch.setattr(geo, "sign_texts", lambda path: ["Sharma Sweets", "Railway Road"])
    photo = os.path.join(os.getcwd(), "resources", "geo_test.jpg")
    os.makedirs(os.path.dirname(photo), exist_ok=True)
    write_photo(photo)
    monkeypatch.setattr(app_module, "BASE_DIR", os.getcwd())
    return "resources/geo_test.jpg"


def sighting_for(owner, image_path):
    case_id = add_case(officer=owner)
    add_match(case_id, "s1", "aligarh", "2026-11-01 10:00:00")
    with __import__("models").get_db_connection() as conn:
        conn.execute("UPDATE public_submissions SET image_path = ? WHERE id = 's1'", (image_path,))
        conn.commit()
    return case_id


def test_case_officer_locates_and_confirms(client, geo_on):
    make_user("owner")
    sighting_for("owner", geo_on)
    login(client, "owner")
    page = client.get("/sighting/s1").get_data(as_text=True)
    assert "Locate this photo" in page
    token = csrf_token(client, "/sighting/s1")
    assert client.post("/sighting/s1/geo", data={"csrf_token": token}).status_code == 302

    [(cand_id, level)] = db_rows("SELECT id, level FROM geo_candidates")
    assert level == "strong"  # two sign texts agree on one place
    page = client.get("/sighting/s1").get_data(as_text=True)
    assert "Candidate 1" in page and 'Sign "Sharma Sweets"' in page

    client.post(f"/sighting/s1/geo/{cand_id}/verdict", data={"verdict": "confirm", "csrf_token": token})
    assert db_rows("SELECT verdict, reviewed_by FROM geo_candidates") == [("confirm", "owner")]
    assert "Verified location" in client.get("/sighting/s1").get_data(as_text=True)
    assert {a for (a,) in db_rows("SELECT action FROM audit_log")} == {"geo_locate", "geo_verdict"}


def test_other_officer_and_public_are_refused(client, geo_on):
    make_user("owner")
    make_user("other")
    sighting_for("owner", geo_on)
    assert client.post("/sighting/s1/geo").status_code in (302, 400)  # public: no session / no CSRF token
    login(client, "other")
    token = csrf_token(client, "/register_case")
    assert "Locate this photo" not in client.get("/sighting/s1").get_data(as_text=True)
    assert client.post("/sighting/s1/geo", data={"csrf_token": token}).status_code == 403
    assert db_rows("SELECT COUNT(*) FROM geo_requests") == [(0,)]


def test_missing_csrf_is_rejected(client, geo_on):
    make_user("owner")
    sighting_for("owner", geo_on)
    login(client, "owner")
    assert client.post("/sighting/s1/geo").status_code == 400


def test_flag_off_hides_the_feature(client, geo_on, monkeypatch):
    monkeypatch.setattr(config, "GEO_ASSIST", False)
    make_user("owner")
    sighting_for("owner", geo_on)
    login(client, "owner")
    assert "Locate this photo" not in client.get("/sighting/s1").get_data(as_text=True)
    token = csrf_token(client, "/register_case")
    assert client.post("/sighting/s1/geo", data={"csrf_token": token}).status_code == 403


def test_rate_limit(client, geo_on, monkeypatch):
    monkeypatch.setattr(config, "GEO_REQUESTS_PER_HOUR", 1)
    make_user("owner")
    sighting_for("owner", geo_on)
    login(client, "owner")
    token = csrf_token(client, "/sighting/s1")
    client.post("/sighting/s1/geo", data={"csrf_token": token})
    client.post("/sighting/s1/geo", data={"csrf_token": token})
    assert db_rows("SELECT COUNT(*) FROM geo_requests") == [(1,)]
