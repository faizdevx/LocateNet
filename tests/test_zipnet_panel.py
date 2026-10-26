"""Missing-person panel of imported ZIPNET records: listing, photos, filters, live hint, safety."""

from __future__ import annotations

import io
import urllib.error

import pytest

import config
from conftest import login, make_user
from services import zipnet_photos
from services import zipnet_store as store
from tests.zipnet_fakes import Env, FakeTransport, make_settings, pages_of
from tests.zipnet_helpers import CANARIES, make_row

MP = "missing_persons"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(config, "ZIPNET_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "ZIPNET_PANEL_PUBLIC", True)
    zipnet_photos.clear_cache()


def seed(zdb, rows):
    Env(zdb, FakeTransport({MP: pages_of(rows, size=10)}), settings={"sections": (MP,), "pages_per_turn": 50}).drain()


def people(n=12):
    return [make_row(MP, 100 - i, Name=f"PERSON {100 - i}") for i in range(n)]


def page(client, qs=""):
    return client.get("/zipnet-records" + qs).get_data(as_text=True)


class FakeResp(io.BytesIO):
    def __init__(self, data, ctype="image/jpeg"):
        super().__init__(data)
        self.headers = {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(monkeypatch, fn):
    monkeypatch.setattr(zipnet_photos.urllib.request, "urlopen", fn)


# --- the page ---------------------------------------------------------------------------------------------------


def test_panel_lists_imported_records_with_photos_and_clear_labels(client, zdb):
    seed(zdb, people())
    html = page(client)
    assert "PERSON 100" in html and "PERSON 89" in html
    assert "External public records" in html and "not LocateNet cases" in html
    assert "Photo as published on ZIPNET" in html and "/zipnet-records/photo/" in html
    assert 'loading="lazy"' in html and "Report a sighting" in html
    assert 'rel="noopener noreferrer nofollow"' in html
    assert "https://zipnet.delhipolice.gov.in/Victims/MissingPersons" in html
    assert 'id="indexed">12<' in html  # live counter


def test_panel_never_shows_prohibited_fields(client, zdb):
    seed(zdb, people())
    html = page(client)
    for canary in CANARIES + ["SECRET LANE", "CANARYRELIGION"]:
        assert canary not in html


def test_markup_in_names_is_stripped_at_import_and_escaped_at_render(client, zdb):
    seed(zdb, [make_row(MP, 5, Name="<script>alert(1)</script>")])
    html = page(client)
    assert "<script>alert(1)</script>" not in html and "alert(1)" in html  # tags removed during normalization
    conn = store.connect(zdb)  # even if hostile markup somehow reached the database, the template escapes it
    conn.execute("UPDATE zipnet_records SET name='<img src=x onerror=alert(2)>'")
    html = page(client)
    assert "<img src=x" not in html and "&lt;img src=x onerror=alert(2)&gt;" in html


def test_empty_state_is_honest(client, zdb):
    html = page(client)
    assert "No ZIPNET records have been imported yet" in html and "never" in html


def test_status_filter_defaults_to_still_missing(client, zdb):
    seed(zdb, [make_row(MP, 9, Name="OPEN ONE"), make_row(MP, 8, Name="TRACED ONE", TracingStatus="Traced")])
    a = page(client)
    assert "OPEN ONE" in a and "TRACED ONE" not in a
    b = page(client, "?status=traced")
    assert "TRACED ONE" in b and "OPEN ONE" not in b
    c = page(client, "?status=all")
    assert "TRACED ONE" in c and "OPEN ONE" in c


def test_search_state_sex_and_photo_filters(client, zdb):
    seed(zdb, [make_row(MP, 9, Name="ANITA", Sex="Female", MissingFrom="near central market"),
               make_row(MP, 8, Name="BHARAT", Sex="Male", State="HARYANA", ImageUrls=[]),
               make_row(MP, 7, Name="CHITRA", Sex="Female")])
    assert "ANITA" in page(client, "?q=central") and "CHITRA" not in page(client, "?q=central")
    assert "BHARAT" in page(client, "?state=haryana") and "ANITA" not in page(client, "?state=haryana")
    assert "ANITA" in page(client, "?sex=Female") and "BHARAT" not in page(client, "?sex=Female")
    assert "BHARAT" not in page(client, "?photo=1") and "ANITA" in page(client, "?photo=1")
    assert "No imported records match" in page(client, "?q=zzzz")
    assert "No imported records match" in page(client, "?q=%25")  # LIKE wildcard is escaped, not match-all


def test_pagination_and_hostile_parameters(client, zdb):
    seed(zdb, [make_row(MP, 1000 - i, Name=f"P{i:03d}", CreatedOn="05/10/2026") for i in range(60)])
    first = page(client)
    assert "Page 1 of 3" in first and "Next →" in first and "P000" in first and "P030" not in first
    assert "P030" in page(client, "?page=2")
    for qs in ("page=-4", "page=abc", "page=99999999", "status=bogus", "q=" + "x" * 500, "state=%27%3B+DROP+TABLE+x%3B--"):
        assert client.get("/zipnet-records?" + qs).status_code == 200
    assert store.connect(zdb).execute("SELECT COUNT(*) FROM zipnet_records").fetchone()[0] == 60


def test_total_is_capped_for_speed_but_says_so(zdb):
    conn = store.connect(zdb)
    with store.transaction(conn):
        conn.executemany(
            "INSERT INTO zipnet_records (section, source_id, title, name, status, image_paths, fingerprint, "
            "source_listing_url, first_seen_at, last_seen_at, last_changed_at, source_created_date) "
            "VALUES ('missing_persons', ?, 'x', 'n', 'Untraced', '[]', 'f', 'u', 'a', 'a', 'a', '2026-01-01')",
            [(str(i),) for i in range(store.COUNT_CAP + 50)])
    out = store.panel_query(conn)
    assert out["total"] == store.COUNT_CAP and out["total_capped"] is True and len(out["items"]) == 24


def test_age_is_shown_as_approximate_as_reported(client, zdb):
    seed(zdb, [make_row(MP, 3, BirthYear="2010")])
    assert "born 2010, as reported" in page(client)


# --- live updates ----------------------------------------------------------------------------------------------------


def test_updates_endpoint_reports_new_records_after_a_later_import(client, zdb):
    seed(zdb, people(5))
    first = client.get("/zipnet-records/updates?newest=0").get_json()
    assert first["indexed"] == 5 and first["newest_id"] == 5 and first["new_records"] == 0
    store.connect(zdb).execute("UPDATE zipnet_checkpoints SET last_incremental_at='2000-01-01T00:00:00.000000+00:00'")
    seed(zdb, [make_row(MP, 200 + i, Name=f"LATE {i}") for i in range(3)] + people(5))
    again = client.get(f"/zipnet-records/updates?newest={first['newest_id']}").get_json()
    assert again["new_records"] == 3 and again["indexed"] == 8 and again["last_success_at"]


def test_panel_shows_live_sync_state(client, zdb, monkeypatch):
    seed(zdb, people(3))
    assert "Live sync on" in page(client)
    monkeypatch.setattr(config, "ZIPNET_SYNC_ENABLED", False)
    assert "Sync is switched off" in page(client)


def test_portal_links_appear_only_when_the_sync_is_on(client, zdb, monkeypatch):
    assert "/zipnet-records" in client.get("/").get_data(as_text=True)
    assert "From ZIPNET (live)" in client.get("/search").get_data(as_text=True)
    monkeypatch.setattr(config, "ZIPNET_SYNC_ENABLED", False)
    assert "/zipnet-records" not in client.get("/").get_data(as_text=True)


# --- access control ---------------------------------------------------------------------------------------------------


def test_private_panel_needs_login(flask_app, zdb, monkeypatch):
    seed(zdb, people(3))
    monkeypatch.setattr(config, "ZIPNET_PANEL_PUBLIC", False)
    c = flask_app.test_client()
    assert c.get("/zipnet-records").status_code == 302
    assert c.get("/zipnet-records/photo/1").status_code == 401
    assert c.get("/zipnet-records/updates").status_code == 401
    make_user("viewer", role="Officer")
    login(c, "viewer")
    assert c.get("/zipnet-records").status_code == 200


# --- photos --------------------------------------------------------------------------------------------------------------


def test_photo_route_serves_the_first_published_image(client, zdb, monkeypatch):
    seed(zdb, people(2))
    seen = []
    patch_urlopen(monkeypatch, lambda req, timeout=0: seen.append(req.full_url) or FakeResp(b"\xff\xd8jpegdata"))
    r = client.get("/zipnet-records/photo/1")
    assert r.status_code == 200 and r.mimetype == "image/jpeg" and r.data == b"\xff\xd8jpegdata"
    assert "max-age" in r.headers["Cache-Control"] and r.headers["X-Content-Type-Options"] == "nosniff"
    assert seen[0].startswith("https://zipnet.delhipolice.gov.in/ImageUploads/")
    client.get("/zipnet-records/photo/1")
    assert len(seen) == 1  # the second view comes from the small in-memory cache


def test_photo_route_404s_for_unknown_records_missing_photos_and_source_errors(client, zdb, monkeypatch):
    seed(zdb, [make_row(MP, 2, ImageUrls=[]), make_row(MP, 1)])
    assert client.get("/zipnet-records/photo/999").status_code == 404
    assert client.get("/zipnet-records/photo/1").status_code == 404  # id 1 = the record without a photo

    def boom(req, timeout=0):
        raise urllib.error.URLError("down")

    patch_urlopen(monkeypatch, boom)
    assert client.get("/zipnet-records/photo/2").status_code == 404


def test_photo_fetch_is_safe(monkeypatch):
    st = make_settings("x.db")
    calls = []
    patch_urlopen(monkeypatch, lambda req, timeout=0: calls.append(req.full_url) or FakeResp(b"img"))
    for bad in ("https://evil.example/a.jpg", "/ImageUploads/../web.config", "//evil.example/ImageUploads/a.jpg",
                "/ImageUploads/2026/Other/a.jpg", "/ImageUploads/a.svg", "http://169.254.169.254/latest"):
        assert zipnet_photos.fetch_photo(st, bad) is None
    assert calls == []  # nothing was requested for any of them
    assert zipnet_photos.fetch_photo(st, "/ImageUploads/ab12-cd34.jpg") == ("image/jpeg", b"img")
    st.base_url = "file:///etc"
    assert zipnet_photos.fetch_photo(st, "/ImageUploads/zz99.jpg") is None


def test_photo_fetch_rejects_non_images_and_oversize(monkeypatch):
    st = make_settings("x.db")
    patch_urlopen(monkeypatch, lambda r, timeout=0: FakeResp(b"<html>", "text/html"))
    assert zipnet_photos.fetch_photo(st, "/ImageUploads/a1.jpg") is None
    patch_urlopen(monkeypatch, lambda r, timeout=0: FakeResp(b"x" * (zipnet_photos.MAX_BYTES + 5)))
    assert zipnet_photos.fetch_photo(st, "/ImageUploads/a2.jpg") is None


def test_photo_proxy_makes_no_request_when_sync_is_disabled(monkeypatch):
    st = make_settings("x.db", enabled=False)
    patch_urlopen(monkeypatch, lambda *a, **k: pytest.fail("network used"))
    assert zipnet_photos.fetch_photo(st, "/ImageUploads/a1.jpg") is None


def test_photo_cache_is_bounded(monkeypatch):
    st = make_settings("x.db")
    monkeypatch.setattr(zipnet_photos, "CACHE_ITEMS", 3)
    patch_urlopen(monkeypatch, lambda r, timeout=0: FakeResp(b"i"))
    for i in range(6):
        zipnet_photos.fetch_photo(st, f"/ImageUploads/p{i}.jpg")
    assert len(zipnet_photos._cache) == 3


# --- separation from face matching ---------------------------------------------------------------------------------------


def test_panel_code_has_no_path_to_face_matching():
    import inspect

    import blueprints.zipnet_panel as panel

    src = inspect.getsource(panel) + inspect.getsource(zipnet_photos)
    for word in ("face_service", "vector_db", "insightface", "case_embeddings", "SearchService"):
        assert word not in src


# --- schema upgrade --------------------------------------------------------------------------------------------------------


def test_upgrade_from_schema_v1_keeps_data_and_adds_the_panel_index(tmp_path):
    conn = store.connect(str(tmp_path / "old.db"))
    with store.transaction(conn):
        for stmt in store.MIGRATIONS[0][1]:
            conn.execute(stmt)
        conn.execute("PRAGMA user_version=1")
        conn.execute("INSERT INTO zipnet_checkpoints (section, updated_at, indexed_count) VALUES ('x','t',9)")
    assert store.migrate(conn) == store.SCHEMA_VERSION >= 2
    names = {r[1] for r in conn.execute("PRAGMA index_list(zipnet_records)")}
    assert "ix_zipnet_records_panel" in names
    assert conn.execute("SELECT indexed_count FROM zipnet_checkpoints").fetchone()[0] == 9
