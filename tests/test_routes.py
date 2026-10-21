from conftest import add_case, csrf_token, db_rows, login, make_user


def test_logged_out_resources_redirect_to_login(client):
    response = client.get("/resources/anything.jpg")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_logged_out_uploads_redirect_to_login(client):
    response = client.get("/uploads/anything.jpg")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_resolve_and_mark_read_reject_get(client):
    assert client.get("/resolve/1").status_code == 405
    assert client.get("/mark_read/1").status_code == 405


def test_post_without_csrf_token_is_rejected(client):
    assert client.post("/login", data={"username": "a", "password": "b"}).status_code == 400


def test_public_search_hides_officer_and_resolved_cases(client):
    add_case(officer="secret_officer", name="Still Missing")
    add_case(officer="secret_officer", name="Already Found", status="F")

    page = client.get("/search").get_data(as_text=True)

    assert "Still Missing" in page
    assert "Aligarh" in page
    assert "secret_officer" not in page
    assert "Already Found" not in page


def test_case_photo_is_public_only_for_missing_cases(client):
    resolved = add_case(name="Found", status="F")
    assert client.get(f"/case_photo/{resolved}").status_code == 404


def test_officer_cannot_resolve_someone_elses_case(client):
    make_user("owner")
    make_user("other")
    case_id = add_case(officer="owner")

    assert login(client, "other").status_code == 302
    client.post(f"/resolve/{case_id}", data={"csrf_token": csrf_token(client, "/register_case")})

    assert db_rows("SELECT status FROM cases WHERE id = ?", (case_id,)) == [("NF",)]


def test_owner_can_resolve_with_post(client):
    make_user("owner")
    case_id = add_case(officer="owner")

    login(client, "owner")
    response = client.post(f"/resolve/{case_id}", data={"csrf_token": csrf_token(client, "/register_case")})

    assert response.status_code == 302
    assert db_rows("SELECT status FROM cases WHERE id = ?", (case_id,)) == [("F",)]
