from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from medical_agent import web
from medical_agent.auth import AuthRepository, hash_password, verify_password
from medical_agent.osteoporosis_db import SQLiteOsteoporosisRepository

DEMO_PASSWORD = "DemoOnly-2026!"


def test_password_is_salted_and_verified() -> None:
    first = hash_password(DEMO_PASSWORD)
    second = hash_password(DEMO_PASSWORD)

    assert first.startswith("scrypt$")
    assert first != second
    assert DEMO_PASSWORD not in first
    assert verify_password(DEMO_PASSWORD, first) is True
    assert verify_password("WrongPassword!", first) is False


def test_session_token_is_not_stored_in_plaintext(tmp_path) -> None:
    database = tmp_path / "auth.sqlite"
    repository = AuthRepository(database)
    repository.initialize()

    authenticated = repository.authenticate("chen", DEMO_PASSWORD)

    assert authenticated is not None
    token, csrf = authenticated
    with sqlite3.connect(database) as connection:
        stored_token, stored_csrf = connection.execute(
            "SELECT token_hash, csrf_hash FROM login_sessions"
        ).fetchone()
    assert token != stored_token
    assert csrf != stored_csrf
    assert repository.get_principal(token).user_id == "doctor-chen"
    assert repository.csrf_is_valid(token, csrf) is True


def test_restart_does_not_restore_a_role_removed_by_admin(tmp_path) -> None:
    repository = AuthRepository(tmp_path / "auth.sqlite")
    repository.initialize()
    assert repository.set_user_roles("doctor-chen", ["administrator"]) is True

    repository.initialize()
    token, _ = repository.authenticate("chen", DEMO_PASSWORD)

    assert repository.get_principal(token).roles == frozenset({"administrator"})


def test_saving_unchanged_roles_does_not_revoke_session(tmp_path) -> None:
    repository = AuthRepository(tmp_path / "auth.sqlite")
    repository.initialize()
    token, _ = repository.authenticate("admin", "AdminOnly-2026!")

    assert repository.set_user_roles("admin-local", ["administrator"]) is True

    assert repository.get_principal(token) is not None


def _client(tmp_path, monkeypatch) -> tuple[TestClient, AuthRepository]:
    repository = AuthRepository(tmp_path / "auth.sqlite")
    repository.initialize()
    monkeypatch.setattr(web, "auth_repository", repository)
    monkeypatch.setattr(web, "SESSION_DB", tmp_path / "sessions.sqlite")
    medical_repository = SQLiteOsteoporosisRepository(tmp_path / "osteoporosis.sqlite")
    monkeypatch.setattr(web, "get_medical_repository", lambda: medical_repository)
    web._initialize_storage()
    return TestClient(web.app), repository


def _login(
    client: TestClient, username: str = "chen", password: str = DEMO_PASSWORD
) -> str:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_web_requires_login_and_csrf(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    assert client.get("/api/bootstrap").status_code == 401
    assert client.post("/api/auth/login", json={"username": "chen", "password": "bad"}).status_code == 401

    csrf = _login(client)
    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    assert bootstrap.json()["user"]["id"] == "doctor-chen"
    assert "password" not in bootstrap.text

    assert client.post("/api/sessions", json={}).status_code == 403
    created = client.post("/api/sessions", json={}, headers={"X-CSRF-Token": csrf})
    assert created.status_code == 200


def test_chat_rejects_client_supplied_actor_identity(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    csrf = _login(client)
    thread_id = client.post(
        "/api/sessions", json={}, headers={"X-CSRF-Token": csrf}
    ).json()["thread_id"]

    response = client.post(
        "/api/chat",
        headers={"X-CSRF-Token": csrf},
        json={
            "actor_id": "doctor-lin",
            "thread_id": thread_id,
            "message": "查询 P10086 的骨密度",
        },
    )

    assert response.status_code == 422


def test_repeated_login_failures_are_rate_limited(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    for _ in range(5):
        response = client.post(
            "/api/auth/login", json={"username": "chen", "password": "wrong"}
        )
        assert response.status_code == 401

    blocked = client.post(
        "/api/auth/login", json={"username": "chen", "password": DEMO_PASSWORD}
    )
    assert blocked.status_code == 429


def test_client_cannot_cross_doctor_session_boundary(tmp_path, monkeypatch) -> None:
    chen_client, _ = _client(tmp_path, monkeypatch)
    chen_csrf = _login(chen_client, "chen")
    thread_id = chen_client.post(
        "/api/sessions", json={}, headers={"X-CSRF-Token": chen_csrf}
    ).json()["thread_id"]

    lin_client = TestClient(web.app)
    _login(lin_client, "lin")
    response = lin_client.get(
        f"/api/sessions/{thread_id}/messages",
        params={"actor_id": "doctor-chen"},
    )

    assert response.status_code == 404


def test_missing_permission_is_denied_and_audited(tmp_path, monkeypatch) -> None:
    client, repository = _client(tmp_path, monkeypatch)
    _login(client)
    with repository._connect() as connection:
        connection.execute(
            "DELETE FROM user_roles WHERE user_id = ? AND role_code = ?",
            ("doctor-chen", "doctor"),
        )

    response = client.get("/api/patients")

    assert response.status_code == 403
    with repository._connect() as connection:
        audit = connection.execute(
            """
            SELECT event_type, outcome, target_id FROM audit_events
            WHERE actor_user_id = ? ORDER BY event_id DESC LIMIT 1
            """,
            ("doctor-chen",),
        ).fetchone()
    assert tuple(audit) == ("authorization", "denied", "patient:read")


def test_doctor_cannot_access_admin_page_or_api(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client, "chen")

    assert client.get("/admin").status_code == 403
    assert client.get("/api/admin/users").status_code == 403


def test_admin_can_create_doctor_and_assign_patient_scope(tmp_path, monkeypatch) -> None:
    admin, repository = _client(tmp_path, monkeypatch)
    csrf = _login(admin, "admin", "AdminOnly-2026!")

    bootstrap = admin.get("/api/bootstrap").json()["user"]
    assert bootstrap["roles"] == ["administrator"]
    assert "patient:read" not in bootstrap["permissions"]
    assert admin.get("/admin").status_code == 200

    created = admin.post(
        "/api/admin/users",
        headers={"X-CSRF-Token": csrf},
        json={
            "username": "doctor.wang",
            "display_name": "王医生",
            "department": "骨质疏松研究组",
            "password": "DoctorWang-2026!",
            "roles": ["doctor"],
        },
    )
    assert created.status_code == 200
    user_id = created.json()["user_id"]

    assigned = admin.put(
        f"/api/admin/users/{user_id}/patient-access",
        headers={"X-CSRF-Token": csrf},
        json={"patient_ids": ["P10086"]},
    )
    assert assigned.status_code == 200

    doctor = TestClient(web.app)
    _login(doctor, "doctor.wang", "DoctorWang-2026!")
    patients = doctor.get("/api/patients")
    assert patients.status_code == 200
    assert [item["patient_id"] for item in patients.json()["patients"]] == ["P10086"]

    events = repository.list_audit_events()
    assert any(event["event_type"] == "user_created" for event in events)
    assert any(event["event_type"] == "patient_access_changed" for event in events)


def test_admin_cannot_disable_or_demote_current_account(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    csrf = _login(client, "admin", "AdminOnly-2026!")

    disabled = client.patch(
        "/api/admin/users/admin-local/status",
        headers={"X-CSRF-Token": csrf},
        json={"is_active": False},
    )
    demoted = client.put(
        "/api/admin/users/admin-local/roles",
        headers={"X-CSRF-Token": csrf},
        json={"roles": ["doctor"]},
    )

    assert disabled.status_code == 400
    assert demoted.status_code == 400
