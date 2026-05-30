from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.deps import get_db
from backend.app.main import app

# Ensure all models are registered before create_all
import backend.app.models  # noqa: F401


# ─── Test DB setup: in-memory SQLite, shared across all sessions ───

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_saved_squads():
    """Wipe the saved_squads table before each test."""
    from backend.app.models import SavedSquad

    db = TestSessionLocal()
    try:
        db.query(SavedSquad).delete()
        db.commit()
    finally:
        db.close()
    yield


@pytest.fixture
def mock_user_a():
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "user-a", "email": "a@example.com"}
        yield mock


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer fake-token"}


def _sample_squad_payload() -> dict:
    """Match SquadGenerateResponse shape."""
    return {
        "formation": "4-3-3",
        "budget": 100.0,
        "total_cost": 95.5,
        "total_predicted_points": 1500.0,
        "players": [
            {
                "id": str(i),
                "name": f"Starter{i}",
                "team": "TeamA",
                "position": "GK" if i == 1 else "DEF" if i <= 5 else "MID" if i <= 8 else "FWD",
                "cost": 5.0,
                "predicted_points": 100.0,
                "is_starter": True,
                "bench_order": None,
                "explanations": [],
            }
            for i in range(1, 12)
        ],
        "bench": [
            {
                "id": str(i),
                "name": f"Bench{i}",
                "team": "TeamB",
                "position": "GK" if i == 12 else "DEF" if i == 13 else "MID" if i == 14 else "FWD",
                "cost": 4.0,
                "predicted_points": 50.0,
                "is_starter": False,
                "bench_order": i - 11,
                "explanations": [],
            }
            for i in range(12, 16)
        ],
    }


def _create_request(name: str) -> dict:
    return {"name": name, "squad": _sample_squad_payload()}


# ─── 1. Auth ───────────────────────────────────────────────────────


def test_list_without_token_returns_401():
    resp = client.get("/squads")
    assert resp.status_code == 401


def test_save_without_token_returns_401():
    resp = client.post("/squads", json=_create_request("My Squad"))
    assert resp.status_code == 401


def test_get_without_token_returns_401():
    resp = client.get("/squads/1")
    assert resp.status_code == 401


def test_delete_without_token_returns_401():
    resp = client.delete("/squads/1")
    assert resp.status_code == 401


# ─── 2. Save then retrieve ─────────────────────────────────────────


def test_save_squad_returns_201_and_appears_in_list(mock_user_a, auth_headers):
    resp = client.post("/squads", json=_create_request("My Squad"), headers=auth_headers)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "My Squad"
    assert body["formation"] == "4-3-3"
    assert body["total_cost"] == 95.5
    assert body["payload"]["formation"] == "4-3-3"
    saved_id = body["id"]

    # Appears in list
    list_resp = client.get("/squads", headers=auth_headers)
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) == 1
    assert items[0]["id"] == saved_id
    assert items[0]["name"] == "My Squad"

    # Can fetch full payload
    get_resp = client.get(f"/squads/{saved_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["payload"]["formation"] == "4-3-3"


# ─── 3. Limit ──────────────────────────────────────────────────────


def test_saving_more_than_10_squads_returns_400(mock_user_a, auth_headers):
    for i in range(10):
        resp = client.post(
            "/squads", json=_create_request(f"Squad {i}"), headers=auth_headers
        )
        assert resp.status_code == 201, resp.text

    resp = client.post(
        "/squads", json=_create_request("Squad 11"), headers=auth_headers
    )
    assert resp.status_code == 400
    assert "limit reached" in resp.json()["detail"].lower()


# ─── 4. Duplicate name ─────────────────────────────────────────────


def test_duplicate_name_returns_400(mock_user_a, auth_headers):
    resp = client.post("/squads", json=_create_request("My Squad"), headers=auth_headers)
    assert resp.status_code == 201

    resp = client.post("/squads", json=_create_request("My Squad"), headers=auth_headers)
    assert resp.status_code == 400
    assert "already exists" in resp.json()["detail"].lower()


def test_duplicate_name_is_case_insensitive(mock_user_a, auth_headers):
    resp = client.post("/squads", json=_create_request("my squad"), headers=auth_headers)
    assert resp.status_code == 201

    resp = client.post("/squads", json=_create_request("MY SQUAD"), headers=auth_headers)
    assert resp.status_code == 400


# ─── 5. Ownership ──────────────────────────────────────────────────


def test_delete_other_users_squad_returns_404(auth_headers):
    # User B creates a squad
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "user-b", "email": "b@example.com"}
        resp = client.post(
            "/squads", json=_create_request("B's Squad"), headers=auth_headers
        )
        assert resp.status_code == 201
        b_squad_id = resp.json()["id"]

    # User A tries to delete it
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "user-a", "email": "a@example.com"}
        resp = client.delete(f"/squads/{b_squad_id}", headers=auth_headers)
        assert resp.status_code == 404


def test_get_other_users_squad_returns_404(auth_headers):
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "user-b", "email": "b@example.com"}
        resp = client.post(
            "/squads", json=_create_request("B's Squad"), headers=auth_headers
        )
        b_squad_id = resp.json()["id"]

    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "user-a", "email": "a@example.com"}
        resp = client.get(f"/squads/{b_squad_id}", headers=auth_headers)
        assert resp.status_code == 404


# ─── 6. Not found ──────────────────────────────────────────────────


def test_get_nonexistent_squad_returns_404(mock_user_a, auth_headers):
    resp = client.get("/squads/99999", headers=auth_headers)
    assert resp.status_code == 404


def test_delete_nonexistent_squad_returns_404(mock_user_a, auth_headers):
    resp = client.delete("/squads/99999", headers=auth_headers)
    assert resp.status_code == 404


# ─── 7. Delete works ───────────────────────────────────────────────


def test_delete_own_squad_returns_204(mock_user_a, auth_headers):
    resp = client.post("/squads", json=_create_request("My Squad"), headers=auth_headers)
    squad_id = resp.json()["id"]

    resp = client.delete(f"/squads/{squad_id}", headers=auth_headers)
    assert resp.status_code == 204

    resp = client.get(f"/squads/{squad_id}", headers=auth_headers)
    assert resp.status_code == 404
