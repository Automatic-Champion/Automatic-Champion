from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from firebase_admin import auth as firebase_auth_sdk

from backend.app.main import app

client = TestClient(app)


VALID_SQUAD_REQUEST = {
    "budget": 100.0,
    "formation": "4-3-3",
    "locked_ids": [],
    "banned_ids": [],
}


def _make_squad() -> list[dict]:
    """Build a valid 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD."""
    players = []
    pid = 1

    def add(pos: str, count: int) -> None:
        nonlocal pid
        for _ in range(count):
            players.append({
                "id": str(pid),
                "name": f"{pos} Player{pid}",
                "position": pos,
                "team": f"Team{chr(65 + (pid % 10))}",
                "cost": 5.0 + pid * 0.5,
                "pred": 40.0 + pid * 2.0,
            })
            pid += 1

    add("GK", 2)
    add("DEF", 5)
    add("MID", 5)
    add("FWD", 3)
    return players


def _mock_optimizer_result() -> dict:
    """Synthetic full-squad result matching the shape build_full_squad() returns."""
    starters = [
        {
            "id": str(i),
            "name": f"Starter{i}",
            "position": "GK" if i == 1 else "DEF" if i <= 5 else "MID" if i <= 8 else "FWD",
            "team": "TeamA",
            "cost": 5.0,
            "pred": 100.0,
            "is_starter": True,
            "bench_order": None,
        }
        for i in range(1, 12)
    ]
    bench = [
        {
            "id": str(i),
            "name": f"Bench{i}",
            "position": "GK" if i == 12 else "DEF" if i == 13 else "MID" if i == 14 else "FWD",
            "team": "TeamB",
            "cost": 4.0,
            "pred": 50.0,
            "is_starter": False,
            "bench_order": i - 11,
        }
        for i in range(12, 16)
    ]
    return {
        "formation": "4-3-3",
        "budget": 100.0,
        "total_cost": 95.5,
        "total_pred": 1500.0,
        "players": starters,
        "bench": bench,
    }


# ─── 1. Public endpoints stay public ──────────────────────────────────────

def test_health_endpoint_is_public():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_players_endpoint_is_public():
    resp = client.get("/players")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ─── 2. Protected endpoint without token → 401 ────────────────────────────

def test_squad_generate_without_token_returns_401():
    resp = client.post("/squad/generate", json=VALID_SQUAD_REQUEST)
    assert resp.status_code == 401
    detail = resp.json()["detail"].lower()
    assert "token" in detail or "auth" in detail


# ─── 3. Protected endpoint with malformed header → 401 ────────────────────

def test_squad_generate_malformed_header_returns_401():
    resp = client.post(
        "/squad/generate",
        json=VALID_SQUAD_REQUEST,
        headers={"Authorization": "notbearer xyz"},
    )
    assert resp.status_code == 401


def test_squad_generate_bearer_with_empty_token_returns_401():
    resp = client.post(
        "/squad/generate",
        json=VALID_SQUAD_REQUEST,
        headers={"Authorization": "Bearer "},
    )
    assert resp.status_code == 401


# ─── 4. Protected endpoint with invalid token → 401 ───────────────────────

@patch("backend.app.auth.firebase_auth.verify_id_token")
def test_squad_generate_invalid_token_returns_401(mock_verify):
    mock_verify.side_effect = firebase_auth_sdk.InvalidIdTokenError("bad token")
    resp = client.post(
        "/squad/generate",
        json=VALID_SQUAD_REQUEST,
        headers={"Authorization": "Bearer fakeToken123"},
    )
    assert resp.status_code == 401
    mock_verify.assert_called_once_with("fakeToken123")


# ─── 5. Protected endpoint with valid token → success ─────────────────────

@patch("src.explainer.explain_squad", return_value={})
@patch("backend.app.routers.squad.generate_optimal_squad")
@patch("backend.app.auth.firebase_auth.verify_id_token")
def test_squad_generate_valid_token_returns_success(
    mock_verify, mock_optimize, mock_explain
):
    mock_verify.return_value = {"uid": "test-user-1", "email": "test@example.com"}
    mock_optimize.return_value = _mock_optimizer_result()

    resp = client.post(
        "/squad/generate",
        json=VALID_SQUAD_REQUEST,
        headers={"Authorization": "Bearer goodToken"},
    )
    assert resp.status_code == 201, resp.text

    data = resp.json()
    assert data["formation"] == "4-3-3"
    assert len(data["players"]) == 11
    assert len(data["bench"]) == 4
    mock_verify.assert_called_once_with("goodToken")
    mock_optimize.assert_called_once()


# ─── 6. Lineup endpoint protected the same way ────────────────────────────

def test_lineup_recommend_without_token_returns_401():
    resp = client.post("/lineup/recommend", json={"squad": _make_squad()})
    assert resp.status_code == 401
    detail = resp.json()["detail"].lower()
    assert "token" in detail or "auth" in detail


@patch("src.explainer.explain_squad", return_value={})
@patch("src.gameweek_predictor.predict_gameweek_points")
@patch("backend.app.auth.firebase_auth.verify_id_token")
def test_lineup_recommend_valid_token_returns_success(
    mock_verify, mock_predict, mock_explain
):
    mock_verify.return_value = {"uid": "test-user-1", "email": "test@example.com"}
    squad = _make_squad()
    mock_predict.return_value = {str(p["id"]): p["pred"] / 38.0 for p in squad}

    resp = client.post(
        "/lineup/recommend",
        json={"squad": squad},
        headers={"Authorization": "Bearer goodToken"},
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert len(data["starters"]) == 11
    assert len(data["bench"]) == 4
    mock_verify.assert_called_once_with("goodToken")
