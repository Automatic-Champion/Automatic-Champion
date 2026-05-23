from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer faketoken"}


@pytest.fixture
def mock_verify_token():
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "test-user", "email": "test@example.com"}
        yield mock


def _make_squad() -> list[dict]:
    """Build a valid 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD."""
    players = []
    pid = 1

    def add(pos: str, count: int) -> None:
        nonlocal pid
        for i in range(count):
            players.append({
                "id": str(pid),
                "name": f"{pos} Player{i+1}",
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


def _mock_gw_predictions(squad, gameweek=None):
    """Return simple predictions based on player pred / 38."""
    return {str(p["id"]): p["pred"] / 38.0 for p in squad}


@patch("src.gameweek_predictor.predict_gameweek_points", side_effect=_mock_gw_predictions)
@patch("src.explainer.explain_squad", return_value={})
def test_recommend_lineup_valid(mock_explain, mock_predict, mock_verify_token, auth_headers):
    squad = _make_squad()
    resp = client.post("/lineup/recommend", json={"squad": squad}, headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["starters"]) == 11
    assert len(data["bench"]) == 4
    assert data["captain_id"] is not None
    assert data["vice_captain_id"] is not None
    assert data["captain_id"] != data["vice_captain_id"]
    assert data["formation"] is not None
    assert data["total_gw_points"] > 0

    # Exactly one captain and one vice-captain among starters
    captains = [s for s in data["starters"] if s["is_captain"]]
    vices = [s for s in data["starters"] if s["is_vice_captain"]]
    assert len(captains) == 1
    assert len(vices) == 1

    # Bench has orders 1-4
    bench_orders = sorted(b["bench_order"] for b in data["bench"])
    assert bench_orders == [1, 2, 3, 4]


@patch("src.gameweek_predictor.predict_gameweek_points", side_effect=_mock_gw_predictions)
def test_recommend_lineup_invalid_squad_size(mock_predict, mock_verify_token, auth_headers):
    squad = _make_squad()[:10]  # Only 10 players
    resp = client.post("/lineup/recommend", json={"squad": squad}, headers=auth_headers)
    assert resp.status_code == 400
    assert "15" in resp.json()["detail"]


@patch("src.gameweek_predictor.predict_gameweek_points", side_effect=_mock_gw_predictions)
@patch("src.explainer.explain_squad", return_value={})
def test_recommend_lineup_with_formation(mock_explain, mock_predict, mock_verify_token, auth_headers):
    squad = _make_squad()
    resp = client.post(
        "/lineup/recommend",
        json={"squad": squad, "formation": "3-5-2"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["formation"] == "3-5-2"

    pos_counts = {}
    for s in data["starters"]:
        pos_counts[s["position"]] = pos_counts.get(s["position"], 0) + 1
    assert pos_counts == {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}


@patch("src.gameweek_predictor.predict_gameweek_points", side_effect=_mock_gw_predictions)
def test_recommend_lineup_invalid_formation(mock_predict, mock_verify_token, auth_headers):
    squad = _make_squad()
    resp = client.post(
        "/lineup/recommend",
        json={"squad": squad, "formation": "1-1-1"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


@patch("src.fpl_api.get_current_gameweek", return_value=12)
@patch("src.gameweek_predictor.predict_gameweek_points", side_effect=_mock_gw_predictions)
@patch("src.explainer.explain_squad", return_value={})
def test_recommend_lineup_null_gameweek_returns_resolved(
    mock_explain, mock_predict, mock_gw, mock_verify_token, auth_headers
):
    squad = _make_squad()
    resp = client.post(
        "/lineup/recommend",
        json={"squad": squad, "gameweek": None},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["gameweek"] == 12


@patch("src.gameweek_predictor.predict_gameweek_points", side_effect=_mock_gw_predictions)
@patch("src.explainer.explain_squad", side_effect=RuntimeError("boom"))
def test_recommend_lineup_explanation_failure_non_fatal(
    mock_explain, mock_predict, mock_verify_token, auth_headers
):
    squad = _make_squad()
    resp = client.post("/lineup/recommend", json={"squad": squad}, headers=auth_headers)
    assert resp.status_code == 200
    # Explanations should be empty but response still valid
    data = resp.json()
    assert len(data["starters"]) == 11
