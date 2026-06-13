from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.optimizer import OptimizationError
from src.team_builder import build_full_squad


def _make_row(player_id: int, name: str, team: str, element_type: int, price_now: int, pred: float) -> dict:
    first, second = name.split(" ", 1)
    return {
        "id": str(player_id),
        "team_name": team,
        "first_name": first,
        "second_name": second,
        "element_type": element_type,
        "price_now": price_now,
        "1_years_past_goals_scored": 1,
        "1_years_past_assists": 1,
        "pred": pred,
    }


def _build_test_data() -> list[dict]:
    """Dataset with enough players for a full 15-player squad (2 GK / 5 DEF / 5 MID / 3 FWD)."""
    return [
        # GK (3 available, need 2)
        _make_row(1, "GK One", "TeamA", 1, 45, 50.0),
        _make_row(2, "GK Two", "TeamB", 1, 40, 45.0),
        _make_row(3, "GK Three", "TeamC", 1, 38, 40.0),
        # DEF (6 available, need 5)
        _make_row(4, "DEF One", "TeamA", 2, 50, 60.0),
        _make_row(5, "DEF Two", "TeamB", 2, 48, 58.0),
        _make_row(6, "DEF Three", "TeamC", 2, 45, 55.0),
        _make_row(7, "DEF Four", "TeamD", 2, 44, 54.0),
        _make_row(8, "DEF Five", "TeamE", 2, 43, 53.0),
        _make_row(9, "DEF Six", "TeamF", 2, 42, 52.0),
        # MID (6 available, need 5)
        _make_row(10, "MID One", "TeamA", 3, 70, 80.0),
        _make_row(11, "MID Two", "TeamB", 3, 68, 78.0),
        _make_row(12, "MID Three", "TeamC", 3, 66, 76.0),
        _make_row(13, "MID Four", "TeamD", 3, 65, 74.0),
        _make_row(14, "MID Five", "TeamE", 3, 63, 72.0),
        _make_row(15, "MID Six", "TeamF", 3, 60, 70.0),
        # FWD (4 available, need 3)
        _make_row(16, "FWD One", "TeamB", 4, 75, 90.0),
        _make_row(17, "FWD Two", "TeamC", 4, 72, 88.0),
        _make_row(18, "FWD Three", "TeamD", 4, 70, 85.0),
        _make_row(19, "FWD Four", "TeamE", 4, 68, 83.0),
    ]


def _build_squad(rows: list[dict], budget: float = 100.0, **kwargs) -> dict:
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "players.csv"
        df.to_csv(data_path, index=False)
        return build_full_squad(
            budget=budget,
            data_path=str(data_path),
            pred_mode="stub",
            **kwargs,
        )


def _position_counts(players: list[dict]) -> dict[str, int]:
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in players:
        counts[p["position"]] += 1
    return counts


# ── TC-02: build with locked / banned constraints (happy paths) ──────


def test_locked_player_appears_in_squad() -> None:
    # id 3 is the lowest-pred GK and would not normally be picked over ids 1 & 2.
    result = _build_squad(_build_test_data(), locked_ids={"3"})

    all_ids = {p["id"] for p in result["players"] + result["bench"]}
    assert "3" in all_ids


def test_banned_player_excluded_from_squad() -> None:
    # id 16 is the highest-pred FWD; banning it forces the other 3 FWDs in.
    result = _build_squad(_build_test_data(), banned_ids={"16"})

    all_players = result["players"] + result["bench"]
    all_ids = {p["id"] for p in all_players}
    assert "16" not in all_ids
    assert _position_counts(all_players) == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


# ── TC-03: infeasible constraints → explain conflict (one per error) ──


def test_locked_and_banned_overlap_raises() -> None:
    with pytest.raises(ValueError, match="Locked and banned overlap"):
        _build_squad(_build_test_data(), locked_ids={"5"}, banned_ids={"5"})


def test_locked_id_not_found_raises() -> None:
    with pytest.raises(ValueError, match="Locked player IDs not found"):
        _build_squad(_build_test_data(), locked_ids={"999"})


def test_locked_over_budget_raises() -> None:
    # budget 5.0 -> 50 tenths; locked ids 10 (70) + 16 (75) = 145 tenths.
    with pytest.raises(ValueError, match="Locked players exceed budget"):
        _build_squad(_build_test_data(), budget=5.0, locked_ids={"10", "16"})


def test_locked_exceeds_position_count_raises() -> None:
    # Three GKs available, squad needs only 2; locking all 3 is infeasible.
    with pytest.raises(ValueError, match="exceed required count for"):
        _build_squad(_build_test_data(), locked_ids={"1", "2", "3"})


def test_locked_exceeds_team_cap_raises() -> None:
    # Add a 4th TeamA player, then lock all 4 (max_per_team default is 3).
    rows = _build_test_data()
    rows.append(_make_row(20, "DEF Seven", "TeamA", 2, 45, 51.0))
    with pytest.raises(ValueError, match="exceed max per team"):
        _build_squad(rows, locked_ids={"1", "4", "10", "20"})


def test_infeasible_budget_raises() -> None:
    # budget 1.0 -> 10 tenths; no valid 15-player squad fits.
    with pytest.raises(ValueError, match="No feasible solution found"):
        _build_squad(_build_test_data(), budget=1.0)


# ── TC-05: router maps optimizer errors to HTTP 400 ──────────────────


client = TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer faketoken"}


@pytest.fixture
def mock_verify_token():
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "test-user", "email": "test@example.com"}
        yield mock


@patch(
    "backend.app.routers.squad.generate_optimal_squad",
    side_effect=ValueError("Locked and banned overlap: ['5']"),
)
def test_generate_squad_value_error_returns_400(mock_generate, mock_verify_token, auth_headers):
    resp = client.post(
        "/squad/generate",
        json={"budget": 100.0, "formation": "4-3-3"},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "Locked and banned overlap" in resp.json()["detail"]


@patch(
    "backend.app.routers.squad.generate_optimal_squad",
    side_effect=OptimizationError("No feasible solution found for the given constraints"),
)
def test_generate_squad_optimization_error_returns_400(mock_generate, mock_verify_token, auth_headers):
    resp = client.post(
        "/squad/generate",
        json={"budget": 100.0, "formation": "4-3-3"},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "No feasible solution found" in resp.json()["detail"]
