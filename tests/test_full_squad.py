from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

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
    """Build a dataset with enough players for a full 15-player squad.

    Need: 2 GK, 5 DEF, 5 MID, 3 FWD = 15 players minimum.
    Provide extras so the optimizer has choices.
    """
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


def test_build_full_squad_returns_15_players() -> None:
    rows = _build_test_data()
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "players.csv"
        df.to_csv(data_path, index=False)

        result = build_full_squad(
            budget=100.0,
            data_path=str(data_path),
            pred_mode="stub",
            formation="4-3-3",
        )

    starters = result["players"]
    bench = result["bench"]

    # 11 starters + 4 bench = 15 total
    assert len(starters) == 11
    assert len(bench) == 4

    # Starters match requested formation (4-3-3)
    starter_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in starters:
        starter_counts[p["position"]] += 1
    assert starter_counts == {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}

    # Full squad has correct position counts (2-5-5-3)
    all_players = starters + bench
    full_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in all_players:
        full_counts[p["position"]] += 1
    assert full_counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    # Budget constraint
    assert result["total_cost_int"] <= result["budget_int"]

    # Max 3 per team
    team_counts: dict[str, int] = {}
    for p in all_players:
        team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1
    assert all(c <= 3 for c in team_counts.values())

    # is_starter / bench_order fields
    assert all(p["is_starter"] is True for p in starters)
    assert all(p["bench_order"] is None for p in starters)
    assert all(p["is_starter"] is False for p in bench)
    assert [p["bench_order"] for p in bench] == [1, 2, 3, 4]

    # Bench ordered by pred descending
    bench_preds = [p["pred"] for p in bench]
    assert bench_preds == sorted(bench_preds, reverse=True)

    # total_pred is starters only
    assert result["total_pred"] == sum(p["pred"] for p in starters)


def test_build_full_squad_different_formation() -> None:
    rows = _build_test_data()
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "players.csv"
        df.to_csv(data_path, index=False)

        result = build_full_squad(
            budget=100.0,
            data_path=str(data_path),
            pred_mode="stub",
            formation="3-5-2",
        )

    starters = result["players"]
    bench = result["bench"]

    assert len(starters) == 11
    assert len(bench) == 4

    starter_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in starters:
        starter_counts[p["position"]] += 1
    assert starter_counts == {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}
