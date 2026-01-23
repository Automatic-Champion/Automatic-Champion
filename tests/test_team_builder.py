from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.team_builder import build_team_433


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


def test_build_team_433_stub_mode() -> None:
    rows = [
        _make_row(1, "GK One", "TeamA", 1, 45, 50.0),
        _make_row(2, "GK Two", "TeamB", 1, 40, 45.0),
        _make_row(3, "DEF One", "TeamA", 2, 50, 60.0),
        _make_row(4, "DEF Two", "TeamA", 2, 48, 58.0),
        _make_row(5, "DEF Three", "TeamB", 2, 45, 55.0),
        _make_row(6, "DEF Four", "TeamC", 2, 44, 54.0),
        _make_row(7, "DEF Five", "TeamD", 2, 43, 53.0),
        _make_row(8, "MID One", "TeamA", 3, 70, 80.0),
        _make_row(9, "MID Two", "TeamB", 3, 68, 78.0),
        _make_row(10, "MID Three", "TeamC", 3, 66, 76.0),
        _make_row(11, "MID Four", "TeamD", 3, 65, 74.0),
        _make_row(12, "FWD One", "TeamA", 4, 75, 90.0),
        _make_row(13, "FWD Two", "TeamB", 4, 72, 88.0),
        _make_row(14, "FWD Three", "TeamC", 4, 70, 85.0),
        _make_row(15, "FWD Four", "TeamD", 4, 68, 83.0),
    ]

    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "players.csv"
        df.to_csv(data_path, index=False)

        result = build_team_433(
            budget=100.0,
            data_path=str(data_path),
            pred_mode="stub",
            method="ilp",
        )

    players = result["players"]
    assert len(players) == 11

    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for player in players:
        counts[player["position"]] += 1
    assert counts == {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}

    assert result["total_cost_int"] <= result["budget_int"]

    team_counts = {}
    for player in players:
        team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1
    assert all(count <= 3 for count in team_counts.values())
