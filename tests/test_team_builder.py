from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

import numpy as np

from src.team_builder import build_team_433, _add_momentum_features, _build_feature_cols, _load_selected_features


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
        "1_years_past_total_points": 50,
        "1_years_past_minutes": 1000,
        "1_years_past_ict_index": 30.0,
        "2_years_past_goals_scored": 0,
        "2_years_past_assists": 0,
        "2_years_past_total_points": 40,
        "2_years_past_minutes": 800,
        "2_years_past_ict_index": 25.0,
        "3_years_past_goals_scored": 0,
        "3_years_past_total_points": 35,
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


def test_add_momentum_features_computes_deltas() -> None:
    df = pd.DataFrame([{
        "1_years_past_total_points": 100,
        "2_years_past_total_points": 80,
        "1_years_past_minutes": 2000,
        "2_years_past_minutes": 1500,
        "1_years_past_ict_index": 50.0,
        "2_years_past_ict_index": 40.0,
        "1_years_past_goals_scored": 10,
        "2_years_past_goals_scored": 7,
    }])
    result = _add_momentum_features(df)
    assert result["momentum_total_points"].iloc[0] == 20
    assert result["momentum_minutes"].iloc[0] == 500
    assert result["momentum_ict_index"].iloc[0] == 10.0
    assert result["momentum_goals_scored"].iloc[0] == 3


def test_add_momentum_features_nan_when_missing_2y() -> None:
    df = pd.DataFrame([{
        "1_years_past_total_points": 100,
        "1_years_past_minutes": 2000,
        "1_years_past_ict_index": 50.0,
        "1_years_past_goals_scored": 10,
    }])
    result = _add_momentum_features(df)
    assert np.isnan(result["momentum_total_points"].iloc[0])
    assert np.isnan(result["momentum_minutes"].iloc[0])


def test_build_feature_cols_includes_multi_year_and_momentum() -> None:
    df = pd.DataFrame([{
        "price_now": 50,
        "1_years_past_goals_scored": 5,
        "2_years_past_goals_scored": 3,
        "3_years_past_goals_scored": 2,
        "momentum_total_points": 10,
        "other_col": "ignore",
    }])
    cols = _build_feature_cols(df)
    assert "price_now" in cols
    assert "1_years_past_goals_scored" in cols
    assert "2_years_past_goals_scored" in cols
    assert "3_years_past_goals_scored" in cols
    assert "momentum_total_points" in cols
    assert "other_col" not in cols


def test_load_selected_features_returns_none_when_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent.json"
        result = _load_selected_features(path)
    assert result is None


def test_load_selected_features_returns_dict_when_exists() -> None:
    expected = {
        "1": ["price_now", "1_years_past_total_points"],
        "2": ["price_now", "1_years_past_clean_sheets"],
        "3": ["price_now", "1_years_past_ict_index"],
        "4": ["price_now", "1_years_past_creativity"],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "selected_features.json"
        with open(path, "w") as f:
            json.dump(expected, f)
        result = _load_selected_features(path)
    assert result == expected
    assert isinstance(result["1"], list)
    assert len(result) == 4
