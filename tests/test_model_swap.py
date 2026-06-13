from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.team_builder import build_full_squad


# ── Dummy model: ignores features, returns a constant for every row ──


class DummyModel:
    """Degenerate stand-in for a trained position model."""

    def __init__(self, constant: float = 5.0):
        self.constant = constant

    def predict(self, X):
        return np.full(len(X), self.constant)


# ── Fixture data (copied from tests/test_full_squad.py) ──────────────


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


def _position_counts(players: list[dict]) -> dict[str, int]:
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in players:
        counts[p["position"]] += 1
    return counts


def _build_with_dummy_models(constant: float, budget: float = 100.0, **kwargs) -> dict:
    """Build a full squad through the real `pred_mode="models"` path with every
    position model swapped for a `DummyModel`.

    `_load_selected_features` is patched to None so `_predict` falls back to the
    feature columns derived from the CSV (the real selected-feature JSON lists
    columns absent from this synthetic data, which would yield an empty subset).
    """
    df = pd.DataFrame(_build_test_data())
    models = {pos: DummyModel(constant) for pos in ("GK", "DEF", "MID", "FWD")}
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "players.csv"
        df.to_csv(data_path, index=False)
        with patch("src.team_builder._load_models", return_value=models), patch(
            "src.team_builder._load_selected_features", return_value=None
        ):
            return build_full_squad(
                budget=budget,
                data_path=str(data_path),
                pred_mode="models",
                **kwargs,
            )


# ── NFR-03: swap model with a dummy → system still works ─────────────


def test_all_dummy_models_still_build_valid_squad() -> None:
    # Core NFR-03 assertion: degenerate models replacing the real ones must
    # still yield a valid 15-player squad — the pipeline is model-agnostic.
    result = _build_with_dummy_models(constant=5.0, formation="4-3-3")

    starters = result["players"]
    bench = result["bench"]
    all_players = starters + bench

    assert len(starters) == 11
    assert len(bench) == 4
    assert len(all_players) == 15

    assert _position_counts(all_players) == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert result["total_cost_int"] <= result["budget_int"]

    team_counts: dict[str, int] = {}
    for p in all_players:
        team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1
    assert all(c <= 3 for c in team_counts.values())


def test_dummy_model_with_constant_zero_still_valid() -> None:
    # Every prediction is identical (and zero) — no signal at all. The ILP must
    # stay feasible and return a complete, valid squad.
    result = _build_with_dummy_models(constant=0.0, formation="4-3-3")

    all_players = result["players"] + result["bench"]

    assert len(all_players) == 15
    assert _position_counts(all_players) == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert result["total_cost_int"] <= result["budget_int"]

    team_counts: dict[str, int] = {}
    for p in all_players:
        team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1
    assert all(c <= 3 for c in team_counts.values())
