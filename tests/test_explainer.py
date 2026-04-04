from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.explainer import explain_selection, explain_squad


class MockModel:
    """A picklable mock model with feature_importances_ and feature_names_in_."""

    def __init__(self, feature_names: list[str], importances: list[float]) -> None:
        self.feature_importances_ = np.array(importances)
        self.feature_names_in_ = np.array(feature_names)


class MockLinearModel:
    """A picklable mock linear model with coef_ and feature_names_in_."""

    def __init__(self, feature_names: list[str], coefficients: list[float]) -> None:
        self.coef_ = np.array(coefficients)
        self.feature_names_in_ = np.array(feature_names)


def _make_mock_model(feature_names: list[str], importances: list[float]) -> MockModel:
    return MockModel(feature_names, importances)


def _make_test_csv(tmpdir: str) -> str:
    rows = [
        {
            "id": "1",
            "team_name": "TeamA",
            "first_name": "John",
            "second_name": "Doe",
            "element_type": 3,
            "price_now": 70,
            "1_years_past_goals_scored": 15,
            "1_years_past_assists": 10,
        },
    ]
    df = pd.DataFrame(rows)
    path = Path(tmpdir) / "players.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_explain_selection_returns_correct_structure() -> None:
    feature_names = ["price_now", "1_years_past_goals_scored", "1_years_past_assists"]
    importances = [0.5, 0.3, 0.2]
    mock_model = _make_mock_model(feature_names, importances)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        models_dir = tmpdir
        # Write mock model
        import joblib

        joblib.dump(mock_model, Path(tmpdir) / "position_model_3.joblib")

        result = explain_selection(
            player_id="1",
            position="MID",
            data_path=data_path,
            models_dir=models_dir,
            top_k=3,
        )

    assert len(result) == 3
    for item in result:
        assert "feature" in item
        assert "value" in item
        assert "importance" in item
        assert "explanation" in item
        assert isinstance(item["importance"], float)
        assert isinstance(item["explanation"], str)

    # Sorted by importance descending
    assert result[0]["importance"] >= result[1]["importance"]
    assert result[1]["importance"] >= result[2]["importance"]

    # First feature should be price_now (highest importance)
    assert result[0]["feature"] == "price_now"


def test_explain_selection_player_not_found() -> None:
    feature_names = ["price_now", "1_years_past_goals_scored"]
    importances = [0.6, 0.4]
    mock_model = _make_mock_model(feature_names, importances)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        import joblib

        joblib.dump(mock_model, Path(tmpdir) / "position_model_3.joblib")

        result = explain_selection(
            player_id="999",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    assert result == []


def test_explain_selection_model_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)

        result = explain_selection(
            player_id="1",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    assert result == []


def test_explain_squad_calls_explain_selection() -> None:
    feature_names = ["price_now", "1_years_past_goals_scored", "1_years_past_assists"]
    importances = [0.5, 0.3, 0.2]
    mock_model = _make_mock_model(feature_names, importances)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        import joblib

        joblib.dump(mock_model, Path(tmpdir) / "position_model_3.joblib")

        players = [{"id": "1", "position": "MID"}]
        result = explain_squad(
            players=players,
            data_path=data_path,
            models_dir=tmpdir,
            top_k=2,
        )

    assert "1" in result
    assert len(result["1"]) == 2


def test_explain_selection_linear_model_coef() -> None:
    """Linear models (Ridge/ElasticNet) use coef_ instead of feature_importances_."""
    feature_names = ["price_now", "1_years_past_goals_scored", "1_years_past_assists"]
    # Negative coefficient should still rank high (absolute value used)
    coefficients = [0.1, -0.8, 0.3]
    mock_model = MockLinearModel(feature_names, coefficients)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        import joblib

        joblib.dump(mock_model, Path(tmpdir) / "position_model_3.joblib")

        result = explain_selection(
            player_id="1",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    assert len(result) == 3
    for item in result:
        assert "feature" in item
        assert "value" in item
        assert "importance" in item
        assert "explanation" in item
        assert isinstance(item["importance"], float)
        assert isinstance(item["explanation"], str)

    # Sorted by normalized abs(coef_) descending: goals_scored(0.8/1.2) > assists(0.3/1.2) > price(0.1/1.2)
    assert result[0]["feature"] == "1_years_past_goals_scored"
    assert abs(result[0]["importance"] - 0.8 / 1.2) < 1e-6
    assert result[1]["feature"] == "1_years_past_assists"
    assert abs(result[1]["importance"] - 0.3 / 1.2) < 1e-6
    assert result[2]["feature"] == "price_now"
    assert abs(result[2]["importance"] - 0.1 / 1.2) < 1e-6
