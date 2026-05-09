from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.explainer import explain_selection, explain_squad


def _make_test_csv(tmpdir: str, num_players: int = 5) -> str:
    """Create a test CSV with multiple players in the same position for comparison."""
    rows = []
    for i in range(1, num_players + 1):
        rows.append({
            "id": str(i),
            "team_name": f"Team{chr(64 + i)}",
            "first_name": f"First{i}",
            "second_name": f"Last{i}",
            "element_type": 3,
            "position": "MID",
            "price_now": 50 + i * 10,
            "total_points": 50 + i * 30,
            "1_years_past_total_points": 40 + i * 25,
            "1_years_past_goals_scored": i * 3,
            "1_years_past_assists": i * 2,
            "1_years_past_minutes": 1000 + i * 400,
            "1_years_past_bonus": i * 5,
            "1_years_past_bps": i * 50,
            "1_years_past_ict_index": i * 40.0,
            "1_years_past_clean_sheets": i,
            "1_years_past_goals_conceded": 30 - i * 3,
        })
    # Add a GK
    rows.append({
        "id": "10",
        "team_name": "TeamGK",
        "first_name": "Keeper",
        "second_name": "One",
        "element_type": 1,
        "position": "GK",
        "price_now": 50,
        "total_points": 120,
        "1_years_past_total_points": 110,
        "1_years_past_goals_scored": 0,
        "1_years_past_assists": 1,
        "1_years_past_minutes": 3200,
        "1_years_past_bonus": 15,
        "1_years_past_bps": 400,
        "1_years_past_ict_index": 20.0,
        "1_years_past_clean_sheets": 14,
        "1_years_past_goals_conceded": 25,
        "1_years_past_gw_saves": 120,
    })
    df = pd.DataFrame(rows)
    path = Path(tmpdir) / "players.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_explain_selection_returns_correct_structure() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        result = explain_selection(
            player_id="5",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    assert len(result) == 3
    for item in result:
        assert "text" in item
        assert "category" in item
        assert isinstance(item["text"], str)
        assert len(item["text"]) > 0
        assert item["category"] in {"performance", "attacking", "defensive", "reliability", "value", "trending"}


def test_explain_selection_player_not_found() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        result = explain_selection(
            player_id="999",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    assert result == []


def test_explain_selection_no_raw_feature_names() -> None:
    """Explanations should not contain raw feature names like '1_years_past_...'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        result = explain_selection(
            player_id="5",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    for item in result:
        assert "1_years_past" not in item["text"]
        assert "element_type" not in item["text"]


def test_different_players_get_different_explanations() -> None:
    """Different players in the same position should get different text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        result1 = explain_selection(
            player_id="1", position="MID", data_path=data_path, models_dir=tmpdir
        )
        result5 = explain_selection(
            player_id="5", position="MID", data_path=data_path, models_dir=tmpdir
        )

    texts1 = {item["text"] for item in result1}
    texts5 = {item["text"] for item in result5}
    assert texts1 != texts5, "Different players should get different explanations"


def test_value_explanation_always_included() -> None:
    """One of the explanations should always be about value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        result = explain_selection(
            player_id="3",
            position="MID",
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    categories = [item["category"] for item in result]
    assert "value" in categories


def test_explain_squad_returns_all_players() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _make_test_csv(tmpdir)
        players = [
            {"id": "1", "position": "MID"},
            {"id": "5", "position": "MID"},
            {"id": "10", "position": "GK"},
        ]
        result = explain_squad(
            players=players,
            data_path=data_path,
            models_dir=tmpdir,
            top_k=3,
        )

    assert "1" in result
    assert "5" in result
    assert "10" in result
    for pid in result:
        assert len(result[pid]) <= 3
        for item in result[pid]:
            assert "text" in item
            assert "category" in item


def test_explain_selection_limited_data() -> None:
    """Player with all-zero stats should get a fallback explanation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = [{
            "id": "99",
            "team_name": "TeamX",
            "first_name": "No",
            "second_name": "Data",
            "element_type": 4,
            "position": "FWD",
            "price_now": 0,
            "total_points": 0,
            "1_years_past_total_points": 0,
            "1_years_past_goals_scored": 0,
            "1_years_past_assists": 0,
            "1_years_past_minutes": 0,
        }]
        df = pd.DataFrame(rows)
        path = Path(tmpdir) / "players.csv"
        df.to_csv(path, index=False)

        result = explain_selection(
            player_id="99",
            position="FWD",
            data_path=str(path),
            models_dir=tmpdir,
            top_k=3,
        )

    assert len(result) >= 1
    assert "text" in result[0]
    assert "category" in result[0]
