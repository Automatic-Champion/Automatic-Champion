from __future__ import annotations

from unittest.mock import patch

import pytest

from src.fpl_api import FPLAPIError
from src.gameweek_predictor import PREDICTOR_VERSION, predict_gameweek_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_squad() -> list[dict]:
    """A minimal squad (only 3 players for predictor tests)."""
    return [
        {"id": "1", "name": "Mohamed Salah", "position": "MID", "team": "Liverpool", "pred": 190.0},
        {"id": "2", "name": "Erling Haaland", "position": "FWD", "team": "Man City", "pred": 200.0},
        {"id": "3", "name": "Unknown Player", "position": "DEF", "team": "Arsenal", "pred": 76.0},
    ]


_MOCK_FPL_PLAYERS = {
    101: {
        "id": 101,
        "web_name": "Salah",
        "position": "MID",
        "team_id": 11,
        "form": 8.0,
        "points_per_game": 7.5,
        "now_cost": 130,
        "ep_next": 9.2,
        "status": "a",
        "chance_of_playing": 100,
        "total_points": 200,
        "minutes": 2000,
    },
    102: {
        "id": 102,
        "web_name": "Haaland",
        "position": "FWD",
        "team_id": 12,
        "form": 6.0,
        "points_per_game": 6.5,
        "now_cost": 140,
        "ep_next": None,  # no ep_next → should fall back to points_per_game
        "status": "a",
        "chance_of_playing": 100,
        "total_points": 180,
        "minutes": 1900,
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictorVersion:
    def test_version_string(self):
        assert PREDICTOR_VERSION == "placeholder-v1"


class TestWithFPLData:
    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_ep_next_used_when_available(self, _mock_gw, _mock_players):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        # Salah matched → ep_next = 9.2
        assert preds["1"] == 9.2

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_points_per_game_fallback(self, _mock_gw, _mock_players):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        # Haaland matched, ep_next is None → points_per_game = 6.5
        assert preds["2"] == 6.5

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_unmatched_falls_back_to_pred(self, _mock_gw, _mock_players):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        # "Unknown Player" not in FPL data → pred / 38
        assert preds["3"] == pytest.approx(76.0 / 38.0)

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_explicit_gameweek(self, _mock_gw, _mock_players):
        squad = _make_squad()
        preds = predict_gameweek_points(squad, gameweek=15)

        # Should still work — gameweek is passed but placeholder doesn't
        # differentiate by GW yet.
        assert len(preds) == 3


class TestFPLAPIFailure:
    @patch("src.gameweek_predictor.get_current_gameweek", side_effect=FPLAPIError("down"))
    def test_all_fallback_on_api_error(self, _mock_gw):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        for player in squad:
            pid = str(player["id"])
            assert preds[pid] == pytest.approx(player["pred"] / 38.0)

    @patch("src.gameweek_predictor.get_current_gameweek", side_effect=ConnectionError("no network"))
    def test_all_fallback_on_network_error(self, _mock_gw):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        for player in squad:
            pid = str(player["id"])
            assert preds[pid] == pytest.approx(player["pred"] / 38.0)
