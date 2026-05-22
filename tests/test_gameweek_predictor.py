from __future__ import annotations

from unittest.mock import patch

import pytest

from src.fpl_api import FPLAPIError
from src.gameweek_predictor import (
    PREDICTOR_VERSION,
    predict_gameweek_points,
    _match_fpl_player,
    _normalize_name,
)


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
        "first_name": "Mohamed",
        "second_name": "Salah",
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
        "first_name": "Erling",
        "second_name": "Haaland",
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


# Extended mock for name-matching tests (covers the hard cases from Bug 2)
_MOCK_FPL_EXTENDED = {
    **_MOCK_FPL_PLAYERS,
    201: {
        "id": 201, "web_name": "Son", "first_name": "Heung-Min", "second_name": "Son",
        "position": "MID", "team_id": 6, "form": 7.0, "points_per_game": 6.0,
        "now_cost": 100, "ep_next": 7.5, "status": "a", "chance_of_playing": 100,
        "total_points": 150, "minutes": 2200,
    },
    202: {
        "id": 202, "web_name": "De Bruyne", "first_name": "Kevin", "second_name": "De Bruyne",
        "position": "MID", "team_id": 12, "form": 5.0, "points_per_game": 5.5,
        "now_cost": 95, "ep_next": 6.0, "status": "a", "chance_of_playing": 75,
        "total_points": 80, "minutes": 1000,
    },
    203: {
        "id": 203, "web_name": "N.Jackson", "first_name": "Nicolas", "second_name": "Jackson",
        "position": "FWD", "team_id": 8, "form": 6.0, "points_per_game": 5.0,
        "now_cost": 75, "ep_next": 5.5, "status": "a", "chance_of_playing": 100,
        "total_points": 120, "minutes": 2000,
    },
    204: {
        "id": 204, "web_name": "B.Silva", "first_name": "Bernardo", "second_name": "Silva",
        "position": "MID", "team_id": 12, "form": 5.0, "points_per_game": 4.5,
        "now_cost": 65, "ep_next": 4.0, "status": "a", "chance_of_playing": 100,
        "total_points": 100, "minutes": 1800,
    },
    205: {
        "id": 205, "web_name": "Alisson", "first_name": "Alisson", "second_name": "Becker",
        "position": "GK", "team_id": 11, "form": 4.0, "points_per_game": 4.5,
        "now_cost": 55, "ep_next": 4.0, "status": "a", "chance_of_playing": 100,
        "total_points": 110, "minutes": 2500,
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictorVersion:
    def test_version_string(self):
        assert PREDICTOR_VERSION == "V8 model"


# When V8 is unavailable (e.g. model files missing or catboost not installed),
# the predictor degrades to the placeholder chain (ep_next → ppg → pred/38).
# The tests below exercise that fallback path by patching the V8 loaders to
# return None.
@patch("src.gameweek_predictor._load_feature_table", return_value=None)
@patch("src.gameweek_predictor._load_models", return_value=None)
class TestWithFPLData:
    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_ep_next_used_when_available(self, _mock_gw, _mock_players, _no_models, _no_table):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        # Salah matched → ep_next = 9.2
        assert preds["1"] == 9.2

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_points_per_game_fallback(self, _mock_gw, _mock_players, _no_models, _no_table):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        # Haaland matched, ep_next is None → points_per_game = 6.5
        assert preds["2"] == 6.5

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_unmatched_falls_back_to_pred(self, _mock_gw, _mock_players, _no_models, _no_table):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        # "Unknown Player" not in FPL data → pred / 38
        assert preds["3"] == pytest.approx(76.0 / 38.0)

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_PLAYERS)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_explicit_gameweek(self, _mock_gw, _mock_players, _no_models, _no_table):
        squad = _make_squad()
        preds = predict_gameweek_points(squad, gameweek=15)

        # Should still work — gameweek is passed but placeholder doesn't
        # differentiate by GW yet.
        assert len(preds) == 3


@patch("src.gameweek_predictor._load_feature_table", return_value=None)
@patch("src.gameweek_predictor._load_models", return_value=None)
class TestFPLAPIFailure:
    @patch("src.gameweek_predictor.get_current_gameweek", side_effect=FPLAPIError("down"))
    def test_all_fallback_on_api_error(self, _mock_gw, _no_models, _no_table):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        for player in squad:
            pid = str(player["id"])
            assert preds[pid] == pytest.approx(player["pred"] / 38.0)

    @patch("src.gameweek_predictor.get_current_gameweek", side_effect=ConnectionError("no network"))
    def test_all_fallback_on_network_error(self, _mock_gw, _no_models, _no_table):
        squad = _make_squad()
        preds = predict_gameweek_points(squad)

        for player in squad:
            pid = str(player["id"])
            assert preds[pid] == pytest.approx(player["pred"] / 38.0)

    @patch("src.gameweek_predictor.get_current_gameweek", side_effect=TypeError("programming bug"))
    def test_non_network_exception_bubbles_up(self, _mock_gw, _no_models, _no_table):
        """TypeError (a programming bug) must NOT be silently swallowed."""
        squad = _make_squad()
        with pytest.raises(TypeError, match="programming bug"):
            predict_gameweek_points(squad)


# ---------------------------------------------------------------------------
# Name matching tests
# ---------------------------------------------------------------------------

def _build_fpl_by_name(fpl_players):
    """Build the fpl_by_name lookup and fpl_list from mock data."""
    fpl_by_name: dict[str, list[dict]] = {}
    fpl_list = list(fpl_players.values())
    for fp in fpl_list:
        key = fp["web_name"].lower()
        fpl_by_name.setdefault(key, []).append(fp)
    return fpl_by_name, fpl_list


class TestNameMatching:
    """Tests for _match_fpl_player with various name formats."""

    def test_web_name_substring_match(self):
        """Salah: web_name 'Salah' is substring of 'Mohamed Salah'."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Mohamed Salah", "position": "MID"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["web_name"] == "Salah"

    def test_son_heungmin(self):
        """Son: web_name 'Son' is substring of 'Son Heung-min'."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Son Heung-min", "position": "MID"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["id"] == 201

    def test_de_bruyne(self):
        """De Bruyne: web_name 'De Bruyne' is substring of 'Kevin De Bruyne'."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Kevin De Bruyne", "position": "MID"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["id"] == 202

    def test_nicolas_jackson(self):
        """N.Jackson: web_name doesn't substring-match, falls back to full name."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Nicolas Jackson", "position": "FWD"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["id"] == 203

    def test_bernardo_silva(self):
        """B.Silva: web_name doesn't substring-match, falls back to full name."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Bernardo Silva", "position": "MID"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["id"] == 204

    def test_alisson(self):
        """Alisson: exact web_name match."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Alisson", "position": "GK"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["id"] == 205

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "MOHAMED SALAH", "position": "MID"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["web_name"] == "Salah"

    def test_unmatched_player_returns_none(self):
        """A completely unknown player should return None."""
        fpl_by_name, fpl_list = _build_fpl_by_name(_MOCK_FPL_EXTENDED)
        player = {"name": "Nonexistent Player", "position": "DEF"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is None

    def test_position_disambiguation(self):
        """When multiple candidates match, prefer same-position."""
        # Create two players with similar names but different positions
        fpl_data = {
            1: {"id": 1, "web_name": "Smith", "first_name": "John", "second_name": "Smith",
                "position": "DEF", "team_id": 1},
            2: {"id": 2, "web_name": "Smith", "first_name": "Adam", "second_name": "Smith",
                "position": "MID", "team_id": 2},
        }
        fpl_by_name, fpl_list = _build_fpl_by_name(fpl_data)
        player = {"name": "Adam Smith", "position": "MID"}
        match = _match_fpl_player(player, fpl_by_name, fpl_list)
        assert match is not None
        assert match["id"] == 2


@patch("src.gameweek_predictor._load_feature_table", return_value=None)
@patch("src.gameweek_predictor._load_models", return_value=None)
class TestNameMatchingIntegration:
    """End-to-end tests: verify matched players get FPL predictions when the
    V8 model path is unavailable (fallback chain)."""

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_EXTENDED)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_nicolas_jackson_gets_fpl_prediction(
        self, _mock_gw, _mock_players, _no_models, _no_table
    ):
        squad = [
            {"id": "10", "name": "Nicolas Jackson", "position": "FWD", "team": "Chelsea", "pred": 100.0},
        ]
        preds = predict_gameweek_points(squad)
        # Should get ep_next=5.5, NOT fallback 100/38=2.63
        assert preds["10"] == 5.5

    @patch("src.gameweek_predictor.get_player_data", return_value=_MOCK_FPL_EXTENDED)
    @patch("src.gameweek_predictor.get_current_gameweek", return_value=10)
    def test_bernardo_silva_gets_fpl_prediction(
        self, _mock_gw, _mock_players, _no_models, _no_table
    ):
        squad = [
            {"id": "11", "name": "Bernardo Silva", "position": "MID", "team": "Man City", "pred": 90.0},
        ]
        preds = predict_gameweek_points(squad)
        assert preds["11"] == 4.0  # ep_next
