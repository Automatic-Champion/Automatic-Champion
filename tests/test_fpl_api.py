"""Tests for src.fpl_api — all HTTP calls are mocked."""

import json
import threading
from unittest import mock

import pytest

from src.fpl_api import (
    FPLAPIError,
    clear_cache,
    fetch_bootstrap,
    fetch_fixtures,
    get_current_gameweek,
    get_fixtures_for_gameweek,
    get_player_data,
    get_player_fixtures,
    get_team_data,
)

# ---------------------------------------------------------------------------
# Fixtures (pytest fixtures, not FPL fixtures)
# ---------------------------------------------------------------------------

MOCK_BOOTSTRAP = {
    "events": [
        {"id": 1, "is_current": False, "is_next": False, "finished": True},
        {"id": 2, "is_current": True, "is_next": False, "finished": False},
        {"id": 3, "is_current": False, "is_next": True, "finished": False},
    ],
    "elements": [
        {
            "id": 10,
            "web_name": "Salah",
            "element_type": 3,
            "team": 14,
            "form": "8.5",
            "points_per_game": "7.2",
            "now_cost": 130,
            "ep_next": "6.1",
            "status": "a",
            "chance_of_playing_next_round": 100,
            "total_points": 180,
            "minutes": 2000,
        },
        {
            "id": 20,
            "web_name": "Haaland",
            "element_type": 4,
            "team": 13,
            "form": "9.0",
            "points_per_game": "6.8",
            "now_cost": 145,
            "ep_next": None,
            "status": "a",
            "chance_of_playing_next_round": None,
            "total_points": 160,
            "minutes": 1800,
        },
    ],
    "teams": [
        {
            "id": 14,
            "name": "Liverpool",
            "short_name": "LIV",
            "strength_attack_home": 1300,
            "strength_attack_away": 1280,
            "strength_defence_home": 1250,
            "strength_defence_away": 1230,
            "strength_overall_home": 1290,
            "strength_overall_away": 1270,
        },
    ],
}

MOCK_FIXTURES = [
    {
        "id": 101,
        "event": 2,
        "team_h": 14,
        "team_a": 13,
        "team_h_difficulty": 3,
        "team_a_difficulty": 4,
        "finished": False,
    },
    {
        "id": 102,
        "event": 3,
        "team_h": 1,
        "team_a": 2,
        "team_h_difficulty": 2,
        "team_a_difficulty": 3,
        "finished": False,
    },
]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module-level cache before every test."""
    clear_cache()
    yield
    clear_cache()


def _mock_urlopen(data):
    """Return a context-manager mock that mimics urllib.request.urlopen."""
    resp = mock.MagicMock()
    resp.status = 200
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = mock.MagicMock(return_value=resp)
    resp.__exit__ = mock.MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetCurrentGameweek:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_returns_current_gw(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_BOOTSTRAP)
        assert get_current_gameweek() == 2

    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_falls_back_to_next_gw(self, mock_open):
        bootstrap = {
            "events": [
                {"id": 5, "is_current": False, "is_next": False},
                {"id": 6, "is_current": False, "is_next": True},
            ]
        }
        mock_open.return_value = _mock_urlopen(bootstrap)
        assert get_current_gameweek() == 6


class TestGetPlayerData:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_maps_players_correctly(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_BOOTSTRAP)
        players = get_player_data()

        assert 10 in players
        salah = players[10]
        assert salah["web_name"] == "Salah"
        assert salah["position"] == "MID"
        assert salah["team_id"] == 14
        assert salah["form"] == 8.5
        assert salah["points_per_game"] == 7.2
        assert salah["now_cost"] == 130
        assert salah["ep_next"] == 6.1
        assert salah["status"] == "a"
        assert salah["chance_of_playing"] == 100
        assert salah["total_points"] == 180
        assert salah["minutes"] == 2000

    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_handles_none_ep_next(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_BOOTSTRAP)
        players = get_player_data()
        assert players[20]["ep_next"] is None
        assert players[20]["chance_of_playing"] is None


class TestGetTeamData:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_maps_teams_correctly(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_BOOTSTRAP)
        teams = get_team_data()

        assert 14 in teams
        liv = teams[14]
        assert liv["name"] == "Liverpool"
        assert liv["short_name"] == "LIV"
        assert liv["strength"]["attack_home"] == 1300
        assert liv["strength"]["defence_away"] == 1230
        assert liv["strength"]["overall_home"] == 1290


class TestGetFixturesForGameweek:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_filters_by_gameweek(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_FIXTURES)
        gw2 = get_fixtures_for_gameweek(2)

        assert len(gw2) == 1
        assert gw2[0]["id"] == 101
        assert gw2[0]["home_team_id"] == 14
        assert gw2[0]["away_team_id"] == 13
        assert gw2[0]["home_difficulty"] == 3
        assert gw2[0]["away_difficulty"] == 4
        assert gw2[0]["finished"] is False


class TestGetPlayerFixtures:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_home_team(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_FIXTURES)
        fixtures = get_player_fixtures(player_team_id=14, gameweek=2)

        assert len(fixtures) == 1
        assert fixtures[0]["opponent_team_id"] == 13
        assert fixtures[0]["is_home"] is True
        assert fixtures[0]["difficulty"] == 3

    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_away_team(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_FIXTURES)
        fixtures = get_player_fixtures(player_team_id=13, gameweek=2)

        assert len(fixtures) == 1
        assert fixtures[0]["opponent_team_id"] == 14
        assert fixtures[0]["is_home"] is False
        assert fixtures[0]["difficulty"] == 4

    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_no_fixtures(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_FIXTURES)
        fixtures = get_player_fixtures(player_team_id=99, gameweek=2)
        assert fixtures == []


class TestCache:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_second_call_uses_cache(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_BOOTSTRAP)

        # First call hits API
        fetch_bootstrap()
        assert mock_open.call_count == 1

        # Second call should use cache
        fetch_bootstrap()
        assert mock_open.call_count == 1


class TestConcurrentCacheAccess:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_concurrent_fetch_bootstrap_no_corruption(self, mock_open):
        mock_open.return_value = _mock_urlopen(MOCK_BOOTSTRAP)
        errors: list[Exception] = []

        def call_bootstrap():
            try:
                fetch_bootstrap()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call_bootstrap) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent fetch_bootstrap raised: {errors}"


class TestMalformedPlayerEntry:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_skips_malformed_player(self, mock_open):
        """get_player_data skips entries missing required keys like 'id'."""
        bootstrap = {
            "events": [],
            "elements": [
                {
                    # Missing "id" key — should be skipped
                    "web_name": "Ghost",
                    "element_type": 3,
                    "team": 1,
                    "form": "0.0",
                    "points_per_game": "0.0",
                    "now_cost": 50,
                    "ep_next": None,
                },
                {
                    "id": 99,
                    "web_name": "Valid",
                    "element_type": 4,
                    "team": 2,
                    "form": "5.0",
                    "points_per_game": "4.0",
                    "now_cost": 80,
                    "ep_next": "3.0",
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "total_points": 50,
                    "minutes": 900,
                },
            ],
            "teams": [],
        }
        mock_open.return_value = _mock_urlopen(bootstrap)
        players = get_player_data()
        # Only the valid player should be present
        assert len(players) == 1
        assert 99 in players
        assert players[99]["web_name"] == "Valid"


class TestErrorHandling:
    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_raises_on_url_error(self, mock_open):
        import urllib.error

        mock_open.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(FPLAPIError, match="unreachable"):
            fetch_bootstrap()

    @mock.patch("src.fpl_api.urllib.request.urlopen")
    def test_raises_on_bad_json(self, mock_open):
        resp = mock.MagicMock()
        resp.status = 200
        resp.read.return_value = b"not json"
        resp.__enter__ = mock.MagicMock(return_value=resp)
        resp.__exit__ = mock.MagicMock(return_value=False)
        mock_open.return_value = resp
        with pytest.raises(FPLAPIError, match="Malformed JSON"):
            fetch_bootstrap()
