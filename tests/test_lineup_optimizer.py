from __future__ import annotations

import pytest

from src.lineup_optimizer import optimize_lineup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_squad() -> list[dict]:
    """A valid 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD."""
    players = [
        # GK (2)
        {"id": "g1", "name": "Goalkeeper A", "position": "GK", "team": "Team1", "cost": 5.0, "pred": 100.0},
        {"id": "g2", "name": "Goalkeeper B", "position": "GK", "team": "Team2", "cost": 4.5, "pred": 80.0},
        # DEF (5)
        {"id": "d1", "name": "Defender A", "position": "DEF", "team": "Team1", "cost": 6.0, "pred": 130.0},
        {"id": "d2", "name": "Defender B", "position": "DEF", "team": "Team2", "cost": 5.5, "pred": 120.0},
        {"id": "d3", "name": "Defender C", "position": "DEF", "team": "Team3", "cost": 5.0, "pred": 110.0},
        {"id": "d4", "name": "Defender D", "position": "DEF", "team": "Team4", "cost": 4.5, "pred": 100.0},
        {"id": "d5", "name": "Defender E", "position": "DEF", "team": "Team5", "cost": 4.0, "pred": 90.0},
        # MID (5)
        {"id": "m1", "name": "Midfielder A", "position": "MID", "team": "Team1", "cost": 8.0, "pred": 180.0},
        {"id": "m2", "name": "Midfielder B", "position": "MID", "team": "Team2", "cost": 7.0, "pred": 160.0},
        {"id": "m3", "name": "Midfielder C", "position": "MID", "team": "Team3", "cost": 6.5, "pred": 150.0},
        {"id": "m4", "name": "Midfielder D", "position": "MID", "team": "Team4", "cost": 6.0, "pred": 140.0},
        {"id": "m5", "name": "Midfielder E", "position": "MID", "team": "Team5", "cost": 5.5, "pred": 120.0},
        # FWD (3)
        {"id": "f1", "name": "Forward A", "position": "FWD", "team": "Team1", "cost": 10.0, "pred": 200.0},
        {"id": "f2", "name": "Forward B", "position": "FWD", "team": "Team2", "cost": 8.0, "pred": 170.0},
        {"id": "f3", "name": "Forward C", "position": "FWD", "team": "Team3", "cost": 6.0, "pred": 130.0},
    ]
    return players


def _make_predictions(squad: list[dict], overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Create GW predictions from squad — uses pred / 38 plus optional overrides."""
    preds = {str(p["id"]): p["pred"] / 38.0 for p in squad}
    if overrides:
        preds.update(overrides)
    return preds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicLineup:
    def test_fixed_formation(self):
        squad = _make_squad()
        preds = _make_predictions(squad)
        result = optimize_lineup(squad, preds, formation="4-3-3")

        assert result["formation"] == "4-3-3"
        assert len(result["starters"]) == 11
        assert len(result["bench"]) == 4

        # Check position counts
        pos_counts = {}
        for s in result["starters"]:
            pos_counts[s["position"]] = pos_counts.get(s["position"], 0) + 1
        assert pos_counts == {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}

    def test_captain_is_highest_predicted(self):
        squad = _make_squad()
        # Give Forward A the highest GW prediction
        preds = _make_predictions(squad, overrides={"f1": 15.0})
        result = optimize_lineup(squad, preds, formation="4-3-3")

        assert result["captain_id"] == "f1"
        captain_in_starters = [s for s in result["starters"] if s["is_captain"]]
        assert len(captain_in_starters) == 1
        assert captain_in_starters[0]["id"] == "f1"

    def test_vice_captain_is_second_highest(self):
        squad = _make_squad()
        preds = _make_predictions(squad, overrides={"f1": 15.0, "m1": 12.0})
        result = optimize_lineup(squad, preds, formation="4-3-3")

        assert result["captain_id"] == "f1"
        assert result["vice_captain_id"] == "m1"

    def test_total_includes_captain_bonus(self):
        squad = _make_squad()
        preds = _make_predictions(squad, overrides={"f1": 15.0})
        result = optimize_lineup(squad, preds, formation="4-3-3")

        starter_pts = sum(s["gw_points"] for s in result["starters"])
        captain_pts = next(s["gw_points"] for s in result["starters"] if s["is_captain"])
        assert result["total_gw_points"] == pytest.approx(starter_pts + captain_pts)


class TestAutoFormation:
    def test_picks_best_formation(self):
        squad = _make_squad()
        # Give midfielders very high scores to favour 3-5-2
        preds = _make_predictions(squad, overrides={
            "m1": 15.0, "m2": 14.0, "m3": 13.0, "m4": 12.0, "m5": 11.0,
        })
        result = optimize_lineup(squad, preds, formation=None)

        # With 5 high-scoring midfielders, 3-5-2 should win
        assert result["formation"] == "3-5-2"
        assert len(result["starters"]) == 11

    def test_auto_returns_valid_result(self):
        squad = _make_squad()
        preds = _make_predictions(squad)
        result = optimize_lineup(squad, preds, formation=None)

        assert "formation" in result
        assert result["formation"] in (
            "4-3-3", "4-4-2", "3-4-3", "3-5-2", "4-5-1", "5-3-2", "5-4-1",
        )
        assert len(result["starters"]) == 11
        assert len(result["bench"]) == 4


class TestBenchOrder:
    def test_bench_gk_always_last(self):
        squad = _make_squad()
        # Give the bench GK very high prediction — should still be last
        preds = _make_predictions(squad, overrides={"g2": 50.0})
        result = optimize_lineup(squad, preds, formation="4-3-3")

        bench = result["bench"]
        assert len(bench) == 4
        # Bench position 4 should be the GK
        last_bench = bench[-1]
        assert last_bench["bench_order"] == 4
        assert last_bench["position"] == "GK"

    def test_bench_outfield_sorted_by_points_desc(self):
        squad = _make_squad()
        preds = _make_predictions(squad)
        result = optimize_lineup(squad, preds, formation="4-3-3")

        bench = result["bench"]
        outfield_bench = [b for b in bench if b["position"] != "GK"]
        pts = [b["gw_points"] for b in outfield_bench]
        assert pts == sorted(pts, reverse=True)


class TestValidation:
    def test_wrong_squad_size(self):
        squad = _make_squad()[:10]
        preds = _make_predictions(squad)
        with pytest.raises(ValueError, match="exactly 15"):
            optimize_lineup(squad, preds)

    def test_wrong_position_counts(self):
        squad = _make_squad()
        # Replace a FWD with an extra GK
        squad[-1] = {
            "id": "g3", "name": "Goalkeeper C", "position": "GK",
            "team": "Team6", "cost": 4.0, "pred": 70.0,
        }
        preds = _make_predictions(squad)
        with pytest.raises(ValueError, match="GK|FWD"):
            optimize_lineup(squad, preds)

    def test_invalid_formation_string(self):
        squad = _make_squad()
        preds = _make_predictions(squad)
        with pytest.raises(ValueError, match="Unsupported formation"):
            optimize_lineup(squad, preds, formation="6-0-4")
