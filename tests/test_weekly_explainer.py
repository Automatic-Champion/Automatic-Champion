from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.weekly_explainer import GENERIC_FALLBACK, VALID_CATEGORIES, explain_weekly_squad

_MODELS_DIR = Path(__file__).resolve().parent.parent / "Weekly Model" / "production" / "models"
_HAVE_MODELS = all((_MODELS_DIR / f"model_{p}.cbm").exists() for p in ("GK", "DEF", "MID", "FWD"))

# Words that signal a genuinely *weekly* (form/fixture) explanation rather than
# a season-long one.
_WEEKLY_WORDS = (
    "gameweek", "form", "fixture", "xg", "xa", "expected", "threat", "ict",
    "minutes", "clean sheet", "goals", "assists", "influence", "bonus", "bps",
    "pts", "home", "away", "saves",
)

# Substrings that must never leak into user-facing text.
_RAW_LEAKS = ("_mean3", "_mean5", "_mean7", "_last", "1_years_past", "element_type", "catboost")


def _salah() -> dict:
    return {"id": "328", "name": "Mohamed Salah", "position": "MID", "team": "Liverpool", "pred": 380.0}


def _mixed_squad() -> list[dict]:
    return [
        {"id": "1", "name": "Keeper", "position": "GK", "team": "Arsenal", "pred": 130.0},
        {"id": "2", "name": "Back", "position": "DEF", "team": "Chelsea", "pred": 150.0},
        {"id": "3", "name": "Mid", "position": "MID", "team": "Liverpool", "pred": 200.0},
        {"id": "4", "name": "Striker", "position": "FWD", "team": "Man City", "pred": 210.0},
    ]


def _assert_well_formed(reasons: list[dict], top_k: int = 3) -> None:
    assert isinstance(reasons, list)
    assert 1 <= len(reasons) <= top_k
    for item in reasons:
        assert set(item.keys()) >= {"text", "category"}
        assert isinstance(item["text"], str) and item["text"].strip()
        assert item["category"] in VALID_CATEGORIES
        low = item["text"].lower()
        assert "last season" not in low
        for leak in _RAW_LEAKS:
            assert leak not in low


# ──────────────────────────────────────────────────────────────────────────────
# Fallback behaviour (no real models needed — deterministic)
# ──────────────────────────────────────────────────────────────────────────────
class TestFallback:
    @patch("src.weekly_explainer.get_feature_rows", return_value={})
    def test_returns_entry_per_player(self, _no_rows):
        squad = _mixed_squad()
        result = explain_weekly_squad(squad, gameweek=10)
        assert set(result.keys()) == {str(p["id"]) for p in squad}
        for reasons in result.values():
            _assert_well_formed(reasons)

    @patch("src.weekly_explainer.get_feature_rows", return_value={})
    def test_models_missing_gives_generic(self, _no_rows):
        result = explain_weekly_squad([_salah()], gameweek=10)
        assert result["328"] == [dict(GENERIC_FALLBACK)]

    @patch("src.weekly_explainer.get_feature_rows", return_value={})
    def test_top_k_respected(self, _no_rows):
        result = explain_weekly_squad(_mixed_squad(), gameweek=10, top_k=2)
        for reasons in result.values():
            assert len(reasons) <= 2

    def test_unknown_player_generic(self):
        # Element id not in the feature table -> no row -> generic fallback.
        squad = [{"id": "999999", "name": "Nobody", "position": "MID", "team": "Nowhere", "pred": 76.0}]
        result = explain_weekly_squad(squad, gameweek=10)
        assert result["999999"] == [dict(GENERIC_FALLBACK)]


# ──────────────────────────────────────────────────────────────────────────────
# Real V8 models (skipped if model files are absent)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _HAVE_MODELS, reason="V8 model files not present")
class TestV8RealModel:
    def test_salah_reasons_are_weekly(self):
        result = explain_weekly_squad([_salah()], gameweek=10)
        reasons = result["328"]
        _assert_well_formed(reasons)
        joined = " ".join(r["text"].lower() for r in reasons)
        assert any(w in joined for w in _WEEKLY_WORDS), f"no weekly concept in: {joined!r}"

    def test_fixture_line_present_and_truthful(self):
        reasons = explain_weekly_squad([_salah()], gameweek=10)["328"]
        fixtures = [r for r in reasons if r["category"] == "fixture"]
        assert fixtures, "expected a fixture reason for Salah at GW10"
        text = fixtures[0]["text"]
        assert text.startswith(("Home", "Away"))
        assert "against" in text
        # Must not claim a (possibly wrong) upcoming gameweek number.
        assert "GW10" not in text

    def test_different_gameweeks_compute(self):
        early = explain_weekly_squad([_salah()], gameweek=5)["328"]
        late = explain_weekly_squad([_salah()], gameweek=35)["328"]
        _assert_well_formed(early)
        _assert_well_formed(late)

    def test_full_squad_positions(self):
        # One real player per position (well-known 2024-25 element ids).
        squad = [
            {"id": "328", "name": "Mohamed Salah", "position": "MID", "team": "Liverpool", "pred": 380.0},
            {"id": "351", "name": "Erling Haaland", "position": "FWD", "team": "Man City", "pred": 360.0},
        ]
        result = explain_weekly_squad(squad, gameweek=10)
        assert set(result.keys()) == {"328", "351"}
        for reasons in result.values():
            _assert_well_formed(reasons)
