"""Gameweek points predictor — V8 per-position CatBoost.

Contract
--------
The function ``predict_gameweek_points`` is the public interface.  Any future
replacement (real ML model, ensemble, etc.) **must** keep the same signature::

    predict_gameweek_points(squad: list[dict], gameweek: int | None = None) -> dict[str, float]

Parameters
    squad : list of player dicts, each with at least ``"id"``, ``"name"``,
            ``"position"``, ``"team"``, ``"pred"`` keys (same format as
            ``build_full_squad()`` output).
    gameweek : optional gameweek number.  If *None*, the current gameweek is
               fetched from the live FPL API.

Returns
    dict mapping player ``"id"`` (str) → predicted gameweek points (float).

Implementation
--------------
Loads the V8 per-position CatBoost models from
``Weekly Model/production/models/model_{POS}.cbm`` and reads each squad
player's most recent 2024-25 feature row from ``test.csv``.  If a player has
no row in the feature table (e.g. recent transfer) the predictor falls back
to FPL's ``ep_next``/``points_per_game``/``pred/38`` chain.  If model loading
fails for any reason, the entire predictor degrades to the placeholder chain
so the endpoint stays alive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.fpl_api import FPLAPIError, get_current_gameweek, get_player_data

logger = logging.getLogger(__name__)

PREDICTOR_VERSION = "V8 model"

# ──────────────────────────────────────────────────────────────────────────────
# Paths — anchored to project root so this module can be imported from CLI,
# backend, or tests regardless of CWD.
# ──────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "Weekly Model" / "production" / "models"
_FEATURE_TABLE_PATH = _PROJECT_ROOT / "Weekly Model" / "production" / "test.csv"

POSITIONS = ("GK", "DEF", "MID", "FWD")
CAT_COLS = ["team", "opponent_team"]

# ──────────────────────────────────────────────────────────────────────────────
# Feature definitions — copied verbatim from notebook cell f3bb2ca4
# (Weekly Model/production/weeklyModels_Production.ipynb). Must match training
# exactly to avoid serving skew.
# ──────────────────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "was_home", "team", "opponent_team",
    "minutes", "starts",
    "value", "selected", "transfers_balance",
    "ict_index", "influence", "creativity", "threat",
    "bps", "bonus",
    "xP",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "team_goals_scored", "team_goals_conceded",
]

ROLL_BASE = [
    "total_points_last",  "total_points_mean3",  "total_points_mean5",  "total_points_mean7",
    "minutes_last",       "minutes_mean3",        "minutes_mean5",       "minutes_mean7",
    "starts_last",        "starts_mean3",         "starts_mean5",
    "ict_index_last",     "ict_index_mean3",      "ict_index_mean5",     "ict_index_mean7",
    "influence_last",     "influence_mean3",      "influence_mean5",
    "creativity_last",    "creativity_mean3",     "creativity_mean5",
    "threat_last",        "threat_mean3",         "threat_mean5",
    "bps_last",           "bps_mean3",            "bps_mean5",
    "bonus_last",         "bonus_mean3",          "bonus_mean5",
    "value_last",         "value_mean3",
    "selected_last",      "selected_mean3",
    "transfers_balance_last", "transfers_balance_mean3",
    "xP_last",            "xP_mean3",             "xP_mean5",
    "expected_goals_last",             "expected_goals_mean3",             "expected_goals_mean5",
    "expected_assists_last",           "expected_assists_mean3",           "expected_assists_mean5",
    "expected_goal_involvements_last", "expected_goal_involvements_mean3", "expected_goal_involvements_mean5",
    "expected_goals_conceded_last",    "expected_goals_conceded_mean3",    "expected_goals_conceded_mean5",
    "team_goals_scored_last",   "team_goals_scored_mean3",   "team_goals_scored_mean5",
    "team_goals_conceded_last", "team_goals_conceded_mean3", "team_goals_conceded_mean5",
]

POS_EXTRAS: dict[str, list[str]] = {
    "GK": [
        "saves", "saves_last", "saves_mean3", "saves_mean5", "saves_mean7",
        "clean_sheets", "clean_sheets_last", "clean_sheets_mean3", "clean_sheets_mean5",
        "goals_conceded", "goals_conceded_last", "goals_conceded_mean3", "goals_conceded_mean5",
    ],
    "DEF": [
        "clean_sheets", "clean_sheets_last", "clean_sheets_mean3", "clean_sheets_mean5",
        "goals_conceded", "goals_conceded_last", "goals_conceded_mean3", "goals_conceded_mean5",
        "goals_scored", "goals_scored_last", "goals_scored_mean3",
        "assists", "assists_last", "assists_mean3",
    ],
    "MID": [
        "goals_scored", "goals_scored_last", "goals_scored_mean3", "goals_scored_mean5",
        "assists", "assists_last", "assists_mean3", "assists_mean5",
    ],
    "FWD": [
        "goals_scored", "goals_scored_last", "goals_scored_mean3", "goals_scored_mean5",
        "assists", "assists_last", "assists_mean3", "assists_mean5",
    ],
}

CLIP_RANGE: dict[str, tuple[float, float]] = {
    "GK":  (-4.0, 20.0),
    "DEF": (-4.0, 25.0),
    "MID": (-4.0, 25.0),
    "FWD": (-4.0, 25.0),
}


def _features_for(pos: str, available_cols: set[str]) -> list[str]:
    """Return the feature subset for *pos*, filtered to columns present in the
    feature table, with order preserved and duplicates removed."""
    candidates = BASE_FEATURES + ROLL_BASE + POS_EXTRAS.get(pos, [])
    seen: set[str] = set()
    out: list[str] = []
    for f in candidates:
        if f in available_cols and f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Lazy loaders — models and feature table are loaded once per process and
# cached in module-level globals. Loading failures degrade to placeholder.
# ──────────────────────────────────────────────────────────────────────────────
_models_cache: Optional[dict] = None
_models_load_failed = False
_feature_table_cache: Optional[pd.DataFrame] = None
_feature_table_load_failed = False


def _load_models() -> Optional[dict]:
    """Load all 4 per-position CatBoost models. Returns None if any fail."""
    global _models_cache, _models_load_failed
    if _models_cache is not None:
        return _models_cache
    if _models_load_failed:
        return None
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("catboost not installed — falling back to placeholder predictor")
        _models_load_failed = True
        return None

    models: dict = {}
    for pos in POSITIONS:
        path = _MODELS_DIR / f"model_{pos}.cbm"
        if not path.exists():
            logger.warning("Model file missing: %s — falling back to placeholder", path)
            _models_load_failed = True
            return None
        try:
            m = CatBoostRegressor()
            m.load_model(str(path))
            models[pos] = m
        except Exception as exc:
            logger.warning("Failed to load %s: %s — falling back to placeholder", path, exc)
            _models_load_failed = True
            return None

    _models_cache = models
    logger.info("V8 CatBoost models loaded from %s", _MODELS_DIR)
    return _models_cache


def _load_feature_table() -> Optional[pd.DataFrame]:
    """Load the 2024-25 frozen feature table indexed by element (FPL ID).
    Casts categorical columns to str (CatBoost requirement) and drops the label
    column. Returns None if the file can't be read."""
    global _feature_table_cache, _feature_table_load_failed
    if _feature_table_cache is not None:
        return _feature_table_cache
    if _feature_table_load_failed:
        return None
    if not _FEATURE_TABLE_PATH.exists():
        logger.warning("Feature table missing: %s — falling back to placeholder", _FEATURE_TABLE_PATH)
        _feature_table_load_failed = True
        return None
    try:
        df = pd.read_csv(_FEATURE_TABLE_PATH)
    except Exception as exc:
        logger.warning("Failed to read %s: %s — falling back to placeholder", _FEATURE_TABLE_PATH, exc)
        _feature_table_load_failed = True
        return None

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "future_points" in df.columns:
        df = df.drop(columns=["future_points"])
    df = df.sort_values("GW").reset_index(drop=True)
    _feature_table_cache = df
    return _feature_table_cache


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def predict_gameweek_points(
    squad: list[dict],
    gameweek: int | None = None,
) -> dict[str, float]:
    """Predict per-player gameweek points for every player in *squad*.

    For each player: look up the most recent row in the V8 feature table by
    FPL element ID, run the position's CatBoost model, clip to the position's
    range. Players with no row in the feature table fall back to the placeholder
    chain (ep_next → points_per_game → pred/38). Total model+feature-table
    loading failure also degrades to the placeholder chain.
    """
    models = _load_models()
    feature_table = _load_feature_table()

    # Build the per-element index for the requested gameweek.
    #
    # Row semantics: a row at GW=N holds the features the model saw going into
    # GW=N, and its label was the player's score in GW=N+1. So to predict
    # gameweek G, we want each player's row at GW = G - 1. If that exact row
    # is missing, we take the latest available row with GW < G. Falling back
    # to the latest row overall is reserved for GW=1 (or when no prior row
    # exists at all).
    rows_by_element: dict[int, pd.Series] = {}
    if models is not None and feature_table is not None and "element" in feature_table.columns:
        if gameweek is None or gameweek <= 1:
            scoped = feature_table
        else:
            scoped = feature_table[feature_table["GW"] < int(gameweek)]
            if scoped.empty:
                scoped = feature_table
        # The DataFrame is sorted ascending by GW, so drop_duplicates(keep='last')
        # leaves one row per element — the latest row within the scoped window.
        latest = scoped.drop_duplicates(subset="element", keep="last")
        rows_by_element = {int(r["element"]): r for _, r in latest.iterrows()}

    predictions: dict[str, float] = {}
    fallback_players: list[dict] = []

    for player in squad:
        pid = str(player["id"])
        pos = player.get("position", "")

        used_model = False
        if models is not None and pos in models and rows_by_element:
            try:
                player_int_id = int(player["id"])
            except (TypeError, ValueError):
                player_int_id = None

            row = rows_by_element.get(player_int_id) if player_int_id is not None else None
            if row is not None:
                feats = _features_for(pos, set(feature_table.columns))
                try:
                    from catboost import Pool
                    X = row[feats].to_frame().T.reset_index(drop=True)
                    for c in CAT_COLS:
                        if c in X.columns:
                            X[c] = X[c].astype(str)
                    cat_in_X = [c for c in CAT_COLS if c in X.columns]
                    pool = Pool(X, cat_features=cat_in_X)
                    raw = float(models[pos].predict(pool)[0])
                    lo, hi = CLIP_RANGE[pos]
                    predictions[pid] = float(np.clip(raw, lo, hi))
                    used_model = True
                except Exception as exc:
                    logger.warning(
                        "V8 predict failed for player %s (%s): %s — using placeholder",
                        pid, player.get("name", "?"), exc,
                    )

        if not used_model:
            fallback_players.append(player)

    # Lazy FPL fetch — only call the FPL API when at least one player needs
    # the placeholder chain. When V8 handles every player (the normal case),
    # this skips two network round-trips and stays resilient to FPL outages.
    if fallback_players:
        fpl_players, fpl_by_name, fpl_list = _fetch_fpl_for_placeholder(gameweek)
        for player in fallback_players:
            pid = str(player["id"])
            predictions[pid] = _placeholder_predict(player, fpl_players, fpl_by_name, fpl_list)

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Placeholder fallback (the original ep_next → ppg → pred/38 chain).
# Used when V8 models are unavailable, the player has no feature row, or
# inference fails for any reason.
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_fpl_for_placeholder(
    gameweek: int | None,
) -> tuple[dict[int, dict] | None, dict[str, list[dict]], list[dict]]:
    """Fetch FPL bootstrap data for the placeholder chain. All failures degrade
    to (None, {}, [])."""
    fpl_players: dict[int, dict] | None = None
    try:
        if gameweek is None:
            gameweek = get_current_gameweek()
        fpl_players = get_player_data()
    except (FPLAPIError, ConnectionError, TimeoutError, OSError, ValueError):
        logger.warning("FPL API unavailable — falling back to season predictions")
        fpl_players = None

    fpl_by_name: dict[str, list[dict]] = {}
    fpl_list: list[dict] = []
    if fpl_players is not None:
        fpl_list = list(fpl_players.values())
        for fp in fpl_list:
            key = fp["web_name"].lower()
            fpl_by_name.setdefault(key, []).append(fp)
    return fpl_players, fpl_by_name, fpl_list


def _placeholder_predict(
    player: dict,
    fpl_players: dict[int, dict] | None,
    fpl_by_name: dict[str, list[dict]],
    fpl_list: list[dict],
) -> float:
    """Original placeholder logic: ep_next → points_per_game → pred / 38."""
    fallback = player["pred"] / 38.0
    if fpl_players is None:
        return fallback
    matched = _match_fpl_player(player, fpl_by_name, fpl_list)
    if matched is None:
        return fallback
    ep = matched.get("ep_next")
    if ep is not None:
        return float(ep)
    if matched.get("points_per_game", 0.0) > 0.0:
        return float(matched["points_per_game"])
    return fallback


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents-ish chars, remove dots/hyphens for fuzzy compare."""
    return name.lower().replace(".", " ").replace("-", " ").strip()


def _word_boundary_match(needle: str, haystack: str) -> bool:
    """Check if needle appears in haystack as a whole word (not mid-word)."""
    haystack_words = haystack.replace("-", " ").split()
    needle_words = needle.replace("-", " ").split()
    if len(needle_words) > 1:
        return all(nw in haystack_words for nw in needle_words)
    return needle in haystack_words


def _match_fpl_player(
    squad_player: dict,
    fpl_by_name: dict[str, list[dict]],
    fpl_list: list[dict],
) -> dict | None:
    """Match a squad player to an FPL API player by name.

    Matching strategies (tried in order):
    1. FPL web_name matches a whole word in the squad player name (case-insensitive).
    2. Squad player's last name is a substring of FPL second_name.
    3. Squad player name is a substring of FPL full name (first_name + second_name).
    4. Normalized token overlap — at least 2 shared tokens and same position.

    If multiple candidates match, prefer same-position matches.
    """
    squad_name_lower = squad_player["name"].lower()
    squad_name_norm = _normalize_name(squad_player["name"])
    squad_position = squad_player.get("position", "")

    candidates: list[dict] = []
    for web_name_lower, fps in fpl_by_name.items():
        if _word_boundary_match(web_name_lower, squad_name_lower):
            candidates.extend(fps)

    if candidates:
        return _pick_best(candidates, squad_position)

    squad_parts = squad_name_lower.split()
    squad_last = squad_parts[-1] if squad_parts else ""

    for fp in fpl_list:
        fpl_second = fp.get("second_name", "").lower()
        fpl_full = f"{fp.get('first_name', '')} {fp.get('second_name', '')}".lower().strip()

        if squad_last and len(squad_last) > 2 and squad_last in fpl_second:
            candidates.append(fp)
            continue

        if squad_name_lower in fpl_full or fpl_full in squad_name_lower:
            candidates.append(fp)
            continue

    if candidates:
        return _pick_best(candidates, squad_position)

    squad_tokens = set(squad_name_norm.split())
    for fp in fpl_list:
        fpl_full_norm = _normalize_name(
            f"{fp.get('first_name', '')} {fp.get('second_name', '')} {fp.get('web_name', '')}"
        )
        fpl_tokens = set(fpl_full_norm.split())
        shared = squad_tokens & fpl_tokens
        if len(shared) >= 2 and fp.get("position") == squad_position:
            candidates.append(fp)

    if candidates:
        return _pick_best(candidates, squad_position)

    return None


def _pick_best(candidates: list[dict], position: str) -> dict:
    """From a list of FPL candidate matches, prefer same-position."""
    for c in candidates:
        if c.get("position") == position:
            return c
    return candidates[0]
