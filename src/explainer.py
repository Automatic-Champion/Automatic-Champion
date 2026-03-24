from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

from src.team_builder import MODEL_FILENAMES, _build_feature_cols

FEATURE_EXPLANATIONS = {
    "price_now": "Cost: a good fit for the budget while keeping quality high.",
    "total_points": "Past points: shows how productive he was over a full season.",
    "minutes": "Reliability: plays regularly, which increases chances of steady points.",
    "gw_games_played": "Reliability: plays regularly, which increases chances of steady points.",
    "gw_minutes_per_game": "Reliability: regular minutes increase steady points.",
    "gw_saves": "Shot-stopping: more saves can add points for goalkeepers.",
    "saves": "Shot-stopping: more saves can add points for goalkeepers.",
    "clean_sheets": "Clean-sheet upside: helps defenders/goalkeepers score extra points.",
    "cs": "Clean-sheet upside: helps defenders/goalkeepers score extra points.",
    "bps": "Bonus potential: higher BPS often means more bonus points.",
    "bonus": "Bonus potential: higher BPS often means more bonus points.",
    "goals_scored": "Goal threat: more goals usually means more points.",
    "goals": "Goal threat: more goals usually means more points.",
    "assists": "Creative output: assists are a key source of points.",
    "expected_goals": "Expected goals: indicates how often he gets good chances to score.",
    "expected_assists": "Expected assists: indicates chance creation for teammates.",
    "xg": "Expected goals: indicates how often he gets good chances to score.",
    "xa": "Expected assists: indicates chance creation for teammates.",
    "ict_index": "Overall involvement: combines influence, creativity and threat.",
    "yellow_cards": "Discipline: fewer cards reduces point deductions.",
    "red_cards": "Discipline: fewer cards reduces point deductions.",
    "saves_per_90": "Keeper performance: more saves can add extra points.",
    "goals_conceded": "Defence record: affects clean sheets and bonus points.",
}

FEATURE_PRETTY = {
    "gw_saves": "saves",
    "gw_games_played": "games played",
    "total_points": "total points",
    "bps": "bonus points system (BPS)",
    "ict_index": "ICT index",
    "goals_scored": "goals scored",
    "goals_conceded": "goals conceded",
    "yellow_cards": "yellow cards",
    "red_cards": "red cards",
}

POSITION_MODEL_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _pretty_feature_name(feature: str) -> str:
    if feature in FEATURE_PRETTY:
        return FEATURE_PRETTY[feature]
    return feature.replace("_", " ")


def _explain_feature(feature_name: str, value: object) -> str:
    if feature_name == "price_now":
        return FEATURE_EXPLANATIONS["price_now"]

    if feature_name.startswith("1_years_past_"):
        base = feature_name.replace("1_years_past_", "")
        pretty = _pretty_feature_name(base)
        why = FEATURE_EXPLANATIONS.get(base) or FEATURE_EXPLANATIONS.get(pretty)
        if why:
            return f"Last season {pretty}: {value} — {why}"
        return f"Last season {pretty}: {value} — This suggests steady returns over the season."

    key = feature_name.lower()
    why = FEATURE_EXPLANATIONS.get(key)
    if why:
        return why

    return "This stat suggests consistent performance in past seasons."


def explain_selection(
    player_id: str,
    position: str,
    data_path: str,
    models_dir: str,
    top_k: int = 3,
) -> list[dict]:
    """Return top-k feature explanations for why a player was selected.

    Each item: {"feature": str, "value": any, "importance": float, "explanation": str}
    """
    if joblib is None:
        warnings.warn("joblib is required to load models for explanations")
        return []

    model_filename = MODEL_FILENAMES.get(position)
    if model_filename is None:
        return []

    model_path = Path(models_dir) / model_filename
    if not model_path.exists():
        warnings.warn(f"Model file not found: {model_path}")
        return []

    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        return []

    data_file = Path(data_path)
    if not data_file.exists():
        return []

    df = pd.read_csv(data_file)
    row = df[df["id"].astype(str) == str(player_id)]
    if row.empty:
        return []
    row = row.iloc[0]

    # Get feature columns — use model's feature_names_in_ if available, else detect
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        try:
            feature_cols = _build_feature_cols(df)
        except ValueError:
            return []

    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        return []

    # Pair features with importances, sort descending
    feat_imp = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for feature, importance in feat_imp[:top_k]:
        raw_value = row.get(feature)
        # Convert numpy types to native Python for JSON serialization
        if hasattr(raw_value, "item"):
            value = raw_value.item()
        else:
            value = raw_value
        results.append({
            "feature": feature,
            "value": value,
            "importance": float(importance),
            "explanation": _explain_feature(feature, value),
        })
    return results


def explain_squad(
    players: list[dict],
    data_path: str,
    models_dir: str,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Return explanations for each player in the squad.

    Returns dict mapping player_id -> list of feature explanations.
    """
    result: dict[str, list[dict]] = {}
    for player in players:
        pid = str(player["id"])
        position = player["position"]
        result[pid] = explain_selection(
            player_id=pid,
            position=position,
            data_path=data_path,
            models_dir=models_dir,
            top_k=top_k,
        )
    return result
