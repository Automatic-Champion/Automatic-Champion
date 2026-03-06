from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

POSITION_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTER_REQUIREMENTS = {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}
TEAM_PLAYER_LIMIT = 3

POSITION_ALIASES = {
    "GK": "GK",
    "GKP": "GK",
    "GOALKEEPER": "GK",
    "DEF": "DEF",
    "D": "DEF",
    "DEFENDER": "DEF",
    "MID": "MID",
    "M": "MID",
    "MIDFIELDER": "MID",
    "FWD": "FWD",
    "FW": "FWD",
    "ST": "FWD",
    "STRIKER": "FWD",
    "FORWARD": "FWD",
}

MODEL_FILES = {
    "GK": "advanced_model_GK.joblib",
    "DEF": "advanced_model_DEF.joblib",
    "MID": "advanced_model_MID.joblib",
    "FWD": "advanced_model_FWD.joblib",
}


class OptimizationError(ValueError):
    """Raised when the heuristic cannot build or adjust a valid squad."""


def _normalize_position(position: str) -> str:
    normalized = POSITION_ALIASES.get(position.upper().strip())
    if normalized is None:
        raise OptimizationError(f"Unsupported position value: {position}")
    return normalized


def _load_models(models_dir: str) -> dict[str, Any]:
    base_path = Path(models_dir)
    models: dict[str, Any] = {}
    for position, filename in MODEL_FILES.items():
        model_path = base_path / filename
        if not model_path.exists():
            raise OptimizationError(f"Model file not found: {model_path}")
        models[position] = joblib.load(model_path)
    return models


def _predict_points(model: Any, player_row: dict[str, Any]) -> float:
    features_source = dict(player_row.get("features", {}))
    for key, value in player_row.items():
        if key not in {"player_id", "name", "position", "price", "team_id", "features"}:
            features_source[key] = value

    if hasattr(model, "feature_names_in_"):
        feature_names = [str(col) for col in model.feature_names_in_]
        row = {feature: features_source.get(feature, 0.0) for feature in feature_names}
        frame = pd.DataFrame([row], columns=feature_names)
    else:
        frame = pd.DataFrame([features_source])

    prediction = model.predict(frame)
    return float(prediction[0])


def _build_initial_squad(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    team_counts: defaultdict[int, int] = defaultdict(int)
    remaining = POSITION_REQUIREMENTS.copy()

    for candidate in sorted(candidates, key=lambda item: item["predicted_points"], reverse=True):
        if remaining[candidate["position"]] == 0:
            continue
        if team_counts[candidate["team_id"]] >= TEAM_PLAYER_LIMIT:
            continue

        selected.append(candidate)
        selected_ids.add(candidate["player_id"])
        team_counts[candidate["team_id"]] += 1
        remaining[candidate["position"]] -= 1

        if all(count == 0 for count in remaining.values()):
            return selected

    raise OptimizationError("Unable to build an initial valid 15-player squad with given constraints.")


def _find_best_swap(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected_ids: set[int],
) -> tuple[int, dict[str, Any]] | None:
    team_counts: defaultdict[int, int] = defaultdict(int)
    for player in selected:
        team_counts[player["team_id"]] += 1

    best_swap: tuple[int, dict[str, Any]] | None = None
    best_score = float("inf")

    for out_index, out_player in enumerate(selected):
        for in_player in candidates:
            if in_player["player_id"] in selected_ids:
                continue
            if in_player["position"] != out_player["position"]:
                continue
            if in_player["price"] >= out_player["price"]:
                continue

            new_in_team_count = team_counts[in_player["team_id"]] + 1
            if in_player["team_id"] == out_player["team_id"]:
                new_in_team_count -= 1
            if new_in_team_count > TEAM_PLAYER_LIMIT:
                continue

            price_saved = out_player["price"] - in_player["price"]
            point_loss = out_player["predicted_points"] - in_player["predicted_points"]
            score = point_loss / price_saved

            if score < best_score:
                best_score = score
                best_swap = (out_index, in_player)

    return best_swap


def _set_starters_and_bench(selected: list[dict[str, Any]]) -> None:
    for player in selected:
        player["is_starter"] = False
        player["bench_order"] = None
        player["selected_position"] = player["position"]

    bench: list[dict[str, Any]] = []
    for position, starter_count in STARTER_REQUIREMENTS.items():
        by_position = [p for p in selected if p["position"] == position]
        by_position.sort(key=lambda item: item["predicted_points"], reverse=True)

        for player in by_position[:starter_count]:
            player["is_starter"] = True

        bench.extend(by_position[starter_count:])

    bench.sort(key=lambda item: item["predicted_points"], reverse=True)
    for index, player in enumerate(bench, start=1):
        player["bench_order"] = index


def generate_optimal_squad(budget: float, players_data: list, models_dir: str) -> dict[str, Any]:
    models = _load_models(models_dir)

    candidates: list[dict[str, Any]] = []
    for raw_player in players_data:
        player = raw_player if isinstance(raw_player, dict) else raw_player.model_dump()
        position = _normalize_position(player["position"])
        model = models[position]

        candidate = {
            "player_id": int(player["player_id"]),
            "name": str(player["name"]),
            "position": position,
            "price": float(player["price"]),
            "team_id": int(player["team_id"]),
            "predicted_points": _predict_points(model, player),
        }
        candidates.append(candidate)

    initial_pool_counts: defaultdict[str, int] = defaultdict(int)
    for player in candidates:
        initial_pool_counts[player["position"]] += 1
    for position, required in POSITION_REQUIREMENTS.items():
        if initial_pool_counts[position] < required:
            raise OptimizationError(
                f"Not enough players in position {position}. Required {required}, got {initial_pool_counts[position]}."
            )

    selected = _build_initial_squad(candidates)
    selected_ids = {player["player_id"] for player in selected}
    total_price = sum(player["price"] for player in selected)

    while total_price > budget:
        swap = _find_best_swap(selected, candidates, selected_ids)
        if swap is None:
            raise OptimizationError(
                "Squad is over budget and no valid cheaper same-position swap is available."
            )

        out_index, in_player = swap
        out_player = selected[out_index]

        selected_ids.remove(out_player["player_id"])
        selected_ids.add(in_player["player_id"])
        selected[out_index] = in_player

        total_price = sum(player["price"] for player in selected)

    _set_starters_and_bench(selected)

    return {
        "selected_players": sorted(
            selected,
            key=lambda item: (item["position"], not item["is_starter"], -item["predicted_points"]),
        ),
        "total_budget_used": total_price,
        "total_predicted_points": sum(player["predicted_points"] for player in selected),
    }
