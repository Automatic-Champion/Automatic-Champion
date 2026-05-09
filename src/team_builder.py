from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

try:
    from ortools.linear_solver import pywraplp
except ImportError:  # pragma: no cover - optional dependency
    pywraplp = None


POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
FORMATION_COUNTS = {
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}
FULL_SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MODEL_FILENAMES = {
    "GK": "position_model_1.joblib",
    "DEF": "position_model_2.joblib",
    "MID": "position_model_3.joblib",
    "FWD": "position_model_4.joblib",
}


@dataclass(frozen=True)
class PlayerRecord:
    idx: int
    player_id: str
    name: str
    team: str
    position: str
    cost_int: int
    cost: float
    pred: float


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _build_feature_cols(df: pd.DataFrame) -> list[str]:
    if "price_now" not in df.columns:
        raise ValueError("Missing required column: price_now")

    numeric_candidates = [
        col
        for col in df.columns
        if any(col.startswith(f"{y}_years_past_") for y in (1, 2, 3))
        or col.startswith("momentum_")
    ]
    for col in numeric_candidates + ["price_now"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df[["price_now"] + numeric_candidates].select_dtypes(
        include="number"
    ).columns.tolist()
    if "price_now" not in numeric_cols:
        raise ValueError("price_now must be numeric")

    feature_cols = ["price_now"] + [
        col for col in numeric_cols
        if col != "price_now"
        and "element_type" not in col
        and (any(col.startswith(f"{y}_years_past_") for y in (1, 2, 3))
             or col.startswith("momentum_"))
    ]
    if len(feature_cols) == 1:
        raise ValueError("No numeric lag or momentum feature columns found")
    return feature_cols


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum (year-over-year delta) features.

    momentum_X = 1_years_past_X - 2_years_past_X
    If either operand is NaN, the result is NaN (filled to 0 later by fillna).
    """
    pairs = [
        ("momentum_total_points", "1_years_past_total_points", "2_years_past_total_points"),
        ("momentum_minutes", "1_years_past_minutes", "2_years_past_minutes"),
        ("momentum_ict_index", "1_years_past_ict_index", "2_years_past_ict_index"),
        ("momentum_goals_scored", "1_years_past_goals_scored", "2_years_past_goals_scored"),
    ]
    for new_col, col_1y, col_2y in pairs:
        if col_1y in df.columns and col_2y in df.columns:
            s1 = pd.to_numeric(df[col_1y], errors="coerce")
            s2 = pd.to_numeric(df[col_2y], errors="coerce")
            df[new_col] = s1 - s2
        else:
            df[new_col] = float("nan")
    return df


def _map_position(series: pd.Series) -> pd.Series:
    def _to_pos(value: object) -> str | None:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return POSITION_MAP.get(number)

    return series.apply(_to_pos)


def _validate_and_filter_element_type(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list]:
    df = df.copy()
    df["element_type"] = pd.to_numeric(df["element_type"], errors="coerce")
    valid_values = set(POSITION_MAP.keys())
    valid_mask = df["element_type"].isin(valid_values)
    invalid_values = sorted(df.loc[~valid_mask, "element_type"].unique().tolist())
    dropped = (~valid_mask).sum()
    return df[valid_mask], int(dropped), invalid_values


def _load_models(models_dir: str) -> dict[str, object]:
    if joblib is None:
        raise ImportError("joblib is required to load models")

    models_path = Path(models_dir)
    models: dict[str, object] = {}
    for position, filename in MODEL_FILENAMES.items():
        path = models_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")
        models[position] = joblib.load(path)
    return models


_SELECTED_FEATURES_PATH = Path(__file__).resolve().parent.parent / "training" / "selected_features.json"


def _load_selected_features(
    path: Path | str | None = None,
) -> dict[str, list[str]] | None:
    """Load per-position selected feature lists from JSON.

    Returns a dict mapping position code string ("1", "2", "3", "4") to
    a list of feature column names.  Returns None if the file doesn't exist.
    """
    if path is None:
        path = _SELECTED_FEATURES_PATH
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _predict(df: pd.DataFrame, feature_cols: list[str], models_dir: str) -> pd.Series:
    models = _load_models(models_dir)
    selected = _load_selected_features()
    preds = pd.Series(index=df.index, dtype="float64")

    pos_to_code = {"GK": "1", "DEF": "2", "MID": "3", "FWD": "4"}

    for position, model in models.items():
        mask = df["position"] == position

        # Use selected features for this position if available
        pos_code = pos_to_code.get(position)
        if selected and pos_code and pos_code in selected:
            pos_features = [f for f in selected[pos_code] if f in df.columns]
        else:
            pos_features = feature_cols

        subset = df.loc[mask, pos_features].fillna(0)
        if subset.empty:
            continue
        if hasattr(model, "predict"):
            pred_values = model.predict(subset)
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(subset)
            if proba.shape[1] < 2:
                raise ValueError(f"predict_proba returned unexpected shape for {position}")
            pred_values = proba[:, 1]
        else:
            raise ValueError(f"Model for {position} has no predict method")

        preds.loc[mask] = pd.to_numeric(pred_values, errors="coerce")

    if preds.isna().any():
        raise ValueError("Prediction produced NaN values")
    return preds


def _build_solver() -> pywraplp.Solver:
    if pywraplp is None:
        raise ImportError("ortools is required for ILP optimization")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No suitable MILP solver available (SCIP/CBC)")
    solver.SetTimeLimit(30_000)
    return solver


def _solve_ilp(
    players: list[PlayerRecord],
    budget_int: int,
    max_per_team: int,
    required_counts: dict[str, int],
    locked_ids: set[str],
    banned_ids: set[str],
) -> list[PlayerRecord]:
    solver = _build_solver()
    x_vars = {}
    for player in players:
        x_vars[player.idx] = solver.BoolVar(f"x_{player.idx}")

    # Budget constraint
    solver.Add(
        solver.Sum(x_vars[p.idx] * p.cost_int for p in players) <= budget_int
    )

    # Formation constraints
    for position, required in required_counts.items():
        solver.Add(
            solver.Sum(x_vars[p.idx] for p in players if p.position == position) == required
        )

    # Team constraints
    team_to_players: dict[str, list[PlayerRecord]] = {}
    for player in players:
        team_to_players.setdefault(player.team, []).append(player)
    for team, team_players in team_to_players.items():
        solver.Add(
            solver.Sum(x_vars[p.idx] for p in team_players) <= max_per_team
        )

    # Locked/Banned constraints
    for player in players:
        if player.player_id in locked_ids:
            solver.Add(x_vars[player.idx] == 1)
        if player.player_id in banned_ids:
            solver.Add(x_vars[player.idx] == 0)

    # Objective
    solver.Maximize(solver.Sum(x_vars[p.idx] * p.pred for p in players))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError("No feasible solution found for the given constraints")

    chosen = [p for p in players if x_vars[p.idx].solution_value() > 0.5]
    if len(chosen) != sum(required_counts.values()):
        raise ValueError("Solver did not return a full XI")
    return chosen


def _normalize_ids(values: set[str | int] | None) -> set[str]:
    if not values:
        return set()
    return {str(value) for value in values}


def _parse_formation(formation: str) -> dict[str, int]:
    if formation not in FORMATION_COUNTS:
        raise ValueError(f"Unsupported formation: {formation}")
    return FORMATION_COUNTS[formation]


def _validate_locked_constraints(
    players: list[PlayerRecord],
    locked_ids: set[str],
    banned_ids: set[str],
    budget_int: int,
    max_per_team: int,
    required_counts: dict[str, int],
) -> None:
    if locked_ids & banned_ids:
        overlap = sorted(locked_ids & banned_ids)
        raise ValueError(f"Locked and banned overlap: {overlap}")

    locked_players = [p for p in players if p.player_id in locked_ids]
    if len(locked_players) != len(locked_ids):
        found_ids = {p.player_id for p in locked_players}
        missing = sorted(locked_ids - found_ids)
        raise ValueError(f"Locked player IDs not found in dataset: {missing}")

    total_cost = sum(p.cost_int for p in locked_players)
    if total_cost > budget_int:
        raise ValueError("Locked players exceed budget")

    counts = {pos: 0 for pos in required_counts}
    for player in locked_players:
        counts[player.position] += 1
    for pos, required in required_counts.items():
        if counts[pos] > required:
            raise ValueError(f"Locked players exceed required count for {pos}")

    team_counts: dict[str, int] = {}
    for player in locked_players:
        team_counts[player.team] = team_counts.get(player.team, 0) + 1
    for team, count in team_counts.items():
        if count > max_per_team:
            raise ValueError(f"Locked players exceed max per team for {team}")


def _prepare_player_df(
    data_path: str,
    models_dir: str,
    pred_mode: str,
) -> tuple[pd.DataFrame, dict]:
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    _validate_columns(df, ["id", "team_name", "first_name", "second_name", "element_type", "price_now"])

    df = df[df["id"].notna() & df["team_name"].notna()]
    df = df[df["id"].astype(str).str.strip() != ""]
    df = df[df["team_name"].astype(str).str.strip() != ""]
    df["id"] = df["id"].astype(float).astype(int).astype(str)

    df["name"] = (
        df[["first_name", "second_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["team"] = df["team_name"].astype(str).str.strip()
    df, dropped_invalid, invalid_values = _validate_and_filter_element_type(df)
    df["position"] = _map_position(df["element_type"])

    df = _add_momentum_features(df)
    feature_cols = _build_feature_cols(df)
    df["price_now"] = pd.to_numeric(df["price_now"], errors="coerce")
    if df["price_now"].isna().any():
        raise ValueError("price_now contains non-numeric values")

    df["cost_int"] = df["price_now"].round().astype(int)
    df["cost"] = df["cost_int"] / 10.0

    if pred_mode == "models":
        df["pred"] = _predict(df, feature_cols, models_dir)
    elif pred_mode == "stub":
        if "pred" not in df.columns:
            raise ValueError("pred_mode=stub requires a pred column")
        df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
        if df["pred"].isna().any():
            raise ValueError("pred column contains non-numeric values")
    else:
        raise ValueError("pred_mode must be 'models' or 'stub'")

    meta = {
        "dropped_invalid_element_type": dropped_invalid,
        "invalid_element_type_values": invalid_values,
    }
    return df, meta


def build_team(
    budget: float,
    data_path: str = "data/players_merged_2024-25.csv",
    models_dir: str = "models",
    max_per_team: int = 3,
    formation: str = "4-3-3",
    method: str = "ilp",
    pred_mode: str = "models",
    locked_ids: set[str | int] | None = None,
    banned_ids: set[str | int] | None = None,
) -> dict:
    """Build an optimal starting XI under budget constraints."""
    if method != "ilp":
        raise ValueError("Only ILP method is supported")

    df, meta = _prepare_player_df(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode=pred_mode,
    )

    required_counts = _parse_formation(formation)

    for position, required in required_counts.items():
        if (df["position"] == position).sum() < required:
            raise ValueError(f"Not enough players for position {position}")

    budget_int = int(round(budget * 10))

    players = [
        PlayerRecord(
            idx=i,
            player_id=str(row["id"]),
            name=row["name"],
            team=row["team"],
            position=row["position"],
            cost_int=int(row["cost_int"]),
            cost=float(row["cost"]),
            pred=float(row["pred"]),
        )
        for i, row in df.iterrows()
    ]

    locked_set = _normalize_ids(locked_ids)
    banned_set = _normalize_ids(banned_ids)
    _validate_locked_constraints(
        players=players,
        locked_ids=locked_set,
        banned_ids=banned_set,
        budget_int=budget_int,
        max_per_team=max_per_team,
        required_counts=required_counts,
    )

    chosen = _solve_ilp(
        players=players,
        budget_int=budget_int,
        max_per_team=max_per_team,
        required_counts=required_counts,
        locked_ids=locked_set,
        banned_ids=banned_set,
    )
    chosen_sorted = sorted(
        chosen,
        key=lambda p: (POSITION_ORDER.index(p.position), p.name),
    )

    total_cost_int = sum(p.cost_int for p in chosen_sorted)
    total_pred = sum(p.pred for p in chosen_sorted)

    return {
        "formation": formation,
        "budget": budget,
        "budget_int": budget_int,
        "total_cost_int": total_cost_int,
        "total_cost": total_cost_int / 10.0,
        "total_pred": float(total_pred),
        "meta": meta,
        "players": [
            {
                "id": p.player_id,
                "name": p.name,
                "team": p.team,
                "position": p.position,
                "cost_int": p.cost_int,
                "cost": p.cost,
                "pred": p.pred,
            }
            for p in chosen_sorted
        ],
    }


def build_team_433(
    budget: float,
    data_path: str = "data/players_merged_2024-25.csv",
    models_dir: str = "models",
    max_per_team: int = 3,
    formation: str = "4-3-3",
    method: str = "ilp",
    pred_mode: str = "models",
    locked_ids: set[str | int] | None = None,
    banned_ids: set[str | int] | None = None,
) -> dict:
    """Backward-compatible wrapper for 4-3-3."""
    return build_team(
        budget=budget,
        data_path=data_path,
        models_dir=models_dir,
        max_per_team=max_per_team,
        formation=formation,
        method=method,
        pred_mode=pred_mode,
        locked_ids=locked_ids,
        banned_ids=banned_ids,
    )


def score_players(
    data_path: str = "data/players_merged_2024-25.csv",
    models_dir: str = "models",
    pred_mode: str = "models",
) -> tuple[pd.DataFrame, dict]:
    """Return the scored player pool and metadata."""
    return _prepare_player_df(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode=pred_mode,
    )


def build_full_squad(
    budget: float,
    data_path: str = "data/players_merged_2024-25.csv",
    models_dir: str = "models",
    max_per_team: int = 3,
    formation: str = "4-3-3",
    method: str = "ilp",
    pred_mode: str = "models",
    locked_ids: set[str | int] | None = None,
    banned_ids: set[str | int] | None = None,
) -> dict:
    """Build an optimal 15-player squad (11 starters + 4 bench) under FPL constraints.

    Selects 2 GK, 5 DEF, 5 MID, 3 FWD in a single ILP solve, then splits into
    starters (per requested formation) and bench (remaining 4, ordered by predicted points).
    """
    if method != "ilp":
        raise ValueError("Only ILP method is supported")

    df, meta = _prepare_player_df(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode=pred_mode,
    )

    formation_counts = _parse_formation(formation)

    for position, required in FULL_SQUAD_COUNTS.items():
        if (df["position"] == position).sum() < required:
            raise ValueError(f"Not enough players for position {position}")

    budget_int = int(round(budget * 10))

    players = [
        PlayerRecord(
            idx=i,
            player_id=str(row["id"]),
            name=row["name"],
            team=row["team"],
            position=row["position"],
            cost_int=int(row["cost_int"]),
            cost=float(row["cost"]),
            pred=float(row["pred"]),
        )
        for i, row in df.iterrows()
    ]

    locked_set = _normalize_ids(locked_ids)
    banned_set = _normalize_ids(banned_ids)
    _validate_locked_constraints(
        players=players,
        locked_ids=locked_set,
        banned_ids=banned_set,
        budget_int=budget_int,
        max_per_team=max_per_team,
        required_counts=FULL_SQUAD_COUNTS,
    )

    chosen = _solve_ilp(
        players=players,
        budget_int=budget_int,
        max_per_team=max_per_team,
        required_counts=FULL_SQUAD_COUNTS,
        locked_ids=locked_set,
        banned_ids=banned_set,
    )

    # Split into starters and bench based on formation
    starters: list[PlayerRecord] = []
    bench: list[PlayerRecord] = []
    for position in POSITION_ORDER:
        pos_players = sorted(
            [p for p in chosen if p.position == position],
            key=lambda p: p.pred,
            reverse=True,
        )
        starter_count = formation_counts[position]
        starters.extend(pos_players[:starter_count])
        bench.extend(pos_players[starter_count:])

    # Sort starters by position then name
    starters_sorted = sorted(
        starters,
        key=lambda p: (POSITION_ORDER.index(p.position), p.name),
    )
    # Sort bench by predicted points descending (best sub first)
    bench_sorted = sorted(bench, key=lambda p: p.pred, reverse=True)

    total_cost_int = sum(p.cost_int for p in chosen)
    total_pred = sum(p.pred for p in starters)

    def _player_dict(p: PlayerRecord, is_starter: bool, bench_order: int | None) -> dict:
        return {
            "id": p.player_id,
            "name": p.name,
            "team": p.team,
            "position": p.position,
            "cost_int": p.cost_int,
            "cost": p.cost,
            "pred": p.pred,
            "is_starter": is_starter,
            "bench_order": bench_order,
        }

    return {
        "formation": formation,
        "budget": budget,
        "budget_int": budget_int,
        "total_cost_int": total_cost_int,
        "total_cost": total_cost_int / 10.0,
        "total_pred": float(total_pred),
        "players": [_player_dict(p, True, None) for p in starters_sorted],
        "bench": [_player_dict(p, False, i + 1) for i, p in enumerate(bench_sorted)],
        "meta": meta,
    }
