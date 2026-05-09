from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

POSITION_NAMES = {"GK": "goalkeeper", "DEF": "defender", "MID": "midfielder", "FWD": "forward"}

# Stats to evaluate, grouped by category.
# Each entry: (csv_column, display_name, category, positions_or_None)
# positions_or_None=None means all positions.
STAT_DEFS: list[tuple[str, str, str, set[str] | None]] = [
    ("1_years_past_total_points", "total points last season", "performance", None),
    ("1_years_past_goals_scored", "goals last season", "attacking", {"MID", "FWD", "DEF"}),
    ("1_years_past_assists", "assists last season", "attacking", {"MID", "FWD", "DEF"}),
    ("1_years_past_clean_sheets", "clean sheets last season", "defensive", {"GK", "DEF"}),
    ("1_years_past_minutes", "minutes played last season", "reliability", None),
    ("1_years_past_gw_saves", "saves last season", "defensive", {"GK"}),
    ("1_years_past_bonus", "bonus points last season", "performance", None),
    ("1_years_past_bps", "BPS last season", "performance", None),
    ("1_years_past_ict_index", "ICT index last season", "performance", None),
    ("1_years_past_goals_conceded", "goals conceded last season", "defensive", {"GK", "DEF"}),
    ("1_years_past_creativity", "creativity last season", "attacking", {"MID", "FWD"}),
    ("1_years_past_threat", "threat last season", "attacking", {"MID", "FWD"}),
    ("1_years_past_influence", "influence last season", "performance", None),
]

# For goals_conceded, lower is better
LOWER_IS_BETTER = {"1_years_past_goals_conceded"}

VALID_CATEGORIES = {"performance", "attacking", "defensive", "reliability", "value", "trending"}

ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure position column is populated from element_type if needed."""
    if df["position"].isna().all() and "element_type" in df.columns:
        df = df.copy()
        df["position"] = df["element_type"].map(ELEMENT_TYPE_TO_POSITION)
    return df


def _compute_position_stats(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter df to the given position."""
    return df[df["position"] == position].copy()


def _explain_selection_preloaded(
    player_id: str,
    position: str,
    df: pd.DataFrame,
    pos_df: pd.DataFrame,
    top_k: int = 3,
) -> list[dict]:
    """Return top-k player-specific explanations using stat comparisons."""
    row = df[df["id"] == str(player_id)]
    if row.empty:
        return []
    row = row.iloc[0]

    pos_name = POSITION_NAMES.get(position, position.lower())
    pos_count = len(pos_df)
    if pos_count == 0:
        return []

    # Compute how impressive each stat is for this player
    candidates: list[tuple[float, dict]] = []

    for col, display_name, category, positions in STAT_DEFS:
        if positions is not None and position not in positions:
            continue
        if col not in df.columns or col not in pos_df.columns:
            continue

        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if hasattr(val, "item"):
            val = val.item()
        if val == 0:
            continue

        pos_values = pos_df[col].dropna()
        if len(pos_values) < 2:
            continue

        avg = pos_values.mean()
        if col in LOWER_IS_BETTER:
            rank = max(1, int((pos_values <= val).sum()))
            percentile = rank / len(pos_values)
        else:
            rank = max(1, int((pos_values >= val).sum()))
            percentile = rank / len(pos_values)

        # Score: lower percentile ratio = more impressive
        score = percentile

        # Build the explanation text
        int_val = int(round(val))
        int_avg = int(round(avg))

        if col == "1_years_past_total_points":
            text = f"{int_val} {display_name} — ranks {_ordinal(rank)} among {pos_name}s (avg: {int_avg})"
        elif col == "1_years_past_goals_scored":
            pct = max(1, int(round(percentile * 100)))
            text = f"{int_val} {display_name} — top {pct}% of {pos_name}s for goal threat"
        elif col == "1_years_past_assists":
            pct = max(1, int(round(percentile * 100)))
            text = f"{int_val} {display_name} — top {pct}% of {pos_name}s for creativity"
        elif col == "1_years_past_clean_sheets":
            text = f"{int_val} {display_name} — ranks {_ordinal(rank)} among {pos_name}s (avg: {int_avg})"
        elif col == "1_years_past_minutes":
            text = f"Played {int_val:,} minutes last season — one of the most reliable starters at {pos_name}"
        elif col == "1_years_past_gw_saves":
            pct = max(1, int(round(percentile * 100)))
            text = f"{int_val} {display_name} — top {pct}% among goalkeepers"
        elif col == "1_years_past_goals_conceded":
            text = f"Only {int_val} {display_name} — ranks {_ordinal(rank)} among {pos_name}s (avg: {int_avg})"
        elif col in ("1_years_past_bonus", "1_years_past_bps"):
            text = f"{int_val} {display_name} — ranks {_ordinal(rank)} among {pos_name}s (avg: {int_avg})"
        elif col == "1_years_past_ict_index":
            text = f"ICT index of {val:.1f} last season — ranks {_ordinal(rank)} among {pos_name}s"
        elif col in ("1_years_past_creativity", "1_years_past_threat", "1_years_past_influence"):
            short_name = display_name.replace(" last season", "")
            text = f"{short_name.capitalize()} score of {val:.1f} — ranks {_ordinal(rank)} among {pos_name}s"
        else:
            text = f"{int_val} {display_name} — ranks {_ordinal(rank)} among {pos_name}s (avg: {int_avg})"

        candidates.append((score, {"text": text, "category": category}))

    # Sort by score (lower = more impressive)
    candidates.sort(key=lambda x: x[0])

    # Pick top stats, leaving room for value explanation
    selected = []
    seen_categories: set[str] = set()
    for _score, exp in candidates:
        # Prefer variety in categories
        if exp["category"] in seen_categories and len(selected) < top_k - 1:
            continue
        selected.append(exp)
        seen_categories.add(exp["category"])
        if len(selected) >= top_k - 1:
            break

    # If we didn't get enough with diversity, fill without the constraint
    if len(selected) < top_k - 1:
        for _score, exp in candidates:
            if exp not in selected:
                selected.append(exp)
                if len(selected) >= top_k - 1:
                    break

    # Always add value explanation
    cost = row.get("price_now")
    pred_points = row.get("total_points")
    if cost is not None and pred_points is not None:
        if hasattr(cost, "item"):
            cost = cost.item()
        if hasattr(pred_points, "item"):
            pred_points = pred_points.item()
        cost_m = cost / 10.0 if cost > 30 else cost  # handle raw price (70 = £7.0m)
        if cost_m > 0:
            if pred_points < 20:
                # Low historical points — value explanation isn't meaningful
                value_text = f"Budget-friendly option at £{cost_m:.1f}m — selected based on model projection."
            else:
                ppm = pred_points / cost_m
                # Position average ppm
                pos_costs = pos_df["price_now"].dropna()
                pos_points = pos_df["total_points"].dropna()
                if len(pos_costs) > 0 and len(pos_points) > 0:
                    avg_cost = pos_costs.mean()
                    avg_cost_m = avg_cost / 10.0 if avg_cost > 30 else avg_cost
                    avg_points = pos_points.mean()
                    avg_ppm = avg_points / avg_cost_m if avg_cost_m > 0 else 0
                    value_text = (
                        f"At £{cost_m:.1f}m, scored {int(round(pred_points))} points last season — "
                        f"{ppm:.1f} pts/£m (position avg: {avg_ppm:.1f})"
                    )
                else:
                    value_text = f"At £{cost_m:.1f}m, scored {int(round(pred_points))} points last season"
            selected.append({"text": value_text, "category": "value"})

    if not selected:
        return [{"text": "Limited historical data available — selected based on model projection.", "category": "performance"}]

    return selected[:top_k]


def explain_selection(
    player_id: str,
    position: str,
    data_path: str,
    models_dir: str,
    top_k: int = 3,
) -> list[dict]:
    """Return top-k player-specific explanations for why a player was selected.

    Each item: {"text": str, "category": str}
    """
    data_file = Path(data_path)
    if not data_file.exists():
        return []

    df = pd.read_csv(data_file)
    df = df[df["id"].notna()]
    df["id"] = df["id"].astype(float).astype(int).astype(str)
    df = _prepare_df(df)
    pos_df = _compute_position_stats(df, position)

    return _explain_selection_preloaded(
        player_id=player_id,
        position=position,
        df=df,
        pos_df=pos_df,
        top_k=top_k,
    )


def explain_squad(
    players: list[dict],
    data_path: str,
    models_dir: str,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Return explanations for each player in the squad.

    Returns dict mapping player_id -> list of explanations.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        return {str(player["id"]): [] for player in players}

    df = pd.read_csv(data_file)
    df = df[df["id"].notna()]
    df["id"] = df["id"].astype(float).astype(int).astype(str)
    df = _prepare_df(df)

    # Pre-compute position DataFrames
    needed_positions = {player["position"] for player in players}
    pos_dfs: dict[str, pd.DataFrame] = {}
    for position in needed_positions:
        pos_dfs[position] = _compute_position_stats(df, position)

    result: dict[str, list[dict]] = {}
    for player in players:
        pid = str(player["id"])
        position = player["position"]
        result[pid] = _explain_selection_preloaded(
            player_id=pid,
            position=position,
            df=df,
            pos_df=pos_dfs.get(position, pd.DataFrame()),
            top_k=top_k,
        )
    return result
