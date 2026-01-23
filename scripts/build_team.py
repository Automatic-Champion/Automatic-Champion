from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

try:
    from ortools.linear_solver import pywraplp
except ImportError:  # pragma: no cover - optional dependency
    pywraplp = None

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.team_builder import build_team, score_players  # noqa: E402
from scripts.top_features_by_position import (
    POSITION_MAP as MODEL_POSITION_MAP,
    SEASONS_TRAIN,
    build_feature_list,
    load_season,
    train_models,
)


POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
FORMATIONS = ["4-3-3", "4-4-2", "3-4-3", "3-5-2", "4-5-1", "5-3-2", "5-4-1"]

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

_EXPLAIN_CACHE = {
    "data_path": None,
    "df": None,
    "feature_cols": None,
    "models": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimal FPL XI")
    parser.add_argument("--budget", type=float, default=None, help="Budget in FPL millions")
    parser.add_argument("--max-per-team", type=int, default=3, help="Max players per team")
    parser.add_argument(
        "--formation",
        type=str,
        default="4-3-3",
        help="Formation (e.g., 4-3-3, 4-4-2, 3-5-2)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/players_merged_2024-25.csv",
        help="Path to merged player dataset",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model files",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output to XI + totals")
    parser.add_argument("--verbose", action="store_true", help="Print debug details")
    parser.add_argument("--out-json", type=str, default=None, help="Write result JSON")
    parser.add_argument("--out-csv", type=str, default=None, help="Write players CSV")
    return parser.parse_args()


def _print_team(result: dict) -> None:
    players = result["players"]
    grouped = {pos: [] for pos in POSITION_ORDER}
    for player in players:
        grouped[player["position"]].append(player)

    for pos in POSITION_ORDER:
        print(f"\n{pos}:")
        for player in grouped[pos]:
            print(
                f"  {player['name']:<25} {player['team']:<20} "
                f"cost={player['cost']:.1f} pred={player['pred']:.2f}"
            )
    print(f"\nTotal predicted points: {result['total_pred']:.2f}")


def _print_bench(bench: list[dict]) -> None:
    grouped = {pos: [] for pos in POSITION_ORDER}
    for player in bench:
        grouped[player["position"]].append(player)

    print("\nBENCH:")
    for pos in POSITION_ORDER:
        print(f"\n{pos}:")
        for player in grouped[pos]:
            print(
                f"  {player['name']:<25} {player['team']:<20} "
                f"cost={player['cost']:.1f} pred={player['pred']:.2f}"
            )
    print("")


def print_budget_status(result: dict, budget: float) -> None:
    remaining = result.get("remaining_budget")
    if remaining is None:
        remaining = budget - result["total_cost"]
    print("Totals:")
    print(f"  total_cost={result['total_cost']:.1f}")
    print(f"  remaining_budget={remaining:.1f}")


def _print_grouped(title: str, players: list[dict]) -> None:
    grouped = {pos: [] for pos in POSITION_ORDER}
    for player in players:
        grouped[player["position"]].append(player)
    print(f"\n{title}")
    for pos in POSITION_ORDER:
        print(f"\n{pos}:")
        for player in grouped[pos]:
            print(
                f"  {player['name']:<25} {player['team']:<20} "
                f"cost={player['cost']:.1f} pred={player['pred']:.2f}"
            )


def _get_mode(result: dict) -> str:
    if "mode" in result:
        return result["mode"]
    return "xi_then_bench"


def _pretty_feature_name(feature: str) -> str:
    if feature in FEATURE_PRETTY:
        return FEATURE_PRETTY[feature]
    return feature.replace("_", " ")


def explain_feature(feature_name: str, value: object, position: str) -> str:
    if feature_name == "price_now":
        try:
            numeric = float(value)
            if numeric >= 30:
                return f"Price: {numeric / 10:.1f} — strong value within budget."
            return f"Price: {numeric:.1f} — strong value within budget."
        except (TypeError, ValueError):
            return "Price: strong value within budget."

    if feature_name.startswith("1_years_past_"):
        base = feature_name.replace("1_years_past_", "")
        pretty = _pretty_feature_name(base)
        why = FEATURE_EXPLANATIONS.get(base) or FEATURE_EXPLANATIONS.get(pretty) or (
            "This suggests steady returns over the season."
        )
        return f"Last season {pretty}: {value} — {why}"

    key = feature_name.lower()
    why = FEATURE_EXPLANATIONS.get(key)
    if why:
        return f"{_pretty_feature_name(feature_name)}: {value} — {why}"

    for pattern, text in FEATURE_EXPLANATIONS.items():
        if key.endswith(pattern):
            return f"{_pretty_feature_name(feature_name)}: {value} — {text}"

    return f"{_pretty_feature_name(feature_name)}: {value} — This stat suggests consistent performance in past seasons."


def _load_explain_resources(data_path: str, models_dir: str) -> tuple[pd.DataFrame, list[str], dict[int, object]]:
    if _EXPLAIN_CACHE["data_path"] != data_path:
        _EXPLAIN_CACHE["data_path"] = data_path
        _EXPLAIN_CACHE["df"] = pd.read_csv(data_path)
        _EXPLAIN_CACHE["feature_cols"] = build_feature_list(_EXPLAIN_CACHE["df"])
        _EXPLAIN_CACHE["models"] = None

    if _EXPLAIN_CACHE["models"] is None:
        models: dict[int, object] = {}
        for position in MODEL_POSITION_MAP:
            path = Path(models_dir) / f"position_model_{position}.joblib"
            if path.exists():
                models[position] = joblib.load(path)
        if len(models) != len(MODEL_POSITION_MAP):
            df_train = pd.concat(
                [load_season(season) for season in SEASONS_TRAIN], ignore_index=True
            )
            feature_cols = build_feature_list(df_train)
            models = train_models(df_train, feature_cols)
            _EXPLAIN_CACHE["feature_cols"] = feature_cols
        _EXPLAIN_CACHE["models"] = models

    return _EXPLAIN_CACHE["df"], _EXPLAIN_CACHE["feature_cols"], _EXPLAIN_CACHE["models"]


def get_top_features_for_player(
    player: dict,
    position: str,
    data_path: str,
    models_dir: str,
    top_k: int = 5,
) -> list[dict]:
    df, feature_cols, models = _load_explain_resources(data_path, models_dir)
    pos_code_map = {label: code for code, label in MODEL_POSITION_MAP.items()}
    pos_code = pos_code_map.get(position)
    if pos_code is None or pos_code not in models:
        return []
    row = df[df["id"].astype(str) == str(player["id"])]
    if row.empty:
        return []
    row = row.iloc[0]
    importances = pd.Series(models[pos_code].feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=False).head(top_k)
    items = []
    for feature, score in top.items():
        value = row.get(feature)
        items.append(
            {
                "feature": feature,
                "value": value,
                "score": float(score),
                "why": explain_feature(feature, value, position),
            }
        )
    return items


def _ensure_explanations_cache(
    result: dict,
    data_path: str,
    models_dir: str,
    top_k: int = 5,
) -> None:
    cache = result.setdefault("explanations_cache", {})
    starters = result.get("starters", result.get("players", []))
    bench = result.get("bench", [])
    for player in starters + bench:
        player_id = str(player["id"])
        if player_id in cache:
            continue
        try:
            cache[player_id] = get_top_features_for_player(
                player=player,
                position=player["position"],
                data_path=data_path,
                models_dir=models_dir,
                top_k=top_k,
            )
        except Exception:
            cache[player_id] = []


def explanations_menu(
    current_result: dict,
    args: argparse.Namespace,
    data_path: str,
    models_dir: str,
    top_k: int = 5,
) -> None:
    starters = current_result.get("starters", current_result.get("players", []))
    bench = current_result.get("bench", [])
    combined = [(player, "STARTER") for player in starters] + [
        (player, "BENCH") for player in bench
    ]
    while True:
        print("\nShow explanations (select a player)")
        menu = {0: "Back"}
        for idx, (player, role) in enumerate(combined, start=1):
            label = f"{player['name']} ({player['position']}, {player['team']}) - {role}"
            menu[idx] = label
        _print_menu(menu)
        choice = _read_menu_choice(menu)
        if choice is None or choice == 0:
            return
        player, role = combined[choice - 1]
        _ensure_explanations_cache(current_result, data_path, models_dir, top_k=top_k)
        items = current_result["explanations_cache"].get(str(player["id"]), [])
        print(f"\n{player['name']} ({player['position']}, {player['team']}) — Top {top_k} model drivers")
        if not items:
            print("  (No explanation available for this player.)")
            input("\nPress Enter to return...")
            continue
        for idx, item in enumerate(items, start=1):
            print(f"  {idx}) {item['why']}")
        input("\nPress Enter to return...")
def print_squad(result: dict, budget: float) -> None:
    remaining = result.get("remaining_budget")
    if remaining is None:
        remaining = budget - result["total_cost"]
    starters = result.get("starters", result.get("players", []))
    bench = result.get("bench", [])
    print(f"Formation: {result.get('formation', 'N/A')}")
    _print_grouped("STARTING XI", starters)
    _print_grouped("BENCH", bench)

    total_pred_starters = sum(player["pred"] for player in starters)
    total_pred_bench = sum(player["pred"] for player in bench) if bench else 0.0
    total_pred_squad = total_pred_starters + total_pred_bench
    print("\nTotals:")
    print(f"  Budget: {budget:.1f}")
    print(f"  Total cost: {result['total_cost']:.1f}")
    print(f"  Remaining budget: {remaining:.1f}")
    print(f"  Total predicted (starters): {total_pred_starters:.2f}")
    print(f"  Total predicted (bench): {total_pred_bench:.2f}")
    print(f"  Total predicted (squad): {total_pred_squad:.2f}")


def _display_team_for_selection(result: dict) -> dict[str, list[dict]]:
    players = result["players"]
    grouped = {pos: [] for pos in POSITION_ORDER}
    for player in players:
        grouped[player["position"]].append(player)

    for pos in POSITION_ORDER:
        print(f"\n{pos}:")
        for idx, player in enumerate(grouped[pos], start=1):
            print(
                f"  [{idx}] {player['name']:<25} {player['team']:<20} "
                f"cost={player['cost']:.1f} pred={player['pred']:.2f}"
            )
    return grouped


def read_choice(prompt: str, choices: set[str]) -> str:
    while True:
        choice = input(prompt).strip()
        if not choice:
            continue
        choice_map = {item.lower(): item for item in choices}
        lowered = choice.lower()
        if lowered in choice_map:
            return choice_map[lowered]
        print("Invalid choice. Please choose one of the displayed options.")


def read_int(
    prompt: str,
    valid_range: range | None = None,
    allow_empty: bool = False,
    invalid_msg: str | None = None,
) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw:
            if allow_empty:
                return -1
            print("Please enter a number.")
            continue
        if not raw.isdigit():
            print("Please enter a number.")
            continue
        value = int(raw)
        if valid_range is None or value in valid_range:
            return value
        if invalid_msg:
            print(invalid_msg)
        elif valid_range is not None:
            print(
                f"Invalid choice. Please enter a number between "
                f"{valid_range.start} and {valid_range.stop - 1}."
            )
        else:
            print("Invalid choice. Please choose one of the displayed options.")


def read_yes_no(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"", "n", "no"}:
            return False
        print("Please enter y/yes or n/no.")


def _format_menu_prompt(keys: list[int]) -> str:
    min_key = min(keys)
    max_key = max(keys)
    is_contiguous = keys == list(range(min_key, max_key + 1))
    if is_contiguous:
        return f"Select an option ({min_key}-{max_key}): "
    key_list = ",".join(str(key) for key in keys)
    return f"Select an option ({key_list}): "


def _print_menu(menu: dict[int, str]) -> str:
    keys = sorted(menu.keys())
    for key in keys:
        print(f"[{key}] {menu[key]}")
    return _format_menu_prompt(keys)


def _read_menu_choice(menu: dict[int, str]) -> int:
    keys = sorted(menu.keys())
    min_key = min(keys)
    max_key = max(keys)
    is_contiguous = keys == list(range(min_key, max_key + 1))
    attempts = 0
    while True:
        if attempts >= 5:
            print("Still no valid selection. Returning to previous menu.")
            return None
        prompt = _format_menu_prompt(keys)
        raw = input(prompt).strip()
        if raw.isdigit():
            value = int(raw)
            if value in menu:
                return value
        if is_contiguous:
            print(f"Invalid choice. Please enter a number between {min_key} and {max_key}.")
        else:
            print("Invalid choice. Please choose one of the displayed options.")
        attempts += 1


def build_candidate_menu(
    title: str,
    candidates: list[object],
    *,
    allow_back: bool = True,
    back_label: str = "Back",
) -> tuple[dict[int, str], dict[int, object]]:
    menu: dict[int, str] = {}
    if allow_back:
        menu[0] = back_label
    mapping: dict[int, object] = {}
    for idx, row in enumerate(candidates, start=1):
        menu[idx] = ""
        mapping[idx] = row
    return menu, mapping


def prompt_menu(menu_dict: dict[int, str], title: str) -> int | None:
    print(title)
    _print_menu(menu_dict)
    return _read_menu_choice(menu_dict)


def handle_no_options(context_message: str, actions: dict[int, str]) -> int | None:
    print(context_message)
    _print_menu(actions)
    return _read_menu_choice(actions)




def _prompt_float(prompt: str, min_value: float) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Enter a valid number.")
            continue
        if value >= min_value:
            return value
        print(f"Enter a number >= {min_value}.")


def _print_verbose(result: dict, quiet: bool, verbose: bool) -> None:
    if quiet or not verbose:
        return

    dropped = result.get("meta", {}).get("dropped_invalid_element_type", 0)
    invalid_values = result.get("meta", {}).get("invalid_element_type_values", [])
    if dropped:
        print(
            f"\nDropped invalid element_type rows: {dropped} (values: {invalid_values})"
        )



def _select_formation(current: str) -> str:
    while True:
        print("\nFormations:")
        menu = {idx: key for idx, key in enumerate(FORMATIONS, start=1)}
        _print_menu(menu)
        choice = _read_menu_choice(menu)
        if choice is None:
            return current
        return menu[choice]


def _team_ids(result: dict) -> set[str]:
    return {str(player["id"]) for player in result["players"]}


def _team_counts(result: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in result["players"]:
        team = player["team"]
        counts[team] = counts.get(team, 0) + 1
    return counts


def _squad_ids(result: dict) -> set[str]:
    ids = {str(player["id"]) for player in result.get("players", [])}
    for player in result.get("bench", []):
        ids.add(str(player["id"]))
    return ids


def _squad_counts(result: dict) -> dict[str, int]:
    counts = _team_counts(result)
    for player in result.get("bench", []):
        team = player["team"]
        counts[team] = counts.get(team, 0) + 1
    return counts


def _position_counts_from_formation(formation: str) -> dict[str, int]:
    parts = formation.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid formation: {formation}")
    defs, mids, fwds = (int(part) for part in parts)
    return {"GK": 1, "DEF": defs, "MID": mids, "FWD": fwds}


def _build_bench_auto(
    xi_result: dict,
    data_path: str,
    models_dir: str,
    budget: float,
    max_per_team: int,
    banned_ids: set[str],
    locked_ids: set[str] | None = None,
) -> list[dict]:
    if pywraplp is None:
        raise RuntimeError("ortools is required for auto bench optimization")

    pool_df, _ = score_players(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode="models",
    )
    xi_ids = {str(player["id"]) for player in xi_result["players"]}
    pool_df = pool_df[~pool_df["id"].astype(str).isin(xi_ids)]
    pool_df = pool_df[~pool_df["id"].astype(str).isin(banned_ids)]

    remaining_budget_int = int(round(budget * 10)) - xi_result["total_cost_int"]
    if remaining_budget_int < 0:
        raise ValueError("XI exceeds budget; cannot build bench")

    team_counts = _team_counts(xi_result)
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No suitable MILP solver available for bench")

    x_vars = {}
    players = list(pool_df.itertuples(index=False))
    for idx, row in enumerate(players):
        x_vars[idx] = solver.BoolVar(f"b_{idx}")

    solver.Add(
        solver.Sum(x_vars[i] * int(row.cost_int) for i, row in enumerate(players))
        <= remaining_budget_int
    )

    required = {"GK": 1, "DEF": 1, "MID": 1, "FWD": 1}
    for pos, count in required.items():
        solver.Add(
            solver.Sum(x_vars[i] for i, row in enumerate(players) if row.position == pos)
            == count
        )

    teams = {}
    for i, row in enumerate(players):
        teams.setdefault(row.team, []).append(i)
    for team, indices in teams.items():
        solver.Add(
            solver.Sum(x_vars[i] for i in indices) + team_counts.get(team, 0)
            <= max_per_team
        )

    if locked_ids:
        locked_set = {str(value) for value in locked_ids}
        for i, row in enumerate(players):
            if str(row.id) in locked_set:
                solver.Add(x_vars[i] == 1)

    solver.Maximize(solver.Sum(x_vars[i] * float(row.pred) for i, row in enumerate(players)))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError("No valid bench found under remaining budget")

    bench = []
    for i, row in enumerate(players):
        if x_vars[i].solution_value() > 0.5:
            bench.append(
                {
                    "id": str(row.id),
                    "name": row.name,
                    "team": row.team,
                    "position": row.position,
                    "cost_int": int(row.cost_int),
                    "cost": float(row.cost),
                    "pred": float(row.pred),
                }
            )
    return bench


def _build_bench_manual(
    xi_result: dict,
    data_path: str,
    models_dir: str,
    budget: float,
    max_per_team: int,
    banned_ids: set[str],
) -> list[dict] | dict | None:
    pool_df, _ = score_players(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode="models",
    )
    selected = []
    squad_ids = {str(player["id"]) for player in xi_result["players"]}
    team_counts = _team_counts(xi_result)
    remaining_budget_int = int(round(budget * 10)) - xi_result["total_cost_int"]

    for pos in POSITION_ORDER:
        while True:
            remaining_budget = remaining_budget_int / 10.0
            print(f"\nRemaining budget: {remaining_budget:.1f}")
            candidates = pool_df[pool_df["position"] == pos].copy()
            candidates = candidates[~candidates["id"].astype(str).isin(banned_ids)]
            candidates = candidates[~candidates["id"].astype(str).isin(squad_ids)]
            candidates = candidates[
                candidates["team"].map(lambda team: team_counts.get(team, 0) < max_per_team)
            ]
            candidates = candidates[candidates["cost_int"].astype(int) <= remaining_budget_int]
            candidates = candidates.sort_values("pred", ascending=False).head(15)

            if candidates.empty:
                action = handle_no_options(
                    f"No valid options available under current constraints (remaining budget: {remaining_budget:.1f}).",
                    {0: "Back", 1: "Switch to Auto"},
                )
                if action == 1:
                    return {"_action": "auto"}
                return None

            print(f"\nTop candidates for {pos}:")
            candidate_rows = list(candidates.itertuples(index=False))
            candidate_menu, candidate_map = build_candidate_menu(
                f"Top candidates for {pos}:",
                candidate_rows,
                allow_back=True,
                back_label="Back",
            )
            for idx, row in candidate_map.items():
                print(
                    f"  [{idx}] {row.name:<25} {row.team:<20} "
                    f"cost={row.cost:.1f} pred={row.pred:.2f}"
                )
            print("  [0] Back")

            choice = _read_menu_choice(candidate_menu)
            if choice is None or choice == 0:
                return None
            picked = candidate_map[choice]
            if int(picked.cost_int) > remaining_budget_int:
                print("Selected player exceeds remaining budget.")
                continue
            if team_counts.get(picked.team, 0) >= max_per_team:
                print(
                    f"Cannot select {picked.name} from {picked.team}: "
                    f"already have {max_per_team} players from that team."
                )
                continue

            selected.append(
                {
                    "id": str(picked.id),
                    "name": picked.name,
                    "team": picked.team,
                    "position": pos,
                    "cost_int": int(picked.cost_int),
                    "cost": float(picked.cost),
                    "pred": float(picked.pred),
                }
            )
            squad_ids.add(str(picked.id))
            team_counts[picked.team] = team_counts.get(picked.team, 0) + 1
            remaining_budget_int -= int(picked.cost_int)
            break

    return selected


def _sort_players(players: list[dict]) -> list[dict]:
    return sorted(players, key=lambda p: (POSITION_ORDER.index(p["position"]), p["name"]))


def _bench_valid(
    xi_result: dict,
    bench: list[dict],
    budget: float,
    max_per_team: int,
) -> bool:
    if len(bench) != 4:
        return False
    bench_ids = {str(player["id"]) for player in bench}
    xi_ids = {str(player["id"]) for player in xi_result["players"]}
    if bench_ids & xi_ids:
        return False
    pos_counts = {pos: 0 for pos in POSITION_ORDER}
    for player in bench:
        pos_counts[player["position"]] += 1
    if any(pos_counts[pos] != 1 for pos in POSITION_ORDER):
        return False

    total_cost_int = xi_result["total_cost_int"] + sum(p["cost_int"] for p in bench)
    if total_cost_int > int(round(budget * 10)):
        return False

    team_counts = _squad_counts({"players": xi_result["players"], "bench": bench})
    return all(count <= max_per_team for count in team_counts.values())


def _update_remaining_budget(result: dict, budget: float) -> None:
    bench_cost = sum(player["cost"] for player in result.get("bench", []))
    remaining = budget - result["total_cost"] - bench_cost
    result["remaining_budget"] = remaining
    result["remaining_budget_int"] = int(round(remaining * 10))


def _require_bench(result: dict) -> None:
    if "bench" not in result or len(result["bench"]) != 4:
        raise RuntimeError("Bench is required but missing (expected 4 bench players).")


def _solve_starters_with_fixed_bench(
    data_path: str,
    models_dir: str,
    formation: str,
    budget: float,
    max_per_team: int,
    bench: list[dict],
    banned_ids: set[str],
) -> dict:
    if pywraplp is None:
        raise RuntimeError("ortools is required for optimization")

    pool_df, _ = score_players(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode="models",
    )
    bench_ids = {str(player["id"]) for player in bench}
    pool_df = pool_df[~pool_df["id"].astype(str).isin(bench_ids)]
    pool_df = pool_df[~pool_df["id"].astype(str).isin(banned_ids)]

    bench_cost_int = sum(player["cost_int"] for player in bench)
    remaining_budget_int = int(round(budget * 10)) - bench_cost_int
    if remaining_budget_int < 0:
        raise ValueError("Bench exceeds budget; cannot rebuild starters")

    counts = _position_counts_from_formation(formation)
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No suitable MILP solver available")

    players = list(pool_df.itertuples(index=False))
    x_vars = {i: solver.BoolVar(f"s_{i}") for i in range(len(players))}

    solver.Add(
        solver.Sum(x_vars[i] * int(row.cost_int) for i, row in enumerate(players))
        <= remaining_budget_int
    )
    for pos, count in counts.items():
        solver.Add(
            solver.Sum(
                x_vars[i] for i, row in enumerate(players) if row.position == pos
            )
            == count
        )

    bench_team_counts = _squad_counts({"players": [], "bench": bench})
    teams = {}
    for i, row in enumerate(players):
        teams.setdefault(row.team, []).append(i)
    for team, indices in teams.items():
        solver.Add(
            solver.Sum(x_vars[i] for i in indices) + bench_team_counts.get(team, 0)
            <= max_per_team
        )

    solver.Maximize(
        solver.Sum(x_vars[i] * float(row.pred) for i, row in enumerate(players))
    )
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError("No valid starters found under constraints")

    starters = [
        {
            "id": str(row.id),
            "name": row.name,
            "team": row.team,
            "position": row.position,
            "cost_int": int(row.cost_int),
            "cost": float(row.cost),
            "pred": float(row.pred),
        }
        for i, row in enumerate(players)
        if x_vars[i].solution_value() > 0.5
    ]
    total_cost_int = sum(p["cost_int"] for p in starters)
    total_pred = sum(p["pred"] for p in starters)
    return {
        "formation": formation,
        "budget": budget,
        "budget_int": int(round(budget * 10)),
        "total_cost_int": total_cost_int,
        "total_cost": total_cost_int / 10.0,
        "total_pred": float(total_pred),
        "players": _sort_players(starters),
        "starters": _sort_players(starters),
        "bench": _sort_players(bench),
    }


def _build_full_squad(
    data_path: str,
    models_dir: str,
    budget: float,
    max_per_team: int,
    formation: str,
    banned_ids: set[str],
) -> dict:
    if pywraplp is None:
        raise RuntimeError("ortools is required for full squad optimization")

    counts = _position_counts_from_formation(formation)
    total_counts = {pos: count + 1 for pos, count in counts.items()}
    pool_df, _ = score_players(
        data_path=data_path,
        models_dir=models_dir,
        pred_mode="models",
    )
    pool_df = pool_df[~pool_df["id"].astype(str).isin(banned_ids)]
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No suitable MILP solver available for squad")

    players = list(pool_df.itertuples(index=False))
    x_vars = {i: solver.BoolVar(f"s_{i}") for i in range(len(players))}
    budget_int = int(round(budget * 10))
    solver.Add(
        solver.Sum(x_vars[i] * int(row.cost_int) for i, row in enumerate(players))
        <= budget_int
    )
    for pos, count in total_counts.items():
        solver.Add(
            solver.Sum(x_vars[i] for i, row in enumerate(players) if row.position == pos)
            == count
        )
    teams = {}
    for i, row in enumerate(players):
        teams.setdefault(row.team, []).append(i)
    for team, indices in teams.items():
        solver.Add(
            solver.Sum(x_vars[i] for i in indices) <= max_per_team
        )
    solver.Maximize(
        solver.Sum(x_vars[i] * float(row.pred) for i, row in enumerate(players))
    )
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError("No valid squad found under constraints")

    squad = [
        {
            "id": str(row.id),
            "name": row.name,
            "team": row.team,
            "position": row.position,
            "cost_int": int(row.cost_int),
            "cost": float(row.cost),
            "pred": float(row.pred),
        }
        for i, row in enumerate(players)
        if x_vars[i].solution_value() > 0.5
    ]
    squad_by_pos = {pos: [] for pos in POSITION_ORDER}
    for player in squad:
        squad_by_pos[player["position"]].append(player)
    bench = []
    starting = []
    for pos, total in total_counts.items():
        sorted_pos = sorted(squad_by_pos[pos], key=lambda p: p["pred"])
        bench.append(sorted_pos[0])
        starting.extend(sorted_pos[1:])

    total_cost_int = sum(p["cost_int"] for p in starting)
    total_pred = sum(p["pred"] for p in starting)
    result = {
        "formation": formation,
        "budget": budget,
        "budget_int": int(round(budget * 10)),
        "total_cost_int": total_cost_int,
        "total_cost": total_cost_int / 10.0,
        "total_pred": float(total_pred),
        "players": _sort_players(starting),
        "starters": _sort_players(starting),
        "bench": _sort_players(bench),
    }
    _update_remaining_budget(result, budget)
    return result


def _replace_with_bench_menu(
    result: dict,
    data_path: str,
    models_dir: str,
    formation: str,
    budget: float,
    max_per_team: int,
    locked_ids: set[str],
    banned_ids: set[str],
    verbose: bool,
) -> dict | None:
    replace_menu = {
        0: "Back",
        1: "Swap starter with bench (same position)",
        2: "Replace a STARTER with a new player",
        3: "Replace a BENCH player with a new player",
    }
    print("\nReplace Menu")
    _print_menu(replace_menu)
    choice = _read_menu_choice(replace_menu)
    if choice is None or choice == 0:
        return None

    starters = result.get("players", [])
    bench = result.get("bench", [])
    team_counts = _squad_counts(result)

    if choice == 1:
        print("\nPositions:")
        pos_menu = {idx: pos for idx, pos in enumerate(POSITION_ORDER, start=1)}
        _print_menu(pos_menu)
        pos_choice_key = _read_menu_choice(pos_menu)
        if pos_choice_key is None:
            return None
        pos_choice = pos_menu[pos_choice_key]
        starters_pos = [p for p in starters if p["position"] == pos_choice]
        bench_pos = [p for p in bench if p["position"] == pos_choice]
        if not starters_pos or not bench_pos:
            print("No players available to swap for that position.")
            return None
        starter_menu = {idx: p for idx, p in enumerate(starters_pos, start=1)}
        bench_menu = {idx: p for idx, p in enumerate(bench_pos, start=1)}
        print("\nStarters:")
        _print_menu({idx: f"{p['name']} ({p['team']})" for idx, p in starter_menu.items()})
        starter_choice = _read_menu_choice(starter_menu)
        if starter_choice is None:
            return None
        print("\nBench:")
        _print_menu({idx: f"{p['name']} ({p['team']})" for idx, p in bench_menu.items()})
        bench_choice = _read_menu_choice(bench_menu)
        if bench_choice is None:
            return None

        starter_player = starter_menu[starter_choice]
        bench_player = bench_menu[bench_choice]
        updated_starters = [p for p in starters if str(p["id"]) != str(starter_player["id"])]
        updated_bench = [p for p in bench if str(p["id"]) != str(bench_player["id"])]
        updated_starters.append(bench_player)
        updated_bench.append(starter_player)
        result["players"] = _sort_players(updated_starters)
        result["bench"] = _sort_players(updated_bench)
        _update_remaining_budget(result, budget)
        _require_bench(result)
        return result

    if choice == 2:
        new_result = _pick_replacement(
            result=result,
            data_path=data_path,
            models_dir=models_dir,
            formation=formation,
            budget=budget,
            max_per_team=max_per_team,
            locked_ids=locked_ids,
            banned_ids=banned_ids,
            verbose=verbose,
            target_players=starters,
            exclude_ids={str(p["id"]) for p in bench},
            team_counts=team_counts,
        )
        if new_result is None:
            return None
        if bench and not _bench_valid(new_result, bench, budget, max_per_team):
            bench = _build_bench_auto(
                xi_result=new_result,
                data_path=data_path,
                models_dir=models_dir,
                budget=budget,
                max_per_team=max_per_team,
                banned_ids=banned_ids,
            )
        new_result["bench"] = bench
        _update_remaining_budget(new_result, budget)
        _require_bench(new_result)
        return new_result

    if choice == 3:
        if not bench:
            print("No bench available.")
            return None

        print("\nPositions:")
        pos_menu = {idx: pos for idx, pos in enumerate(POSITION_ORDER, start=1)}
        _print_menu(pos_menu)
        pos_choice_key = _read_menu_choice(pos_menu)
        if pos_choice_key is None:
            return None
        pos_choice = pos_menu[pos_choice_key]
        bench_pos = [p for p in bench if p["position"] == pos_choice]
        if not bench_pos:
            print("No bench players in that position.")
            return None
        bench_menu = {idx: p for idx, p in enumerate(bench_pos, start=1)}
        _print_menu({idx: f"{p['name']} ({p['team']})" for idx, p in bench_menu.items()})
        bench_choice = _read_menu_choice(bench_menu)
        if bench_choice is None:
            return None
        selected_bench = bench_menu[bench_choice]
        selected_id = str(selected_bench["id"])
        budget_int = int(round(budget * 10))
        bench_cost_int = sum(p.get("cost_int", int(round(p["cost"] * 10))) for p in bench)
        selected_bench_cost_int = selected_bench.get("cost_int", int(round(selected_bench["cost"] * 10)))
        remaining_budget_int = budget_int - result["total_cost_int"] - (bench_cost_int - selected_bench_cost_int)

        prev_banned = set(banned_ids)
        banned_ids.add(selected_id)
        locked_bench_ids = {str(p["id"]) for p in bench if str(p["id"]) != selected_id}
        if read_yes_no("\nPick a specific replacement? (y/n): "):
            pool_df, _ = score_players(
                data_path=data_path,
                models_dir=models_dir,
                pred_mode="models",
            )
            candidates = pool_df[pool_df["position"] == pos_choice].copy()
            squad_ids = _squad_ids(result)
            candidates = candidates[~candidates["id"].astype(str).isin(banned_ids)]
            candidates = candidates[~candidates["id"].astype(str).isin(squad_ids)]
            squad_team_counts = _squad_counts(result)
            candidates = candidates[candidates["cost_int"].astype(int) <= remaining_budget_int]
            candidates = candidates[
                candidates["team"].map(lambda team: squad_team_counts.get(team, 0) < max_per_team)
            ]
            candidates = candidates.sort_values("pred", ascending=False).head(10)

            if candidates.empty:
                action = handle_no_options(
                    "No valid options available under current constraints.",
                    {0: "Back", 1: "Switch to Auto"},
                )
                if action == 1:
                    try:
                        updated_bench = _build_bench_auto(
                            xi_result=result,
                            data_path=data_path,
                            models_dir=models_dir,
                            budget=budget,
                            max_per_team=max_per_team,
                            banned_ids=banned_ids,
                            locked_ids=locked_bench_ids,
                        )
                    except ValueError as exc:
                        print(f"No valid bench found: {exc}")
                        return None
                    result["bench"] = updated_bench
                    _update_remaining_budget(result, budget)
                    _require_bench(result)
                    return result
                return None

            print("\nTop candidates:")
            candidate_rows = list(candidates.itertuples(index=False))
            candidate_menu, candidate_map = build_candidate_menu(
                "Top candidates:",
                candidate_rows,
                allow_back=True,
                back_label="Back",
            )
            for idx, row in candidate_map.items():
                print(
                    f"  [{idx}] {row.name:<25} {row.team:<20} "
                    f"cost={row.cost:.1f} pred={row.pred:.2f}"
                )
            print("  [0] Back")

            while True:
                choice_idx = _read_menu_choice(candidate_menu)
                if choice_idx is None or choice_idx == 0:
                    return None
                chosen_row = candidate_map[choice_idx]
                team = chosen_row.team
                if squad_team_counts.get(team, 0) >= max_per_team:
                    print(
                        f"Cannot select {chosen_row.name} from {team}: "
                        f"already have {max_per_team} players from that team."
                    )
                    continue

                locked_id = str(chosen_row.id)
                try:
                    updated_bench = _build_bench_auto(
                        xi_result=result,
                        data_path=data_path,
                        models_dir=models_dir,
                        budget=budget,
                        max_per_team=max_per_team,
                        banned_ids=banned_ids,
                        locked_ids=locked_bench_ids | {locked_id},
                    )
                except ValueError as exc:
                    print(f"No valid bench found: {exc}")
                    continue

                locked_in_bench = any(str(p["id"]) == locked_id for p in updated_bench)
                if not locked_in_bench:
                    print(
                        f"Could not lock {chosen_row.name} into the bench under current constraints. "
                        "Choose another player."
                    )
                    continue

                result["bench"] = updated_bench
                _update_remaining_budget(result, budget)
                return result
        else:
            try:
                updated_bench = _build_bench_auto(
                    xi_result=result,
                    data_path=data_path,
                    models_dir=models_dir,
                    budget=budget,
                    max_per_team=max_per_team,
                    banned_ids=banned_ids,
                    locked_ids=locked_bench_ids,
                )
            except ValueError as exc:
                banned_ids.clear()
                banned_ids.update(prev_banned)
                print(f"No valid bench found: {exc}")
                return None

            result["bench"] = updated_bench
            _update_remaining_budget(result, budget)
            _require_bench(result)
            return result

    return None


def _pick_replacement(
    result: dict,
    data_path: str,
    models_dir: str,
    formation: str,
    budget: float,
    max_per_team: int,
    locked_ids: set[str],
    banned_ids: set[str],
    verbose: bool,
    target_players: list[dict] | None = None,
    exclude_ids: set[str] | None = None,
    team_counts: dict[str, int] | None = None,
) -> dict | None:
    base_players = target_players if target_players is not None else result["players"]
    grouped = {pos: [] for pos in POSITION_ORDER}
    for player in base_players:
        grouped[player["position"]].append(player)
    for pos in POSITION_ORDER:
        print(f"\n{pos}:")
        for idx, player in enumerate(grouped[pos], start=1):
            print(
                f"  [{idx}] {player['name']:<25} {player['team']:<20} "
                f"cost={player['cost']:.1f} pred={player['pred']:.2f}"
            )

    print("\nPositions:")
    pos_menu = {idx: pos for idx, pos in enumerate(POSITION_ORDER, start=1)}
    _print_menu(pos_menu)
    pos_choice_key = _read_menu_choice(pos_menu)
    if pos_choice_key is None:
        return None
    pos_choice = pos_menu[pos_choice_key]
    players_in_pos = grouped[pos_choice]
    if not players_in_pos:
        print("No players in that position.")
        return None

    replace_menu = {idx: player for idx, player in enumerate(players_in_pos, start=1)}
    idx_choice = _read_menu_choice(replace_menu)
    if idx_choice is None:
        return None
    selected = players_in_pos[idx_choice - 1]
    selected_id = str(selected["id"])
    budget_int = int(round(budget * 10))
    bench_cost_int = sum(p.get("cost_int", int(round(p["cost"] * 10))) for p in result.get("bench", []))
    selected_cost_int = selected.get("cost_int", int(round(selected["cost"] * 10)))

    prev_banned = set(banned_ids)
    prev_locked = set(locked_ids)
    banned_ids.add(selected_id)

    try:
        new_result = build_team(
            budget=budget,
            data_path=data_path,
            models_dir=models_dir,
            max_per_team=max_per_team,
            formation=formation,
            method="ilp",
            pred_mode="models",
            locked_ids=locked_ids,
            banned_ids=banned_ids,
        )
    except ValueError:
        banned_ids.clear()
        banned_ids.update(prev_banned)
        print(
            "No valid team found under the current budget/formation with the selected constraints. "
            "Undoing the last change."
        )
        return None

    if read_yes_no("\nPick a specific replacement? (y/n): "):
        pool_df, _ = score_players(
            data_path=data_path,
            models_dir=models_dir,
            pred_mode="models",
        )
        candidates = pool_df[pool_df["position"] == pos_choice].copy()
        current_ids = _team_ids(new_result)
        if exclude_ids:
            current_ids = current_ids | exclude_ids
        candidates = candidates[~candidates["id"].astype(str).isin(banned_ids)]
        candidates = candidates[~candidates["id"].astype(str).isin(current_ids)]
        active_team_counts = team_counts or _team_counts(result)
        remaining_budget_int = budget_int - bench_cost_int - (result["total_cost_int"] - selected_cost_int)
        candidates = candidates[candidates["cost_int"].astype(int) <= remaining_budget_int]
        candidates = candidates[
            candidates["team"].map(lambda team: active_team_counts.get(team, 0) < max_per_team)
        ]
        candidates = candidates.sort_values("pred", ascending=False).head(10)

        if candidates.empty:
            action = handle_no_options(
                "No valid options available under current constraints.",
                {0: "Back", 1: "Switch to Auto"},
            )
            if action == 1:
                return new_result
            return None

        print("\nTop candidates:")
        candidate_rows = list(candidates.itertuples(index=False))
        candidate_menu, candidate_map = build_candidate_menu(
            "Top candidates:",
            candidate_rows,
            allow_back=True,
            back_label="Back",
        )
        for idx, row in candidate_map.items():
            print(
                f"  [{idx}] {row.name:<25} {row.team:<20} "
                f"cost={row.cost:.1f} pred={row.pred:.2f}"
            )
        print("  [0] Back")

        while True:
            lock_choice = _read_menu_choice(candidate_menu)
            if lock_choice is None or lock_choice == 0:
                return new_result

            chosen_row = candidate_map[lock_choice]
            team = chosen_row.team
            if active_team_counts.get(team, 0) >= max_per_team:
                print(
                    f"Cannot select {chosen_row.name} from {team}: "
                    f"already have {max_per_team} players from that team."
                )
                continue

            locked_id = str(chosen_row.id)
            locked_ids.add(locked_id)
            try:
                locked_result = build_team(
                    budget=budget,
                    data_path=data_path,
                    models_dir=models_dir,
                    max_per_team=max_per_team,
                    formation=formation,
                    method="ilp",
                    pred_mode="models",
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                )
            except ValueError as exc:
                locked_ids.clear()
                locked_ids.update(prev_locked)
                banned_ids.clear()
                banned_ids.update(prev_banned)
                if verbose:
                    print(f"Locking failed, reverting: {exc}")
                else:
                    print(
                        "No valid team found under the current budget/formation with the selected constraints. "
                        "Undoing the last change."
                    )
                return None

            locked_in_team = any(
                str(player["id"]) == locked_id for player in locked_result["players"]
            )
            if not locked_in_team:
                locked_ids.clear()
                locked_ids.update(prev_locked)
                print(
                    f"Could not lock {chosen_row.name} into the team under current constraints. "
                    "Choose another player."
                )
                continue

            return locked_result

    return new_result


def _interactive_menu(args: argparse.Namespace) -> None:
    state = {
        "budget": 100.0,
        "formation": args.formation,
        "mode": None,
        "bench_mode": "auto",
    }
    locked_ids: set[str] = set()
    banned_ids: set[str] = set()
    current_result: dict | None = None

    while True:
        if current_result is None:
            print("\nMain Menu")
            menu = {0: "Exit", 1: "Create team"}
            _print_menu(menu)
            choice = _read_menu_choice(menu)
            if choice is None:
                continue

            if choice == 0:
                return

            state["budget"] = _prompt_float("Enter budget (e.g., 100.0): ", 1.0)
            state["formation"] = _select_formation(state["formation"])

            build_menu = {
                1: "Build Starting XI then Bench (4)",
                2: "Build Full Squad (15) in one solve",
            }
            print("\nBuild mode:")
            _print_menu(build_menu)
            build_choice = _read_menu_choice(build_menu)
            if build_choice is None:
                continue

            try:
                if build_choice == 1:
                    state["mode"] = "xi_then_bench"
                    current_result = build_team(
                        budget=state["budget"],
                        data_path=args.data,
                        models_dir=args.models_dir,
                        max_per_team=args.max_per_team,
                        formation=state["formation"],
                        method="ilp",
                        pred_mode="models",
                        locked_ids=locked_ids,
                        banned_ids=banned_ids,
                    )
                    current_result["mode"] = state["mode"]
                    current_result["starters"] = current_result["players"]

                    bench_menu = {1: "Manual (pick players)", 2: "Auto (optimize)"}
                    print("\nBuild bench:")
                    _print_menu(bench_menu)
                    bench_choice = _read_menu_choice(bench_menu)
                    if bench_choice is None:
                        continue

                    if bench_choice == 2:
                        state["bench_mode"] = "auto"
                        bench = _build_bench_auto(
                            xi_result=current_result,
                            data_path=args.data,
                            models_dir=args.models_dir,
                            budget=state["budget"],
                            max_per_team=args.max_per_team,
                            banned_ids=banned_ids,
                        )
                    else:
                        state["bench_mode"] = "manual"
                        bench = _build_bench_manual(
                            xi_result=current_result,
                            data_path=args.data,
                            models_dir=args.models_dir,
                            budget=state["budget"],
                            max_per_team=args.max_per_team,
                            banned_ids=banned_ids,
                        )
                        if isinstance(bench, dict) and bench.get("_action") == "auto":
                            state["bench_mode"] = "auto"
                            bench = _build_bench_auto(
                                xi_result=current_result,
                                data_path=args.data,
                                models_dir=args.models_dir,
                                budget=state["budget"],
                                max_per_team=args.max_per_team,
                                banned_ids=banned_ids,
                            )
                        if bench is None:
                            current_result = None
                            continue

                    current_result["bench"] = bench
                    current_result["starters"] = current_result["players"]
                    _update_remaining_budget(current_result, state["budget"])
                    _require_bench(current_result)
                    print_squad(current_result, state["budget"])
                    _print_verbose(current_result, args.quiet, args.verbose)
                else:
                    state["mode"] = "full_15"
                    current_result = _build_full_squad(
                        data_path=args.data,
                        models_dir=args.models_dir,
                        budget=state["budget"],
                        max_per_team=args.max_per_team,
                        formation=state["formation"],
                        banned_ids=banned_ids,
                    )
                    current_result["mode"] = state["mode"]
                    _require_bench(current_result)
                    print_squad(current_result, state["budget"])
                    _print_verbose(current_result, args.quiet, args.verbose)

            except ValueError as exc:
                print(f"No valid team found: {exc}")
                current_result = None
                continue
            continue

        print("\nMain Menu")
        menu = {
            0: "Exit",
            1: "Replace a player",
            2: "Change formation & regenerate",
            3: "Change budget & regenerate",
            4: "Show explanations",
            5: "Save outputs",
        }
        _print_menu(menu)
        choice = _read_menu_choice(menu)
        if choice is None:
            continue

        if choice == 0:
            return
        if choice == 1:
            has_bench = bool(current_result.get("bench")) and len(current_result["bench"]) == 4
            if has_bench:
                new_result = _replace_with_bench_menu(
                    result=current_result,
                    data_path=args.data,
                    models_dir=args.models_dir,
                    formation=state["formation"],
                    budget=state["budget"],
                    max_per_team=args.max_per_team,
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                    verbose=args.verbose,
                )
            else:
                new_result = _pick_replacement(
                    result=current_result,
                    data_path=args.data,
                    models_dir=args.models_dir,
                    formation=state["formation"],
                    budget=state["budget"],
                    max_per_team=args.max_per_team,
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                    verbose=args.verbose,
                )
            if new_result is None:
                continue
            current_result = new_result
            current_result["mode"] = state["mode"]
            current_result["starters"] = current_result.get("players", current_result.get("starters", []))
            _update_remaining_budget(current_result, state["budget"])
            _require_bench(current_result)
            print_squad(current_result, state["budget"])
            _print_verbose(current_result, args.quiet, args.verbose)
        elif choice == 2:
            state["formation"] = _select_formation(state["formation"])
            try:
                if state["mode"] == "full_15":
                    new_result = _build_full_squad(
                        data_path=args.data,
                        models_dir=args.models_dir,
                        budget=state["budget"],
                        max_per_team=args.max_per_team,
                        formation=state["formation"],
                        banned_ids=banned_ids,
                    )
                else:
                    bench = current_result.get("bench", [])
                    if bench:
                        bench_menu = {
                            1: "Auto rebuild bench",
                            2: "Keep current bench (lock)",
                        }
                        print("\nBench after regenerate:")
                        _print_menu(bench_menu)
                        bench_choice = _read_menu_choice(bench_menu)
                        if bench_choice is None:
                            continue
                        if bench_choice == 2:
                            new_result = _solve_starters_with_fixed_bench(
                                data_path=args.data,
                                models_dir=args.models_dir,
                                formation=state["formation"],
                                budget=state["budget"],
                                max_per_team=args.max_per_team,
                                bench=bench,
                                banned_ids=banned_ids,
                            )
                            new_result["bench"] = bench
                        else:
                            new_result = build_team(
                                budget=state["budget"],
                                data_path=args.data,
                                models_dir=args.models_dir,
                                max_per_team=args.max_per_team,
                                formation=state["formation"],
                                method="ilp",
                                pred_mode="models",
                                locked_ids=locked_ids,
                                banned_ids=banned_ids,
                            )
                            new_bench = _build_bench_auto(
                                xi_result=new_result,
                                data_path=args.data,
                                models_dir=args.models_dir,
                                budget=state["budget"],
                                max_per_team=args.max_per_team,
                                banned_ids=banned_ids,
                            )
                            new_result["bench"] = new_bench
                    else:
                        new_result = build_team(
                            budget=state["budget"],
                            data_path=args.data,
                            models_dir=args.models_dir,
                            max_per_team=args.max_per_team,
                            formation=state["formation"],
                            method="ilp",
                            pred_mode="models",
                            locked_ids=locked_ids,
                            banned_ids=banned_ids,
                        )
            except ValueError as exc:
                print(f"No valid team found under the new formation: {exc}")
                continue
            current_result = new_result
            current_result["mode"] = state["mode"]
            current_result["starters"] = current_result.get("players", current_result.get("starters", []))
            _update_remaining_budget(current_result, state["budget"])
            _require_bench(current_result)
            print_squad(current_result, state["budget"])
            _print_verbose(current_result, args.quiet, args.verbose)
        elif choice == 3:
            state["budget"] = _prompt_float("Enter budget (e.g., 100.0): ", 1.0)
            try:
                if state["mode"] == "full_15":
                    current_result = _build_full_squad(
                        data_path=args.data,
                        models_dir=args.models_dir,
                        budget=state["budget"],
                        max_per_team=args.max_per_team,
                        formation=state["formation"],
                        banned_ids=banned_ids,
                    )
                else:
                    bench = current_result.get("bench", [])
                    if bench:
                        bench_menu = {
                            1: "Auto rebuild bench",
                            2: "Keep current bench (lock)",
                        }
                        print("\nBench after regenerate:")
                        _print_menu(bench_menu)
                        bench_choice = _read_menu_choice(bench_menu)
                        if bench_choice is None:
                            continue
                        if bench_choice == 2:
                            current_result = _solve_starters_with_fixed_bench(
                                data_path=args.data,
                                models_dir=args.models_dir,
                                formation=state["formation"],
                                budget=state["budget"],
                                max_per_team=args.max_per_team,
                                bench=bench,
                                banned_ids=banned_ids,
                            )
                            current_result["bench"] = bench
                        else:
                            current_result = build_team(
                                budget=state["budget"],
                                data_path=args.data,
                                models_dir=args.models_dir,
                                max_per_team=args.max_per_team,
                                formation=state["formation"],
                                method="ilp",
                                pred_mode="models",
                                locked_ids=locked_ids,
                                banned_ids=banned_ids,
                            )
                            new_bench = _build_bench_auto(
                                xi_result=current_result,
                                data_path=args.data,
                                models_dir=args.models_dir,
                                budget=state["budget"],
                                max_per_team=args.max_per_team,
                                banned_ids=banned_ids,
                            )
                            current_result["bench"] = new_bench
                    else:
                        current_result = build_team(
                            budget=state["budget"],
                            data_path=args.data,
                            models_dir=args.models_dir,
                            max_per_team=args.max_per_team,
                            formation=state["formation"],
                            method="ilp",
                            pred_mode="models",
                            locked_ids=locked_ids,
                            banned_ids=banned_ids,
                        )
            except ValueError as exc:
                print(f"No valid team found under the new budget: {exc}")
                continue
            current_result["mode"] = state["mode"]
            current_result["starters"] = current_result.get("players", current_result.get("starters", []))
            _update_remaining_budget(current_result, state["budget"])
            _require_bench(current_result)
            print_squad(current_result, state["budget"])
            _print_verbose(current_result, args.quiet, args.verbose)
        elif choice == 4:
            explanations_menu(
                current_result,
                args,
                data_path=args.data,
                models_dir=args.models_dir,
                top_k=5,
            )
        elif choice == 5:
            timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            outputs_dir = ROOT_DIR / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            csv_path = outputs_dir / f"team_{timestamp}.csv"
            json_path = outputs_dir / f"team_{timestamp}.json"
            _require_bench(current_result)
            starters = current_result.get("starters", current_result.get("players", []))
            bench = current_result.get("bench", [])
            total_cost = current_result["total_cost"]
            remaining = current_result.get("remaining_budget")
            if remaining is None:
                remaining = state["budget"] - total_cost
            total_pred_starters = sum(player["pred"] for player in starters)
            total_pred_bench = sum(player["pred"] for player in bench) if bench else 0.0
            mode = _get_mode(current_result)
            _ensure_explanations_cache(
                current_result,
                data_path=args.data,
                models_dir=args.models_dir,
            )

            header = ["position", "name", "team", "cost", "pred", "id"]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["STARTING_XI"])
                writer.writerow(header)
                for player in starters:
                    writer.writerow([
                        player["position"],
                        player["name"],
                        player["team"],
                        player["cost"],
                        player["pred"],
                        player["id"],
                    ])
                if bench:
                    writer.writerow([])
                    writer.writerow(["BENCH"])
                    writer.writerow(header)
                    for player in bench:
                        writer.writerow([
                            player["position"],
                            player["name"],
                            player["team"],
                            player["cost"],
                            player["pred"],
                            player["id"],
                        ])
                writer.writerow([])
                writer.writerow(["TOTALS"])
                writer.writerow(["formation", current_result.get("formation", "N/A")])
                writer.writerow(["mode", mode])
                writer.writerow(["total_cost", f"{total_cost:.1f}"])
                writer.writerow(["remaining_budget", f"{remaining:.1f}"])
                writer.writerow(["total_pred_starters", f"{total_pred_starters:.2f}"])
                writer.writerow(["total_pred_bench", f"{total_pred_bench:.2f}"])
                writer.writerow(
                    ["total_pred_squad", f"{(total_pred_starters + total_pred_bench):.2f}"]
                )
                writer.writerow([])
                writer.writerow(["EXPLANATIONS"])
                writer.writerow(
                    ["player_id", "name", "position", "team", "role", "rank", "feature", "value", "score", "why"]
                )
                combined = [(p, "STARTER") for p in starters] + [(p, "BENCH") for p in bench]
                for player, role in combined:
                    items = current_result["explanations_cache"].get(str(player["id"]), [])
                    for idx, item in enumerate(items, start=1):
                        writer.writerow(
                            [
                                player["id"],
                                player["name"],
                                player["position"],
                                player["team"],
                                role,
                                idx,
                                item.get("feature"),
                                item.get("value"),
                                item.get("score"),
                                item.get("why"),
                            ]
                        )

            payload = dict(current_result)
            payload["starters"] = starters
            payload["bench"] = bench
            payload["remaining_budget"] = remaining
            payload["mode"] = mode
            payload["explanations"] = current_result.get("explanations_cache", {})
            _write_json(str(json_path), payload)

            print(f"Saved: outputs/team_{timestamp}.csv")
            print(f"Saved: outputs/team_{timestamp}.json")


def _write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(value: object) -> object:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        np = None
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_csv(path: str, players: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["position", "name", "team", "cost", "pred", "id"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for player in players:
            writer.writerow({
                "position": player["position"],
                "name": player["name"],
                "team": player["team"],
                "cost": player["cost"],
                "pred": player["pred"],
                "id": player["id"],
            })




def main() -> None:
    args = parse_args()
    if args.budget is None:
        _interactive_menu(args)
        return

    result = build_team(
        budget=args.budget,
        data_path=args.data,
        models_dir=args.models_dir,
        max_per_team=args.max_per_team,
        formation=args.formation,
        method="ilp",
        pred_mode="models",
    )

    print_squad(result, args.budget)

    if args.out_json:
        _write_json(args.out_json, result)
    if args.out_csv:
        _write_csv(args.out_csv, result["players"])

    _print_verbose(result, args.quiet, args.verbose)


if __name__ == "__main__":
    main()
