from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.team_builder import build_team, score_players  # noqa: E402


POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
FORMATIONS = ["4-3-3", "4-4-2", "3-4-3", "3-5-2", "4-5-1", "5-3-2", "5-4-1"]


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

    print("\nTotals:")
    print(f"  total_cost={result['total_cost']:.1f}")
    print(f"  total_pred={result['total_pred']:.2f}")


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
        print(f"Invalid choice. Options: {', '.join(sorted(choices))}")


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
        else:
            print("Invalid selection. Please choose one of the numbers shown.")


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
    while True:
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
        return menu[choice]


def _team_ids(result: dict) -> set[str]:
    return {str(player["id"]) for player in result["players"]}


def _team_counts(result: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in result["players"]:
        team = player["team"]
        counts[team] = counts.get(team, 0) + 1
    return counts


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
) -> dict | None:
    grouped = _display_team_for_selection(result)
    print("\nPositions:")
    pos_menu = {idx: pos for idx, pos in enumerate(POSITION_ORDER, start=1)}
    _print_menu(pos_menu)
    pos_choice = pos_menu[_read_menu_choice(pos_menu)]
    players_in_pos = grouped[pos_choice]
    if not players_in_pos:
        print("No players in that position.")
        return None

    replace_menu = {idx: player for idx, player in enumerate(players_in_pos, start=1)}
    idx_choice = _read_menu_choice(replace_menu)
    selected = players_in_pos[idx_choice - 1]
    selected_id = str(selected["id"])

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
        candidates = candidates[~candidates["id"].astype(str).isin(banned_ids)]
        candidates = candidates[~candidates["id"].astype(str).isin(current_ids)]
        team_counts = _team_counts(result)
        candidates = candidates[
            candidates["team"].map(lambda team: team_counts.get(team, 0) < max_per_team)
        ]
        candidates = candidates.sort_values("pred", ascending=False).head(10)

        if candidates.empty:
            print("No candidates available for that position.")
            return new_result

        print("\nTop candidates:")
        candidate_rows = list(candidates.itertuples(index=False))
        candidate_menu: dict[int, object] = {0: "Skip locking a replacement"}
        for idx, row in enumerate(candidate_rows, start=1):
            candidate_menu[idx] = row
            print(
                f"  [{idx}] {row.name:<25} {row.team:<20} "
                f"cost={row.cost:.1f} pred={row.pred:.2f}"
            )
        print("  [0] Skip locking a replacement")

        while True:
            lock_choice = _read_menu_choice(candidate_menu)
            if lock_choice == 0:
                return new_result

            chosen_row = candidate_menu[lock_choice]
            team = chosen_row.team
            if team_counts.get(team, 0) >= max_per_team:
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
    budget = 100.0
    formation = args.formation
    locked_ids: set[str] = set()
    banned_ids: set[str] = set()
    current_result: dict | None = None

    while True:
        if current_result is None:
            print("\nMain Menu")
            menu = {0: "Exit", 1: "Create team"}
            _print_menu(menu)
            choice = _read_menu_choice(menu)

            if choice == 0:
                return

            budget = _prompt_float("Enter budget (e.g., 100.0): ", 1.0)
            formation = _select_formation(formation)
            try:
                current_result = build_team(
                    budget=budget,
                    data_path=args.data,
                    models_dir=args.models_dir,
                    max_per_team=args.max_per_team,
                    formation=formation,
                    method="ilp",
                    pred_mode="models",
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                )
            except ValueError as exc:
                print(f"No valid team found: {exc}")
                current_result = None
                continue
            _print_team(current_result)
            _print_verbose(current_result, args.quiet, args.verbose)
            continue

        print("\nMain Menu")
        menu = {
            0: "Exit",
            1: "Replace a player",
            2: "Change formation & regenerate",
            3: "Change budget & regenerate",
            4: "Save outputs",
        }
        _print_menu(menu)
        choice = _read_menu_choice(menu)

        if choice == 0:
            return
        if choice == 1:
            new_result = _pick_replacement(
                result=current_result,
                data_path=args.data,
                models_dir=args.models_dir,
                formation=formation,
                budget=budget,
                max_per_team=args.max_per_team,
                locked_ids=locked_ids,
                banned_ids=banned_ids,
                verbose=args.verbose,
            )
            if new_result is None:
                continue
            current_result = new_result
            _print_team(current_result)
            _print_verbose(current_result, args.quiet, args.verbose)
        elif choice == 2:
            new_formation = _select_formation(formation)
            try:
                new_result = build_team(
                    budget=budget,
                    data_path=args.data,
                    models_dir=args.models_dir,
                    max_per_team=args.max_per_team,
                    formation=new_formation,
                    method="ilp",
                    pred_mode="models",
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                )
            except ValueError as exc:
                print(f"No valid team found under the new formation: {exc}")
                continue
            formation = new_formation
            current_result = new_result
            _print_team(current_result)
            _print_verbose(current_result, args.quiet, args.verbose)
        elif choice == 3:
            budget = _prompt_float("Enter budget (e.g., 100.0): ", 1.0)
            try:
                current_result = build_team(
                    budget=budget,
                    data_path=args.data,
                    models_dir=args.models_dir,
                    max_per_team=args.max_per_team,
                    formation=formation,
                    method="ilp",
                    pred_mode="models",
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                )
            except ValueError as exc:
                print(f"No valid team found under the new budget: {exc}")
                continue
            _print_team(current_result)
            _print_verbose(current_result, args.quiet, args.verbose)
        elif choice == 4:
            timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            outputs_dir = ROOT_DIR / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            csv_path = outputs_dir / f"team_{timestamp}.csv"
            json_path = outputs_dir / f"team_{timestamp}.json"
            _write_csv(str(csv_path), current_result["players"])
            _write_json(str(json_path), current_result)
            print(f"Saved to outputs/team_{timestamp}.csv and .json")


def _write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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

    _print_team(result)

    if args.out_json:
        _write_json(args.out_json, result)
    if args.out_csv:
        _write_csv(args.out_csv, result["players"])

    _print_verbose(result, args.quiet, args.verbose)


if __name__ == "__main__":
    main()
