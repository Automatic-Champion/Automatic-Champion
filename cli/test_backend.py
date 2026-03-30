#!/usr/bin/env python3
"""Automatic Champion — Interactive Backend Tester CLI.

Exercises both UC1 (squad generation) and UC2 (lineup recommendation)
end-to-end against the running FastAPI backend.

Usage:
    python cli/test_backend.py [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request

# ── ANSI helpers ──────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"

from src.team_builder import (
    FORMATION_COUNTS,
    FULL_SQUAD_COUNTS as REQUIRED,
    POSITION_MAP as POS_MAP,
    POSITION_ORDER,
)

POS_ORDER = {pos: i for i, pos in enumerate(POSITION_ORDER)}
FORMATIONS = list(FORMATION_COUNTS.keys())

# ── CSV loading ───────────────────────────────────────────────────────

def load_players(csv_path: str) -> list[dict]:
    """Load players from the merged CSV file."""
    players: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("id", "").strip()
            et = row.get("element_type", "").strip()
            if not pid or not et:
                continue
            element_type = int(et)
            price_raw = row.get("price_now", "0")
            price = float(price_raw) / 10.0 if float(price_raw) > 20 else float(price_raw)
            tp_raw = row.get("total_points", "0")
            total_points = float(tp_raw) if tp_raw else 0.0
            players.append({
                "id": int(pid),
                "first_name": row.get("first_name", ""),
                "second_name": row.get("second_name", ""),
                "name": f"{row.get('first_name', '')} {row.get('second_name', '')}".strip(),
                "team_name": row.get("team_name", ""),
                "position": POS_MAP.get(element_type, "??"),
                "price": price,
                "total_points": total_points,
            })
    return players


def search_players(players: list[dict], query: str, limit: int = 15) -> list[dict]:
    """Search players by name, team, or position keyword."""
    q = query.strip().upper()
    if q in ("GK", "DEF", "MID", "FWD"):
        matches = [p for p in players if p["position"] == q]
    else:
        ql = query.strip().lower()
        matches = [
            p for p in players
            if ql in p["name"].lower() or ql in p["team_name"].lower()
        ]
    matches.sort(key=lambda p: p["price"], reverse=True)
    return matches[:limit]


# ── HTTP helpers ──────────────────────────────────────────────────────

def api_post(base_url: str, path: str, body: dict) -> dict:
    """POST JSON to backend. Raises on error."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to {base_url} — is the backend running?\n  ({e.reason})"
        ) from e


# ── Display helpers ───────────────────────────────────────────────────

def print_header(text: str) -> None:
    print(f"\n{BOLD}═══ {text} ═══{RESET}")


def print_error(text: str) -> None:
    print(f"{RED}Error: {text}{RESET}")


def display_squad_player(p: dict, prefix: str = "  ", show_pred: bool = True) -> None:
    pos = p.get("position", "??")
    name = p.get("name", "?")
    team = p.get("team", "?")
    cost = p.get("cost", 0)
    pred = p.get("predicted_points", 0)
    line = f"{prefix}{pos:<4} {name} ({team})"
    line = f"{line:<50} £{cost:<6.1f}"
    if show_pred:
        line += f" pred={pred:.1f}"
    print(line)


def display_squad(squad_data: dict) -> None:
    """Display a full squad response from /squad/generate."""
    formation = squad_data.get("formation", "?")
    budget = squad_data.get("budget", 0)
    total_cost = squad_data.get("total_cost", 0)
    total_pred = squad_data.get("total_predicted_points", 0)

    print_header("Squad Built")
    print(f"Formation: {formation} | Budget: {budget} | Cost: {total_cost:.1f} | Predicted Pts: {total_pred:.1f}")

    starters = squad_data.get("players", [])
    bench = squad_data.get("bench", [])

    # Sort starters by position
    starters_sorted = sorted(starters, key=lambda p: POS_ORDER.get(p.get("position", ""), 9))

    print(f"\n{GREEN}{BOLD}STARTERS:{RESET}")
    for p in starters_sorted:
        display_squad_player(p, prefix=f"  {GREEN}")
    print(RESET, end="")

    print(f"\n{YELLOW}{BOLD}BENCH:{RESET}")
    for i, p in enumerate(bench, 1):
        display_squad_player(p, prefix=f"  {YELLOW}{i}. ")
    print(RESET, end="")

    # Explanations
    all_players = starters + bench
    has_explanations = any(p.get("explanations") for p in all_players)
    if has_explanations:
        print(f"\n{CYAN}{BOLD}EXPLANATIONS:{RESET}")
        for p in all_players:
            for exp in p.get("explanations", []):
                feature = exp.get("feature", "")
                text = exp.get("explanation", "")
                if text:
                    print(f"  {CYAN}{p['name']}: {feature} — {text}{RESET}")
    print()


def display_lineup(lineup_data: dict) -> None:
    """Display a lineup recommendation from /lineup/recommend."""
    formation = lineup_data.get("formation", "?")
    gw = lineup_data.get("gameweek", "?")
    total_pts = lineup_data.get("total_gw_points", 0)
    captain_id = lineup_data.get("captain_id", "")
    vc_id = lineup_data.get("vice_captain_id", "")

    print_header("Lineup Recommendation")
    print(f"Formation: {formation} | Gameweek: {gw} | Total GW Pts: {total_pts:.1f}")

    starters = lineup_data.get("starters", [])
    bench = lineup_data.get("bench", [])

    starters_sorted = sorted(starters, key=lambda p: POS_ORDER.get(p.get("position", ""), 9))

    print(f"\n{GREEN}{BOLD}STARTING XI:{RESET}")
    for p in starters_sorted:
        suffix = ""
        if p.get("is_captain"):
            suffix = f"  {BOLD}← CAPTAIN (2×){RESET}{GREEN}"
        elif p.get("is_vice_captain"):
            suffix = f"  {BOLD}← VICE-CAPTAIN{RESET}{GREEN}"
        gw_pts = p.get("gw_points", 0)
        name = p.get("name", "?")
        team = p.get("team", "?")
        pos = p.get("position", "??")
        line = f"  {GREEN}{pos:<4} {name} ({team})"
        line = f"{line:<55} gw={gw_pts:.1f}{suffix}"
        print(line)
    print(RESET, end="")

    print(f"\n{YELLOW}{BOLD}BENCH:{RESET}")
    bench_sorted = sorted(bench, key=lambda b: b.get("bench_order", 99))
    for b in bench_sorted:
        order = b.get("bench_order", "?")
        gw_pts = b.get("gw_points", 0)
        name = b.get("name", "?")
        pos = b.get("position", "??")
        print(f"  {YELLOW}{order}. {pos:<4} {name:<30} gw={gw_pts:.1f}{RESET}")

    # Captain info
    cap_name = next((s["name"] for s in starters if str(s.get("id")) == str(captain_id)), "?")
    cap_pts = next((s["gw_points"] for s in starters if str(s.get("id")) == str(captain_id)), 0)
    vc_name = next((s["name"] for s in starters if str(s.get("id")) == str(vc_id)), "?")
    vc_pts = next((s["gw_points"] for s in starters if str(s.get("id")) == str(vc_id)), 0)
    print(f"\n  Captain: {cap_name} ({cap_pts:.1f} × 2 = {cap_pts * 2:.1f} pts)")
    print(f"  Vice-Captain: {vc_name} ({vc_pts:.1f} pts)")

    # Explanations
    all_p = starters + bench
    has_explanations = any(p.get("explanations") for p in all_p)
    if has_explanations:
        print(f"\n{CYAN}{BOLD}EXPLANATIONS:{RESET}")
        for p in all_p:
            for exp in p.get("explanations", []):
                feature = exp.get("feature", "")
                text = exp.get("explanation", "")
                if text:
                    print(f"  {CYAN}{p['name']}: {feature} — {text}{RESET}")
    print()


# ── Player search UI ─────────────────────────────────────────────────

def show_player_results(results: list[dict], start: int = 1) -> None:
    """Print a numbered list of players."""
    if not results:
        print("  No players found.")
        return
    print(f"  {'#':<4} {'ID':<6} {'Name':<30} {'Team':<20} {'Pos':<5} {'Price'}")
    print(f"  {'─'*4} {'─'*6} {'─'*30} {'─'*20} {'─'*5} {'─'*6}")
    for i, p in enumerate(results, start):
        print(f"  {i:<4} {p['id']:<6} {p['name']:<30} {p['team_name']:<20} {p['position']:<5} £{p['price']:.1f}")


def player_search_pick(players: list[dict], prompt_label: str = "Search player") -> int | None:
    """Search-and-pick loop. Returns player ID or None."""
    while True:
        query = input(f"  {prompt_label} (or 'done'): ").strip()
        if query.lower() == "done":
            return None
        results = search_players(players, query)
        show_player_results(results)
        if not results:
            continue
        pick = input("  Pick # (or Enter to search again): ").strip()
        if not pick:
            continue
        try:
            idx = int(pick)
            if 1 <= idx <= len(results):
                return results[idx - 1]["id"]
            else:
                print_error("Invalid number.")
        except ValueError:
            print_error("Enter a number.")
    return None


def player_search_loop(players: list[dict], label: str) -> list[int]:
    """Collect multiple player IDs via search-and-pick."""
    ids: list[int] = []
    print(f"  Search and pick players to {label}. Type 'done' when finished.")
    while True:
        pid = player_search_pick(players, f"{label} player")
        if pid is None:
            break
        ids.append(pid)
        p = next((pl for pl in players if pl["id"] == pid), None)
        name = p["name"] if p else str(pid)
        print(f"  {GREEN}Added: {name} (ID {pid}). {len(ids)} selected so far.{RESET}")
    return ids


# ── Squad storage ────────────────────────────────────────────────────

class SquadStore:
    def __init__(self):
        self.current: dict | None = None  # raw API response
        self.saved: dict[str, dict] = {}  # name -> squad data

    def set_current(self, squad_data: dict) -> None:
        self.current = squad_data

    def save(self, name: str) -> None:
        if self.current:
            self.saved[name] = self.current

    def load(self, name: str) -> bool:
        if name in self.saved:
            self.current = self.saved[name]
            return True
        return False

    def get_lineup_squad(self) -> list[dict] | None:
        """Convert current squad to LineupPlayerInput format."""
        if not self.current:
            return None
        result = []
        for p in self.current.get("players", []) + self.current.get("bench", []):
            result.append({
                "id": str(p["id"]),
                "name": p["name"],
                "position": p["position"],
                "team": p["team"],
                "cost": p["cost"],
                "pred": p.get("predicted_points", 0),
            })
        return result


# ── Menu actions ──────────────────────────────────────────────────────

def action_build_squad(base_url: str, players: list[dict], store: SquadStore) -> None:
    """Option 1 — Build Squad (UC1)."""
    print_header("Build Squad (UC1)")

    # Budget
    budget_str = input(f"  Budget [{BOLD}100.0{RESET}]: ").strip()
    budget = float(budget_str) if budget_str else 100.0

    # Formation
    print(f"  Formations: {', '.join(FORMATIONS)}")
    while True:
        formation = input(f"  Formation [{BOLD}4-3-3{RESET}]: ").strip()
        if not formation:
            formation = "4-3-3"
            break
        if formation in FORMATIONS:
            break
        print_error(f"Invalid formation '{formation}'. Choose from: {', '.join(FORMATIONS)}")

    # Locked players
    locked_ids: list[int] = []
    lock = input("  Lock specific players? (y/N): ").strip().lower()
    if lock == "y":
        locked_ids = player_search_loop(players, "lock")

    # Banned players
    banned_ids: list[int] = []
    ban = input("  Ban specific players? (y/N): ").strip().lower()
    if ban == "y":
        banned_ids = player_search_loop(players, "ban")

    print(f"\n  {DIM}Calling POST /squad/generate ...{RESET}")
    body = {
        "budget": budget,
        "formation": formation,
        "locked_ids": locked_ids,
        "banned_ids": banned_ids,
    }

    try:
        result = api_post(base_url, "/squad/generate", body)
    except ConnectionError as e:
        print_error(str(e))
        return
    except RuntimeError as e:
        print_error(str(e))
        return

    store.set_current(result)
    display_squad(result)

    save = input("  Save this squad? (Y/n): ").strip().lower()
    if save != "n":
        name = input("  Squad name: ").strip() or "squad-1"
        store.save(name)
        print(f"  {GREEN}Saved as '{name}'.{RESET}")


def action_enter_squad(base_url: str, players: list[dict], store: SquadStore) -> None:
    """Option 2 — Enter My Squad Manually."""
    print_header("Enter My Squad Manually")
    print("  Build a 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD")
    print("  Commands: search term | 'undo' | 'done' | 'auto'")

    squad: list[dict] = []
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    total_cost = 0.0

    def show_progress():
        parts = [f"{pos}: {counts[pos]}/{REQUIRED[pos]}" for pos in ["GK", "DEF", "MID", "FWD"]]
        print(f"\n  {' | '.join(parts)} | Cost: £{total_cost:.1f}")

    show_progress()

    while True:
        query = input("\n  Search (or undo/done/auto): ").strip()
        if not query:
            continue

        if query.lower() == "undo":
            if squad:
                removed = squad.pop()
                counts[removed["position"]] -= 1
                total_cost -= removed["price"]
                print(f"  Removed: {removed['name']}")
                show_progress()
            else:
                print("  Nothing to undo.")
            continue

        if query.lower() == "done":
            errors = []
            for pos, req in REQUIRED.items():
                if counts[pos] != req:
                    errors.append(f"{pos}: have {counts[pos]}, need {req}")
            if errors:
                print_error("Squad incomplete: " + ", ".join(errors))
                continue
            break

        if query.lower() == "auto":
            # Fill remaining slots via optimizer with current picks as locked
            locked_ids = [p["id"] for p in squad]
            budget_str = input(f"  Budget for auto-fill [{BOLD}100.0{RESET}]: ").strip()
            remaining_budget = float(budget_str) if budget_str else 100.0
            print(f"  Formations: {', '.join(FORMATIONS)}")
            formation_str = input(f"  Formation for auto-fill [{BOLD}4-3-3{RESET}]: ").strip()
            auto_formation = formation_str if formation_str in FORMATIONS else "4-3-3"
            print(f"  {DIM}Auto-filling remaining slots via optimizer...{RESET}")
            body = {
                "budget": remaining_budget,
                "formation": auto_formation,
                "locked_ids": locked_ids,
                "banned_ids": [],
            }
            try:
                result = api_post(base_url, "/squad/generate", body)
                store.set_current(result)
                display_squad(result)
                save = input("  Save this squad? (Y/n): ").strip().lower()
                if save != "n":
                    name = input("  Squad name: ").strip() or "manual-squad"
                    store.save(name)
                    print(f"  {GREEN}Saved as '{name}'.{RESET}")
                return
            except (ConnectionError, RuntimeError) as e:
                print_error(str(e))
                continue

        # Search
        results = search_players(players, query)
        # Filter out already-picked players
        picked_ids = {p["id"] for p in squad}
        results = [r for r in results if r["id"] not in picked_ids]
        # Filter out positions that are full
        results = [r for r in results if counts[r["position"]] < REQUIRED[r["position"]]]

        show_player_results(results)
        if not results:
            continue

        pick = input("  Pick # (or Enter to skip): ").strip()
        if not pick:
            continue
        try:
            idx = int(pick)
            if 1 <= idx <= len(results):
                p = results[idx - 1]
                squad.append(p)
                counts[p["position"]] += 1
                total_cost += p["price"]
                print(f"  {GREEN}Added: {p['name']} ({p['position']}, £{p['price']:.1f}){RESET}")
                show_progress()
                if sum(counts.values()) == 15:
                    print(f"\n  {GREEN}Squad complete!{RESET}")
                    break
            else:
                print_error("Invalid number.")
        except ValueError:
            print_error("Enter a number.")

    # Split into starters (1 GK + 10 best outfield) and bench
    gks = [p for p in squad if p["position"] == "GK"]
    outfield = [p for p in squad if p["position"] != "GK"]
    outfield.sort(key=lambda p: p.get("total_points", 0), reverse=True)
    starters = gks[:1] + outfield[:10]
    bench = gks[1:] + outfield[10:]

    total_pred = sum(p.get("total_points", 0) for p in squad)

    # Build a squad-like response for storage
    squad_data = {
        "formation": "4-3-3",
        "budget": 100.0,
        "total_cost": total_cost,
        "total_predicted_points": total_pred,
        "players": [
            {
                "id": p["id"],
                "name": p["name"],
                "team": p["team_name"],
                "position": p["position"],
                "cost": p["price"],
                "predicted_points": p.get("total_points", 0),
                "is_starter": True,
                "explanations": [],
            }
            for p in starters
        ],
        "bench": [
            {
                "id": p["id"],
                "name": p["name"],
                "team": p["team_name"],
                "position": p["position"],
                "cost": p["price"],
                "predicted_points": p.get("total_points", 0),
                "is_starter": False,
                "bench_order": i + 1,
                "explanations": [],
            }
            for i, p in enumerate(bench)
        ],
    }
    store.set_current(squad_data)
    print_header("Your Squad")
    display_squad(squad_data)

    save = input("  Save this squad? (Y/n): ").strip().lower()
    if save != "n":
        name = input("  Squad name: ").strip() or "manual-squad"
        store.save(name)
        print(f"  {GREEN}Saved as '{name}'.{RESET}")


def action_recommend_lineup(base_url: str, store: SquadStore) -> None:
    """Option 3 — Recommend Lineup (UC2)."""
    print_header("Recommend Lineup (UC2)")

    lineup_squad = store.get_lineup_squad()
    if not lineup_squad:
        print_error("No squad loaded. Use option 1, 2, or 6 first.")
        return

    print(f"  Squad: {len(lineup_squad)} players loaded.")

    gw_str = input("  Gameweek (Enter for auto): ").strip()
    gameweek = int(gw_str) if gw_str else None

    formation_str = input("  Formation (Enter for auto-pick best): ").strip()
    formation = formation_str if formation_str in FORMATIONS else None

    print(f"\n  {DIM}Calling POST /lineup/recommend ...{RESET}")
    body: dict = {"squad": lineup_squad}
    if gameweek is not None:
        body["gameweek"] = gameweek
    if formation is not None:
        body["formation"] = formation

    try:
        result = api_post(base_url, "/lineup/recommend", body)
    except ConnectionError as e:
        print_error(str(e))
        return
    except RuntimeError as e:
        print_error(str(e))
        return

    display_lineup(result)


def action_browse_players(players: list[dict]) -> None:
    """Option 4 — Browse Players."""
    while True:
        print_header("Browse Players")
        query = input("  Search players (name, team, or position): ").strip()
        if not query:
            return
        results = search_players(players, query, limit=20)
        show_player_results(results)
        again = input("\n  Search again? (Y/n): ").strip().lower()
        if again == "n":
            return


def action_view_squad(store: SquadStore) -> None:
    """Option 5 — View Current Squad."""
    if not store.current:
        print("\n  No squad loaded.")
        return
    display_squad(store.current)


def action_saved_squads(store: SquadStore) -> None:
    """Option 6 — Saved Squads."""
    print_header("Saved Squads")
    if not store.saved:
        print("  No saved squads.")
        return
    names = list(store.saved.keys())
    for i, name in enumerate(names, 1):
        s = store.saved[name]
        n_players = len(s.get("players", [])) + len(s.get("bench", []))
        cost = s.get("total_cost", 0)
        print(f"  [{i}] {name} ({n_players} players, £{cost:.1f})")

    pick = input("\n  Load # (or Enter to cancel): ").strip()
    if not pick:
        return
    try:
        idx = int(pick)
        if 1 <= idx <= len(names):
            name = names[idx - 1]
            store.load(name)
            print(f"  {GREEN}Loaded '{name}' as current squad.{RESET}")
            display_squad(store.current)
        else:
            print_error("Invalid number.")
    except ValueError:
        print_error("Enter a number.")


def action_quick_test(base_url: str, store: SquadStore) -> None:
    """Option 7 — Quick Test (UC1 + UC2 with defaults)."""
    print_header("Quick Test — UC1 (defaults)")
    print(f"  {DIM}POST /squad/generate (budget=100, formation=4-3-3) ...{RESET}")

    try:
        squad_result = api_post(base_url, "/squad/generate", {
            "budget": 100.0,
            "formation": "4-3-3",
            "locked_ids": [],
            "banned_ids": [],
        })
    except (ConnectionError, RuntimeError) as e:
        print_error(str(e))
        return

    store.set_current(squad_result)
    display_squad(squad_result)

    # Now UC2
    print_header("Quick Test — UC2 (auto lineup)")
    lineup_squad = store.get_lineup_squad()
    if not lineup_squad:
        print_error("Failed to build lineup squad from UC1 result.")
        return

    print(f"  {DIM}POST /lineup/recommend (auto gameweek, auto formation) ...{RESET}")
    try:
        lineup_result = api_post(base_url, "/lineup/recommend", {
            "squad": lineup_squad,
        })
    except (ConnectionError, RuntimeError) as e:
        print_error(str(e))
        return

    display_lineup(lineup_result)


# ── Main menu ─────────────────────────────────────────────────────────

def main_menu() -> str:
    print(f"""
{BOLD}========================================
  Automatic Champion — Backend Tester
========================================{RESET}

  [1] Build Squad (UC1)
  [2] Enter My Squad Manually
  [3] Recommend Lineup (UC2)
  [4] Browse Players
  [5] View Current Squad
  [6] Saved Squads
  [7] Quick Test
  [0] Exit
""")
    return input("  Choose: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic Champion — Interactive Backend Tester"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    base_url = args.url

    # Find CSV relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "players_merged_2024-25.csv")

    if not os.path.exists(csv_path):
        print_error(f"CSV not found at {csv_path}")
        sys.exit(1)

    players = load_players(csv_path)
    print(f"  {DIM}Loaded {len(players)} players from CSV.{RESET}")

    store = SquadStore()

    while True:
        choice = main_menu()
        try:
            if choice == "1":
                action_build_squad(base_url, players, store)
            elif choice == "2":
                action_enter_squad(base_url, players, store)
            elif choice == "3":
                action_recommend_lineup(base_url, store)
            elif choice == "4":
                action_browse_players(players)
            elif choice == "5":
                action_view_squad(store)
            elif choice == "6":
                action_saved_squads(store)
            elif choice == "7":
                action_quick_test(base_url, store)
            elif choice == "0":
                print("  Goodbye!")
                break
            elif choice == "":
                continue
            else:
                print_error("Invalid choice.")
        except KeyboardInterrupt:
            print("\n  (interrupted)")
        except EOFError:
            print("\n  Goodbye!")
            break


if __name__ == "__main__":
    main()
