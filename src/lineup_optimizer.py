"""Lineup optimizer — pick the best starting 11 from a 15-player FPL squad.

Uses Integer Linear Programming (OR-Tools) to maximise predicted gameweek
points subject to FPL formation rules.  Also selects captain, vice-captain,
and determines bench order.
"""

from __future__ import annotations

from src.team_builder import (
    FORMATION_COUNTS,
    FULL_SQUAD_COUNTS,
    POSITION_ORDER,
    _build_solver,
)


def optimize_lineup(
    squad: list[dict],
    gw_predictions: dict[str, float],
    formation: str | None = None,
) -> dict:
    """Select the optimal starting 11, captain, vice-captain, and bench order.

    Parameters
    ----------
    squad : list[dict]
        Exactly 15 player dicts, each with ``"id"``, ``"name"``, ``"position"``,
        ``"team"``, ``"cost"``, ``"pred"`` keys.
    gw_predictions : dict[str, float]
        Mapping of player ``"id"`` (str) → predicted gameweek points.
    formation : str | None
        e.g. ``"4-3-3"``.  If *None*, tries all 7 valid formations and picks
        the one that maximises total GW points.

    Returns
    -------
    dict
        Result with ``starters``, ``bench``, ``captain_id``,
        ``vice_captain_id``, ``total_gw_points``, ``formation``.
    """
    _validate_squad(squad)

    if formation is not None:
        if formation not in FORMATION_COUNTS:
            raise ValueError(f"Unsupported formation: {formation}")
        return _solve_for_formation(squad, gw_predictions, formation)

    # Try every formation, keep the best
    best: dict | None = None
    for f in FORMATION_COUNTS:
        result = _solve_for_formation(squad, gw_predictions, f)
        if best is None or result["total_gw_points"] > best["total_gw_points"]:
            best = result
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_squad(squad: list[dict]) -> None:
    if len(squad) != 15:
        raise ValueError(
            f"Squad must have exactly 15 players, got {len(squad)}"
        )

    pos_counts: dict[str, int] = {}
    for p in squad:
        pos = p["position"]
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    for pos, required in FULL_SQUAD_COUNTS.items():
        actual = pos_counts.get(pos, 0)
        if actual != required:
            raise ValueError(
                f"Squad must have exactly {required} {pos}, got {actual}"
            )


# ---------------------------------------------------------------------------
# ILP solve for a single formation
# ---------------------------------------------------------------------------

def _solve_for_formation(
    squad: list[dict],
    gw_predictions: dict[str, float],
    formation: str,
) -> dict:
    required = FORMATION_COUNTS[formation]
    solver = _build_solver()

    # Decision variables — one per player
    x = {}
    for i, player in enumerate(squad):
        x[i] = solver.BoolVar(f"x_{i}")

    # Points lookup (default 0 if missing)
    pts = [gw_predictions.get(str(p["id"]), 0.0) for p in squad]

    # Exactly 11 starters
    solver.Add(solver.Sum(x[i] for i in range(15)) == 11)

    # Position constraints
    for pos, count in required.items():
        solver.Add(
            solver.Sum(
                x[i] for i, p in enumerate(squad) if p["position"] == pos
            )
            == count
        )

    # Maximise predicted GW points
    solver.Maximize(solver.Sum(x[i] * pts[i] for i in range(15)))

    status = solver.Solve()
    from ortools.linear_solver import pywraplp

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError(f"No feasible lineup for formation {formation}")

    # Partition into starters / bench
    starter_indices = [i for i in range(15) if x[i].solution_value() > 0.5]
    bench_indices = [i for i in range(15) if x[i].solution_value() <= 0.5]

    # Captain = starter with highest predicted points
    starter_indices_sorted = sorted(starter_indices, key=lambda i: pts[i], reverse=True)
    captain_idx = starter_indices_sorted[0]
    vice_captain_idx = starter_indices_sorted[1]

    captain_id = str(squad[captain_idx]["id"])
    vice_captain_id = str(squad[vice_captain_idx]["id"])

    # Total points: sum of starters + captain bonus (captain counted twice)
    total = sum(pts[i] for i in starter_indices) + pts[captain_idx]

    # Sort starters by position then name
    starters_sorted = sorted(
        starter_indices,
        key=lambda i: (POSITION_ORDER.index(squad[i]["position"]), squad[i]["name"]),
    )

    starters = []
    for i in starters_sorted:
        p = squad[i]
        starters.append({
            "id": str(p["id"]),
            "name": p["name"],
            "position": p["position"],
            "team": p["team"],
            "gw_points": pts[i],
            "is_captain": i == captain_idx,
            "is_vice_captain": i == vice_captain_idx,
        })

    # Bench order: GK always last (position 4), others by predicted pts desc
    bench_gk = [i for i in bench_indices if squad[i]["position"] == "GK"]
    bench_outfield = [i for i in bench_indices if squad[i]["position"] != "GK"]
    bench_outfield_sorted = sorted(bench_outfield, key=lambda i: pts[i], reverse=True)
    bench_ordered = bench_outfield_sorted + bench_gk

    bench = []
    for order, i in enumerate(bench_ordered, start=1):
        p = squad[i]
        bench.append({
            "id": str(p["id"]),
            "name": p["name"],
            "position": p["position"],
            "team": p["team"],
            "gw_points": pts[i],
            "bench_order": order,
        })

    return {
        "formation": formation,
        "starters": starters,
        "bench": bench,
        "captain_id": captain_id,
        "vice_captain_id": vice_captain_id,
        "total_gw_points": total,
    }
