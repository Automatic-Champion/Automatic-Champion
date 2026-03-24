"""Gameweek points predictor — placeholder implementation.

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
"""

from __future__ import annotations

import logging

from src.fpl_api import FPLAPIError, get_current_gameweek, get_player_data

logger = logging.getLogger(__name__)

PREDICTOR_VERSION = "placeholder-v1"


def predict_gameweek_points(
    squad: list[dict],
    gameweek: int | None = None,
) -> dict[str, float]:
    """Predict per-player gameweek points for every player in *squad*.

    Placeholder logic (in priority order):
    1. ``ep_next`` from the FPL API (official expected-points estimate).
    2. ``points_per_game`` from the FPL API.
    3. ``player["pred"] / 38.0`` (season prediction ÷ 38 gameweeks).

    If the FPL API is unreachable, all players fall back to option 3.
    """
    # ------------------------------------------------------------------
    # Fetch FPL data (gracefully degrade on failure)
    # ------------------------------------------------------------------
    fpl_players: dict[int, dict] | None = None
    try:
        if gameweek is None:
            gameweek = get_current_gameweek()
        fpl_players = get_player_data()
    except (FPLAPIError, Exception):
        logger.warning("FPL API unavailable — falling back to season predictions")
        fpl_players = None

    # Build a lookup: lowercase web_name → list of FPL player dicts
    fpl_by_name: dict[str, list[dict]] = {}
    if fpl_players is not None:
        for fp in fpl_players.values():
            key = fp["web_name"].lower()
            fpl_by_name.setdefault(key, []).append(fp)

    # ------------------------------------------------------------------
    # Predict for each squad player
    # ------------------------------------------------------------------
    predictions: dict[str, float] = {}
    for player in squad:
        pid = str(player["id"])
        fallback = player["pred"] / 38.0

        if fpl_players is None:
            predictions[pid] = fallback
            continue

        matched = _match_fpl_player(player, fpl_by_name)
        if matched is None:
            predictions[pid] = fallback
            continue

        # Priority: ep_next > points_per_game > fallback
        ep = matched.get("ep_next")
        if ep is not None:
            predictions[pid] = float(ep)
        elif matched.get("points_per_game", 0.0) > 0.0:
            predictions[pid] = float(matched["points_per_game"])
        else:
            predictions[pid] = fallback

    return predictions


def _match_fpl_player(
    squad_player: dict,
    fpl_by_name: dict[str, list[dict]],
) -> dict | None:
    """Match a squad player to an FPL API player by name (substring, case-insensitive).

    If multiple FPL players share the same web_name, prefer the one whose
    position matches the squad player's position.
    """
    squad_name_lower = squad_player["name"].lower()

    candidates: list[dict] = []
    for web_name_lower, fps in fpl_by_name.items():
        if web_name_lower in squad_name_lower:
            candidates.extend(fps)

    if not candidates:
        return None

    # Prefer same-position match
    for c in candidates:
        if c["position"] == squad_player["position"]:
            return c

    return candidates[0]
