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
    except (FPLAPIError, ConnectionError, TimeoutError, OSError, ValueError):
        logger.warning("FPL API unavailable — falling back to season predictions")
        fpl_players = None

    # Build lookups for name matching
    fpl_by_name: dict[str, list[dict]] = {}
    fpl_list: list[dict] = []
    if fpl_players is not None:
        fpl_list = list(fpl_players.values())
        for fp in fpl_list:
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

        matched = _match_fpl_player(player, fpl_by_name, fpl_list)
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


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents-ish chars, remove dots/hyphens for fuzzy compare."""
    return name.lower().replace(".", " ").replace("-", " ").strip()


def _word_boundary_match(needle: str, haystack: str) -> bool:
    """Check if needle appears in haystack as a whole word (not mid-word)."""
    haystack_words = haystack.replace("-", " ").split()
    needle_words = needle.replace("-", " ").split()
    # Multi-word web_names (e.g. "de bruyne"): check all words present
    if len(needle_words) > 1:
        return all(nw in haystack_words for nw in needle_words)
    # Single word: must be an exact word match in the haystack
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

    # Strategy 1: web_name as whole-word match in squad name
    candidates: list[dict] = []
    for web_name_lower, fps in fpl_by_name.items():
        if _word_boundary_match(web_name_lower, squad_name_lower):
            candidates.extend(fps)

    if candidates:
        return _pick_best(candidates, squad_position)

    # Strategy 2 & 3: match against FPL full name
    # Extract squad player's last name (last word)
    squad_parts = squad_name_lower.split()
    squad_last = squad_parts[-1] if squad_parts else ""

    for fp in fpl_list:
        fpl_second = fp.get("second_name", "").lower()
        fpl_full = f"{fp.get('first_name', '')} {fp.get('second_name', '')}".lower().strip()

        # Strategy 2: squad last name matches FPL second_name
        if squad_last and len(squad_last) > 2 and squad_last in fpl_second:
            candidates.append(fp)
            continue

        # Strategy 3: squad full name is a substring of FPL full name (or vice versa)
        if squad_name_lower in fpl_full or fpl_full in squad_name_lower:
            candidates.append(fp)
            continue

    if candidates:
        return _pick_best(candidates, squad_position)

    # Strategy 4: normalized token overlap (handles dots, hyphens)
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
