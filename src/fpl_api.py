"""FPL API wrapper — fetches player form, fixture difficulty, and gameweek data.

Caches responses in memory with a configurable TTL (default 5 minutes).
Uses only stdlib (urllib.request) for HTTP.
"""

import json
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Optional

from src.team_builder import POSITION_MAP

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class FPLAPIError(Exception):
    """Raised when the FPL API is unreachable, returns non-200, or bad JSON."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://fantasy.premierleague.com/api"
_BOOTSTRAP_URL = f"{_BASE_URL}/bootstrap-static/"
_FIXTURES_URL = f"{_BASE_URL}/fixtures/"
_USER_AGENT = "AutomaticChampion/1.0"
_TIMEOUT = 10  # seconds
_DEFAULT_TTL = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}  # key -> (timestamp, data)
_cache_ttl: int = _DEFAULT_TTL
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Invalidate all cached API responses."""
    with _cache_lock:
        _cache.clear()


def _get_cached(key: str) -> Optional[object]:
    with _cache_lock:
        if key in _cache:
            ts, data = _cache[key]
            if time.time() - ts < _cache_ttl:
                return data
            del _cache[key]
        return None


def _set_cached(key: str, data: object) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), data)

# ---------------------------------------------------------------------------
# Raw HTTP helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str) -> object:
    """Fetch a URL and return parsed JSON. Raises FPLAPIError on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                raise FPLAPIError(f"FPL API returned status {resp.status} for {url}")
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise FPLAPIError(f"FPL API HTTP error {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise FPLAPIError(f"FPL API unreachable: {exc}") from exc

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise FPLAPIError(f"Malformed JSON from {url}: {exc}") from exc

# ---------------------------------------------------------------------------
# Core fetchers (cached)
# ---------------------------------------------------------------------------

def fetch_bootstrap() -> dict:
    """Fetch and cache the bootstrap-static endpoint."""
    cached = _get_cached("bootstrap")
    if cached is not None:
        return cached
    data = _fetch_json(_BOOTSTRAP_URL)
    _set_cached("bootstrap", data)
    return data


def fetch_fixtures() -> list[dict]:
    """Fetch and cache the fixtures endpoint."""
    cached = _get_cached("fixtures")
    if cached is not None:
        return cached
    data = _fetch_json(_FIXTURES_URL)
    _set_cached("fixtures", data)
    return data

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def get_current_gameweek() -> int:
    """Return the current gameweek number.

    Uses ``is_current``; falls back to ``is_next`` if no GW is current.
    """
    bootstrap = fetch_bootstrap()
    events = bootstrap.get("events", [])
    for ev in events:
        if ev.get("is_current"):
            return ev["id"]
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    raise FPLAPIError("Could not determine current gameweek from API data")


def get_player_data() -> dict[int, dict]:
    """Return a dict mapping player ID → player info dict."""
    bootstrap = fetch_bootstrap()
    players: dict[int, dict] = {}
    for p in bootstrap.get("elements", []):
        try:
            pid = p["id"]
            players[pid] = {
                "id": pid,
                "web_name": p["web_name"],
                "first_name": p.get("first_name", ""),
                "second_name": p.get("second_name", ""),
                "position": POSITION_MAP.get(p["element_type"], "UNK"),
                "team_id": p["team"],
                "form": float(p["form"]) if p.get("form") is not None else 0.0,
                "points_per_game": float(p["points_per_game"]) if p.get("points_per_game") is not None else 0.0,
                "now_cost": p["now_cost"],
                "ep_next": float(p["ep_next"]) if p.get("ep_next") is not None else None,
                "status": p.get("status", ""),
                "chance_of_playing": p.get("chance_of_playing_next_round"),
                "total_points": p.get("total_points", 0),
                "minutes": p.get("minutes", 0),
            }
        except KeyError as exc:
            print(f"Warning: skipping malformed player entry (missing {exc})", file=sys.stderr)
            continue
    return players


def get_team_data() -> dict[int, dict]:
    """Return a dict mapping team ID → team info dict."""
    bootstrap = fetch_bootstrap()
    teams: dict[int, dict] = {}
    for t in bootstrap.get("teams", []):
        tid = t["id"]
        teams[tid] = {
            "id": tid,
            "name": t["name"],
            "short_name": t["short_name"],
            "strength": {
                "attack_home": t["strength_attack_home"],
                "attack_away": t["strength_attack_away"],
                "defence_home": t["strength_defence_home"],
                "defence_away": t["strength_defence_away"],
                "overall_home": t["strength_overall_home"],
                "overall_away": t["strength_overall_away"],
            },
        }
    return teams


def get_fixtures_for_gameweek(gameweek: int) -> list[dict]:
    """Return fixtures for a specific gameweek."""
    fixtures = fetch_fixtures()
    result = []
    for f in fixtures:
        if f.get("event") == gameweek:
            result.append({
                "id": f["id"],
                "home_team_id": f["team_h"],
                "away_team_id": f["team_a"],
                "home_difficulty": f["team_h_difficulty"],
                "away_difficulty": f["team_a_difficulty"],
                "finished": f.get("finished", False),
            })
    return result


def get_player_fixtures(player_team_id: int, gameweek: int) -> list[dict]:
    """Return fixture(s) for a specific team in a specific gameweek.

    Each dict: ``opponent_team_id``, ``is_home`` (bool), ``difficulty`` (int 1-5).
    """
    gw_fixtures = get_fixtures_for_gameweek(gameweek)
    result = []
    for f in gw_fixtures:
        if f["home_team_id"] == player_team_id:
            result.append({
                "opponent_team_id": f["away_team_id"],
                "is_home": True,
                "difficulty": f["home_difficulty"],
            })
        elif f["away_team_id"] == player_team_id:
            result.append({
                "opponent_team_id": f["home_team_id"],
                "is_home": False,
                "difficulty": f["away_difficulty"],
            })
    return result
