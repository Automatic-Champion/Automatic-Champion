"""Weekly (V8) per-player prediction explainer.

For each squad player this surfaces the top reasons the V8 weekly model
predicted their gameweek points, using the model's own *per-prediction* SHAP
contributions over the exact feature row the predictor scored (obtained via
``gameweek_predictor.get_feature_rows`` so the explanation can't drift from the
number on screen).

Output shape matches ``src.explainer.explain_squad`` so the API/UI treat season
and weekly explanations identically::

    {player_id: [{"text": str, "category": str}, ...]}

Design notes
------------
* The chosen gameweek drives the explanation by selecting which feature row is
  used (a different gameweek → a different row → different reasons).
* The feature table is the frozen 2024-25 snapshot, and ``opponent_team`` /
  ``was_home`` describe the row's own match — so the fixture reason is phrased
  truthfully ("Home/Away fixture against X") without asserting an upcoming
  schedule that may not match.
* Price/ownership features (``value``/``selected``/``transfers_balance``) tend to
  dominate SHAP for premium players, so at most one may appear in a player's
  reasons — letting genuine form/fixture/attacking signals win the other slots.
* All model-facing strings say "V8 model"; the underlying library is never named.
* The function never raises: a missing model/row degrades to a global-importance
  heuristic and finally a single generic message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.gameweek_predictor import CAT_COLS, _load_models, get_feature_rows

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FEATURE_IMPORTANCE_PATH = (
    _PROJECT_ROOT / "Weekly Model" / "production" / "feature_importance.csv"
)

# Price/ownership features — at most one may surface per player.
_PRICE_OWNERSHIP = {"value", "selected", "transfers_balance"}

# Features owned by the dedicated fixture line — never surfaced generically.
_FIXTURE_COLS = {"was_home", "opponent_team", "team"}

# Concepts where a LOW (incl. zero) value is a positive signal, so a zero must
# NOT be filtered out (e.g. "conceded just 0"). Everything else with a ~0 value
# is a non-reason ("0.00 xG" for a keeper) and is skipped.
_LOWER_IS_BETTER = {"goals_conceded", "expected_goals_conceded", "team_goals_conceded"}

VALID_CATEGORIES = {
    "form", "fixture", "attacking", "defensive",
    "reliability", "performance", "value", "trending",
}

GENERIC_FALLBACK = {
    "text": "Selected by the V8 model based on recent form and fixtures.",
    "category": "performance",
}


# ──────────────────────────────────────────────────────────────────────────────
# Rolling-window suffix handling
# ──────────────────────────────────────────────────────────────────────────────
_WINDOW_PHRASE = {
    "_last": "last gameweek",
    "_mean3": "over the last 3 gameweeks",
    "_mean5": "over the last 5 gameweeks",
    "_mean7": "over the last 7 gameweeks",
}
_WINDOW_SUFFIXES = ("_mean7", "_mean5", "_mean3", "_last")


def _split_window(feature: str) -> tuple[str, str]:
    """Return ``(base_concept, window_phrase)`` for a feature column.

    ``ict_index_mean3`` -> ``("ict_index", "over the last 3 gameweeks")``;
    a bare current-GW column -> ``("ict_index", "")``.
    """
    for suf in _WINDOW_SUFFIXES:
        if feature.endswith(suf):
            return feature[: -len(suf)], _WINDOW_PHRASE[suf]
    return feature, ""


def _w(window: str) -> str:
    return f" {window}" if window else ""


def _i(v) -> str:
    return f"{int(round(float(v)))}"


def _f1(v) -> str:
    return f"{float(v):.1f}"


def _f2(v) -> str:
    return f"{float(v):.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# base concept -> (category, builder(value, window_phrase) -> friendly text)
# ──────────────────────────────────────────────────────────────────────────────
_CONCEPTS: dict[str, tuple[str, "callable"]] = {
    # form
    "total_points": ("form", lambda v, w: f"Returning {_f1(v)} pts{_w(w)} — in good form"),
    "xP":           ("form", lambda v, w: f"Projected ~{_f1(v)} pts{_w(w)}"),
    # reliability
    "minutes": ("reliability", lambda v, w: f"Played {_i(v)} minutes{_w(w)} — a nailed-on starter"),
    "starts":  ("reliability", lambda v, w: f"Starting regularly{_w(w)}"),
    # attacking
    "expected_goals":             ("attacking", lambda v, w: f"{_f2(v)} expected goals (xG){_w(w)}"),
    "expected_assists":           ("attacking", lambda v, w: f"{_f2(v)} expected assists (xA){_w(w)}"),
    "expected_goal_involvements": ("attacking", lambda v, w: f"{_f2(v)} expected goal involvements{_w(w)}"),
    "goals_scored":      ("attacking", lambda v, w: f"{_i(v)} goals{_w(w)}"),
    "assists":           ("attacking", lambda v, w: f"{_i(v)} assists{_w(w)}"),
    "threat":            ("attacking", lambda v, w: f"High threat rating ({_i(v)}){_w(w)}"),
    "creativity":        ("attacking", lambda v, w: f"High creativity ({_i(v)}){_w(w)}"),
    "team_goals_scored": ("attacking", lambda v, w: f"Team scoring freely ({_i(v)} goals){_w(w)}"),
    # performance
    "ict_index": ("performance", lambda v, w: f"ICT index of {_f1(v)}{_w(w)}"),
    "influence": ("performance", lambda v, w: f"Strong influence ({_f1(v)}){_w(w)}"),
    "bps":       ("performance", lambda v, w: f"{_i(v)} BPS{_w(w)} — in line for bonus"),
    "bonus":     ("performance", lambda v, w: f"{_i(v)} bonus points{_w(w)}"),
    # defensive
    "clean_sheets":            ("defensive", lambda v, w: f"{_i(v)} clean sheets{_w(w)}"),
    "goals_conceded":          ("defensive", lambda v, w: f"Conceded just {_i(v)}{_w(w)} — solid defensively"),
    "expected_goals_conceded": ("defensive", lambda v, w: f"Low expected goals conceded ({_f1(v)}){_w(w)}"),
    "saves":                   ("defensive", lambda v, w: f"{_i(v)} saves{_w(w)}"),
    "team_goals_conceded":     ("defensive", lambda v, w: f"Team defence holding firm ({_i(v)} conceded){_w(w)}"),
    # value / trending
    "value":    ("value", lambda v, w: f"Priced at £{float(v) / 10.0:.1f}m"),
    "selected": ("trending", lambda v, w: "Popular among managers — a trending pick"),
    "transfers_balance": (
        "trending",
        lambda v, w: "Net transfers in — momentum building"
        if float(v) >= 0
        else "Steady ownership trend",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature importance (lazy cache) — used only for the no-SHAP fallback.
# ──────────────────────────────────────────────────────────────────────────────
_importance_cache: Optional[dict[str, dict[str, float]]] = None
_importance_load_failed = False


def _load_importance() -> dict[str, dict[str, float]]:
    """position -> {feature: importance}. Empty dict if the file is missing/bad."""
    global _importance_cache, _importance_load_failed
    if _importance_cache is not None:
        return _importance_cache
    if _importance_load_failed:
        return {}
    if not _FEATURE_IMPORTANCE_PATH.exists():
        _importance_load_failed = True
        return {}
    try:
        df = pd.read_csv(_FEATURE_IMPORTANCE_PATH)
    except Exception:
        logger.warning("Failed to read weekly feature_importance.csv", exc_info=True)
        _importance_load_failed = True
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        out.setdefault(str(r["position"]), {})[str(r["feature"])] = float(r["importance"])
    _importance_cache = out
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reason builders
# ──────────────────────────────────────────────────────────────────────────────
def _fixture_reason(row: pd.Series) -> Optional[dict]:
    """Truthful home/away + opponent line, or None when not usable."""
    opp = row.get("opponent_team")
    if opp is None:
        return None
    opp = str(opp).strip()
    if opp in ("", "nan", "None"):
        return None
    team = str(row.get("team", "")).strip()
    if opp == team:  # double-GW / blank-row artifact
        return None
    where = "Home" if bool(row.get("was_home")) else "Away"
    return {"text": f"{where} fixture against {opp}", "category": "fixture"}


def _usable_value(row: pd.Series, feat: str):
    val = row.get(feat)
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def _is_zeroish(base: str, val) -> bool:
    """True when *val* is effectively zero AND zero is not itself a good reason.

    Filters non-reasons like "0.00 xG"/"0 goals" for a goalkeeper, while keeping
    low/zero defensive concepts (goals conceded) that are positive signals.
    """
    if base in _LOWER_IS_BETTER:
        return False
    try:
        return abs(float(val)) < 0.05
    except (TypeError, ValueError):
        return True


def _fill_reasons(reasons: list[dict], ranked, top_k: int) -> list[dict]:
    """Append concept reasons under the price/ownership cap until full.

    ``ranked`` is an iterable of ``(base_concept, feature, value)`` already in
    priority order.
    """
    price_used = False
    for base, feat, val in ranked:
        if len(reasons) >= top_k:
            break
        if base in _PRICE_OWNERSHIP:
            if price_used:
                continue
            price_used = True
        category, builder = _CONCEPTS[base]
        _b, phrase = _split_window(feat)
        reasons.append({"text": builder(val, phrase), "category": category})
    return reasons


def _reasons_from_shap(row: pd.Series, contribs: dict[str, float], top_k: int) -> list[dict]:
    reasons: list[dict] = []
    fx = _fixture_reason(row)
    if fx:
        reasons.append(fx)

    # Collapse each base concept to its highest-|contribution| window.
    best: dict[str, tuple[float, float, str, object]] = {}
    for feat, contrib in contribs.items():
        if feat in _FIXTURE_COLS:
            continue
        base, _phrase = _split_window(feat)
        if base not in _CONCEPTS:
            continue
        val = _usable_value(row, feat)
        if val is None or _is_zeroish(base, val):
            continue
        a = abs(contrib)
        if base not in best or a > best[base][0]:
            best[base] = (a, contrib, feat, val)

    # Keep only features that pushed the prediction UP, strongest first.
    ranked = sorted(
        (b for b in best.items() if b[1][1] > 0),
        key=lambda kv: kv[1][1],
        reverse=True,
    )
    return _fill_reasons(reasons, [(b, v[2], v[3]) for b, v in ranked], top_k)


def _reasons_from_importance(
    pos: str, row: pd.Series, feats: list[str], importance: dict[str, dict[str, float]], top_k: int
) -> list[dict]:
    reasons: list[dict] = []
    fx = _fixture_reason(row)
    if fx:
        reasons.append(fx)

    imp = importance.get(pos, {})
    best: dict[str, tuple[float, str, object]] = {}
    for feat in feats:
        if feat in _FIXTURE_COLS:
            continue
        base, _phrase = _split_window(feat)
        if base not in _CONCEPTS:
            continue
        val = _usable_value(row, feat)
        if val is None or _is_zeroish(base, val):
            continue
        score = imp.get(feat, 0.0)
        if base not in best or score > best[base][0]:
            best[base] = (score, feat, val)

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    return _fill_reasons(reasons, [(b, v[1], v[2]) for b, v in ranked], top_k)


# ──────────────────────────────────────────────────────────────────────────────
# SHAP (batched per position)
# ──────────────────────────────────────────────────────────────────────────────
def _shap_for_rows(rows: list[pd.Series], feats: list[str], model) -> Optional[list[dict[str, float]]]:
    """Per-row {feature: contribution} dicts via CatBoost SHAP. None on failure."""
    try:
        from catboost import Pool

        X = pd.concat([r[feats].to_frame().T for r in rows], ignore_index=True)[feats]
        for c in CAT_COLS:
            if c in X.columns:
                X[c] = X[c].astype(str)
        cat_in_X = [c for c in CAT_COLS if c in X.columns]
        pool = Pool(X, cat_features=cat_in_X)
        shap = model.get_feature_importance(pool, type="ShapValues")
        out: list[dict[str, float]] = []
        for i in range(len(rows)):
            contribs = shap[i][:-1]  # drop trailing bias/expected-value column
            out.append({feats[j]: float(contribs[j]) for j in range(len(feats))})
        return out
    except Exception as exc:
        logger.warning("Weekly SHAP computation failed: %s — using importance fallback", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def explain_weekly_squad(
    squad: list[dict],
    gameweek: int | None,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Return up to ``top_k`` weekly reasons per player.

    Mirrors ``src.explainer.explain_squad``'s return shape. Always returns an
    entry for every squad player; never raises.
    """
    rows = get_feature_rows(squad, gameweek)
    models = _load_models() if rows else None
    importance = _load_importance()

    # Batch SHAP by position (≤4 model calls instead of one per player).
    shap_by_pid: dict[str, dict[str, float]] = {}
    if models is not None and rows:
        by_pos: dict[str, list[str]] = {}
        for pid, (pos, _row, _feats) in rows.items():
            by_pos.setdefault(pos, []).append(pid)
        for pos, pids in by_pos.items():
            model = models.get(pos)
            if model is None:
                continue
            feats = rows[pids[0]][2]
            contribs = _shap_for_rows([rows[pid][1] for pid in pids], feats, model)
            if contribs is None:
                continue
            for pid, c in zip(pids, contribs):
                shap_by_pid[pid] = c

    result: dict[str, list[dict]] = {}
    for player in squad:
        pid = str(player["id"])
        reasons: list[dict] = []
        if pid in rows:
            pos, row, feats = rows[pid]
            if pid in shap_by_pid:
                reasons = _reasons_from_shap(row, shap_by_pid[pid], top_k)
            else:
                reasons = _reasons_from_importance(pos, row, feats, importance, top_k)
        if not reasons:
            reasons = [dict(GENERIC_FALLBACK)]
        result[pid] = reasons[:top_k]
    return result
