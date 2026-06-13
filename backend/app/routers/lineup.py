from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..schemas import (
    LineupRecommendRequest,
    LineupRecommendResponse,
    LineupStarterResponse,
    LineupBenchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lineup", tags=["lineup"])


@router.post("/recommend", response_model=LineupRecommendResponse)
def recommend_lineup(
    payload: LineupRecommendRequest,
    current_user: dict = Depends(get_current_user),
) -> LineupRecommendResponse:
    from src.fpl_api import get_current_gameweek
    from src.gameweek_predictor import PREDICTOR_VERSION, predict_gameweek_points_with_meta
    from src.lineup_optimizer import optimize_lineup

    # Convert request models to plain dicts for src/ functions
    squad = [p.model_dump() for p in payload.squad]

    # Resolve gameweek before passing to predictor
    resolved_gameweek = payload.gameweek
    if resolved_gameweek is None:
        resolved_gameweek = get_current_gameweek()

    # Predict gameweek points (with degradation metadata so we can warn the
    # user when predictions fell back to season-average estimates — TC-06).
    gw_predictions, meta = predict_gameweek_points_with_meta(squad, resolved_gameweek)

    warnings: list[str] = []
    if not meta["model_available"]:
        warnings.append(
            "Weekly prediction model unavailable — these picks use season-average "
            "estimates and may be less accurate. Try again later."
        )
    elif meta["fallback_player_ids"]:
        n = len(meta["fallback_player_ids"])
        warnings.append(
            f"{n} player(s) have no recent gameweek data; their points use "
            "season-average estimates."
        )

    # Optimize lineup
    try:
        result = optimize_lineup(squad, gw_predictions, payload.formation)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # Generate weekly (V8) per-player explanations (non-fatal). These explain
    # the model's gameweek prediction, unlike the season explainer used by UC1.
    explanations: dict[str, list[dict]] = {}
    try:
        from src.weekly_explainer import explain_weekly_squad

        explanations = explain_weekly_squad(squad, resolved_gameweek, top_k=3)
    except Exception:
        logger.warning("Explanation generation failed", exc_info=True)

    starters = [
        LineupStarterResponse(
            id=s["id"],
            name=s["name"],
            position=s["position"],
            team=s["team"],
            gw_points=s["gw_points"],
            is_captain=s["is_captain"],
            is_vice_captain=s["is_vice_captain"],
            explanations=explanations.get(s["id"], []),
        )
        for s in result["starters"]
    ]

    bench = [
        LineupBenchResponse(
            id=b["id"],
            name=b["name"],
            position=b["position"],
            team=b["team"],
            gw_points=b["gw_points"],
            bench_order=b["bench_order"],
            explanations=explanations.get(b["id"], []),
        )
        for b in result["bench"]
    ]

    return LineupRecommendResponse(
        formation=result["formation"],
        gameweek=resolved_gameweek,
        captain_id=result["captain_id"],
        vice_captain_id=result["vice_captain_id"],
        total_gw_points=result["total_gw_points"],
        starters=starters,
        bench=bench,
        predictor_version=PREDICTOR_VERSION,
        warnings=warnings,
    )
