from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    SquadGenerateRequest,
    SquadGenerateResponse,
    SquadPlayerResponse,
)
from ..services.optimizer import OptimizationError, generate_optimal_squad

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/squad", tags=["squad"])

DATA_PATH = "data/players_merged_2024-25.csv"
MODELS_DIR = "models"


def _build_player_response(
    player: dict,
    explanations: dict[str, list[dict]],
) -> SquadPlayerResponse:
    pid = str(player["id"])
    return SquadPlayerResponse(
        id=pid,
        name=str(player["name"]),
        team=str(player["team"]),
        position=str(player["position"]),
        cost=float(player["cost"]),
        predicted_points=float(player["pred"]),
        is_starter=bool(player["is_starter"]),
        bench_order=int(player["bench_order"]) if player.get("bench_order") is not None else None,
        explanations=explanations.get(pid, []),
    )


@router.post("/generate", response_model=SquadGenerateResponse, status_code=status.HTTP_201_CREATED)
def generate_squad(payload: SquadGenerateRequest) -> SquadGenerateResponse:
    locked = {str(pid) for pid in payload.locked_ids} if payload.locked_ids else None
    banned = {str(pid) for pid in payload.banned_ids} if payload.banned_ids else None

    try:
        result = generate_optimal_squad(
            budget=payload.budget,
            formation=payload.formation,
            data_path=DATA_PATH,
            models_dir=MODELS_DIR,
            locked_ids=locked,
            banned_ids=banned,
        )
    except (OptimizationError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Generate explanations (non-fatal)
    explanations: dict[str, list[dict]] = {}
    try:
        from src.explainer import explain_squad

        explanations = explain_squad(
            players=result["players"] + result["bench"],
            data_path=DATA_PATH,
            models_dir=MODELS_DIR,
            top_k=3,
        )
    except Exception:
        logger.warning("Explanation generation failed", exc_info=True)

    all_players = [_build_player_response(p, explanations) for p in result["players"]]
    bench = [_build_player_response(p, explanations) for p in result["bench"]]

    return SquadGenerateResponse(
        formation=result["formation"],
        budget=result["budget"],
        total_cost=result["total_cost"],
        total_predicted_points=result["total_pred"],
        players=all_players,
        bench=bench,
    )
