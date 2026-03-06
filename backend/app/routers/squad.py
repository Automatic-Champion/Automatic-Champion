from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import get_db
from ..models import Constraint_Set, Player, Season_Team, Season_Team_Player, User
from ..schemas import (
    SeasonTeamPlayerResponse,
    SeasonTeamResponse,
    SquadGenerateRequest,
    SquadGenerateResponse,
    SquadPlayerSelectionResponse,
)
from ..services.optimizer import OptimizationError, generate_optimal_squad

router = APIRouter(prefix="/squad", tags=["squad"])


@router.post("/generate", response_model=SquadGenerateResponse, status_code=status.HTTP_201_CREATED)
def generate_squad(payload: SquadGenerateRequest, db: Session = Depends(get_db)) -> SquadGenerateResponse:
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    constraint_set = db.get(Constraint_Set, payload.constraint_id)
    if constraint_set is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Constraint set not found")
    if constraint_set.user_id != payload.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Constraint set does not belong to the specified user",
        )

    existing_season_team = db.get(Season_Team, payload.season_team_id)
    if existing_season_team is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="season_team_id already exists",
        )

    try:
        result = generate_optimal_squad(
            budget=payload.budget,
            players_data=[player.model_dump() for player in payload.players_data],
            models_dir=payload.models_dir,
        )
    except OptimizationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    selected_players = result["selected_players"]
    selected_player_ids = [player["player_id"] for player in selected_players]

    existing_player_ids = set(
        db.scalars(select(Player.player_id).where(Player.player_id.in_(selected_player_ids))).all()
    )
    missing_player_ids = [player_id for player_id in selected_player_ids if player_id not in existing_player_ids]
    if missing_player_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Cannot persist season squad because some selected players are missing "
                f"from Player table: {missing_player_ids}"
            ),
        )

    season_team = Season_Team(
        season_team_id=payload.season_team_id,
        user_id=payload.user_id,
        constraint_id=payload.constraint_id,
        created_at=datetime.utcnow(),
        total_budget_used=float(result["total_budget_used"]),
    )
    db.add(season_team)

    season_team_players: list[Season_Team_Player] = []
    for player in selected_players:
        season_team_player = Season_Team_Player(
            season_team_id=payload.season_team_id,
            player_id=player["player_id"],
            is_starter=player["is_starter"],
            bench_order=player["bench_order"],
            selected_position=player["selected_position"],
        )
        season_team_players.append(season_team_player)
        db.add(season_team_player)

    db.commit()
    db.refresh(season_team)

    season_team_response = SeasonTeamResponse.model_validate(season_team)
    season_team_players_response = [
        SeasonTeamPlayerResponse.model_validate(season_team_player)
        for season_team_player in season_team_players
    ]
    selected_players_response = [
        SquadPlayerSelectionResponse(
            player_id=player["player_id"],
            name=player["name"],
            position=player["position"],
            team_id=player["team_id"],
            price=player["price"],
            predicted_points=player["predicted_points"],
            is_starter=player["is_starter"],
            bench_order=player["bench_order"],
            selected_position=player["selected_position"],
        )
        for player in selected_players
    ]

    return SquadGenerateResponse(
        season_team=season_team_response,
        season_team_players=season_team_players_response,
        selected_players=selected_players_response,
    )
