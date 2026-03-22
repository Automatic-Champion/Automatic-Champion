from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    user_id: str


class UserResponse(UserBase):
    user_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConstraintSetBase(BaseModel):
    user_id: str
    type: str
    value_json: dict[str, Any]


class ConstraintSetCreate(ConstraintSetBase):
    constraint_id: str


class ConstraintSetResponse(ConstraintSetBase):
    constraint_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TeamBase(BaseModel):
    team_name: str


class TeamCreate(TeamBase):
    team_id: int


class TeamResponse(TeamBase):
    team_id: int

    model_config = ConfigDict(from_attributes=True)


class PlayerBase(BaseModel):
    name: str
    position: str
    price: float
    team_id: int


class PlayerCreate(PlayerBase):
    player_id: int


class PlayerResponse(PlayerBase):
    player_id: int

    model_config = ConfigDict(from_attributes=True)


class SeasonTeamBase(BaseModel):
    user_id: str
    constraint_id: str
    total_budget_used: float


class SeasonTeamCreate(SeasonTeamBase):
    season_team_id: str


class SeasonTeamResponse(SeasonTeamBase):
    season_team_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SeasonTeamPlayerBase(BaseModel):
    season_team_id: str
    player_id: int
    is_starter: bool
    bench_order: int | None = None
    selected_position: str


class SeasonTeamPlayerCreate(SeasonTeamPlayerBase):
    pass


class SeasonTeamPlayerResponse(SeasonTeamPlayerBase):
    model_config = ConfigDict(from_attributes=True)


class SquadGenerateRequest(BaseModel):
    budget: float = 100.0
    formation: str = "4-3-3"
    locked_ids: list[int] = Field(default_factory=list)
    banned_ids: list[int] = Field(default_factory=list)


class SquadPlayerResponse(BaseModel):
    id: str
    name: str
    team: str
    position: str
    cost: float
    predicted_points: float
    is_starter: bool
    bench_order: int | None = None
    explanations: list[dict[str, Any]] = Field(default_factory=list)


class SquadGenerateResponse(BaseModel):
    formation: str
    budget: float
    total_cost: float
    total_predicted_points: float
    players: list[SquadPlayerResponse]
    bench: list[SquadPlayerResponse]
