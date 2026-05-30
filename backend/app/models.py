from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    __tablename__ = "User"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    constraint_sets: Mapped[list["Constraint_Set"]] = relationship(
        "Constraint_Set", back_populates="user", cascade="all, delete-orphan"
    )
    season_teams: Mapped[list["Season_Team"]] = relationship(
        "Season_Team", back_populates="user", cascade="all, delete-orphan"
    )


class Constraint_Set(Base):
    __tablename__ = "Constraint_Set"

    constraint_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("User.user_id"), nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    value_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="constraint_sets")
    season_teams: Mapped[list["Season_Team"]] = relationship(
        "Season_Team", back_populates="constraint_set", cascade="all, delete-orphan"
    )


class Team(Base):
    __tablename__ = "Team"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_name: Mapped[str] = mapped_column(String, nullable=False)

    players: Mapped[list["Player"]] = relationship(
        "Player", back_populates="team", cascade="all, delete-orphan"
    )


class Player(Base):
    __tablename__ = "Player"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("Team.team_id"), nullable=False)

    team: Mapped[Team] = relationship("Team", back_populates="players")
    season_team_players: Mapped[list["Season_Team_Player"]] = relationship(
        "Season_Team_Player", back_populates="player", cascade="all, delete-orphan"
    )


class Season_Team(Base):
    __tablename__ = "Season_Team"

    season_team_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("User.user_id"), nullable=False)
    constraint_id: Mapped[str] = mapped_column(ForeignKey("Constraint_Set.constraint_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    total_budget_used: Mapped[float] = mapped_column(Float, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="season_teams")
    constraint_set: Mapped[Constraint_Set] = relationship("Constraint_Set", back_populates="season_teams")
    season_team_players: Mapped[list["Season_Team_Player"]] = relationship(
        "Season_Team_Player", back_populates="season_team", cascade="all, delete-orphan"
    )


class Season_Team_Player(Base):
    __tablename__ = "Season_Team_Player"

    season_team_id: Mapped[str] = mapped_column(
        ForeignKey("Season_Team.season_team_id"), primary_key=True
    )
    player_id: Mapped[int] = mapped_column(ForeignKey("Player.player_id"), primary_key=True)
    is_starter: Mapped[bool] = mapped_column(Boolean, nullable=False)
    bench_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_position: Mapped[str] = mapped_column(String, nullable=False)

    season_team: Mapped[Season_Team] = relationship("Season_Team", back_populates="season_team_players")
    player: Mapped[Player] = relationship("Player", back_populates="season_team_players")


class SavedSquad(Base):
    __tablename__ = "saved_squads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_uid: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
