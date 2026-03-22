#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required for this script. Install with: pip install pandas"
    ) from exc

from sqlalchemy.dialects.postgresql import insert

# Allow importing from backend/app when running this script directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.database import SessionLocal
from app.models import Player, Team

TEAMS_CSV_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/teams.csv"
)
PLAYERS_CSV_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv"
)

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


def _load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    teams_df = pd.read_csv(TEAMS_CSV_URL)
    players_df = pd.read_csv(PLAYERS_CSV_URL)
    return teams_df, players_df


def _build_team_rows(teams_df: pd.DataFrame) -> list[dict]:
    required_columns = {"id", "name"}
    missing = required_columns - set(teams_df.columns)
    if missing:
        raise ValueError(f"Teams CSV is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, row in teams_df.iterrows():
        rows.append(
            {
                "team_id": int(row["id"]),
                "team_name": str(row["name"]).strip(),
            }
        )
    return rows


def _build_player_rows(players_df: pd.DataFrame) -> list[dict]:
    required_columns = {"id", "first_name", "second_name", "element_type", "now_cost", "team"}
    missing = required_columns - set(players_df.columns)
    if missing:
        raise ValueError(f"Players CSV is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, row in players_df.iterrows():
        element_type = int(row["element_type"])
        position = POSITION_MAP.get(element_type)
        if position is None:
            continue

        full_name = f"{str(row['first_name']).strip()} {str(row['second_name']).strip()}".strip()

        rows.append(
            {
                "player_id": int(row["id"]),
                "name": full_name,
                "position": position,
                "price": float(row["now_cost"]) / 10.0,
                "team_id": int(row["team"]),
            }
        )
    return rows


def _upsert_teams(team_rows: list[dict], db_session) -> int:
    if not team_rows:
        return 0

    stmt = insert(Team).values(team_rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Team.team_id],
        set_={
            "team_name": stmt.excluded.team_name,
        },
    )
    db_session.execute(stmt)
    return len(team_rows)


def _upsert_players(player_rows: list[dict], db_session) -> int:
    if not player_rows:
        return 0

    stmt = insert(Player).values(player_rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Player.player_id],
        set_={
            "name": stmt.excluded.name,
            "position": stmt.excluded.position,
            "price": stmt.excluded.price,
            "team_id": stmt.excluded.team_id,
        },
    )
    db_session.execute(stmt)
    return len(player_rows)


def main() -> None:
    teams_df, players_df = _load_dataframes()
    team_rows = _build_team_rows(teams_df)
    player_rows = _build_player_rows(players_df)

    db = SessionLocal()
    try:
        team_count = _upsert_teams(team_rows, db)
        player_count = _upsert_players(player_rows, db)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    print(f"Teams upserted: {team_count}")
    print(f"Players upserted: {player_count}")


if __name__ == "__main__":
    main()
