from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .auth import init_firebase
from .config import DATA_PATH
from .database import engine
from .routers.lineup import router as lineup_router
from .routers.squad import router as squad_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    yield


app = FastAPI(title="Automatic Champion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return {"status": "degraded", "database": "disconnected"}


POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@app.get("/players")
def list_players() -> list[dict]:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(float).astype(int)
    df["name"] = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.strip()
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["cost"] = df["price_now"] / 10
    players = df[["id", "name", "position", "team_name", "cost"]].copy()
    players = players.rename(columns={"team_name": "team"})
    players = players.dropna(subset=["position"])
    return players.to_dict(orient="records")


app.include_router(squad_router)
app.include_router(lineup_router)
