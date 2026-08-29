import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .auth import init_firebase
from .config import DATA_PATH
from .database import engine
from .routers.lineup import router as lineup_router
from .routers.saved_squads import router as saved_squads_router
from .routers.squad import router as squad_router

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    yield


app = FastAPI(title="Automatic Champion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return unhandled errors as JSON *with* CORS headers.

    Starlette's ServerErrorMiddleware wraps the CORS middleware, so a bare 500
    reaches the browser without CORS headers and the frontend reports it as
    "cannot connect to the backend" instead of the real error. Re-attaching the
    headers here lets genuine server errors surface in the UI.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    headers = {}
    origin = request.headers.get("origin")
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Vary"] = "Origin"
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=headers,
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
app.include_router(saved_squads_router)
