from fastapi import FastAPI
from sqlalchemy import text

from .database import engine
from .routers.lineup import router as lineup_router
from .routers.squad import router as squad_router

app = FastAPI(title="Automatic Champion API")


@app.get("/health")
def health_check() -> dict[str, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return {"status": "degraded", "database": "disconnected"}


app.include_router(squad_router)
app.include_router(lineup_router)
