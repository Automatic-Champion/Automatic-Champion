from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from .deps import get_db
from .routers.lineup import router as lineup_router
from .routers.squad import router as squad_router

app = FastAPI(title="Automatic Champion API")


@app.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict[str, str]:
    _ = db
    return {"status": "ok"}


app.include_router(squad_router)
app.include_router(lineup_router)
