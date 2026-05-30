from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..deps import get_db
from ..models import SavedSquad
from ..schemas import (
    SavedSquadCreateRequest,
    SavedSquadResponse,
    SavedSquadSummary,
    SquadGenerateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/squads", tags=["saved_squads"])

MAX_SAVED_SQUADS = 10


def _to_summary(row: SavedSquad) -> SavedSquadSummary:
    payload = row.payload_json or {}
    return SavedSquadSummary(
        id=row.id,
        name=row.name,
        formation=str(payload.get("formation", "")),
        total_cost=float(payload.get("total_cost", 0.0)),
        total_predicted_points=float(payload.get("total_predicted_points", 0.0)),
        created_at=row.created_at,
    )


def _to_response(row: SavedSquad) -> SavedSquadResponse:
    summary = _to_summary(row)
    return SavedSquadResponse(
        **summary.model_dump(),
        payload=SquadGenerateResponse(**row.payload_json),
    )


@router.get("", response_model=list[SavedSquadSummary])
def list_saved_squads(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SavedSquadSummary]:
    uid = current_user["uid"]
    rows = (
        db.query(SavedSquad)
        .filter(SavedSquad.user_uid == uid)
        .order_by(SavedSquad.created_at.desc())
        .all()
    )
    return [_to_summary(r) for r in rows]


@router.post("", response_model=SavedSquadResponse, status_code=status.HTTP_201_CREATED)
def save_squad(
    payload: SavedSquadCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SavedSquadResponse:
    uid = current_user["uid"]

    count = db.query(SavedSquad).filter(SavedSquad.user_uid == uid).count()
    if count >= MAX_SAVED_SQUADS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Squad limit reached. Delete a saved squad to add a new one.",
        )

    name = payload.name.strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Squad name cannot be empty.",
        )

    existing = (
        db.query(SavedSquad)
        .filter(SavedSquad.user_uid == uid)
        .filter(func.lower(SavedSquad.name) == name.lower())
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A squad named '{name}' already exists.",
        )

    row = SavedSquad(
        user_uid=uid,
        name=name,
        payload_json=payload.squad.model_dump(mode="json"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _to_response(row)


@router.get("/{squad_id}", response_model=SavedSquadResponse)
def get_saved_squad(
    squad_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SavedSquadResponse:
    uid = current_user["uid"]
    row = (
        db.query(SavedSquad)
        .filter(SavedSquad.id == squad_id, SavedSquad.user_uid == uid)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Squad not found")
    return _to_response(row)


@router.delete("/{squad_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_squad(
    squad_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    uid = current_user["uid"]
    row = (
        db.query(SavedSquad)
        .filter(SavedSquad.id == squad_id, SavedSquad.user_uid == uid)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Squad not found")
    db.delete(row)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
