from __future__ import annotations

import logging
import os
from pathlib import Path

import firebase_admin
from fastapi import Header, HTTPException, status
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

logger = logging.getLogger(__name__)


def init_firebase() -> None:
    if firebase_admin._apps:
        return

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS not set — auth endpoints will return 503"
        )
        return

    if not Path(cred_path).is_file():
        logger.warning(
            "Firebase service account file not found at %s — auth endpoints will return 503",
            cred_path,
        )
        return

    try:
        cred = credentials.Certificate(cred_path)
        app = firebase_admin.initialize_app(cred)
    except Exception as exc:
        logger.warning("Failed to initialize Firebase Admin SDK: %s", exc)
        return

    project_id = app.project_id or "<unknown>"
    logger.info("Firebase Admin SDK initialized for project: %s", project_id)


def get_current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )

    try:
        decoded = firebase_auth.verify_id_token(token)
    except ValueError as exc:
        # firebase_admin raises ValueError when no app is initialized
        if not firebase_admin._apps:
            logger.warning("Firebase Admin SDK not initialized — returning 503")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable",
            ) from exc
        logger.info("Firebase token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        ) from exc
    except Exception as exc:
        logger.info("Firebase token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        ) from exc

    return decoded
