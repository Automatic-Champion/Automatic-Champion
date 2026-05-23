from __future__ import annotations

import logging
import os
from pathlib import Path

import firebase_admin
from fastapi import Header, HTTPException, status
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

logger = logging.getLogger(__name__)

_DEFAULT_CREDENTIALS_PATH = "/Users/yuvaldavidovits/firebase-admin-automatic-champion.json"


def _initialize_firebase() -> None:
    if firebase_admin._apps:
        return

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", _DEFAULT_CREDENTIALS_PATH)
    if not Path(cred_path).is_file():
        logger.error("Firebase service account file not found at %s", cred_path)
        raise RuntimeError(
            f"Firebase service account file not found at {cred_path}. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to the JSON path."
        )

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    except Exception as exc:
        logger.error("Failed to initialize Firebase Admin SDK: %s", exc)
        raise


_initialize_firebase()


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
    except Exception as exc:
        logger.info("Firebase token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        ) from exc

    return decoded
