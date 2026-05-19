from __future__ import annotations

import hashlib
import pwd
import secrets

import fastapi

import qjob.api.auth as auth
import qjob.api.schemas as schemas
import qjob.core.database as database
import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Router

router = fastapi.APIRouter(prefix="/auth", tags=["auth"])


# --------------------------------------------------------------------------------------
# POST /auth/token


@router.post(
    "/token",
    response_model=schemas.TokenResponse,
    status_code=201,
    summary="Create an API token",
)
def create_token(
    body: schemas.TokenCreateRequest,
    _admin: str = fastapi.Depends(auth.require_admin),
) -> schemas.TokenResponse:
    """
    Generate a new API token for *username* (admin only).

    Requires admin privileges.  *username* must exist as an OS user and must
    not already have a token.  The raw token is returned once and never stored
    — only its SHA-256 hash is kept in the database.
    """

    try:
        pwd.getpwnam(body.username)
    except KeyError:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"OS user {body.username!r} does not exist.",
        )

    with database.get_session() as session:
        existing = (
            session.query(models.ApiToken)
            .filter(models.ApiToken.username == body.username)
            .first()
        )
    if existing is not None:
        raise fastapi.HTTPException(
            status_code=409,
            detail=f"User {body.username!r} already has a token. Revoke it first.",
        )

    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with database.get_session() as session:
        row = models.ApiToken(username=body.username, token_hash=token_hash)
        session.add(row)

    return schemas.TokenResponse(token=token, username=body.username)
