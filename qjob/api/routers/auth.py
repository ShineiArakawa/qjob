from __future__ import annotations

import hashlib
import secrets

import fastapi

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
def create_token(body: schemas.TokenCreateRequest) -> schemas.TokenResponse:
    """
    Generate a new API token for *username*.

    The raw token is returned once and never stored — only its SHA-256 hash
    is kept in the database.  Save the token to ``~/.config/qjob/token``.

    Parameters
    ----------
    body : schemas.TokenCreateRequest
        The username to associate with the new token.

    Returns
    -------
    schemas.TokenResponse
        The raw token and the associated username.
    """

    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with database.get_session() as session:
        row = models.ApiToken(username=body.username, token_hash=token_hash)
        session.add(row)

    return schemas.TokenResponse(token=token, username=body.username)
