from __future__ import annotations

import grp
import hashlib
import os

import fastapi
import fastapi.security

import qjob.core.database as database
import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Internal helpers

_bearer = fastapi.security.HTTPBearer()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def is_admin(username: str) -> bool:
    """Return True if *username* has admin privileges."""
    admin_env = os.environ.get("QJOB_ADMIN_USERS", "root")
    if username in [u.strip() for u in admin_env.split(",")]:
        return True
    try:
        group = grp.getgrnam("qjob_admin")
        return username in group.gr_mem
    except KeyError:
        return False


# --------------------------------------------------------------------------------------
# FastAPI dependencies


def get_current_user(
    credentials: fastapi.security.HTTPAuthorizationCredentials = fastapi.Depends(_bearer),
) -> str:
    """Validate the Bearer token and return the authenticated username."""
    token_hash = _hash_token(credentials.credentials)
    with database.get_session() as session:
        row = (
            session.query(models.ApiToken)
            .filter(models.ApiToken.token_hash == token_hash)
            .first()
        )
    if row is None:
        raise fastapi.HTTPException(status_code=401, detail="Invalid or missing token.")
    return row.username


def require_admin(
    username: str = fastapi.Depends(get_current_user),
) -> str:
    """Validate the Bearer token and require admin privileges."""
    if not is_admin(username):
        raise fastapi.HTTPException(status_code=403, detail="Admin privileges required.")
    return username
