from __future__ import annotations

import contextlib
import logging
import typing

import fastapi
import uvicorn

import qjob.api.routers.jobs as jobs_router
import qjob.api.routers.resources as resources_router
import qjob.api.schemas as schemas
import qjob.core.database as database

# --------------------------------------------------------------------------------------
# Module logger

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants

_DEFAULT_HOST: str = "127.0.0.1"
_DEFAULT_PORT: int = 8000


# --------------------------------------------------------------------------------------
# Application factory


def create_app(db_url: str | None = None) -> fastapi.FastAPI:
    """
    Create and configure the FastAPI application.

    The scheduler is started as a background task on application startup
    and stopped gracefully on shutdown.

    Parameters
    ----------
    db_url : str | None
        SQLAlchemy database URL.  When None, ``database.init_db()`` uses
        ``QJOB_DB_URL`` from the environment.

    Returns
    -------
    fastapi.FastAPI
        The configured application instance.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: fastapi.FastAPI) -> typing.AsyncGenerator[None, None]:
        database.init_db(db_url)
        logger.info("Database initialised.")
        yield

    app = fastapi.FastAPI(
        title="qjob API",
        description="Lightweight job scheduler for research servers.",
        version="0.2.0",
        lifespan=lifespan,
    )

    # -- Exception handlers ------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: fastapi.Request,
        exc:     Exception,
    ) -> fastapi.responses.JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content=schemas.ErrorResponse(detail="Internal server error.").model_dump(),
        )

    # -- Routers -----------------------------------------------------------------------

    app.include_router(jobs_router.router)
    app.include_router(resources_router.router)

    # -- Health check ------------------------------------------------------------------

    @app.get("/health", tags=["meta"], summary="Health check")
    def health() -> dict[str, str]:
        """Return a simple liveness indicator."""
        return {"status": "ok"}

    return app


# --------------------------------------------------------------------------------------
# Server entry point


def serve(
    host:      str = _DEFAULT_HOST,
    port:      int = _DEFAULT_PORT,
    log_level: str = "info",
    db_url:    str | None = None,
    reload:    bool = False,
    workers:   int = 1,
) -> None:
    """
    Start the uvicorn server with the qjob FastAPI application.

    Parameters
    ----------
    host : str
        Network interface to bind to.  Defaults to ``"127.0.0.1"``.
    port : int
        TCP port to listen on.  Defaults to ``8000``.
    log_level : str
        Uvicorn log level string (debug/info/warning/error).
    db_url : str | None
        Database URL passed through to ``create_app()``.  Ignored when
        ``workers > 1``; set ``QJOB_DB_URL`` in the environment instead.
    reload : bool
        Enable auto-reload for development.  Cannot be combined with
        ``workers > 1``.
    workers : int
        Number of uvicorn worker processes.  Values greater than 1 require
        ``QJOB_DB_URL`` to be set in the environment.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If both *reload* and *workers* > 1 are requested.
    """

    if reload and workers > 1:
        raise ValueError("--reload and --workers cannot be combined.")

    if workers > 1:
        # Multi-worker mode requires an import string, not an app object.
        uvicorn.run(
            "qjob.api.server:app",
            host=host,
            port=port,
            log_level=log_level.lower(),
            workers=workers,
        )
    else:
        uvicorn.run(
            create_app(db_url=db_url),
            host=host,
            port=port,
            log_level=log_level.lower(),
            reload=reload,
        )


# Module-level app instance used when ``workers > 1``.
# The DB URL is read from QJOB_DB_URL at startup time via the lifespan.
app = create_app()
