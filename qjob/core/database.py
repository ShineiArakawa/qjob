from __future__ import annotations

import contextlib
import os
import typing

import sqlalchemy
import sqlalchemy.orm

import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Module-level engine and session factory
#
# Both are initialised lazily by calling ``init_db()``.  All other modules must
# call ``get_session()`` rather than constructing sessions directly.

_engine:         sqlalchemy.engine.Engine | None = None
_SessionFactory: sqlalchemy.orm.sessionmaker | None = None


# --------------------------------------------------------------------------------------
# Public API


def init_db(url: str | None = None) -> sqlalchemy.engine.Engine:
    """
    Initialise the database engine and create all tables.

    This function is idempotent: calling it multiple times with the same URL
    has no effect after the first call.

    Parameters
    ----------
    url : str | None
        SQLAlchemy database URL.  When *None*, the value of the environment
        variable ``QJOB_DB_URL`` is used.  If that variable is also unset,
        the default ``sqlite:///qjob.db`` (in the current working directory)
        is used.

    Returns
    -------
    sqlalchemy.engine.Engine
        The initialised engine.

    Raises
    ------
    RuntimeError
        If called a second time with a *different* URL than the first call.
    """

    global _engine, _SessionFactory

    resolved_url = url or os.environ.get("QJOB_DB_URL", "sqlite:///qjob.db")

    if _engine is not None:
        # Compare normalised URL objects instead of raw strings.
        existing = sqlalchemy.engine.make_url(str(_engine.url))
        requested = sqlalchemy.engine.make_url(resolved_url)
        if existing != requested:
            raise RuntimeError(
                f"init_db() already called with url={_engine.url!r}; "
                f"cannot reinitialise with url={resolved_url!r}."
            )
        return _engine

    connect_args = {}
    if resolved_url.startswith("sqlite"):
        # Allow the same SQLite connection to be used across threads.
        connect_args["check_same_thread"] = False

    _engine = sqlalchemy.create_engine(
        resolved_url,
        connect_args=connect_args,
        echo=False,
    )
    _SessionFactory = sqlalchemy.orm.sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    # Create tables that do not yet exist (safe to run repeatedly).
    models.Base.metadata.create_all(_engine)

    # Insert the default resource row if the table is empty.
    _ensure_default_resource()

    return _engine


def get_engine() -> sqlalchemy.engine.Engine:
    """
    Return the active engine.

    Parameters
    ----------
    None

    Returns
    -------
    sqlalchemy.engine.Engine
        The engine created by ``init_db()``.

    Raises
    ------
    RuntimeError
        If ``init_db()`` has not been called yet.
    """

    if _engine is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")
    return _engine


@contextlib.contextmanager
def get_session() -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    """
    Yield a transactional database session, committing on success and
    rolling back on any exception.

    Parameters
    ----------
    None

    Yields
    ------
    sqlalchemy.orm.Session
        An open session bound to the engine created by ``init_db()``.

    Raises
    ------
    RuntimeError
        If ``init_db()`` has not been called yet.

    Examples
    --------
    >>> with database.get_session() as session:
    ...     session.add(job)
    """

    if _SessionFactory is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")

    session: sqlalchemy.orm.Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db() -> None:
    """
    Drop all tables and reinitialise the database from scratch.

    This function is intended for use in tests only.  It resets the
    module-level engine and session factory so that ``init_db()`` can
    be called again with a different URL.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global _engine, _SessionFactory

    if _engine is not None:
        models.Base.metadata.drop_all(_engine)
        _engine.dispose()

    _engine = None
    _SessionFactory = None


# --------------------------------------------------------------------------------------
# Private helpers


def _ensure_default_resource() -> None:
    """Insert a default Resource row (id=1) if none exists."""

    if _SessionFactory is None:
        return

    session: sqlalchemy.orm.Session = _SessionFactory()
    try:
        exists = session.get(models.Resource, 1)
        if exists is None:
            session.add(models.Resource(id=1))
            session.commit()
    finally:
        session.close()
