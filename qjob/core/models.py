from __future__ import annotations

import datetime
import enum
import uuid

import sqlalchemy
import sqlalchemy.orm

import qjob.core.parser as parser

# --------------------------------------------------------------------------------------
# Declarative base


class Base(sqlalchemy.orm.DeclarativeBase):
    pass

# --------------------------------------------------------------------------------------
# Enums


class JobStatus(str, enum.Enum):
    """Lifecycle states of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------------------
# ORM models


class Job(Base):
    """
    Represents a single submitted job.

    Attributes
    ----------
    id : str
        UUID primary key assigned at submission time.
    user : str
        OS username of the submitting user.
    name : str | None
        Human-readable job name from the --name directive.
    script_path : str
        Absolute path to the submitted shell script.
    status : JobStatus
        Current lifecycle state of the job.
    req_cpus : int
        Number of CPU cores requested.
    req_gpus : int
        Number of GPUs requested.
    req_mem_mb : int
        Amount of memory requested in megabytes.
    walltime_sec : int | None
        Maximum allowed wall-clock time in seconds. None means unlimited.
    priority : int
        Scheduling priority score (0–100).
    submitted_at : datetime.datetime
        UTC timestamp when the job was submitted.
    started_at : datetime.datetime | None
        UTC timestamp when execution began.
    finished_at : datetime.datetime | None
        UTC timestamp when execution ended (success, failure, or cancel).
    exit_code : int | None
        Process exit code. None while the job has not yet finished.
    pid : int | None
        OS process ID of the running job. None before start.
    assigned_cpus : str | None
        JSON-encoded list of assigned CPU core indices, e.g. "[0,1,2,3]".
    assigned_gpus : str | None
        JSON-encoded list of assigned GPU device IDs, e.g. "[0,1]".
    log_stdout : str | None
        Absolute path to the file capturing the job's stdout.
    log_stderr : str | None
        Absolute path to the file capturing the job's stderr.
    """

    __tablename__ = "jobs"

    # -- Identity -----------------------------------------------------------------------
    id: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.String(64), nullable=False
    )
    name: sqlalchemy.orm.Mapped[str | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.String(128), nullable=True
    )
    script_path: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=False
    )

    # -- Status -------------------------------------------------------------------------
    status: sqlalchemy.orm.Mapped[JobStatus] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Enum(JobStatus),
        nullable=False,
        default=JobStatus.QUEUED,
    )

    # -- Resource requests --------------------------------------------------------------
    req_cpus: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=1
    )
    req_gpus: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=0
    )
    req_mem_mb: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=1024
    )
    walltime_sec: sqlalchemy.orm.Mapped[int | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=True
    )
    priority: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=50
    )

    # -- Timestamps ---------------------------------------------------------------------
    submitted_at: sqlalchemy.orm.Mapped[datetime.datetime] = sqlalchemy.orm.mapped_column(
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
    started_at: sqlalchemy.orm.Mapped[datetime.datetime | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.DateTime(timezone=True), nullable=True
    )
    finished_at: sqlalchemy.orm.Mapped[datetime.datetime | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.DateTime(timezone=True), nullable=True
    )

    # -- Runtime info -------------------------------------------------------------------
    exit_code: sqlalchemy.orm.Mapped[int | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=True
    )
    pid: sqlalchemy.orm.Mapped[int | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=True
    )
    assigned_cpus: sqlalchemy.orm.Mapped[str | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=True
    )
    assigned_gpus: sqlalchemy.orm.Mapped[str | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=True
    )

    # -- Log paths ----------------------------------------------------------------------
    log_stdout: sqlalchemy.orm.Mapped[str | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=True
    )
    log_stderr: sqlalchemy.orm.Mapped[str | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=True
    )

    # -- Indexes ------------------------------------------------------------------------
    __table_args__ = (
        sqlalchemy.Index("ix_jobs_status",   "status"),
        sqlalchemy.Index("ix_jobs_user",     "user"),
        sqlalchemy.Index("ix_jobs_priority", "priority"),
    )

    # -- Factory ------------------------------------------------------------------------

    @classmethod
    def from_directives(
        cls,
        directives: parser.JobDirectives,
        user: str,
    ) -> Job:
        """
        Construct a Job instance from parsed #QJOB directives.

        Parameters
        ----------
        directives : parser.JobDirectives
            Parsed directives returned by ``parser.parse_script()``.
        user : str
            OS username of the submitting user.

        Returns
        -------
        Job
            A new Job instance with status QUEUED. Not yet added to any session.
        """

        return cls(
            id=str(uuid.uuid4()),
            user=user,
            name=directives.name,
            script_path=directives.script_path or "",
            status=JobStatus.QUEUED,
            req_cpus=directives.cpus,
            req_gpus=directives.gpus,
            req_mem_mb=directives.mem_mb,
            walltime_sec=directives.walltime_sec,
            priority=directives.priority,
        )

    # -- Helpers ------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Job id={self.id!r} name={self.name!r} "
            f"status={self.status} user={self.user!r}>"
        )


class Resource(Base):
    """
    Represents the total available resources on this server.

    Only a single row (id=1) is used. Administrators update this row
    to reflect actual hardware capacity or to reserve resources.

    Attributes
    ----------
    id : int
        Always 1. Enforced by the application layer.
    total_cpus : int
        Total number of CPU cores available for job scheduling.
    total_gpus : int
        Total number of GPU devices available for job scheduling.
    total_mem_mb : int
        Total memory available in megabytes.
    updated_at : datetime.datetime
        UTC timestamp of the last update.
    """

    __tablename__ = "resources"

    id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, primary_key=True, default=1
    )
    total_cpus: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=1
    )
    total_gpus: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=0
    )
    total_mem_mb: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=1024
    )
    updated_at: sqlalchemy.orm.Mapped[datetime.datetime] = sqlalchemy.orm.mapped_column(
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<Resource cpus={self.total_cpus} "
            f"gpus={self.total_gpus} mem_mb={self.total_mem_mb}>"
        )
