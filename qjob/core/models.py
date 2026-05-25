from __future__ import annotations

import datetime
import enum
import json
import secrets

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

    # autopep8: off
    QUEUED        = "queued"
    RUNNING       = "running"
    CANCELLING    = "cancelling"
    DONE          = "done"
    FAILED        = "failed"
    CANCELLED     = "cancelled"
    # autopep8: on


# --------------------------------------------------------------------------------------
# ORM models


class Job(Base):
    """
    Represents a single submitted job.

    Attributes
    ----------
    id : str
        12-character lowercase hex ID assigned at submission time.
    user : str
        OS username of the submitting user.
    name : str | None
        Human-readable job name from the --name directive.
    script_path : str
        Absolute path to the submitted shell script.
    workdir : str | None
        Directory used as the subprocess working directory.
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
        sqlalchemy.String(12), primary_key=True, default=lambda: secrets.token_hex(6)
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
    workdir: sqlalchemy.orm.Mapped[str | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=True
    )

    # -- Status -------------------------------------------------------------------------
    status: sqlalchemy.orm.Mapped[JobStatus] = sqlalchemy.orm.mapped_column(
        sqlalchemy.String(16),
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
        sqlalchemy.Index("ix_jobs_submitted_at", "submitted_at"),
        sqlalchemy.Index("ix_jobs_user_submitted_at", "user", "submitted_at"),
        sqlalchemy.Index("ix_jobs_status_submitted_at", "status", "submitted_at"),
        sqlalchemy.Index(
            "ix_jobs_user_status_submitted_at",
            "user",
            "status",
            "submitted_at",
        ),
    )

    # -- Factory ------------------------------------------------------------------------

    @classmethod
    def from_directives(
        cls,
        directives: parser.JobDirectives,
        user: str,
        workdir: str | None = None,
    ) -> Job:
        """
        Construct a Job instance from parsed #QJOB directives.

        Parameters
        ----------
        directives : parser.JobDirectives
            Parsed directives returned by ``parser.parse_script()``.
        user : str
            OS username of the submitting user.
        workdir : str | None
            Directory used as the subprocess working directory.

        Returns
        -------
        Job
            A new Job instance with status QUEUED. Not yet added to any session.
        """

        return cls(
            id=secrets.token_hex(6),
            user=user,
            name=directives.name,
            script_path=directives.script_path or "",
            workdir=workdir,
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
    Represents the total and currently used resources on this server.

    Only a single row (id=1) is used.  The ``used_*`` columns are updated
    atomically inside DB transactions by the scheduler, replacing the
    in-memory ResourcePool that was used in earlier versions.  This makes
    resource accounting safe across multiple processes and workers.

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
    used_cpus : int
        CPU cores currently allocated to running jobs.
    used_gpus : int
        GPU devices currently allocated to running jobs.
    used_mem_mb : int
        Memory currently allocated to running jobs in megabytes.
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
    gpu_ids: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Text, nullable=False, default="[]"
    )
    total_mem_mb: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=1024
    )
    max_walltime_sec: sqlalchemy.orm.Mapped[int | None] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=True
    )
    used_cpus: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=0
    )
    used_gpus: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=0
    )
    used_mem_mb: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, nullable=False, default=0
    )
    updated_at: sqlalchemy.orm.Mapped[datetime.datetime] = sqlalchemy.orm.mapped_column(
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    # -- Properties --------------------------------------------------------------------

    @property
    def free_cpus(self) -> int:
        """Number of unallocated CPU cores."""
        return self.total_cpus - self.used_cpus

    @property
    def free_gpus(self) -> int:
        """Number of unallocated GPU devices."""
        return self.total_gpus - self.used_gpus

    @property
    def configured_gpu_ids(self) -> list[int]:
        """GPU device IDs managed by qjob."""

        ids = json.loads(self.gpu_ids) if self.gpu_ids else []
        if ids or self.total_gpus == 0:
            return list(ids)
        return list(range(self.total_gpus))

    def set_configured_gpu_ids(self, gpu_ids: list[int]) -> None:
        """Persist the managed GPU device ID list and derived count."""

        self.gpu_ids = json.dumps(gpu_ids)
        self.total_gpus = len(gpu_ids)

    @property
    def free_mem_mb(self) -> int:
        """Unallocated memory in megabytes."""
        return self.total_mem_mb - self.used_mem_mb

    def can_fit(self, job: Job) -> bool:
        """
        Return True if this resource row has enough free capacity for *job*.

        Parameters
        ----------
        job : Job
            The job whose resource requirements are checked.

        Returns
        -------
        bool
            True when CPUs, GPUs, and memory can all be satisfied.
        """

        return (
            job.req_cpus <= self.free_cpus
            and job.req_gpus <= self.free_gpus
            and job.req_mem_mb <= self.free_mem_mb
        )

    def __repr__(self) -> str:
        return (
            f"<Resource total=({self.total_cpus}cpu/{self.total_gpus}gpu/"
            f"{self.total_mem_mb}mb) "
            f"used=({self.used_cpus}cpu/{self.used_gpus}gpu/{self.used_mem_mb}mb)>"
        )


class ApiToken(Base):
    """
    Represents a user API token for authentication.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    username : str
        OS username the token belongs to.
    token_hash : str
        SHA-256 hex digest of the raw token.
    created_at : datetime.datetime
        UTC timestamp when the token was created.
    """

    __tablename__ = "api_tokens"

    id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.Integer, primary_key=True, autoincrement=True
    )
    username: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.String(64), nullable=False, index=True
    )
    token_hash: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.String(64), nullable=False, unique=True
    )
    created_at: sqlalchemy.orm.Mapped[datetime.datetime] = sqlalchemy.orm.mapped_column(
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<ApiToken id={self.id} username={self.username!r}>"
