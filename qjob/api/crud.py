from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import signal

import coolname
import sqlalchemy.orm

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.parser as parser

# --------------------------------------------------------------------------------------
# Constants

DEFAULT_LOG_MAX_BYTES: int = 1024 * 1024
MAX_LOG_MAX_BYTES:     int = 16 * 1024 * 1024
DEFAULT_JOB_LIST_LIMIT: int = 100
MAX_JOB_LIST_LIMIT:     int = 1000
JOB_LIST_SORT_KEYS: tuple[str, ...] = (
    "submitted",
    "started",
    "finished",
    "priority",
    "user",
)


def _normalise_gpu_ids(gpu_ids: list[int]) -> list[int]:
    """Validate and return GPU IDs in administrator-specified order."""

    normalised: list[int] = []
    for gpu_id in gpu_ids:
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
            raise ValueError("gpu_ids must contain only integer GPU IDs.")
        if gpu_id < 0:
            raise ValueError("gpu_ids must contain only non-negative GPU IDs.")
        normalised.append(gpu_id)
    if len(set(normalised)) != len(normalised):
        raise ValueError("gpu_ids must not contain duplicates.")
    return normalised

# --------------------------------------------------------------------------------------
# Return value data classes
#
# These data classes are the public contract of the CRUD layer.
# Routers depend only on these types, never on ORM models directly.


@dataclasses.dataclass
class JobInfo:
    """
    Summarised view of a single job for display purposes.

    Attributes
    ----------
    id : str
        UUID of the job.
    user : str
        Submitting user.
    name : str | None
        Human-readable job name.
    status : str
        Current lifecycle status string.
    req_cpus : int
        Number of CPU cores requested.
    req_gpus : int
        Number of GPUs requested.
    req_mem_mb : int
        Memory requested in megabytes.
    priority : int
        Scheduling priority score (0–100).
    submitted_at : datetime.datetime | None
        UTC timestamp when the job was submitted.
    started_at : datetime.datetime | None
        UTC timestamp when execution began.
    finished_at : datetime.datetime | None
        UTC timestamp when execution ended.
    exit_code : int | None
        Process exit code.
    log_stdout : str | None
        Path to the stdout log file.
    log_stderr : str | None
        Path to the stderr log file.
    workdir : str | None
        Directory used as the subprocess working directory.
    """

    id:           str
    user:         str
    name:         str | None
    status:       str
    req_cpus:     int
    req_gpus:     int
    req_mem_mb:   int
    priority:     int
    submitted_at: datetime.datetime | None
    started_at:   datetime.datetime | None
    finished_at:  datetime.datetime | None
    exit_code:    int | None
    log_stdout:   str | None
    log_stderr:   str | None
    workdir:      str | None


@dataclasses.dataclass
class ResourceInfo:
    """
    Current resource availability summary.

    Attributes
    ----------
    total_cpus : int
        Total CPU cores configured.
    total_gpus : int
        Total GPU devices configured.
    total_mem_mb : int
        Total memory configured in megabytes.
    used_cpus : int
        CPU cores currently allocated to running jobs.
    used_gpus : int
        GPU devices currently allocated to running jobs.
    used_mem_mb : int
        Memory currently allocated to running jobs in megabytes.
    """

    total_cpus:       int
    total_gpus:       int
    total_mem_mb:     int
    max_walltime_sec: int | None
    used_cpus:        int
    used_gpus:        int
    used_mem_mb:      int
    gpu_ids:          list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class JobListPage:
    """
    Paginated job query result.

    Attributes
    ----------
    jobs : list[JobInfo]
        Jobs in the requested page.
    total : int
        Total matching jobs before pagination.
    limit : int
        Maximum requested page size.
    offset : int
        Number of matching rows skipped.
    """

    jobs:   list[JobInfo]
    total:  int
    limit:  int
    offset: int


# --------------------------------------------------------------------------------------
# Job operations


def submit_job(
    script_path: str,
    user:        str,
    workdir:     str | None = None,
) -> JobInfo:
    """
    Parse a shell script and enqueue it as a new job.

    Parameters
    ----------
    script_path : str
        Path to the shell script containing ``#QJOB`` directives.
    user : str
        Authenticated username of the submitting user.
    workdir : str | None
        Directory used as the subprocess working directory. Defaults to the
        submitted script's parent directory.

    Returns
    -------
    JobInfo
        The newly created job.

    Raises
    ------
    FileNotFoundError
        If *script_path* does not exist.
    parser.DirectiveParseError
        If any ``#QJOB`` directive is malformed.
    """

    resolved_user = user

    # --------------------------------------------------------------------------------------
    # Parse directives from the script file.

    directives = parser.parse_script(pathlib.Path(script_path))

    # --------------------------------------------------------------------------------------
    # If no name directive was given, generate a random human-readable name.

    if directives.name is None:
        directives.name = coolname.generate_slug(2)

    # --------------------------------------------------------------------------------------
    # Check current resource limits and apply defaults as needed.

    with database.get_session() as session:
        resource_row: models.Resource | None = session.get(models.Resource, 1)

        total_cpus = resource_row.total_cpus if resource_row is not None else None
        total_gpus = len(resource_row.configured_gpu_ids) if resource_row is not None else None
        total_mem_mb = resource_row.total_mem_mb if resource_row is not None else None
        max_walltime = resource_row.max_walltime_sec if resource_row is not None else None

    if total_cpus is not None and (directives.cpus < 1 or directives.cpus > total_cpus):
        raise ValueError(
            f"Requested CPU count {directives.cpus} is out of bounds "
            f"(1–{total_cpus})."
        )

    if total_gpus is not None and (directives.gpus < 0 or directives.gpus > total_gpus):
        raise ValueError(
            f"Requested GPU count {directives.gpus} is out of bounds "
            f"(0–{total_gpus})."
        )

    if total_mem_mb is not None and (directives.mem_mb < 1 or directives.mem_mb > total_mem_mb):
        raise ValueError(
            f"Requested memory {directives.mem_mb} MB is out of bounds "
            f"(1–{total_mem_mb} MB)."
        )

    if max_walltime is not None:
        if directives.walltime_sec is None:
            directives.walltime_sec = max_walltime
        elif directives.walltime_sec > max_walltime:
            raise ValueError(
                f"Requested walltime {directives.walltime_sec}s exceeds "
                f"the maximum allowed {max_walltime}s."
            )

    # --------------------------------------------------------------------------------------
    # Create a new Job record in the database with the parsed directives and defaults.

    resolved_workdir = _resolve_workdir(workdir, directives.script_path)
    job = models.Job.from_directives(
        directives,
        user=resolved_user,
        workdir=resolved_workdir,
    )

    with database.get_session() as session:
        session.add(job)

    return _job_to_info(job)


def get_job(job_id: str) -> JobInfo | None:
    """
    Return details of a single job.

    Parameters
    ----------
    job_id : str
        UUID of the job to look up.

    Returns
    -------
    JobInfo | None
        The job, or ``None`` if no job with that ID exists.
    """

    with database.get_session() as session:
        job = session.get(models.Job, job_id)
        if job is None:
            return None
        return _job_to_info(job)


def list_jobs(
    user:   str | None = None,
    status: str | None = None,
    states: list[str] | None = None,
    since:  datetime.datetime | None = None,
    sort:   str = "submitted",
    limit:  int = DEFAULT_JOB_LIST_LIMIT,
    offset: int = 0,
) -> JobListPage:
    """
    Return a list of jobs, optionally filtered by user and/or state.

    Parameters
    ----------
    user : str | None
        When given, only jobs submitted by this user are returned.
    status : str | None
        Legacy single-state filter.  When given, only jobs in this state are returned.
    states : list[str] | None
        When given, only jobs in these states are returned.
    since : datetime.datetime | None
        When given, only jobs submitted at or after this time are returned.
    sort : str
        Sort key: submitted, started, finished, priority, or user.
    limit : int
        Maximum number of jobs to return.
    offset : int
        Number of matching jobs to skip.

    Returns
    -------
    JobListPage
        Matching page ordered by submission time descending plus total count.

    Raises
    ------
    ValueError
        If *status*, *states*, *since*, *sort*, *limit*, or *offset* is invalid.
    """

    if limit <= 0:
        raise ValueError("limit must be greater than 0.")
    if limit > MAX_JOB_LIST_LIMIT:
        raise ValueError(f"limit must be <= {MAX_JOB_LIST_LIMIT}.")
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0.")
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=datetime.timezone.utc)
    if sort not in JOB_LIST_SORT_KEYS:
        raise ValueError(
            f"Invalid sort {sort!r}. Valid values: {list(JOB_LIST_SORT_KEYS)}"
        )

    if status is not None and states is not None:
        raise ValueError("status and states cannot both be specified.")

    requested_states = (
        states if states is not None else ([status] if status is not None else None)
    )
    status_filters: list[models.JobStatus] | None = None
    if requested_states is not None:
        if not requested_states:
            raise ValueError("states must not be empty.")
        status_filters = []
        seen: set[models.JobStatus] = set()
        valid = [s.value for s in models.JobStatus]
        for state in requested_states:
            try:
                status_filter = models.JobStatus(state)
            except ValueError:
                raise ValueError(
                    f"Invalid state {state!r}. Valid values: {valid}"
                )
            if status_filter in seen:
                raise ValueError("states must not contain duplicates.")
            seen.add(status_filter)
            status_filters.append(status_filter)

    with database.get_session() as session:
        query = session.query(models.Job)
        if user is not None:
            query = query.filter(models.Job.user == user)
        if status_filters is not None:
            query = query.filter(models.Job.status.in_(status_filters))
        if since is not None:
            query = query.filter(models.Job.submitted_at >= since)
        total = query.count()
        query = (
            query.order_by(*_job_list_order_by(sort))
            .offset(offset)
            .limit(limit)
        )
        jobs = query.all()
        return JobListPage(
            jobs=[_job_to_info(j) for j in jobs],
            total=total,
            limit=limit,
            offset=offset,
        )


def _job_list_order_by(sort: str) -> tuple[object, ...]:
    """Return SQLAlchemy order-by clauses for a job list sort key."""

    if sort == "submitted":
        return (models.Job.submitted_at.desc(),)
    if sort == "started":
        return (
            models.Job.started_at.desc().nullslast(),
            models.Job.submitted_at.desc(),
        )
    if sort == "finished":
        return (
            models.Job.finished_at.desc().nullslast(),
            models.Job.submitted_at.desc(),
        )
    if sort == "priority":
        return (models.Job.priority.desc(), models.Job.submitted_at.desc())
    if sort == "user":
        return (models.Job.user.asc(), models.Job.submitted_at.desc())
    raise ValueError(
        f"Invalid sort {sort!r}. Valid values: {list(JOB_LIST_SORT_KEYS)}"
    )


def cancel_job(job_id: str, user: str, admin: bool = False) -> JobInfo | None:
    """
    Request cancellation of a queued or running job.

    Only the submitting user (or an admin) may cancel a job.

    Parameters
    ----------
    job_id : str
        UUID of the job to cancel.
    user : str
        Authenticated username of the requesting user.
    admin : bool
        When True, ownership check is skipped (admin cancellation).

    Returns
    -------
    JobInfo | None
        The updated job, or ``None`` if the job was not found.

    Raises
    ------
    PermissionError
        If *user* is not the job owner and *admin* is False.
    ValueError
        If the job is already in a terminal state.
    """

    resolved_user = user

    with database.get_session() as session:
        job = session.get(models.Job, job_id)
        if job is None:
            return None

        if not admin and job.user != resolved_user:
            raise PermissionError(
                f"User {resolved_user!r} is not allowed to cancel "
                f"job {job_id!r} owned by {job.user!r}."
            )

        terminal = {
            models.JobStatus.DONE,
            models.JobStatus.FAILED,
            models.JobStatus.CANCELLED,
            models.JobStatus.CANCELLING,
        }
        if job.status in terminal:
            job_status = job.status.value if isinstance(job.status, models.JobStatus) else job.status
            raise ValueError(
                f"Job {job_id!r} is already in terminal state {job_status!r}."
            )

        if job.status == models.JobStatus.RUNNING and job.pid is not None:
            try:
                if os.name == "posix":
                    os.killpg(job.pid, signal.SIGTERM)
                else:
                    os.kill(job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            job.status = models.JobStatus.CANCELLING
        else:
            # QUEUED jobs have no process — terminate immediately.
            job.status = models.JobStatus.CANCELLED
            job.finished_at = datetime.datetime.now(datetime.timezone.utc)

        return _job_to_info(job)


def get_log(
    job_id:    str,
    stream:    str = "stdout",
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
) -> str:
    """
    Return the log content for a job.

    Parameters
    ----------
    job_id : str
        UUID of the job.
    stream : str
        Which log stream to read: ``"stdout"`` or ``"stderr"``.
    max_bytes : int
        Maximum number of bytes to read from the end of the log file.

    Returns
    -------
    str
        The log content, or an explanatory message if not yet available.

    Raises
    ------
    ValueError
        If *stream* is not ``"stdout"`` or ``"stderr"``.
    """

    if stream not in ("stdout", "stderr"):
        raise ValueError(f"stream must be 'stdout' or 'stderr', got {stream!r}.")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than 0.")
    if max_bytes > MAX_LOG_MAX_BYTES:
        raise ValueError(f"max_bytes must be <= {MAX_LOG_MAX_BYTES}.")

    with database.get_session() as session:
        job = session.get(models.Job, job_id)
        if job is None:
            return f"Job {job_id!r} not found."

        log_path = job.log_stdout if stream == "stdout" else job.log_stderr
        if log_path is None:
            return (
                f"Log not yet available for job {job_id!r} "
                f"(status: {job.status if isinstance(job.status, str) else job.status.value})."
            )

        path = pathlib.Path(log_path)
        if not path.exists():
            return f"Log file not found: {log_path}"

        return _read_log_tail(path, max_bytes=max_bytes)


# --------------------------------------------------------------------------------------
# Resource operations


def get_resources() -> ResourceInfo:
    """
    Return the current resource configuration and usage summary.

    Usage counts are read from the ``resources`` table which is kept
    up-to-date by the scheduler via atomic DB transactions.

    Parameters
    ----------
    None

    Returns
    -------
    ResourceInfo
        Total and used resource counts.
    """

    with database.get_session() as session:
        row: models.Resource | None = session.get(models.Resource, 1)
        if row is None:
            return ResourceInfo(
                total_cpus=0, total_gpus=0, total_mem_mb=0,
                max_walltime_sec=None,
                gpu_ids=[],
                used_cpus=0,  used_gpus=0,  used_mem_mb=0,
            )

        gpu_ids = row.configured_gpu_ids
        return ResourceInfo(
            total_cpus=row.total_cpus,
            total_gpus=len(gpu_ids),
            total_mem_mb=row.total_mem_mb,
            max_walltime_sec=row.max_walltime_sec,
            gpu_ids=gpu_ids,
            used_cpus=row.used_cpus,
            used_gpus=row.used_gpus,
            used_mem_mb=row.used_mem_mb,
        )


def set_resources(
    total_cpus:       int | None = None,
    total_gpus:       int | None = None,
    total_mem_mb:     int | None = None,
    max_walltime_sec: int | None = None,
    gpu_ids:          list[int] | None = None,
) -> ResourceInfo:
    """
    Update the resource limits.

    Parameters
    ----------
    total_cpus : int | None
        New total CPU core count.
    total_gpus : int | None
        New total GPU device count.
    total_mem_mb : int | None
        New total memory in megabytes.

    Returns
    -------
    ResourceInfo
        The updated resource configuration.

    Raises
    ------
    ValueError
        If all arguments are ``None``.
    """

    if total_cpus is None and total_gpus is None and total_mem_mb is None and max_walltime_sec is None and gpu_ids is None:
        raise ValueError("At least one resource field must be specified.")
    if total_cpus is not None and total_cpus <= 0:
        raise ValueError("total_cpus must be greater than 0.")
    if total_gpus is not None and total_gpus < 0:
        raise ValueError("total_gpus must be greater than or equal to 0.")
    if total_mem_mb is not None and total_mem_mb <= 0:
        raise ValueError("total_mem_mb must be greater than 0.")
    if max_walltime_sec is not None and max_walltime_sec <= 0:
        raise ValueError("max_walltime_sec must be greater than 0.")

    normalised_gpu_ids = _normalise_gpu_ids(gpu_ids) if gpu_ids is not None else None
    if (
        total_gpus is not None
        and normalised_gpu_ids is not None
        and total_gpus != len(normalised_gpu_ids)
    ):
        raise ValueError("total_gpus must match the number of gpu_ids when both are set.")

    with database.get_session() as session:
        row: models.Resource | None = session.get(models.Resource, 1)
        if row is None:
            row = models.Resource(id=1)
            session.add(row)

        if total_cpus is not None:
            row.total_cpus = total_cpus
        if total_gpus is not None:
            row.set_configured_gpu_ids(list(range(total_gpus)))
        if normalised_gpu_ids is not None:
            row.set_configured_gpu_ids(normalised_gpu_ids)
        if total_mem_mb is not None:
            row.total_mem_mb = total_mem_mb
        if max_walltime_sec is not None:
            row.max_walltime_sec = max_walltime_sec

        row.updated_at = datetime.datetime.now(datetime.timezone.utc)

        current_gpu_ids = row.configured_gpu_ids
        return ResourceInfo(
            total_cpus=row.total_cpus,
            total_gpus=len(current_gpu_ids),
            total_mem_mb=row.total_mem_mb,
            max_walltime_sec=row.max_walltime_sec,
            gpu_ids=current_gpu_ids,
            used_cpus=row.used_cpus,
            used_gpus=row.used_gpus,
            used_mem_mb=row.used_mem_mb,
        )


# --------------------------------------------------------------------------------------
# Private helpers


def _job_to_info(job: models.Job) -> JobInfo:
    """
    Convert an ORM Job instance to a JobInfo data class.

    Parameters
    ----------
    job : models.Job
        The ORM instance to convert.

    Returns
    -------
    JobInfo
        A plain data object safe to use outside a DB session.
    """

    status = job.status if isinstance(job.status, str) else job.status.value

    return JobInfo(
        id=job.id,
        user=job.user,
        name=job.name,
        status=status,
        req_cpus=job.req_cpus,
        req_gpus=job.req_gpus,
        req_mem_mb=job.req_mem_mb,
        priority=job.priority,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        exit_code=job.exit_code,
        log_stdout=job.log_stdout,
        log_stderr=job.log_stderr,
        workdir=job.workdir,
    )


def _resolve_workdir(workdir: str | None, script_path: str | None) -> str:
    """
    Return an absolute working directory for a submitted job.

    When *workdir* is omitted, the script's parent directory is used.
    """

    if workdir is None:
        if not script_path:
            raise ValueError("script_path is required to infer workdir.")
        return str(pathlib.Path(script_path).resolve().parent)

    path = pathlib.Path(workdir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Working directory not found: {workdir}")
    if not path.is_dir():
        raise NotADirectoryError(f"Working directory is not a directory: {workdir}")
    return str(path)


def _read_log_tail(path: pathlib.Path, max_bytes: int) -> str:
    """
    Read at most *max_bytes* from the end of *path*.

    This keeps log responses bounded even when stdout/stderr grows very large.
    A short marker is prepended when older content was omitted.
    """

    size = path.stat().st_size
    offset = max(0, size - max_bytes)

    with path.open("rb") as f:
        if offset:
            f.seek(offset)
        data = f.read(max_bytes)

    text = data.decode("utf-8", errors="replace")
    if offset:
        return (
            f"[log truncated: showing last {len(data)} of {size} bytes]\n"
            f"{text}"
        )
    return text


def _compute_usage(
    session: sqlalchemy.orm.Session,
) -> tuple[int, int, int]:
    """
    Compute total CPU, GPU, and memory usage from currently running jobs.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An open DB session.

    Returns
    -------
    tuple[int, int, int]
        A ``(used_cpus, used_gpus, used_mem_mb)`` triple.
    """

    running: list[models.Job] = (
        session.query(models.Job)
        .filter(models.Job.status == models.JobStatus.RUNNING)
        .all()
    )

    used_cpus = 0
    used_gpus = 0
    used_mem_mb = 0

    for job in running:
        cpu_ids: list[int] = json.loads(job.assigned_cpus) if job.assigned_cpus else []
        gpu_ids: list[int] = json.loads(job.assigned_gpus) if job.assigned_gpus else []
        used_cpus += len(cpu_ids)
        used_gpus += len(gpu_ids)
        used_mem_mb += job.req_mem_mb

    return used_cpus, used_gpus, used_mem_mb
