from __future__ import annotations

import dataclasses
import datetime
import getpass
import json
import os
import pathlib
import signal

import sqlalchemy.orm

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.parser as parser

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

    total_cpus:   int
    total_gpus:   int
    total_mem_mb: int
    used_cpus:    int
    used_gpus:    int
    used_mem_mb:  int


# --------------------------------------------------------------------------------------
# Job operations


def submit_job(script_path: str, user: str | None = None) -> JobInfo:
    """
    Parse a shell script and enqueue it as a new job.

    Parameters
    ----------
    script_path : str
        Path to the shell script containing ``#QJOB`` directives.
    user : str | None
        Username of the submitting user.  Defaults to the OS login name.

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

    resolved_user = user or getpass.getuser()
    directives = parser.parse_script(pathlib.Path(script_path))
    job = models.Job.from_directives(directives, user=resolved_user)

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
) -> list[JobInfo]:
    """
    Return a list of jobs, optionally filtered by user and/or status.

    Parameters
    ----------
    user : str | None
        When given, only jobs submitted by this user are returned.
    status : str | None
        When given, only jobs in this status are returned.
        Must be one of: queued, running, done, failed, cancelled.

    Returns
    -------
    list[JobInfo]
        Matching jobs ordered by submission time descending.

    Raises
    ------
    ValueError
        If *status* is not a valid ``JobStatus`` value.
    """

    if status is not None:
        try:
            status_filter = models.JobStatus(status)
        except ValueError:
            valid = [s.value for s in models.JobStatus]
            raise ValueError(
                f"Invalid status {status!r}. Valid values: {valid}"
            )

    with database.get_session() as session:
        query = session.query(models.Job)
        if user is not None:
            query = query.filter(models.Job.user == user)
        if status is not None:
            query = query.filter(models.Job.status == status_filter)
        query = query.order_by(models.Job.submitted_at.desc())
        jobs = query.all()
        return [_job_to_info(j) for j in jobs]


def cancel_job(job_id: str, user: str | None = None) -> JobInfo | None:
    """
    Request cancellation of a queued or running job.

    Only the submitting user (or root) may cancel a job.

    Parameters
    ----------
    job_id : str
        UUID of the job to cancel.
    user : str | None
        The requesting user.  Defaults to the OS login name.

    Returns
    -------
    JobInfo | None
        The updated job, or ``None`` if the job was not found.

    Raises
    ------
    PermissionError
        If *user* is not the job owner and not root.
    ValueError
        If the job is already in a terminal state.
    """

    resolved_user = user or getpass.getuser()

    with database.get_session() as session:
        job = session.get(models.Job, job_id)
        if job is None:
            return None

        if job.user != resolved_user and resolved_user != "root":
            raise PermissionError(
                f"User {resolved_user!r} is not allowed to cancel "
                f"job {job_id!r} owned by {job.user!r}."
            )

        terminal = {
            models.JobStatus.DONE,
            models.JobStatus.FAILED,
            models.JobStatus.CANCELLED,
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

        job.status = models.JobStatus.CANCELLED
        job.finished_at = datetime.datetime.now(datetime.timezone.utc)
        return _job_to_info(job)


def get_log(job_id: str, stream: str = "stdout") -> str:
    """
    Return the log content for a job.

    Parameters
    ----------
    job_id : str
        UUID of the job.
    stream : str
        Which log stream to read: ``"stdout"`` or ``"stderr"``.

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

        return path.read_text(encoding="utf-8", errors="replace")


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
                used_cpus=0,  used_gpus=0,  used_mem_mb=0,
            )

        return ResourceInfo(
            total_cpus=row.total_cpus,
            total_gpus=row.total_gpus,
            total_mem_mb=row.total_mem_mb,
            used_cpus=row.used_cpus,
            used_gpus=row.used_gpus,
            used_mem_mb=row.used_mem_mb,
        )


def set_resources(
    total_cpus:   int | None = None,
    total_gpus:   int | None = None,
    total_mem_mb: int | None = None,
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

    if total_cpus is None and total_gpus is None and total_mem_mb is None:
        raise ValueError("At least one resource field must be specified.")

    with database.get_session() as session:
        row: models.Resource | None = session.get(models.Resource, 1)
        if row is None:
            row = models.Resource(id=1)
            session.add(row)

        if total_cpus is not None:
            row.total_cpus = total_cpus
        if total_gpus is not None:
            row.total_gpus = total_gpus
        if total_mem_mb is not None:
            row.total_mem_mb = total_mem_mb

        row.updated_at = datetime.datetime.now(datetime.timezone.utc)

        return ResourceInfo(
            total_cpus=row.total_cpus,
            total_gpus=row.total_gpus,
            total_mem_mb=row.total_mem_mb,
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
    )


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
