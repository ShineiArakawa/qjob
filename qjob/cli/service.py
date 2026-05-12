from __future__ import annotations

import asyncio
import dataclasses
import datetime
import getpass
import os
import typing

import httpx

import qjob.core.parser as parser

# --------------------------------------------------------------------------------------
# Server connection settings

_DEFAULT_API_URL: str = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT: float = 10.0


def _api_url() -> str:
    """Return the base URL of the qjob API server."""
    return os.environ.get("QJOB_API_URL", _DEFAULT_API_URL).rstrip("/")


def _async_client() -> httpx.AsyncClient:
    """Return a configured async httpx client."""
    return httpx.AsyncClient(base_url=_api_url(), timeout=_DEFAULT_TIMEOUT)


def _run(coro: typing.Coroutine) -> typing.Any:
    """
    Run a coroutine from synchronous code.

    Parameters
    ----------
    coro : typing.Coroutine
        The coroutine to execute.

    Returns
    -------
    typing.Any
        The return value of the coroutine.
    """

    return asyncio.run(coro)


# --------------------------------------------------------------------------------------
# Return value data classes


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
# Job operations — public synchronous API
#
# Each public function delegates to an async counterpart via _run().
# This keeps the CLI interface synchronous while using AsyncClient internally.


def submit_job(script_path: str, user: str | None = None) -> JobInfo:
    """
    Submit a shell script to the job queue via the API.

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
        If the script file does not exist (validated locally before the request).
    parser.DirectiveParseError
        If a ``#QJOB`` directive is malformed (validated locally before the request).
    ConnectionError
        If the API server is unreachable.
    """

    # Validate locally before making the network round trip.
    parser.parse_script(script_path)
    return _run(_async_submit_job(script_path, user))


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

    return _run(_async_get_job(job_id))


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

    Returns
    -------
    list[JobInfo]
        Matching jobs ordered by submission time descending.

    Raises
    ------
    ValueError
        If *status* is not a valid ``JobStatus`` value.
    """

    return _run(_async_list_jobs(user, status))


def cancel_job(job_id: str, user: str | None = None) -> JobInfo | None:
    """
    Request cancellation of a queued or running job.

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
        If the user is not the job owner and not root.
    ValueError
        If the job is already in a terminal state.
    """

    return _run(_async_cancel_job(job_id, user))


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
    return _run(_async_get_log(job_id, stream))


# --------------------------------------------------------------------------------------
# Resource operations


def get_resources() -> ResourceInfo:
    """
    Return the current resource configuration and usage summary.

    Parameters
    ----------
    None

    Returns
    -------
    ResourceInfo
        Total and used resource counts.
    """

    return _run(_async_get_resources())


def set_resources(
    total_cpus:   int | None = None,
    total_gpus:   int | None = None,
    total_mem_mb: int | None = None,
) -> ResourceInfo:
    """
    Update the resource limits (admin only).

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
        If all arguments are None.
    """

    if total_cpus is None and total_gpus is None and total_mem_mb is None:
        raise ValueError("At least one resource field must be specified.")
    return _run(_async_set_resources(total_cpus, total_gpus, total_mem_mb))


# --------------------------------------------------------------------------------------
# Async implementations


async def _async_submit_job(
    script_path: str,
    user:        str | None,
) -> JobInfo:
    """Async implementation of submit_job."""

    resolved_user = user or getpass.getuser()
    async with _async_client() as client:
        response = await client.post(
            "/jobs",
            json={"script_path": script_path, "user": resolved_user},
        )
        _raise_for_status(response)
    return _parse_job(response.json())


async def _async_get_job(job_id: str) -> JobInfo | None:
    """Async implementation of get_job."""

    async with _async_client() as client:
        response = await client.get(f"/jobs/{job_id}")
    if response.status_code == 404:
        return None
    _raise_for_status(response)
    return _parse_job(response.json())


async def _async_list_jobs(
    user:   str | None,
    status: str | None,
) -> list[JobInfo]:
    """Async implementation of list_jobs."""

    params: dict[str, str] = {}
    if user is not None:
        params["user"] = user
    if status is not None:
        params["status"] = status

    async with _async_client() as client:
        response = await client.get("/jobs", params=params)
    if response.status_code == 400:
        raise ValueError(response.json().get("detail", "Invalid request."))
    _raise_for_status(response)
    return [_parse_job(j) for j in response.json()["jobs"]]


async def _async_cancel_job(
    job_id: str,
    user:   str | None,
) -> JobInfo | None:
    """Async implementation of cancel_job."""

    resolved_user = user or getpass.getuser()
    async with _async_client() as client:
        response = await client.delete(
            f"/jobs/{job_id}", params={"user": resolved_user}
        )
    if response.status_code == 404:
        return None
    if response.status_code == 403:
        raise PermissionError(response.json().get("detail", "Permission denied."))
    if response.status_code == 409:
        raise ValueError(response.json().get("detail", "Job is in a terminal state."))
    _raise_for_status(response)
    return _parse_job(response.json())


async def _async_get_log(job_id: str, stream: str) -> str:
    """Async implementation of get_log."""

    async with _async_client() as client:
        response = await client.get(
            f"/jobs/{job_id}/log", params={"stream": stream}
        )
    if response.status_code == 400:
        raise ValueError(response.json().get("detail", "Invalid request."))
    _raise_for_status(response)
    return response.json()["content"]


async def _async_get_resources() -> ResourceInfo:
    """Async implementation of get_resources."""

    async with _async_client() as client:
        response = await client.get("/resources")
    _raise_for_status(response)
    return _parse_resource(response.json())


async def _async_set_resources(
    total_cpus:   int | None,
    total_gpus:   int | None,
    total_mem_mb: int | None,
) -> ResourceInfo:
    """Async implementation of set_resources."""

    body: dict[str, typing.Any] = {}
    if total_cpus is not None:
        body["total_cpus"] = total_cpus
    if total_gpus is not None:
        body["total_gpus"] = total_gpus
    if total_mem_mb is not None:
        body["total_mem_mb"] = total_mem_mb

    async with _async_client() as client:
        response = await client.put("/resources", json=body)
    if response.status_code == 400:
        raise ValueError(response.json().get("detail", "Invalid request."))
    _raise_for_status(response)
    return _parse_resource(response.json())


# --------------------------------------------------------------------------------------
# Private helpers


def _raise_for_status(response: httpx.Response) -> None:
    """
    Raise a ConnectionError with a human-readable message on HTTP errors.

    Parameters
    ----------
    response : httpx.Response
        The response to check.

    Returns
    -------
    None

    Raises
    ------
    ConnectionError
        If the response status code indicates an unhandled error.
    """

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        raise ConnectionError(
            f"API request failed [{exc.response.status_code}]: {detail}"
        ) from exc


def _parse_job(data: dict) -> JobInfo:
    """
    Parse a job JSON response dict into a JobInfo data class.

    Parameters
    ----------
    data : dict
        Raw JSON dict from the API response.

    Returns
    -------
    JobInfo
        The parsed job.
    """

    def _dt(val: str | None) -> datetime.datetime | None:
        if val is None:
            return None
        return datetime.datetime.fromisoformat(val)

    return JobInfo(
        id=data["id"],
        user=data["user"],
        name=data.get("name"),
        status=data["status"],
        req_cpus=data["req_cpus"],
        req_gpus=data["req_gpus"],
        req_mem_mb=data["req_mem_mb"],
        priority=data["priority"],
        submitted_at=_dt(data.get("submitted_at")),
        started_at=_dt(data.get("started_at")),
        finished_at=_dt(data.get("finished_at")),
        exit_code=data.get("exit_code"),
        log_stdout=data.get("log_stdout"),
        log_stderr=data.get("log_stderr"),
    )


def _parse_resource(data: dict) -> ResourceInfo:
    """
    Parse a resource JSON response dict into a ResourceInfo data class.

    Parameters
    ----------
    data : dict
        Raw JSON dict from the API response.

    Returns
    -------
    ResourceInfo
        The parsed resource info.
    """

    return ResourceInfo(
        total_cpus=data["total_cpus"],
        total_gpus=data["total_gpus"],
        total_mem_mb=data["total_mem_mb"],
        used_cpus=data["used_cpus"],
        used_gpus=data["used_gpus"],
        used_mem_mb=data["used_mem_mb"],
    )
