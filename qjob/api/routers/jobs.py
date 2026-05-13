from __future__ import annotations

import typing

import fastapi

import qjob.api.crud as crud
import qjob.api.schemas as schemas

# --------------------------------------------------------------------------------------
# Router

router = fastapi.APIRouter(prefix="/jobs", tags=["jobs"])


# --------------------------------------------------------------------------------------
# POST /jobs


@router.post(
    "",
    response_model=schemas.JobResponse,
    status_code=201,
    summary="Submit a new job",
)
def submit_job(body: schemas.JobSubmitRequest) -> schemas.JobResponse:
    """
    Parse a shell script and enqueue it as a new job.

    Parameters
    ----------
    body : schemas.JobSubmitRequest
        Request body containing the script path and optional username.

    Returns
    -------
    schemas.JobResponse
        The newly created job.

    Raises
    ------
    fastapi.HTTPException
        404 if the script file does not exist.
        422 if a #QJOB directive is malformed.
    """

    try:
        info = crud.submit_job(
            script_path=body.script_path,
            user=body.user,
            workdir=body.workdir,
        )
    except FileNotFoundError as exc:
        raise fastapi.HTTPException(status_code=404, detail=str(exc))
    except NotADirectoryError as exc:
        raise fastapi.HTTPException(status_code=422, detail=str(exc))
    except crud.parser.DirectiveParseError as exc:
        raise fastapi.HTTPException(status_code=422, detail=str(exc))

    return _info_to_response(info)


# --------------------------------------------------------------------------------------
# GET /jobs


@router.get(
    "",
    response_model=schemas.JobListResponse,
    summary="List jobs",
)
def list_jobs(
    user:   typing.Optional[str] = fastapi.Query(None, description="Filter by username."),
    status: typing.Optional[str] = fastapi.Query(None, description="Filter by status."),
    limit:  int = fastapi.Query(
        crud.DEFAULT_JOB_LIST_LIMIT,
        ge=1,
        le=crud.MAX_JOB_LIST_LIMIT,
        description="Maximum number of jobs to return.",
    ),
    offset: int = fastapi.Query(
        0,
        ge=0,
        description="Number of matching jobs to skip.",
    ),
) -> schemas.JobListResponse:
    """
    Return jobs optionally filtered by user and/or status.

    Parameters
    ----------
    user : str | None
        When given, only jobs submitted by this user are returned.
    status : str | None
        When given, only jobs in this status are returned.
    limit : int
        Maximum number of jobs to return.
    offset : int
        Number of matching jobs to skip.

    Returns
    -------
    schemas.JobListResponse
        Matching jobs ordered by submission time descending.

    Raises
    ------
    fastapi.HTTPException
        400 if *status* is not a valid value.
    """

    try:
        page = crud.list_jobs(user=user, status=status, limit=limit, offset=offset)
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc))

    return schemas.JobListResponse(
        jobs=[_info_to_response(j) for j in page.jobs],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


# --------------------------------------------------------------------------------------
# GET /jobs/{job_id}


@router.get(
    "/{job_id}",
    response_model=schemas.JobResponse,
    summary="Get job details",
)
def get_job(job_id: str) -> schemas.JobResponse:
    """
    Return details of a single job.

    Parameters
    ----------
    job_id : str
        UUID of the job to look up.

    Returns
    -------
    schemas.JobResponse
        The job details.

    Raises
    ------
    fastapi.HTTPException
        404 if no job with *job_id* exists.
    """

    info = crud.get_job(job_id)
    if info is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Job {job_id!r} not found."
        )

    return _info_to_response(info)


# --------------------------------------------------------------------------------------
# DELETE /jobs/{job_id}


@router.delete(
    "/{job_id}",
    response_model=schemas.JobResponse,
    summary="Cancel a job",
)
def cancel_job(
    job_id: str,
    user:   typing.Optional[str] = fastapi.Query(
        None, description="Requesting username."
    ),
) -> schemas.JobResponse:
    """
    Cancel a queued or running job.

    Parameters
    ----------
    job_id : str
        UUID of the job to cancel.
    user : str | None
        The requesting user.  Defaults to the OS login name.

    Returns
    -------
    schemas.JobResponse
        The updated job.

    Raises
    ------
    fastapi.HTTPException
        404 if the job does not exist.
        403 if the requesting user does not own the job.
        409 if the job is already in a terminal state.
    """

    try:
        info = crud.cancel_job(job_id, user=user)
    except PermissionError as exc:
        raise fastapi.HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=409, detail=str(exc))

    if info is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Job {job_id!r} not found."
        )

    return _info_to_response(info)


# --------------------------------------------------------------------------------------
# GET /jobs/{job_id}/log


@router.get(
    "/{job_id}/log",
    response_model=schemas.JobLogResponse,
    summary="Get job log",
)
def get_log(
    job_id: str,
    stream: str = fastapi.Query("stdout", description="Log stream: stdout or stderr."),
    max_bytes: int = fastapi.Query(
        crud.DEFAULT_LOG_MAX_BYTES,
        ge=1,
        le=crud.MAX_LOG_MAX_BYTES,
        description="Maximum bytes to return from the end of the log.",
    ),
) -> schemas.JobLogResponse:
    """
    Return the log content for a job.

    Parameters
    ----------
    job_id : str
        UUID of the job.
    stream : str
        Which log stream to read: ``"stdout"`` or ``"stderr"``.
    max_bytes : int
        Maximum bytes to return from the end of the log.

    Returns
    -------
    schemas.JobLogResponse
        The log content.

    Raises
    ------
    fastapi.HTTPException
        400 if *stream* is not ``"stdout"`` or ``"stderr"``.
    """

    try:
        content = crud.get_log(job_id, stream=stream, max_bytes=max_bytes)
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc))

    return schemas.JobLogResponse(job_id=job_id, stream=stream, content=content)


# --------------------------------------------------------------------------------------
# Private helpers


def _info_to_response(info: crud.JobInfo) -> schemas.JobResponse:
    """
    Convert a JobInfo data class to a JobResponse schema.

    Parameters
    ----------
    info : crud.JobInfo
        The service-layer data object.

    Returns
    -------
    schemas.JobResponse
        The API response model.
    """

    return schemas.JobResponse(
        id=info.id,
        user=info.user,
        name=info.name,
        status=info.status,
        req_cpus=info.req_cpus,
        req_gpus=info.req_gpus,
        req_mem_mb=info.req_mem_mb,
        priority=info.priority,
        submitted_at=info.submitted_at,
        started_at=info.started_at,
        finished_at=info.finished_at,
        exit_code=info.exit_code,
        log_stdout=info.log_stdout,
        log_stderr=info.log_stderr,
        workdir=info.workdir,
    )
