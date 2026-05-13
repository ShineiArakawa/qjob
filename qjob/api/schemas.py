from __future__ import annotations

import datetime

import pydantic

# --------------------------------------------------------------------------------------
# Job schemas


class JobSubmitRequest(pydantic.BaseModel):
    """
    Request body for POST /jobs.

    Attributes
    ----------
    script_path : str
        Absolute path to the shell script on the server.
    user : str | None
        Submitting username.  When None the server resolves it from the OS.
    """

    script_path: str
    user:        str | None = None


class JobResponse(pydantic.BaseModel):
    """
    Full representation of a job returned by the API.

    Attributes
    ----------
    id : str
        UUID of the job.
    user : str
        Submitting user.
    name : str | None
        Human-readable job name from the --name directive.
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

    model_config = pydantic.ConfigDict(from_attributes=True)


class JobListResponse(pydantic.BaseModel):
    """
    Paginated list of jobs.

    Attributes
    ----------
    jobs : list[JobResponse]
        The matching jobs.
    total : int
        Total number of matching jobs (before pagination).
    limit : int
        Maximum page size requested.
    offset : int
        Number of matching rows skipped.
    """

    jobs:   list[JobResponse]
    total:  int
    limit:  int
    offset: int


class JobLogResponse(pydantic.BaseModel):
    """
    Log content for a single job stream.

    Attributes
    ----------
    job_id : str
        UUID of the job.
    stream : str
        Which stream this content belongs to: ``"stdout"`` or ``"stderr"``.
    content : str
        The log text.
    """

    job_id:   str
    stream:   str
    content:  str


# --------------------------------------------------------------------------------------
# Resource schemas


class ResourceResponse(pydantic.BaseModel):
    """
    Current resource configuration and usage.

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


class ResourceUpdateRequest(pydantic.BaseModel):
    """
    Request body for PUT /resources.

    Only the fields that are not None are updated.

    Attributes
    ----------
    total_cpus : int | None
        New total CPU core count.
    total_gpus : int | None
        New total GPU device count.
    total_mem_mb : int | None
        New total memory in megabytes.
    """

    total_cpus: int | None = pydantic.Field(default=None, gt=0)
    total_gpus: int | None = pydantic.Field(default=None, ge=0)
    total_mem_mb: int | None = pydantic.Field(default=None, gt=0)

    @pydantic.model_validator(mode="after")
    def at_least_one_field(self) -> ResourceUpdateRequest:
        """
        Validate that at least one field is provided.

        Returns
        -------
        ResourceUpdateRequest
            The validated model instance.

        Raises
        ------
        ValueError
            If all fields are None.
        """

        if all(v is None for v in (self.total_cpus, self.total_gpus, self.total_mem_mb)):
            raise ValueError("At least one of total_cpus, total_gpus, total_mem_mb must be set.")
        return self


# --------------------------------------------------------------------------------------
# Error schema


class ErrorResponse(pydantic.BaseModel):
    """
    Standard error response body.

    Attributes
    ----------
    detail : str
        Human-readable error message.
    """

    detail: str
