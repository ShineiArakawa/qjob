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
    workdir : str | None
        Working directory to use when running the script.
    """

    script_path: str
    workdir:     str | None = None


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

    total_cpus:       int
    total_gpus:       int
    total_mem_mb:     int
    max_walltime_sec: int | None
    gpu_ids:          list[int]
    used_cpus:        int
    used_gpus:        int
    used_mem_mb:      int


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

    total_cpus:       int | None = pydantic.Field(default=None, gt=0)
    total_gpus:       int | None = pydantic.Field(default=None, ge=0)
    gpu_ids:          list[int] | None = None
    total_mem_mb:     int | None = pydantic.Field(default=None, gt=0)
    max_walltime_sec: int | None = pydantic.Field(default=None, gt=0)

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

        if all(v is None for v in (self.total_cpus, self.total_gpus, self.gpu_ids, self.total_mem_mb, self.max_walltime_sec)):
            raise ValueError("At least one of total_cpus, total_gpus, gpu_ids, total_mem_mb, max_walltime_sec must be set.")
        if self.gpu_ids is not None:
            for gpu_id in self.gpu_ids:
                if isinstance(gpu_id, bool) or gpu_id < 0:
                    raise ValueError("gpu_ids must contain non-negative integer GPU IDs.")
            if len(set(self.gpu_ids)) != len(self.gpu_ids):
                raise ValueError("gpu_ids must not contain duplicates.")
        if self.total_gpus is not None and self.gpu_ids is not None and self.total_gpus != len(self.gpu_ids):
            raise ValueError("total_gpus must match the number of gpu_ids when both are set.")
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


# --------------------------------------------------------------------------------------
# Auth schemas


class TokenCreateRequest(pydantic.BaseModel):
    """
    Request body for POST /auth/token.

    Attributes
    ----------
    username : str
        OS username to associate with the new token.
    """

    username: str


class TokenResponse(pydantic.BaseModel):
    """
    Response for POST /auth/token.

    Attributes
    ----------
    token : str
        The raw token.  Shown once — store it securely.
    username : str
        The username associated with this token.
    """

    token:    str
    username: str
