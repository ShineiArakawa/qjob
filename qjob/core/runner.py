from __future__ import annotations

import asyncio
import datetime
import logging
import os
import pathlib
import signal

import qjob.core.database as database
import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Module logger

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants

_LOG_DIR_ENV:        str = "QJOB_LOG_DIR"
_DEFAULT_LOG_DIR:    str = "/tmp/qjob_logs"
_SIGTERM_GRACE_SEC:  float = 10.0   # Seconds between SIGTERM and SIGKILL.


# --------------------------------------------------------------------------------------
# Public API


def start_job(job: models.Job, resource_pool: object) -> None:
    """
    Launch a job as a subprocess and schedule its monitoring coroutine.

    This function is intentionally synchronous so that the scheduler can
    call it without ``await``.  Internally it schedules an async monitoring
    task on the running event loop.

    Parameters
    ----------
    job : models.Job
        The job to launch.  ``job.assigned_cpus`` and ``job.assigned_gpus``
        must already be set by the scheduler.
    resource_pool : object
        The pool used to release resources when the job finishes.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If there is no running event loop (i.e. called outside asyncio context).
    OSError
        If the log directory cannot be created.
    """

    loop = asyncio.get_event_loop()
    loop.create_task(
        _run_job(job, resource_pool),
        name=f"job-{job.id}",
    )


# --------------------------------------------------------------------------------------
# Core async runner


async def _run_job(job: models.Job, resource_pool: object) -> None:
    """
    Async coroutine that launches, monitors, and finalises a single job.

    Parameters
    ----------
    job : models.Job
        The job to execute.
    resource_pool : object
        ResourcePool instance.  Its ``release()`` method is called on
        job completion regardless of outcome.

    Returns
    -------
    None
    """

    log_dir = _ensure_log_dir(job)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    env = _build_env(job)

    try:
        process = await _spawn(job, env, stdout_path, stderr_path)
    except Exception:
        logger.exception("Failed to spawn subprocess for job %s.", job.id)
        _mark_failed(job, resource_pool)
        return

    _mark_running(job, process.pid, stdout_path, stderr_path)

    exit_code = await _wait_with_walltime(job, process)

    _mark_finished(job, exit_code, resource_pool)


# --------------------------------------------------------------------------------------
# Subprocess helpers


async def _spawn(
    job:         models.Job,
    env:         dict[str, str],
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> asyncio.subprocess.Process:
    """
    Start the job's shell script as an async subprocess.

    CPU affinity is applied via ``taskset -c`` when assigned CPU IDs are
    present.

    Parameters
    ----------
    job : models.Job
        The job whose ``script_path`` is executed.
    env : dict[str, str]
        Environment variables for the child process.
    stdout_path : pathlib.Path
        File path where stdout is captured.
    stderr_path : pathlib.Path
        File path where stderr is captured.

    Returns
    -------
    asyncio.subprocess.Process
        The running process.
    """

    import json

    cpu_ids: list[int] = json.loads(job.assigned_cpus) if job.assigned_cpus else []

    if cpu_ids:
        # Pin the process to specific CPU cores via taskset.
        cpu_list = ",".join(str(c) for c in cpu_ids)
        argv = ["taskset", "-c", cpu_list, "bash", job.script_path]
    else:
        argv = ["bash", job.script_path]

    with stdout_path.open("wb") as fout, stderr_path.open("wb") as ferr:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=fout,
            stderr=ferr,
            env=env,
            start_new_session=True,
        )

    logger.debug(
        "Spawned job %s as PID %d (cpus=%s).",
        job.id, process.pid, cpu_ids,
    )
    return process


async def _wait_with_walltime(
    job:     models.Job,
    process: asyncio.subprocess.Process,
) -> int:
    """
    Wait for the process to finish, enforcing walltime if set.

    If the job exceeds its walltime, sends SIGTERM and waits
    ``_SIGTERM_GRACE_SEC`` seconds before escalating to SIGKILL.

    Parameters
    ----------
    job : models.Job
        The running job.  ``job.walltime_sec`` may be None (no limit).
    process : asyncio.subprocess.Process
        The running subprocess.

    Returns
    -------
    int
        The process exit code.  Returns a negative value when the process
        was killed by a signal (standard POSIX convention).
    """

    wait_coro = process.wait()

    if job.walltime_sec is None:
        return await wait_coro

    try:
        return await asyncio.wait_for(wait_coro, timeout=float(job.walltime_sec))
    except asyncio.TimeoutError:
        logger.warning(
            "Job %s exceeded walltime (%ds). Sending SIGTERM.",
            job.id, job.walltime_sec,
        )
        _send_signal(process, signal.SIGTERM)

        try:
            return await asyncio.wait_for(
                process.wait(), timeout=_SIGTERM_GRACE_SEC
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Job %s did not exit after SIGTERM. Sending SIGKILL.", job.id
            )
            _send_signal(process, signal.SIGKILL)
            return await process.wait()


def _send_signal(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """
    Send *sig* to *process*, ignoring ProcessLookupError if it already exited.

    Parameters
    ----------
    process : asyncio.subprocess.Process
        Target process.
    sig : signal.Signals
        Signal to deliver.

    Returns
    -------
    None
    """

    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass

# --------------------------------------------------------------------------------------
# Environment helpers


def _build_env(job: models.Job) -> dict[str, str]:
    """
    Build the subprocess environment for *job*.

    Starts from the current process environment, then sets
    ``CUDA_VISIBLE_DEVICES`` from the assigned GPU IDs, and overlays
    QJOB metadata variables the script can reference.

    Parameters
    ----------
    job : models.Job
        The job whose ``assigned_gpus`` determines GPU visibility.

    Returns
    -------
    dict[str, str]
        Full environment mapping for the child process.
    """

    import json

    env = os.environ.copy()

    gpu_ids: list[int] = json.loads(job.assigned_gpus) if job.assigned_gpus else []
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids) if gpu_ids else ""

    # Metadata variables available inside the script.
    env["QJOB_JOB_ID"] = job.id
    env["QJOB_JOB_NAME"] = job.name or ""
    env["QJOB_USER"] = job.user

    return env


# --------------------------------------------------------------------------------------
# DB state transition helpers


def _mark_running(
    job:         models.Job,
    pid:         int,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> None:
    """
    Persist the RUNNING state, PID, and log paths to the database.

    Parameters
    ----------
    job : models.Job
        The job that has just been spawned.
    pid : int
        OS process ID of the subprocess.
    stdout_path : pathlib.Path
        Path to the captured stdout file.
    stderr_path : pathlib.Path
        Path to the captured stderr file.

    Returns
    -------
    None
    """

    now = datetime.datetime.now(datetime.timezone.utc)

    with database.get_session() as session:
        stored = session.get(models.Job, job.id)
        if stored is None:
            return
        stored.status = models.JobStatus.RUNNING
        stored.pid = pid
        stored.started_at = now
        stored.log_stdout = str(stdout_path)
        stored.log_stderr = str(stderr_path)

    # Keep the in-memory object in sync so callers can read without a DB round trip.
    job.status = models.JobStatus.RUNNING
    job.pid = pid
    job.started_at = now

    logger.info("Job %s is now RUNNING (pid=%d).", job.id, pid)


def _mark_finished(
    job:           models.Job,
    exit_code:     int,
    resource_pool: object,
) -> None:
    """
    Persist the terminal state and release resources.

    Parameters
    ----------
    job : models.Job
        The job that has finished.
    exit_code : int
        The process exit code.
    resource_pool : object
        ResourcePool whose ``release()`` is called to free CPU/GPU/memory.

    Returns
    -------
    None
    """

    final_status = (
        models.JobStatus.DONE if exit_code == 0 else models.JobStatus.FAILED
    )
    now = datetime.datetime.now(datetime.timezone.utc)

    with database.get_session() as session:
        stored = session.get(models.Job, job.id)
        if stored is None:
            return

        if stored.status == models.JobStatus.CANCELLED:
            stored.exit_code = exit_code
            stored.finished_at = stored.finished_at or now
            final_status = models.JobStatus.CANCELLED
        else:
            stored.status = final_status
            stored.exit_code = exit_code
            stored.finished_at = now

    job.status = final_status

    if resource_pool is not None:
        resource_pool.release(job)

    logger.info(
        "Job %s finished — status=%s exit_code=%d.",
        job.id, final_status, exit_code,
    )


def _mark_failed(job: models.Job, resource_pool: object) -> None:
    """
    Mark *job* as FAILED when the subprocess could not be spawned at all.

    Parameters
    ----------
    job : models.Job
        The job that failed to start.
    resource_pool : object
        ResourcePool whose ``release()`` is called.

    Returns
    -------
    None
    """

    now = datetime.datetime.now(datetime.timezone.utc)

    with database.get_session() as session:
        stored = session.get(models.Job, job.id)
        if stored is None:
            return
        stored.status = models.JobStatus.FAILED
        stored.finished_at = now

    job.status = models.JobStatus.FAILED

    if resource_pool is not None:
        resource_pool.release(job)

    logger.error("Job %s failed to start.", job.id)


# --------------------------------------------------------------------------------------
# Log directory helpers


def _ensure_log_dir(job: models.Job) -> pathlib.Path:
    """
    Create and return the per-job log directory.

    The path is ``<QJOB_LOG_DIR>/<job.id>/``.  The base directory is taken
    from the ``QJOB_LOG_DIR`` environment variable, defaulting to
    ``/tmp/qjob_logs``.

    Parameters
    ----------
    job : models.Job
        The job whose ID names the subdirectory.

    Returns
    -------
    pathlib.Path
        The created (or already existing) log directory.
    """

    base = pathlib.Path(os.environ.get(_LOG_DIR_ENV, _DEFAULT_LOG_DIR))
    log_dir = base / job.id
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
