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

_SIGTERM_GRACE_SEC: float = 10.0   # Seconds between SIGTERM and SIGKILL.

# --------------------------------------------------------------------------------------
# Active subprocess registry

_active_processes: dict[str, asyncio.subprocess.Process] = {}
_active_tasks:     set[asyncio.Task] = set()


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
    task = loop.create_task(
        _run_job(job, resource_pool),
        name=f"job-{job.id}",
    )
    _active_tasks.add(task)
    task.add_done_callback(_active_tasks.discard)


async def shutdown_active_jobs(grace_sec: float = _SIGTERM_GRACE_SEC) -> None:
    """
    Terminate all subprocesses currently managed by this runner.

    Sends SIGTERM to each active job, waits up to *grace_sec*, then escalates
    any remaining processes to SIGKILL. The per-job monitor tasks are then
    awaited so they can persist final DB state and release resources.

    Parameters
    ----------
    grace_sec : float
        Seconds to wait after SIGTERM before sending SIGKILL.

    Returns
    -------
    None
    """

    processes = [
        process
        for process in _active_processes.values()
        if process.returncode is None
    ]
    if not processes:
        return

    logger.info("Terminating %d active job subprocess(es).", len(processes))

    for process in processes:
        _send_signal(process, signal.SIGTERM)

    tasks = [task for task in _active_tasks if not task.done()]
    waitables = tasks or [asyncio.create_task(process.wait()) for process in processes]

    _, pending = await asyncio.wait(waitables, timeout=grace_sec)
    if pending:
        still_running = [
            process
            for process in processes
            if process.returncode is None
        ]
        for process in still_running:
            _send_signal(process, signal.SIGKILL)
        await asyncio.gather(*pending, return_exceptions=True)

    await asyncio.gather(*waitables, return_exceptions=True)


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

    stdout_path, stderr_path = _log_paths(job)

    env = _build_env(job)

    process: asyncio.subprocess.Process | None = None

    try:
        process = await _spawn(job, env, stdout_path, stderr_path)
    except Exception:
        logger.exception("Failed to spawn subprocess for job %s.", job.id)
        _mark_failed(job, resource_pool)
        return

    _active_processes[job.id] = process

    try:
        _mark_running(job, process.pid, stdout_path, stderr_path)

        wait_task = asyncio.create_task(_wait_with_walltime(job, process))
        watchdog_task = asyncio.create_task(_cancel_watchdog(job, process))

        try:
            exit_code = await wait_task
        finally:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task, return_exceptions=True)

        _mark_finished(job, exit_code, resource_pool)
    finally:
        _active_processes.pop(job.id, None)


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

    cwd = job.workdir or str(pathlib.Path(job.script_path).resolve().parent)

    with stdout_path.open("wb") as fout, stderr_path.open("wb") as ferr:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=fout,
            stderr=ferr,
            env=env,
            cwd=cwd,
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


async def _cancel_watchdog(
    job:     models.Job,
    process: asyncio.subprocess.Process,
) -> None:
    """
    Poll DB every 2 s for CANCELLING status and escalate to SIGKILL after grace period.

    SIGTERM is already sent by cancel_job() in the API layer. This coroutine only
    handles the case where the process ignores SIGTERM and must be force-killed.
    """
    while process.returncode is None:
        await asyncio.sleep(2.0)
        if process.returncode is not None:
            return
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
        if stored is None or stored.status == models.JobStatus.CANCELLING:
            try:
                await asyncio.wait_for(process.wait(), timeout=_SIGTERM_GRACE_SEC)
            except asyncio.TimeoutError:
                logger.warning(
                    "Job %s did not exit after SIGTERM. Sending SIGKILL.", job.id
                )
                _send_signal(process, signal.SIGKILL)
                await process.wait()
            return


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

        if stored.status in (
            models.JobStatus.CANCELLED,
            models.JobStatus.CANCELLING,
        ):
            stored.status = models.JobStatus.CANCELLED
            stored.exit_code = exit_code
            stored.finished_at = now
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
# Log path helpers


def _log_paths(job: models.Job) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Return stdout/stderr log paths next to the submitted script.

    The files are named ``{script_name}.{job_id}.stdout.log`` and
    ``{script_name}.{job_id}.stderr.log``.
    """

    script_path = pathlib.Path(job.script_path).resolve()
    script_dir = script_path.parent
    script_name = script_path.name
    return (
        script_dir / f"{script_name}.{job.id}.stdout.log",
        script_dir / f"{script_name}.{job.id}.stderr.log",
    )
