from __future__ import annotations

import asyncio
import datetime
import logging
import os
import pathlib
import pwd
import signal
import subprocess

import qjob.core.database as database
import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Module logger

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants

_SIGTERM_GRACE_SEC: float = 10.0   # Seconds between SIGTERM and SIGKILL.
_LOGIN_PATH_TIMEOUT_SEC: float = 5.0   # Seconds allowed for the login-shell PATH probe.

# PATH used when neither the login shell nor /etc/login.defs can be consulted.
_FALLBACK_PATH: str = "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"

# The login shell may print banners, so the probed PATH is tagged with a marker.
_PATH_PROBE_MARKER:  str = "__QJOB_PATH__="
_PATH_PROBE_COMMAND: str = f"printf '\\n{_PATH_PROBE_MARKER}%s\\n' \"$PATH\""

# Login shells that cannot run the probe command.
_NON_LOGIN_SHELLS: frozenset[str] = frozenset({"nologin", "false", "true", "sync"})

# Variables that describe the daemon's own process and must not leak into jobs.
_DAEMON_ONLY_ENV: frozenset[str] = frozenset({
    # -- systemd service manager state ---------------------------------------------
    "INVOCATION_ID", "JOURNAL_STREAM", "NOTIFY_SOCKET", "LISTEN_FDS", "LISTEN_FDNAMES",
    "LISTEN_PID", "MAINPID", "MANAGERPID", "SYSTEMD_EXEC_PID", "WATCHDOG_PID",
    "WATCHDOG_USEC", "RUNTIME_DIRECTORY", "STATE_DIRECTORY", "CACHE_DIRECTORY",
    "LOGS_DIRECTORY", "CONFIGURATION_DIRECTORY",
    # -- session state of whoever started the daemon --------------------------------
    "XDG_RUNTIME_DIR", "XDG_SESSION_ID", "XDG_SESSION_TYPE", "XDG_SESSION_CLASS",
    "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT",
    "SSH_TTY", "SUDO_USER", "SUDO_UID", "SUDO_GID", "SUDO_COMMAND",
    "MAIL", "PWD", "OLDPWD", "_",
    # -- the daemon's own Python interpreter ----------------------------------------
    # Left in place these hijack `python` inside every job.
    "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
    "PYTHONEXECUTABLE", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL",
    "CONDA_EXE", "CONDA_PYTHON_EXE", "CONDA_PROMPT_MODIFIER", "_CE_CONDA", "_CE_M",
    "LD_PRELOAD",
    # -- daemon configuration, including database credentials -----------------------
    "QJOB_DB_URL", "QJOB_DB_POOL_ENABLED", "QJOB_DB_POOL_SIZE", "QJOB_DB_MAX_OVERFLOW",
    "QJOB_ADMIN_USERS", "QJOB_TOKEN_PATH", "QJOB_JOB_PATH",
})

# --------------------------------------------------------------------------------------
# Active subprocess registry

_active_processes: dict[str, asyncio.subprocess.Process] = {}
_active_tasks:     set[asyncio.Task] = set()

# Login PATH per username, resolved once per daemon lifetime (see _user_path()).
_user_path_cache: dict[str, str] = {}


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

    process: asyncio.subprocess.Process | None = None

    try:
        # _build_env() may probe the user's login shell, which is blocking.
        env = await asyncio.to_thread(_build_env, job)
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


def _make_preexec_fn(username: str):
    """Return a callable that drops privileges to *username* in the child process.

    Privilege drop is only performed when the daemon runs as root.  When not
    root the function is a no-op (the child inherits the daemon's identity).
    """
    pw = pwd.getpwnam(username)
    uid, gid = pw.pw_uid, pw.pw_gid
    running_as_root = os.getuid() == 0

    def _drop_privileges() -> None:
        if not running_as_root:
            return
        os.initgroups(username, gid)
        os.setgid(gid)
        os.setuid(uid)

    return _drop_privileges


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

    stdout_path.touch(mode=0o644)
    stderr_path.touch(mode=0o644)
    if os.getuid() == 0:
        pw = pwd.getpwnam(job.user)
        os.chown(stdout_path, pw.pw_uid, pw.pw_gid)
        os.chown(stderr_path, pw.pw_uid, pw.pw_gid)

    with stdout_path.open("wb") as fout, stderr_path.open("wb") as ferr:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=fout,
            stderr=ferr,
            env=env,
            cwd=cwd,
            start_new_session=True,
            preexec_fn=_make_preexec_fn(job.user),
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

    Jobs run as ``bash <script>``, which is neither a login nor an interactive
    shell, so no profile or rc file is read by the child: what this function
    returns is the whole environment the script sees.

    The mapping starts from the daemon environment, drops the variables that
    describe the daemon itself (see ``_DAEMON_ONLY_ENV``), overlays the login
    identity of the submitting user (``HOME``, ``USER``, ``LOGNAME``, ``SHELL``
    and ``PATH``), sets ``CUDA_VISIBLE_DEVICES`` from the assigned GPU IDs, and
    finally adds the QJOB metadata variables.

    Parameters
    ----------
    job : models.Job
        The job whose ``user`` determines the login identity and whose
        ``assigned_gpus`` determines GPU visibility.

    Returns
    -------
    dict[str, str]
        Full environment mapping for the child process.
    """

    import json

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _DAEMON_ONLY_ENV
    }

    env.update(_user_identity_env(job.user))

    gpu_ids: list[int] = json.loads(job.assigned_gpus) if job.assigned_gpus else []
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids) if gpu_ids else ""

    # Metadata variables available inside the script.
    env["QJOB_JOB_ID"] = job.id
    env["QJOB_JOB_NAME"] = job.name or ""
    env["QJOB_USER"] = job.user

    return env


def _user_identity_env(username: str) -> dict[str, str]:
    """
    Return the login-identity variables of *username*.

    Without these the job would inherit the identity of the daemon (``HOME``
    pointing at ``/root`` and the minimal PATH systemd gives a service), even
    though ``_make_preexec_fn()`` has already dropped privileges to *username*.

    Parameters
    ----------
    username : str
        OS username the job runs as.

    Returns
    -------
    dict[str, str]
        ``HOME``, ``USER``, ``LOGNAME``, ``SHELL`` and ``PATH``.  Empty when
        *username* is unknown to the password database, in which case the job
        keeps the daemon values.
    """

    try:
        pw = pwd.getpwnam(username)
    except KeyError:
        logger.warning(
            "User %r is not in the password database; "
            "the job inherits the daemon HOME and PATH.",
            username,
        )
        return {}

    return {
        "HOME":    pw.pw_dir,
        "USER":    pw.pw_name,
        "LOGNAME": pw.pw_name,
        "SHELL":   pw.pw_shell or "/bin/sh",
        "PATH":    _user_path(pw),
    }


def _user_path(pw: pwd.struct_passwd) -> str:
    """
    Return the PATH the user of *pw* would get in a login shell.

    Resolution order:

    1. ``QJOB_JOB_PATH`` from the daemon environment, when set.  A deployment
       can use it to pin one PATH for every job.
    2. The user's own login shell, which applies ``/etc/profile``,
       ``~/.profile`` and friends, so conda/pyenv/``~/.local/bin`` entries are
       picked up.  Probed at most once per user per daemon lifetime.
    3. ``ENV_PATH`` (``ENV_SUPATH`` for root) from ``/etc/login.defs``.
    4. ``_FALLBACK_PATH``.

    Parameters
    ----------
    pw : pwd.struct_passwd
        Password database entry of the job user.

    Returns
    -------
    str
        A non-empty PATH value.
    """

    override = os.environ.get("QJOB_JOB_PATH")
    if override:
        return override

    cached = _user_path_cache.get(pw.pw_name)
    if cached is not None:
        return cached

    path = (
        _login_shell_path(pw)
        or _login_defs_path(pw.pw_name)
        or _FALLBACK_PATH
    )
    path = _prepend_user_bin_dirs(path, pw.pw_dir)

    _user_path_cache[pw.pw_name] = path
    logger.debug("Resolved PATH for user %s: %s", pw.pw_name, path)
    return path


def _login_shell_path(pw: pwd.struct_passwd) -> str | None:
    """
    Ask the user's login shell for its PATH.

    Runs ``runuser -l`` (falling back to ``su -l``) when the daemon is root, or
    the shell directly when the daemon already runs as that same user.  The
    value is tagged with ``_PATH_PROBE_MARKER`` so banners printed by the
    profile scripts cannot be mistaken for the PATH.

    Parameters
    ----------
    pw : pwd.struct_passwd
        Password database entry of the job user.

    Returns
    -------
    str | None
        The PATH reported by the login shell, or None when it could not be
        obtained (no privileges, missing ``runuser``/``su``, non-login shell,
        probe timeout, or unparsable output).
    """

    shell = pw.pw_shell or "/bin/sh"
    if pathlib.Path(shell).name in _NON_LOGIN_SHELLS:
        return None

    if os.getuid() == 0:
        candidates = [
            ["runuser", "-l", pw.pw_name, "-c", _PATH_PROBE_COMMAND],
            ["su", "-l", pw.pw_name, "-c", _PATH_PROBE_COMMAND],
        ]
    elif pw.pw_uid == os.getuid():
        candidates = [[shell, "-l", "-c", _PATH_PROBE_COMMAND]]
    else:
        # Becoming another user requires root.
        return None

    # A minimal environment so the login shell starts from the system defaults.
    probe_env = {
        "HOME":    pw.pw_dir,
        "USER":    pw.pw_name,
        "LOGNAME": pw.pw_name,
        "SHELL":   shell,
        "PATH":    _login_defs_path(pw.pw_name) or _FALLBACK_PATH,
        "TERM":    "dumb",
    }

    for argv in candidates:
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_LOGIN_PATH_TIMEOUT_SEC,
                env=probe_env,
            )
        except FileNotFoundError:
            continue
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "Could not read the login PATH of %s via %s (%s).",
                pw.pw_name, argv[0], exc,
            )
            continue

        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(_PATH_PROBE_MARKER):
                path = line[len(_PATH_PROBE_MARKER):].strip()
                if path:
                    return path

        logger.warning(
            "%s returned no PATH for user %s (exit code %d).",
            argv[0], pw.pw_name, completed.returncode,
        )

    return None


def _login_defs_path(username: str) -> str | None:
    """
    Return the default PATH configured in ``/etc/login.defs``.

    Parameters
    ----------
    username : str
        Job user; ``root`` reads ``ENV_SUPATH``, anyone else ``ENV_PATH``.

    Returns
    -------
    str | None
        The configured PATH, or None when the file or the key is absent.
    """

    key = "ENV_SUPATH" if username == "root" else "ENV_PATH"

    try:
        lines = pathlib.Path("/etc/login.defs").read_text().splitlines()
    except OSError:
        return None

    for raw in lines:
        line = raw.strip()
        if not line.startswith(key):
            continue
        value = line[len(key):].strip()
        if value.startswith("PATH="):
            value = value[len("PATH="):].strip()
        if value:
            return value

    return None


def _prepend_user_bin_dirs(path: str, home: str) -> str:
    """
    Prepend ``~/.local/bin`` and ``~/bin`` to *path* when they exist.

    This mirrors the default ``~/.profile`` of Debian-like systems, so that
    ``pip install --user`` executables are reachable even when the PATH came
    from ``/etc/login.defs`` or the fallback.

    Parameters
    ----------
    path : str
        PATH resolved so far.
    home : str
        Home directory of the job user.

    Returns
    -------
    str
        PATH with the per-user bin directories in front.
    """

    entries = path.split(os.pathsep)

    for candidate in (os.path.join(home, "bin"), os.path.join(home, ".local", "bin")):
        if candidate in entries:
            continue
        try:
            if not os.path.isdir(candidate):
                continue
        except OSError:
            continue
        entries.insert(0, candidate)

    return os.pathsep.join(entries)


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
