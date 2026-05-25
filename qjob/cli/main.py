from __future__ import annotations

import asyncio
import datetime
import importlib.metadata
import os
import pathlib
import typing

import typer

import qjob.cli.dashboard as dashboard
import qjob.cli.service as service
import qjob.core.database as database

# --------------------------------------------------------------------------------------
# Constants

_DEFAULT_STATUS_LIMIT = 20
_MAX_STATUS_LIMIT = 1000
_JOB_STATES = ["queued", "running", "cancelling", "done", "failed", "cancelled"]
_JOB_SORT_KEYS = ["submitted", "started", "finished", "priority", "user"]

_SYSTEMD_DIR = pathlib.Path("/etc/systemd/system")
_SERVER_UNIT = "qjob-server.service"
_SCHEDULER_UNIT = "qjob-scheduler.service"

# --------------------------------------------------------------------------------------
# Typer application

app = typer.Typer(
    name="qjob",
    help="A lightweight job scheduler designed for single-node lab-scale GPU servers.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

admin_app = typer.Typer(
    help="Administrative commands.",
    no_args_is_help=True,
)
app.add_typer(admin_app, name="admin")


def _version_callback(value: bool) -> None:
    if value:
        version = importlib.metadata.version("qjob")
        typer.echo(f"qjob {version}")
        raise typer.Exit()


@app.callback()
def _main(
    version: typing.Optional[bool] = typer.Option(
        None, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    env_file: pathlib.Path = typer.Option(
        pathlib.Path(".env"), "--env-file",
        help="Path to .env file (loaded if it exists).",
        is_eager=True,
    ),
) -> None:
    # Load environment variables from the specified .env file, if it exists.
    import dotenv
    dotenv.load_dotenv(env_file, override=False)


# --------------------------------------------------------------------------------------
# Shell completion helpers


def _complete_status(incomplete: str) -> list[str]:
    return [v for v in _JOB_STATES if v.startswith(incomplete)]


def _complete_state_filter(incomplete: str) -> list[str]:
    prefix, separator, current = incomplete.rpartition(",")
    base = f"{prefix}{separator}" if separator else ""
    return [f"{base}{v}" for v in _JOB_STATES if v.startswith(current)]


def _complete_sort(incomplete: str) -> list[str]:
    return [v for v in _JOB_SORT_KEYS if v.startswith(incomplete)]


def _complete_job_id_filtered(
    incomplete: str,
    status_filter: list[str] | None = None,
) -> list[str]:
    try:
        jobs = service.list_jobs(user=None)
        jobs = [j for j in jobs if j.id.startswith(incomplete)]
        if status_filter:
            jobs = [j for j in jobs if j.status in status_filter]
        return [j.id for j in jobs]
    except Exception:
        return []


def _parse_gpu_ids(value: str) -> list[int]:
    """Parse a comma-separated GPU ID list."""

    if not value.strip():
        return []
    ids: list[int] = []
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            raise ValueError("GPU ID list must not contain empty entries.")
        try:
            gpu_id = int(part)
        except ValueError:
            raise ValueError(f"GPU ID must be an integer, got {part!r}.")
        if gpu_id < 0:
            raise ValueError("GPU IDs must be greater than or equal to 0.")
        ids.append(gpu_id)
    if len(set(ids)) != len(ids):
        raise ValueError("GPU IDs must not contain duplicates.")
    return ids


def _parse_states(value: str) -> list[str]:
    """Parse a comma-separated job state list."""

    if not value.strip():
        raise ValueError("state filter must not be empty.")
    states: list[str] = []
    for raw in value.split(","):
        state = raw.strip()
        if not state:
            raise ValueError("state filter must not contain empty entries.")
        if state not in _JOB_STATES:
            raise ValueError(
                f"Invalid state {state!r}. Valid values: {', '.join(_JOB_STATES)}."
            )
        states.append(state)
    if len(set(states)) != len(states):
        raise ValueError("state filter must not contain duplicates.")
    return states


def _parse_since(value: str) -> datetime.datetime:
    """Parse an absolute or relative submitted-at lower bound."""

    raw = value.strip()
    if not raw:
        raise ValueError("since filter must not be empty.")

    unit = raw[-1].lower()
    amount_text = raw[:-1]
    if unit in ("m", "h", "d") and amount_text:
        try:
            amount = float(amount_text)
        except ValueError:
            amount = -1
        if amount <= 0:
            raise ValueError("relative since filter must be a positive duration.")
        if unit == "m":
            delta = datetime.timedelta(minutes=amount)
        elif unit == "h":
            delta = datetime.timedelta(hours=amount)
        else:
            delta = datetime.timedelta(days=amount)
        return datetime.datetime.now(datetime.timezone.utc) - delta

    try:
        if len(raw) == 10:
            parsed_date = datetime.date.fromisoformat(raw)
            parsed = datetime.datetime.combine(parsed_date, datetime.time.min)
        else:
            parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            "since filter must be a duration like 24h or an ISO date/time."
        )

    if parsed.tzinfo is None:
        local_tz = datetime.datetime.now().astimezone().tzinfo
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(datetime.timezone.utc)


def _parse_sort(value: str) -> str:
    """Validate a job list sort key."""

    if value not in _JOB_SORT_KEYS:
        raise ValueError(
            f"Invalid sort {value!r}. Valid values: {', '.join(_JOB_SORT_KEYS)}."
        )
    return value


def _complete_job_id(incomplete: str) -> list[str]:
    return _complete_job_id_filtered(incomplete)


def _complete_cancellable_job_id(incomplete: str) -> list[str]:
    return _complete_job_id_filtered(incomplete, status_filter=["queued", "running"])


# --------------------------------------------------------------------------------------
# submit


@app.command()
def submit(
    script: str = typer.Argument(..., help="Path to the shell script to submit."),
) -> None:
    """
    Submit a shell script to the job queue.

    The script must contain #QJOB directives in its leading comment block.
    """

    try:
        info = service.submit_job(script_path=script)
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Submitted job {info.id}")
    _print_job_table([info])


# --------------------------------------------------------------------------------------
# status


@app.command()
def status(
    job_id: typing.Optional[str] = typer.Argument(
        None, help="Job ID to inspect. Omit to list all jobs.",
        autocompletion=_complete_job_id,
    ),
    user: typing.Optional[str] = typer.Option(
        None, "--user", "-u", help="Filter by username."
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help=f"Show all matching jobs instead of the latest {_DEFAULT_STATUS_LIMIT}.",
    ),
    all_users: bool = typer.Option(
        False,
        "--all-users",
        help="Show jobs from all users instead of the current user.",
    ),
    limit: typing.Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help=f"Show the latest N matching jobs. Defaults to {_DEFAULT_STATUS_LIMIT}.",
    ),
    state_filter: typing.Optional[str] = typer.Option(
        None, "--state", "-s",
        help="Filter by state. Comma-separated values are accepted.",
        autocompletion=_complete_state_filter,
    ),
    since_filter: typing.Optional[str] = typer.Option(
        None,
        "--since",
        help=(
            "Show jobs submitted since a duration or ISO date/time, "
            "e.g. 24h or 2026-05-01."
        ),
    ),
    sort_key: str = typer.Option(
        "submitted",
        "--sort",
        help="Sort by submitted, started, finished, priority, or user.",
        autocompletion=_complete_sort,
    ),
) -> None:
    """
    Show job status.

    Without arguments, lists jobs submitted by the current user.
    """

    if all_users and user is not None:
        typer.echo("Error: --all-users and --user cannot be combined.", err=True)
        raise typer.Exit(code=1)

    if show_all and limit is not None:
        typer.echo("Error: --all and --limit cannot be combined.", err=True)
        raise typer.Exit(code=1)

    if limit is not None and limit <= 0:
        typer.echo("Error: --limit must be greater than 0.", err=True)
        raise typer.Exit(code=1)
    if limit is not None and limit > _MAX_STATUS_LIMIT:
        typer.echo(f"Error: --limit must be <= {_MAX_STATUS_LIMIT}.", err=True)
        raise typer.Exit(code=1)

    try:
        states = _parse_states(state_filter) if state_filter is not None else None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        since = _parse_since(since_filter) if since_filter is not None else None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        sort = _parse_sort(sort_key)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if job_id is not None:
        try:
            info = service.get_job(job_id)
        except ConnectionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        if info is None:
            typer.echo(f"Error: Job {job_id!r} not found.", err=True)
            raise typer.Exit(code=1)
        _print_job_detail(info)
        return

    resolved_user = None if all_users else (user or os.environ.get("USER"))
    try:
        if show_all:
            jobs = service.list_jobs(
                user=resolved_user,
                all_users=all_users,
                states=states,
                since=since,
                sort=sort,
            )
            truncated = False
        else:
            display_limit = limit or _DEFAULT_STATUS_LIMIT
            request_limit = min(display_limit + 1, _MAX_STATUS_LIMIT)
            jobs = service.list_jobs(
                user=resolved_user,
                all_users=all_users,
                limit=request_limit,
                states=states,
                since=since,
                sort=sort,
            )
            truncated = len(jobs) > display_limit
            jobs = jobs[:display_limit]
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if not jobs:
        typer.echo("No jobs found.")
        return

    _print_job_table(jobs, truncated=truncated)


# --------------------------------------------------------------------------------------
# cancel


@app.command()
def cancel(
    job_ids: typing.Optional[typing.List[str]] = typer.Argument(
        None, help="IDs of the jobs to cancel.", autocompletion=_complete_cancellable_job_id,
    ),
    all_jobs: bool = typer.Option(
        False, "--all", "-a", help="Cancel all your queued and running jobs."
    ),
) -> None:
    """Cancel one or more queued or running jobs."""

    if all_jobs and job_ids:
        typer.echo("Error: --all cannot be combined with explicit job IDs.", err=True)
        raise typer.Exit(code=1)

    if not all_jobs and not job_ids:
        typer.echo("Error: specify at least one JOB_ID or use --all.", err=True)
        raise typer.Exit(code=1)

    if all_jobs:
        current_user = os.environ.get("USER")
        try:
            cancellable = [
                j for j in service.list_jobs(user=current_user)
                if j.status in ("queued", "running")
            ]
        except ConnectionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        if not cancellable:
            typer.echo("No cancellable jobs found.")
            return

        job_ids = [j.id for j in cancellable]

    # Validate all job IDs before cancelling any.
    not_found: list[str] = []
    for job_id in job_ids:
        try:
            info = service.get_job(job_id)
        except ConnectionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        if info is None:
            not_found.append(job_id)

    if not_found:
        for job_id in not_found:
            typer.echo(f"Error: Job {job_id!r} not found.", err=True)
        raise typer.Exit(code=1)

    exit_code = 0
    for job_id in job_ids:
        try:
            info = service.cancel_job(job_id)  # authenticated via token
        except PermissionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            exit_code = 1
            continue
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            exit_code = 1
            continue
        except ConnectionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            exit_code = 1
            continue

        if info is None:
            typer.echo(f"Error: Job {job_id!r} not found.", err=True)
            exit_code = 1
        else:
            typer.echo(f"Cancelled job {info.id}")

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# --------------------------------------------------------------------------------------
# log


@app.command()
def log(
    job_id: str = typer.Argument(..., help="ID of the job whose log to display.", autocompletion=_complete_job_id),
    stderr: bool = typer.Option(
        False, "--stderr", "-e", help="Show stderr instead of stdout."
    ),
) -> None:
    """Print the stdout (or stderr) log of a job."""

    stream = "stderr" if stderr else "stdout"
    try:
        content = service.get_log(job_id, stream=stream)
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(content, nl=False)


# --------------------------------------------------------------------------------------
# resources


@app.command()
def resources() -> None:
    """Show current resource availability."""

    try:
        info = service.get_resources()
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    _print_resources(info)


@app.command()
def dash(
    refresh: float = typer.Option(
        2.0, "--refresh", "-r", help="Seconds between screen refreshes."
    ),
) -> None:
    """
    Open the live TUI dashboard showing job status and resource usage.

    Connects to the local database directly.  Press Ctrl+C to exit.
    """

    database.init_db()
    dashboard.run(refresh_interval=refresh)


# --------------------------------------------------------------------------------------
# admin up / down


@admin_app.command("up")
def up() -> None:
    """Start qjob services (API server + scheduler) via systemd."""

    import subprocess

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin up' must be run as root.", err=True)
        raise typer.Exit(code=1)

    for unit in (_SERVER_UNIT, _SCHEDULER_UNIT):
        if not (_SYSTEMD_DIR / unit).exists():
            typer.echo(
                f"Error: {unit} not found. Run 'qjob admin install' first.", err=True
            )
            raise typer.Exit(code=1)

    try:
        subprocess.run(
            ["systemctl", "start", _SERVER_UNIT, _SCHEDULER_UNIT], check=True
        )
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Error: systemctl exited with code {exc.returncode}.", err=True)
        raise typer.Exit(code=1)

    typer.echo("qjob services started.")


@admin_app.command("down")
def down() -> None:
    """Stop qjob services (scheduler + API server) via systemd."""

    import subprocess

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin down' must be run as root.", err=True)
        raise typer.Exit(code=1)

    try:
        subprocess.run(
            ["systemctl", "stop", _SCHEDULER_UNIT, _SERVER_UNIT], check=True
        )
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Error: systemctl exited with code {exc.returncode}.", err=True)
        raise typer.Exit(code=1)

    typer.echo("qjob services stopped.")


# --------------------------------------------------------------------------------------
# admin sub-commands


@admin_app.command("create-token")
def admin_create_token(
    username: str = typer.Argument(..., help="OS username to create a token for."),
) -> None:
    """
    Create an API token directly in the database (root only, no HTTP required).

    Use this command to bootstrap the admin token before the server is running,
    or when the API server is unavailable.  The token is saved to the target
    user's ~/.config/qjob/token and also printed to stdout.
    """

    import hashlib
    import pwd
    import secrets
    import stat

    import qjob.core.models as _models

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin create-token' must be run as root.", err=True)
        raise typer.Exit(code=1)

    try:
        pw = pwd.getpwnam(username)
    except KeyError:
        typer.echo(f"Error: OS user {username!r} does not exist.", err=True)
        raise typer.Exit(code=1)

    try:
        database.init_db()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    with database.get_session() as session:
        existing = (
            session.query(_models.ApiToken)
            .filter(_models.ApiToken.username == username)
            .first()
        )
    if existing is not None:
        typer.echo(f"Error: User {username!r} already has a token. Revoke it first.", err=True)
        raise typer.Exit(code=1)

    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with database.get_session() as session:
        row = _models.ApiToken(username=username, token_hash=token_hash)
        session.add(row)

    token_path = pathlib.Path(pw.pw_dir) / ".config" / "qjob" / "token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token)
    token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    os.chown(token_path, pw.pw_uid, pw.pw_gid)

    typer.echo(f"Token for {username}: {token}")
    typer.echo(f"Token saved to {token_path}")


@admin_app.command("init-token")
def admin_init_token(
    username: typing.Optional[str] = typer.Option(
        None, "--username", "-u",
        help="OS username to create a token for (default: current user).",
    ),
) -> None:
    """
    Create an API token via the API server (admin privileges required).

    When --username is omitted the token is for the current OS user and is
    saved to ~/.config/qjob/token automatically.  When --username names
    another user the token is printed — distribute it to that user manually.
    """

    import getpass
    import stat

    current_user = getpass.getuser()
    target = username or current_user

    try:
        token = service.create_token(target)
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if target == current_user:
        token_path = service._TOKEN_PATH
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token)
        token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
        typer.echo(f"Token saved to {token_path}")
    else:
        typer.echo(f"Token for {target}: {token}")
        typer.echo("Save this token — it will not be shown again.")

    typer.echo(f"Authenticated as: {target}")


@admin_app.command("set-resources")
def admin_set_resources(
    cpus: typing.Optional[int] = typer.Option(
        None, "--cpus", help="Total number of CPU cores."
    ),
    gpus: typing.Optional[int] = typer.Option(
        None, "--gpus", help="Total number of GPU devices."
    ),
    gpu_ids: typing.Optional[str] = typer.Option(
        None, "--gpu-ids", help="Comma-separated managed GPU device IDs, e.g. 0,2,5."
    ),
    mem: typing.Optional[str] = typer.Option(
        None, "--mem", help="Total memory (e.g. 64G, 512M, 65536)."
    ),
    max_walltime: typing.Optional[str] = typer.Option(
        None, "--max-walltime", help="Maximum allowed walltime per job (HH:MM:SS or MM:SS)."
    ),
) -> None:
    """Update the available resource limits."""

    if cpus is None and gpus is None and gpu_ids is None and mem is None and max_walltime is None:
        typer.echo(
            "Error: specify at least one of --cpus, --gpus, --gpu-ids, --mem, --max-walltime.",
            err=True,
        )
        raise typer.Exit(code=1)

    import qjob.core.parser as _parser

    mem_mb: int | None = None
    if mem is not None:
        try:
            # Bare integer (no unit) is treated as MB for admin use.
            mem_mb = int(mem) if mem.strip().isdigit() else _parser._parse_mem(mem)
        except (_parser.DirectiveParseError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

    max_walltime_sec: int | None = None
    if max_walltime is not None:
        try:
            max_walltime_sec = _parser._parse_walltime(max_walltime)
        except _parser.DirectiveParseError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

    parsed_gpu_ids: list[int] | None = None
    if gpu_ids is not None:
        try:
            parsed_gpu_ids = _parse_gpu_ids(gpu_ids)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

    try:
        info = service.set_resources(
            total_cpus=cpus,
            total_gpus=gpus,
            total_mem_mb=mem_mb,
            max_walltime_sec=max_walltime_sec,
            gpu_ids=parsed_gpu_ids,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Resources updated.")
    _print_resources(info)


@admin_app.command("list-jobs")
def admin_list_jobs(
    status_filter: typing.Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status.", autocompletion=_complete_status,
    ),
) -> None:
    """List all jobs from all users (admin view)."""

    try:
        states = [status_filter] if status_filter is not None else None
        jobs = service.list_jobs(user=None, all_users=True, states=states)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except ConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if not jobs:
        typer.echo("No jobs found.")
        return

    _print_job_table(jobs)


@admin_app.command("serve")
def admin_serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-H", help="Network interface to bind."
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="TCP port to listen on."
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="Uvicorn log level: debug/info/warning/error."
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (development only)."
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of uvicorn worker processes."
    ),
) -> None:
    """
    Start the qjob API server (FastAPI + uvicorn).

    For multi-process deployments use --workers N and run 'qjob admin scheduler'
    as a separate process.  Press Ctrl+C to stop.
    """

    import qjob.api.server as server

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin serve' must be run as root.", err=True)
        raise typer.Exit(code=1)

    if reload and workers > 1:
        typer.echo("Error: --reload and --workers cannot be combined.", err=True)
        raise typer.Exit(code=1)

    server.serve(
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        workers=workers,
    )


@admin_app.command("scheduler")
def admin_scheduler(
    poll_interval: float = typer.Option(
        2.0, "--poll-interval", help="Seconds between scheduling ticks."
    ),
    max_workers: int = typer.Option(
        64, "--max-workers", help="Maximum number of concurrently running jobs."
    ),
) -> None:
    """
    Start the standalone job scheduler process.

    Only one scheduler may run at a time; a second invocation will exit
    immediately with an error.  Press Ctrl+C to stop gracefully.
    """

    import qjob.core.scheduler as _scheduler

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin scheduler' must be run as root.", err=True)
        raise typer.Exit(code=1)

    try:
        database.init_db()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    sched = _scheduler.Scheduler(
        poll_interval=poll_interval,
        max_workers=max_workers,
        install_signal_handlers=True,
    )

    try:
        asyncio.run(sched.start())
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@admin_app.command("install")
def admin_install(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-H", help="Host for the API server."
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port for the API server."
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="Uvicorn log level for the API server."
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of uvicorn worker processes."
    ),
    poll_interval: float = typer.Option(
        2.0, "--poll-interval", help="Scheduler poll interval in seconds."
    ),
    max_workers: int = typer.Option(
        64, "--max-workers", help="Maximum concurrent jobs for the scheduler."
    ),
    svc_env_file: typing.Optional[pathlib.Path] = typer.Option(
        None, "--svc-env-file",
        help="Absolute path to .env file embedded in the unit files (optional).",
    ),
    enable: bool = typer.Option(
        True, "--enable/--no-enable",
        help="Enable services to start automatically on boot.",
    ),
) -> None:
    """
    Generate and install systemd unit files for the qjob services.

    Writes qjob-server.service and qjob-scheduler.service to
    /etc/systemd/system/, reloads the systemd daemon, and optionally
    enables both services.  Run 'qjob admin up' to start them immediately.
    """

    import subprocess

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin install' must be run as root.", err=True)
        raise typer.Exit(code=1)

    qjob_bin = _resolve_qjob_bin()

    env_file_str: str | None = None
    if svc_env_file is not None:
        env_file_str = str(svc_env_file.resolve())

    server_path = _SYSTEMD_DIR / _SERVER_UNIT
    scheduler_path = _SYSTEMD_DIR / _SCHEDULER_UNIT

    server_path.write_text(
        _server_unit_content(qjob_bin, host, port, log_level, workers, env_file_str)
    )
    typer.echo(f"Written {server_path}")

    scheduler_path.write_text(
        _scheduler_unit_content(qjob_bin, poll_interval, max_workers, env_file_str)
    )
    typer.echo(f"Written {scheduler_path}")

    try:
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        typer.echo("Reloaded systemd daemon.")

        if enable:
            subprocess.run(
                ["systemctl", "enable", _SERVER_UNIT, _SCHEDULER_UNIT], check=True
            )
            typer.echo("Services enabled (will start on boot).")
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Error: systemctl exited with code {exc.returncode}.", err=True)
        raise typer.Exit(code=1)

    typer.echo("Run 'qjob admin up' to start the services now.")


@admin_app.command("uninstall")
def admin_uninstall() -> None:
    """
    Stop, disable, and remove qjob systemd unit files.

    Stops the services if running, disables them, removes the unit files
    from /etc/systemd/system/, and reloads the systemd daemon.
    """

    import subprocess

    if os.getuid() != 0:
        typer.echo("Error: 'qjob admin uninstall' must be run as root.", err=True)
        raise typer.Exit(code=1)

    # Stop and disable — ignore errors (services may not be running/enabled).
    subprocess.run(["systemctl", "stop", _SCHEDULER_UNIT, _SERVER_UNIT])
    subprocess.run(["systemctl", "disable", _SERVER_UNIT, _SCHEDULER_UNIT])

    for path in (_SYSTEMD_DIR / _SERVER_UNIT, _SYSTEMD_DIR / _SCHEDULER_UNIT):
        if path.exists():
            path.unlink()
            typer.echo(f"Removed {path}")
        else:
            typer.echo(f"Not found, skipping: {path}")

    try:
        subprocess.run(["systemctl", "daemon-reload"], check=True)
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Error: systemctl exited with code {exc.returncode}.", err=True)
        raise typer.Exit(code=1)

    typer.echo("Services uninstalled.")


# --------------------------------------------------------------------------------------
# systemd unit-file helpers


def _resolve_qjob_bin() -> str:
    """Return the absolute path to the qjob executable."""
    import shutil
    path = shutil.which("qjob")
    if path:
        return str(pathlib.Path(path).resolve())
    # Fallback: derive from the package location (works inside a venv).
    import sys
    return str(pathlib.Path(sys.argv[0]).resolve())


def _server_unit_content(
    qjob_bin:  str,
    host:      str,
    port:      int,
    log_level: str,
    workers:   int,
    env_file:  str | None,
) -> str:
    env_flag = f"--env-file {env_file} " if env_file else ""
    return (
        "[Unit]\n"
        "Description=qjob API server\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={qjob_bin} {env_flag}"
        f"admin serve --host {host} --port {port} "
        f"--log-level {log_level} --workers {workers}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _scheduler_unit_content(
    qjob_bin:      str,
    poll_interval: float,
    max_workers:   int,
    env_file:      str | None,
) -> str:
    env_flag = f"--env-file {env_file} " if env_file else ""
    return (
        "[Unit]\n"
        "Description=qjob job scheduler\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={qjob_bin} {env_flag}"
        f"admin scheduler --poll-interval {poll_interval} --max-workers {max_workers}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


# --------------------------------------------------------------------------------------
# Display helpers


def _print_job_table(
    jobs:      list[service.JobInfo],
    truncated: bool = False,
) -> None:
    """Print a compact table of jobs to stdout."""

    header = (
        f"{'ID':<36}  {'USER':<10}  {'NAME':<20}  "
        f"{'STATUS':<10}  {'CPU':>3}  {'GPU':>3}  {'PRI':>3}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for j in jobs:
        name = (j.name or "")[:20]
        typer.echo(
            f"{j.id:<36}  {j.user:<10}  {name:<20}  {j.status:<10}  "
            f"{j.req_cpus:>3}  {j.req_gpus:>3}  {j.priority:>3}"
        )
    if truncated:
        typer.echo("...")


def _print_job_detail(info: service.JobInfo) -> None:
    """Print detailed information for a single job."""

    def _fmt(dt: object) -> str:
        return str(dt) if dt is not None else "—"

    lines = [
        f"ID           : {info.id}",
        f"Name         : {info.name or '—'}",
        f"User         : {info.user}",
        f"Status       : {info.status}",
        f"CPUs         : {info.req_cpus}",
        f"GPUs         : {info.req_gpus}",
        f"Memory       : {info.req_mem_mb} MB",
        f"Priority     : {info.priority}",
        f"Submitted    : {_fmt(info.submitted_at)}",
        f"Started      : {_fmt(info.started_at)}",
        f"Finished     : {_fmt(info.finished_at)}",
        f"Exit code    : {info.exit_code if info.exit_code is not None else '—'}",
        f"Workdir      : {info.workdir or '—'}",
        f"Stdout log   : {info.log_stdout or '—'}",
        f"Stderr log   : {info.log_stderr or '—'}",
    ]
    typer.echo("\n".join(lines))


def _fmt_walltime(sec: int | None) -> str:
    """Format seconds as HH:MM:SS, or '—' if None."""
    if sec is None:
        return "—"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_resources(info: service.ResourceInfo) -> None:
    """Print a resource summary table."""

    typer.echo(f"{'RESOURCE':<12}  {'TOTAL':>10}  {'USED':>8}  {'FREE':>8}")
    typer.echo("-" * 44)
    typer.echo(
        f"{'CPUs':<12}  {info.total_cpus:>10}  {info.used_cpus:>8}  "
        f"{info.total_cpus - info.used_cpus:>8}"
    )
    typer.echo(
        f"{'GPUs':<12}  {info.total_gpus:>10}  {info.used_gpus:>8}  "
        f"{info.total_gpus - info.used_gpus:>8}"
    )
    if info.gpu_ids:
        typer.echo(f"{'GPU IDs':<12}  {','.join(str(i) for i in info.gpu_ids):>10}")
    typer.echo(
        f"{'Memory(MB)':<12}  {info.total_mem_mb:>10}  {info.used_mem_mb:>8}  "
        f"{info.total_mem_mb - info.used_mem_mb:>8}"
    )
    typer.echo(
        f"{'Walltime':<12}  {_fmt_walltime(info.max_walltime_sec):>10}"
    )


# --------------------------------------------------------------------------------------
# Entry point

if __name__ == "__main__":
    app()
