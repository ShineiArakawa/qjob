from __future__ import annotations

import asyncio
import importlib.metadata
import os
import typing

import typer

import qjob.api.server as server
import qjob.cli.dashboard as dashboard
import qjob.cli.service as service
import qjob.core.database as database
import qjob.core.scheduler as _scheduler

# --------------------------------------------------------------------------------------
# Constants

_DEFAULT_STATUS_LIMIT = 10

# --------------------------------------------------------------------------------------
# Typer application

app = typer.Typer(
    name="qjob",
    help="Lightweight job scheduler for research servers.",
    no_args_is_help=True,
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
) -> None:
    pass


# --------------------------------------------------------------------------------------
# Shell completion helpers


def _complete_status(incomplete: str) -> list[str]:
    values = ["queued", "running", "done", "failed", "cancelled"]
    return [v for v in values if v.startswith(incomplete)]


def _complete_job_id(incomplete: str) -> list[str]:
    try:
        jobs = service.list_jobs(user=None)
        return [j.id for j in jobs if j.id.startswith(incomplete)]
    except Exception:
        return []


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
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except ConnectionError as exc:
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
    all_users: bool = typer.Option(
        False, "--all", "-a", help="Show jobs from all users (default: current user)."
    ),
    status_filter: typing.Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status: queued, running, done, failed, cancelled.",
        autocompletion=_complete_status,
    ),
    all_jobs: bool = typer.Option(
        False, "--all-jobs", help="Show all matching jobs instead of the latest 10."
    ),
) -> None:
    """
    Show job status.

    Without arguments, lists jobs submitted by the current user.
    """

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
        if all_jobs:
            jobs = service.list_jobs(user=resolved_user, status=status_filter)
            truncated = False
        else:
            jobs = service.list_jobs(
                user=resolved_user,
                status=status_filter,
                limit=_DEFAULT_STATUS_LIMIT + 1,
            )
            truncated = len(jobs) > _DEFAULT_STATUS_LIMIT
            jobs = jobs[:_DEFAULT_STATUS_LIMIT]
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
    job_ids: typing.List[str] = typer.Argument(
        ..., help="IDs of the jobs to cancel.", autocompletion=_complete_job_id
    ),
) -> None:
    """Cancel one or more queued or running jobs."""

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
            info = service.cancel_job(job_id)
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
def dashboard_cmd(
    refresh: float = typer.Option(
        3.0, "--refresh", "-r", help="Seconds between screen refreshes."
    ),
) -> None:
    """
    Open the live TUI dashboard showing resources and job status.

    Requires the qjob API server to be running.  Press Ctrl+C to exit.
    """

    dashboard.run(refresh_interval=refresh)


# --------------------------------------------------------------------------------------
# serve  (replaces Phase 1 daemon command)


@app.command()
def serve(
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

    For multi-process deployments use --workers N and run 'qjob scheduler'
    as a separate process.  Press Ctrl+C to stop.
    """

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


# --------------------------------------------------------------------------------------
# scheduler


@app.command()
def scheduler(
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


# --------------------------------------------------------------------------------------
# dashboard


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
# admin sub-commands


@admin_app.command("set-resources")
def admin_set_resources(
    cpus: typing.Optional[int] = typer.Option(
        None, "--cpus", help="Total number of CPU cores."
    ),
    gpus: typing.Optional[int] = typer.Option(
        None, "--gpus", help="Total number of GPU devices."
    ),
    mem: typing.Optional[int] = typer.Option(
        None, "--mem", help="Total memory in megabytes."
    ),
) -> None:
    """Update the available resource limits."""

    if cpus is None and gpus is None and mem is None:
        typer.echo("Error: specify at least one of --cpus, --gpus, --mem.", err=True)
        raise typer.Exit(code=1)

    try:
        info = service.set_resources(
            total_cpus=cpus,
            total_gpus=gpus,
            total_mem_mb=mem,
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
        jobs = service.list_jobs(user=None, status=status_filter)
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


def _print_resources(info: service.ResourceInfo) -> None:
    """Print a resource summary table."""

    typer.echo(f"{'RESOURCE':<10}  {'TOTAL':>8}  {'USED':>8}  {'FREE':>8}")
    typer.echo("-" * 40)
    typer.echo(
        f"{'CPUs':<10}  {info.total_cpus:>8}  {info.used_cpus:>8}  "
        f"{info.total_cpus - info.used_cpus:>8}"
    )
    typer.echo(
        f"{'GPUs':<10}  {info.total_gpus:>8}  {info.used_gpus:>8}  "
        f"{info.total_gpus - info.used_gpus:>8}"
    )
    typer.echo(
        f"{'Memory(MB)':<10}  {info.total_mem_mb:>8}  {info.used_mem_mb:>8}  "
        f"{info.total_mem_mb - info.used_mem_mb:>8}"
    )


# --------------------------------------------------------------------------------------
# Entry point

if __name__ == "__main__":
    app()
