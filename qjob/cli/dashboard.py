from __future__ import annotations

import datetime
import time

import rich.box
import rich.console
import rich.layout
import rich.live
import rich.panel
import rich.table
import rich.text

import qjob.cli.service as service

# --------------------------------------------------------------------------------------
# Constants

_REFRESH_INTERVAL: float = 3.0    # Seconds between dashboard refreshes.
_MAX_QUEUE_ROWS:   int = 20     # Maximum queued jobs displayed.
_MAX_DONE_ROWS:    int = 10     # Maximum recent done/failed jobs displayed.


# --------------------------------------------------------------------------------------
# Public API


def run(refresh_interval: float = _REFRESH_INTERVAL) -> None:
    """
    Start the Rich TUI dashboard and block until the user presses Ctrl+C.

    The dashboard polls the API every *refresh_interval* seconds and redraws
    all panes in-place using ``rich.live.Live``.

    Parameters
    ----------
    refresh_interval : float
        Seconds between data fetches and screen refreshes.

    Returns
    -------
    None
    """

    console = rich.console.Console()

    with rich.live.Live(
        _build_layout(refresh_interval),
        console=console,
        screen=True,
        refresh_per_second=4,
    ) as live:
        try:
            while True:
                live.update(_build_layout(refresh_interval))
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            pass


# --------------------------------------------------------------------------------------
# Layout builder


def _build_layout(refresh_interval: float) -> rich.layout.Layout:
    """
    Fetch current data and assemble the full dashboard layout.

    Parameters
    ----------
    refresh_interval : float
        Displayed in the footer so the user knows how often data refreshes.

    Returns
    -------
    rich.layout.Layout
        The assembled layout ready for rendering.
    """

    try:
        resources = service.get_resources()
        all_jobs = service.list_jobs(all_users=True)
        error_msg = None
    except ConnectionError as exc:
        resources = None
        all_jobs = []
        error_msg = str(exc)

    running = [j for j in all_jobs if j.status in ("running", "cancelling")]
    queued = [j for j in all_jobs if j.status == "queued"]
    done = [j for j in all_jobs if j.status == "done"]
    failed = [j for j in all_jobs if j.status == "failed"]
    recent_terminal = sorted(
        [j for j in all_jobs if j.status in ("done", "failed", "cancelled")],
        key=lambda j: j.finished_at or datetime.datetime.min,
        reverse=True,
    )[:_MAX_DONE_ROWS]

    layout = rich.layout.Layout()
    layout.split_column(
        rich.layout.Layout(name="header",   size=3),
        rich.layout.Layout(name="middle",   size=8),
        rich.layout.Layout(name="running",  minimum_size=6),
        rich.layout.Layout(name="queued",   minimum_size=6),
        rich.layout.Layout(name="recent",   minimum_size=4),
        rich.layout.Layout(name="footer",   size=1),
    )
    layout["middle"].split_row(
        rich.layout.Layout(name="resources", ratio=1),
        rich.layout.Layout(name="stats",     ratio=1),
    )

    layout["header"].update(_render_header(error_msg))
    layout["resources"].update(_render_resources(resources))
    layout["stats"].update(
        _render_stats(
            total=len(all_jobs),
            running=len(running),
            queued=len(queued),
            done=len(done),
            failed=len(failed),
        )
    )
    layout["running"].update(_render_running_jobs(running))
    layout["queued"].update(_render_queued_jobs(queued))
    layout["recent"].update(_render_recent_jobs(recent_terminal))
    layout["footer"].update(
        rich.text.Text(
            f" Refreshing every {refresh_interval:.0f}s  —  Press Ctrl+C to exit",
            style="dim",
        )
    )

    return layout


# --------------------------------------------------------------------------------------
# Pane renderers


def _render_header(error_msg: str | None) -> rich.panel.Panel:
    """
    Render the top header panel with title and current timestamp.

    Parameters
    ----------
    error_msg : str | None
        When not None an error banner is shown instead of the normal header.

    Returns
    -------
    rich.panel.Panel
        The rendered header.
    """

    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    if error_msg:
        content = rich.text.Text(
            f"  Cannot reach API server: {error_msg}",
            style="bold red",
        )
    else:
        content = rich.text.Text.assemble(
            ("  qjob", "bold cyan"),
            "  Job Scheduler Dashboard    ",
            (now, "dim"),
        )

    return rich.panel.Panel(content, style="bold")


def _render_resources(
    resources: service.ResourceInfo | None,
) -> rich.panel.Panel:
    """
    Render the resource usage panel with ASCII progress bars.

    Parameters
    ----------
    resources : service.ResourceInfo | None
        Current resource data, or None if the server is unreachable.

    Returns
    -------
    rich.panel.Panel
        The rendered resource panel.
    """

    if resources is None:
        return rich.panel.Panel(
            rich.text.Text("Unavailable", style="dim"),
            title="Resources",
        )

    def _bar(used: int, total: int, color: str) -> rich.text.Text:
        if total == 0:
            return rich.text.Text("N/A", style="dim")
        ratio = used / total
        filled = int(ratio * 20)
        bar_str = "█" * filled + "░" * (20 - filled)
        style = color if ratio < 0.8 else ("yellow" if ratio < 0.95 else "red")
        return rich.text.Text.assemble(
            (bar_str, style),
            f"  {ratio * 100:.0f}%",
        )

    free_cpus = resources.total_cpus - resources.used_cpus
    free_gpus = resources.total_gpus - resources.used_gpus
    free_mem_gb = (resources.total_mem_mb - resources.used_mem_mb) / 1024

    table = rich.table.Table.grid(padding=(0, 2))
    table.add_column(justify="right",  style="dim", width=8)
    table.add_column(justify="left",   width=28)
    table.add_column(justify="right",  width=18)

    table.add_row(
        "CPU",
        _bar(resources.used_cpus, resources.total_cpus, "green"),
        f"{resources.used_cpus}/{resources.total_cpus}  ({free_cpus} free)",
    )
    table.add_row(
        "GPU",
        _bar(resources.used_gpus, resources.total_gpus, "blue"),
        f"{resources.used_gpus}/{resources.total_gpus}  ({free_gpus} free)",
    )
    table.add_row(
        "Memory",
        _bar(resources.used_mem_mb, resources.total_mem_mb, "magenta"),
        f"{resources.used_mem_mb // 1024}G/{resources.total_mem_mb // 1024}G"
        f"  ({free_mem_gb:.1f}G free)",
    )

    return rich.panel.Panel(table, title="[bold]Resources[/bold]")


def _render_stats(
    total:   int,
    running: int,
    queued:  int,
    done:    int,
    failed:  int,
) -> rich.panel.Panel:
    """
    Render the job count statistics panel.

    Parameters
    ----------
    total : int
        Total number of known jobs.
    running : int
        Jobs currently running.
    queued : int
        Jobs waiting in the queue.
    done : int
        Jobs that completed successfully.
    failed : int
        Jobs that failed or were cancelled.

    Returns
    -------
    rich.panel.Panel
        The rendered stats panel.
    """

    table = rich.table.Table.grid(padding=(0, 4))
    table.add_column(justify="right", style="dim", width=10)
    table.add_column(justify="left")

    table.add_row("Total",   rich.text.Text(str(total),   style="bold"))
    table.add_row("Running", rich.text.Text(str(running), style="bold green"))
    table.add_row("Queued",  rich.text.Text(str(queued),  style="bold yellow"))
    table.add_row("Done",    rich.text.Text(str(done),    style="bold cyan"))
    table.add_row("Failed",  rich.text.Text(str(failed),  style="bold red"))

    return rich.panel.Panel(table, title="[bold]Statistics[/bold]")


def _render_running_jobs(jobs: list[service.JobInfo]) -> rich.panel.Panel:
    """
    Render the running jobs table with elapsed time.

    Parameters
    ----------
    jobs : list[service.JobInfo]
        Currently running jobs.

    Returns
    -------
    rich.panel.Panel
        The rendered panel.
    """

    table = rich.table.Table(
        show_header=True,
        header_style="bold green",
        box=rich.box.SIMPLE,
        expand=True,
    )
    table.add_column("ID",      width=36)
    table.add_column("User",    width=10)
    table.add_column("Name",    width=20)
    table.add_column("CPU",     width=4,  justify="right")
    table.add_column("GPU",     width=4,  justify="right")
    table.add_column("Mem MB",  width=7,  justify="right")
    table.add_column("Elapsed", width=10, justify="right")

    now = datetime.datetime.now(datetime.timezone.utc)

    if not jobs:
        table.add_row(
            rich.text.Text("No running jobs", style="dim"),
            "", "", "", "", "", "",
        )
    else:
        for j in jobs:
            table.add_row(
                j.id,
                j.user,
                (j.name or "")[:20],
                str(j.req_cpus),
                str(j.req_gpus),
                str(j.req_mem_mb),
                _elapsed(j.started_at, now),
            )

    return rich.panel.Panel(
        table,
        title=f"[bold green]Running  ({len(jobs)})[/bold green]",
    )


def _render_queued_jobs(jobs: list[service.JobInfo]) -> rich.panel.Panel:
    """
    Render the queued jobs table (up to ``_MAX_QUEUE_ROWS`` rows).

    Parameters
    ----------
    jobs : list[service.JobInfo]
        Jobs currently waiting in the queue, sorted by priority.

    Returns
    -------
    rich.panel.Panel
        The rendered panel.
    """

    display = jobs[:_MAX_QUEUE_ROWS]

    table = rich.table.Table(
        show_header=True,
        header_style="bold yellow",
        box=rich.box.SIMPLE,
        expand=True,
    )
    table.add_column("ID",      width=36)
    table.add_column("User",    width=10)
    table.add_column("Name",    width=20)
    table.add_column("CPU",     width=4,  justify="right")
    table.add_column("GPU",     width=4,  justify="right")
    table.add_column("Pri",     width=4,  justify="right")
    table.add_column("Waiting", width=10, justify="right")

    now = datetime.datetime.now(datetime.timezone.utc)

    if not display:
        table.add_row(
            rich.text.Text("Queue is empty", style="dim"),
            "", "", "", "", "", "",
        )
    else:
        for j in display:
            table.add_row(
                j.id,
                j.user,
                (j.name or "")[:20],
                str(j.req_cpus),
                str(j.req_gpus),
                str(j.priority),
                _elapsed(j.submitted_at, now),
            )

    suffix = f"  (+{len(jobs) - _MAX_QUEUE_ROWS} more)" if len(jobs) > _MAX_QUEUE_ROWS else ""
    return rich.panel.Panel(
        table,
        title=f"[bold yellow]Queued  ({len(jobs)}){suffix}[/bold yellow]",
    )


def _render_recent_jobs(jobs: list[service.JobInfo]) -> rich.panel.Panel:
    """
    Render the recently completed or failed jobs table.

    Parameters
    ----------
    jobs : list[service.JobInfo]
        Recently finished jobs (done / failed / cancelled), newest first.

    Returns
    -------
    rich.panel.Panel
        The rendered panel.
    """

    table = rich.table.Table(
        show_header=True,
        header_style="bold cyan",
        box=rich.box.SIMPLE,
        expand=True,
    )
    table.add_column("ID",       width=36)
    table.add_column("User",     width=10)
    table.add_column("Name",     width=20)
    table.add_column("Status",   width=10)
    table.add_column("Exit",     width=5,  justify="right")
    table.add_column("Finished", width=20)

    if not jobs:
        table.add_row(
            rich.text.Text("No completed jobs", style="dim"),
            "", "", "", "", "",
        )
    else:
        for j in jobs:
            status_style = "green" if j.status == "done" else "red"
            finished_str = str(j.finished_at)[:19] if j.finished_at else "—"
            table.add_row(
                j.id,
                j.user,
                (j.name or "")[:20],
                rich.text.Text(j.status, style=status_style),
                str(j.exit_code) if j.exit_code is not None else "—",
                finished_str,
            )

    return rich.panel.Panel(
        table,
        title="[bold cyan]Recently Completed[/bold cyan]",
    )


# --------------------------------------------------------------------------------------
# Utilities


def _elapsed(
    start: datetime.datetime | None,
    now:   datetime.datetime,
) -> str:
    """
    Format the elapsed time between *start* and *now* as ``HH:MM:SS``.

    Parameters
    ----------
    start : datetime.datetime | None
        Start timestamp.  Returns ``"—"`` when None.
    now : datetime.datetime
        Reference time (typically UTC now).

    Returns
    -------
    str
        Elapsed time string, or ``"—"`` when *start* is None.
    """

    if start is None:
        return "—"

    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)

    delta = max(datetime.timedelta(0), now - start)
    total_s = int(delta.total_seconds())
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
