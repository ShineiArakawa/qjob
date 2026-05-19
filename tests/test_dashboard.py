from __future__ import annotations

import datetime

import rich.console

import qjob.cli.dashboard as dashboard
import qjob.cli.service as service

# --------------------------------------------------------------------------------------
# Helper


def _render_to_str(renderable) -> str:
    """
    Render a Rich renderable to a plain string for assertion.

    Parameters
    ----------
    renderable : object
        Any Rich renderable (Panel, Table, Text, Layout, …).

    Returns
    -------
    str
        The rendered output with ANSI codes stripped.
    """

    console = rich.console.Console(
        width=200,
        highlight=False,
        no_color=True,
        force_terminal=False,
    )
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def _make_job_info(**kwargs) -> service.JobInfo:
    """Return a JobInfo with sensible defaults overridden by *kwargs*."""

    defaults = dict(
        id="aaaaaaaa-0000-0000-0000-000000000001",
        user="alice",
        name="test-job",
        status="queued",
        req_cpus=2,
        req_gpus=1,
        req_mem_mb=4096,
        priority=50,
        submitted_at=datetime.datetime(2025, 1, 1, 12, 0, 0,
                                       tzinfo=datetime.timezone.utc),
        started_at=None,
        finished_at=None,
        exit_code=None,
        log_stdout=None,
        log_stderr=None,
        workdir=None,
    )
    defaults.update(kwargs)
    return service.JobInfo(**defaults)


def _make_resource_info(**kwargs) -> service.ResourceInfo:
    """Return a ResourceInfo with sensible defaults overridden by *kwargs*."""

    defaults = dict(
        total_cpus=32,
        total_gpus=4,
        total_mem_mb=131072,
        max_walltime_sec=None,
        used_cpus=8,
        used_gpus=2,
        used_mem_mb=32768,
    )
    defaults.update(kwargs)
    return service.ResourceInfo(**defaults)


# --------------------------------------------------------------------------------------
# _elapsed


class TestElapsed:
    """_elapsed() formats timedeltas correctly."""

    def _now(self) -> datetime.datetime:
        return datetime.datetime(2025, 6, 1, 12, 0, 0,
                                 tzinfo=datetime.timezone.utc)

    def test_returns_dash_when_start_is_none(self):
        assert dashboard._elapsed(None, self._now()) == "—"

    def test_zero_elapsed(self):
        now = self._now()
        assert dashboard._elapsed(now, now) == "00:00:00"

    def test_one_hour(self):
        now = self._now()
        start = now - datetime.timedelta(hours=1)
        assert dashboard._elapsed(start, now) == "01:00:00"

    def test_mixed_hms(self):
        now = self._now()
        start = now - datetime.timedelta(hours=2, minutes=30, seconds=15)
        assert dashboard._elapsed(start, now) == "02:30:15"

    def test_naive_start_treated_as_utc(self):
        now = self._now()
        start = datetime.datetime(2025, 6, 1, 11, 0, 0)  # naive
        assert dashboard._elapsed(start, now) == "01:00:00"

    def test_never_returns_negative(self):
        now = self._now()
        start = now + datetime.timedelta(hours=1)  # future start
        assert dashboard._elapsed(start, now) == "00:00:00"


# --------------------------------------------------------------------------------------
# _render_header


class TestRenderHeader:
    """_render_header() shows title or error message."""

    def test_no_error_contains_qjob(self):
        output = _render_to_str(dashboard._render_header(None))
        assert "qjob" in output.lower()

    def test_error_message_is_displayed(self):
        output = _render_to_str(dashboard._render_header("Connection refused"))
        assert "Connection refused" in output

    def test_no_error_contains_timestamp_digits(self):
        output = _render_to_str(dashboard._render_header(None))
        assert any(c.isdigit() for c in output)


# --------------------------------------------------------------------------------------
# _render_resources


class TestRenderResources:
    """_render_resources() shows usage bars and free counts."""

    def test_unavailable_when_none(self):
        output = _render_to_str(dashboard._render_resources(None))
        assert "Unavailable" in output

    def test_cpu_usage_displayed(self):
        info = _make_resource_info(total_cpus=32, used_cpus=8)
        output = _render_to_str(dashboard._render_resources(info))
        assert "8" in output
        assert "32" in output

    def test_gpu_usage_displayed(self):
        info = _make_resource_info(total_gpus=4, used_gpus=2)
        output = _render_to_str(dashboard._render_resources(info))
        assert "2" in output
        assert "4" in output

    def test_memory_in_gigabytes(self):
        info = _make_resource_info(total_mem_mb=131072, used_mem_mb=65536)
        output = _render_to_str(dashboard._render_resources(info))
        assert "128" in output  # 131072 MB = 128 GB

    def test_zero_total_gpu_shows_na(self):
        info = _make_resource_info(total_gpus=0, used_gpus=0)
        output = _render_to_str(dashboard._render_resources(info))
        assert "N/A" in output


# --------------------------------------------------------------------------------------
# _render_stats


class TestRenderStats:
    """_render_stats() shows correct job counts and labels."""

    def test_all_counts_present(self):
        output = _render_to_str(
            dashboard._render_stats(
                total=10, running=3, queued=5, done=1, failed=1
            )
        )
        for n in ("10", "3", "5"):
            assert n in output

    def test_labels_present(self):
        output = _render_to_str(
            dashboard._render_stats(
                total=0, running=0, queued=0, done=0, failed=0
            )
        )
        for label in ("Total", "Running", "Queued", "Done", "Failed"):
            assert label in output


# --------------------------------------------------------------------------------------
# _render_running_jobs


class TestRenderRunningJobs:
    """_render_running_jobs() shows running jobs or a placeholder."""

    def test_empty_shows_placeholder(self):
        output = _render_to_str(dashboard._render_running_jobs([]))
        assert "No running jobs" in output

    def test_job_id_displayed(self):
        job = _make_job_info(
            status="running",
            started_at=datetime.datetime(2025, 1, 1, 10, 0, 0,
                                         tzinfo=datetime.timezone.utc),
        )
        output = _render_to_str(dashboard._render_running_jobs([job]))
        assert job.id in output

    def test_job_user_displayed(self):
        job = _make_job_info(
            status="running",
            user="bob",
            started_at=datetime.datetime.now(datetime.timezone.utc),
        )
        output = _render_to_str(dashboard._render_running_jobs([job]))
        assert "bob" in output

    def test_count_in_title(self):
        jobs = [
            _make_job_info(
                id=f"aaaaaaaa-0000-0000-0000-{i:012d}",
                status="running",
                started_at=datetime.datetime.now(datetime.timezone.utc),
            )
            for i in range(3)
        ]
        output = _render_to_str(dashboard._render_running_jobs(jobs))
        assert "3" in output


# --------------------------------------------------------------------------------------
# _render_queued_jobs


class TestRenderQueuedJobs:
    """_render_queued_jobs() shows queued jobs or a placeholder."""

    def test_empty_shows_placeholder(self):
        output = _render_to_str(dashboard._render_queued_jobs([]))
        assert "Queue is empty" in output

    def test_priority_displayed(self):
        job = _make_job_info(status="queued", priority=80)
        output = _render_to_str(dashboard._render_queued_jobs([job]))
        assert "80" in output

    def test_overflow_shows_more_suffix(self):
        jobs = [
            _make_job_info(id=f"aaaaaaaa-0000-0000-0000-{i:012d}", status="queued")
            for i in range(25)
        ]
        output = _render_to_str(dashboard._render_queued_jobs(jobs))
        assert "more" in output

    def test_no_overflow_within_limit(self):
        jobs = [
            _make_job_info(id=f"aaaaaaaa-0000-0000-0000-{i:012d}", status="queued")
            for i in range(5)
        ]
        output = _render_to_str(dashboard._render_queued_jobs(jobs))
        assert "more" not in output


# --------------------------------------------------------------------------------------
# _render_recent_jobs


class TestRenderRecentJobs:
    """_render_recent_jobs() shows completed jobs or a placeholder."""

    def test_empty_shows_placeholder(self):
        output = _render_to_str(dashboard._render_recent_jobs([]))
        assert "No completed jobs" in output

    def test_done_job_displayed(self):
        job = _make_job_info(
            status="done",
            exit_code=0,
            finished_at=datetime.datetime(2025, 6, 1, 10, 0, 0,
                                          tzinfo=datetime.timezone.utc),
        )
        output = _render_to_str(dashboard._render_recent_jobs([job]))
        assert job.id in output
        assert "done" in output

    def test_failed_job_and_exit_code_displayed(self):
        job = _make_job_info(
            status="failed",
            exit_code=1,
            finished_at=datetime.datetime(2025, 6, 1, 10, 0, 0,
                                          tzinfo=datetime.timezone.utc),
        )
        output = _render_to_str(dashboard._render_recent_jobs([job]))
        assert "failed" in output
        assert "1" in output

    def test_exit_code_dash_when_none(self):
        job = _make_job_info(
            status="done",
            exit_code=None,
            finished_at=datetime.datetime(2025, 6, 1, 10, 0, 0,
                                          tzinfo=datetime.timezone.utc),
        )
        output = _render_to_str(dashboard._render_recent_jobs([job]))
        assert "—" in output


# --------------------------------------------------------------------------------------
# _build_layout smoke tests


class TestBuildLayout:
    """_build_layout() assembles a renderable layout without errors."""

    def test_renders_without_error(self, monkeypatch):
        monkeypatch.setattr(service, "get_resources",
                            lambda: _make_resource_info())
        monkeypatch.setattr(
            service, "list_jobs",
            lambda **_: [
                _make_job_info(
                    status="running",
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                ),
                _make_job_info(
                    id="bbbbbbbb-0000-0000-0000-000000000002",
                    status="queued",
                ),
            ],
        )
        output = _render_to_str(dashboard._build_layout(refresh_interval=3.0))
        assert len(output) > 0

    def test_connection_error_shows_error_in_header(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise ConnectionError("Connection refused")

        monkeypatch.setattr(service, "get_resources", _raise)
        output = _render_to_str(dashboard._build_layout(refresh_interval=3.0))
        assert "Connection refused" in output
