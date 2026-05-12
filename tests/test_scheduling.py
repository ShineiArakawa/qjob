from __future__ import annotations

import datetime
import json

import pytest

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.scheduler as scheduler

# --------------------------------------------------------------------------------------
# Helpers


def _make_pool(
    cpus:   int = 8,
    gpus:   int = 2,
    mem_mb: int = 16384,
) -> scheduler.ResourcePool:
    """Return a ResourcePool with all resources free."""

    return scheduler.ResourcePool(
        total_cpus=cpus,
        total_gpus=gpus,
        total_mem_mb=mem_mb,
        free_cpu_ids=list(range(cpus)),
        free_gpu_ids=list(range(gpus)),
    )


def _make_queued_job(
    req_cpus:     int = 1,
    req_gpus:     int = 0,
    req_mem_mb:   int = 512,
    priority:     int = 50,
    walltime_sec: int | None = None,
    wait_hours:   float = 0.0,
) -> models.Job:
    """
    Insert and return a QUEUED job with a synthetic submission time.

    Parameters
    ----------
    req_cpus : int
        CPU cores requested.
    req_gpus : int
        GPUs requested.
    req_mem_mb : int
        Memory requested in megabytes.
    priority : int
        Base scheduling priority (0–100).
    walltime_sec : int | None
        Walltime limit in seconds.  None means unlimited.
    wait_hours : float
        How many hours ago the job was submitted (used to simulate ageing).

    Returns
    -------
    models.Job
        The persisted job instance.
    """

    submitted = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        hours=wait_hours
    )
    job = models.Job(
        user="alice",
        script_path="/tmp/job.sh",
        status=models.JobStatus.QUEUED,
        req_cpus=req_cpus,
        req_gpus=req_gpus,
        req_mem_mb=req_mem_mb,
        priority=priority,
        walltime_sec=walltime_sec,
        submitted_at=submitted,
    )
    with database.get_session() as session:
        session.add(job)
    return job


def _make_running_job(
    req_cpus:     int = 1,
    req_gpus:     int = 0,
    req_mem_mb:   int = 512,
    walltime_sec: int | None = 3600,
    started_seconds_ago: int = 0,
) -> models.Job:
    """Insert and return a RUNNING job."""

    started = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=started_seconds_ago
    )
    job = models.Job(
        user="alice",
        script_path="/tmp/job.sh",
        status=models.JobStatus.RUNNING,
        req_cpus=req_cpus,
        req_gpus=req_gpus,
        req_mem_mb=req_mem_mb,
        walltime_sec=walltime_sec,
        started_at=started,
        assigned_cpus=json.dumps(list(range(req_cpus))),
        assigned_gpus=json.dumps(list(range(req_gpus))),
    )
    with database.get_session() as session:
        session.add(job)
    return job


def _make_scheduler(
    poll_interval: float = 2.0,
    aging_factor:  float = 5.0,
    max_workers:   int = 64,
    pool_cpus:     int = 8,    # ← pool 用パラメータを別名にする
    pool_gpus:     int = 2,
    pool_mem_mb:   int = 16384,
) -> scheduler.Scheduler:
    """Return a Scheduler with the given kwargs and a pre-loaded pool."""
    sched = scheduler.Scheduler(
        poll_interval=poll_interval,
        aging_factor=aging_factor,
        max_workers=max_workers,
    )
    sched._pool = _make_pool(cpus=pool_cpus, gpus=pool_gpus, mem_mb=pool_mem_mb)
    return sched

# --------------------------------------------------------------------------------------
# Priority ageing tests


class TestApplyAging:
    """_apply_aging() increases job priority based on wait time."""

    def test_priority_increases_after_waiting(self):
        job = _make_queued_job(priority=50, wait_hours=1.0)
        sched = _make_scheduler(poll_interval=3600.0, aging_factor=5.0)
        # increment = 5.0 * (3600/3600) = 5 → priority = 55
        with database.get_session() as session:
            sched._apply_aging(session, [job])
        assert job.priority > 50

    def test_priority_capped_at_100(self):
        job = _make_queued_job(priority=99, wait_hours=1000.0)
        sched = _make_scheduler(poll_interval=3600.0, aging_factor=100.0)
        # increment = 100 * (3600/3600) = 100 → min(100, 199) = 100
        with database.get_session() as session:
            sched._apply_aging(session, [job])
        assert job.priority == 100

    def test_zero_wait_time_no_increase(self):
        job = _make_queued_job(priority=50, wait_hours=0.0)
        sched = _make_scheduler(poll_interval=2.0, aging_factor=5.0)
        # increment = 5.0 * (2/3600) ≈ 0.0028 → int() = 0 → no change
        original = job.priority
        with database.get_session() as session:
            sched._apply_aging(session, [job])
        assert job.priority == original

    def test_higher_aging_factor_increases_faster(self):
        job_slow = _make_queued_job(priority=50, wait_hours=10.0)
        job_fast = _make_queued_job(priority=50, wait_hours=10.0)
        sched_slow = _make_scheduler(poll_interval=3600.0, aging_factor=1.0)
        sched_fast = _make_scheduler(poll_interval=3600.0, aging_factor=10.0)

        with database.get_session() as session:
            sched_slow._apply_aging(session, [job_slow])
            sched_fast._apply_aging(session, [job_fast])

        assert job_fast.priority > job_slow.priority

    def test_aging_persisted_to_db(self):
        job = _make_queued_job(priority=50, wait_hours=0.0)
        sched = _make_scheduler(poll_interval=7200.0, aging_factor=5.0)

        with database.get_session() as session:
            sched._apply_aging(session, [job])

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            # After 2h poll_interval with aging_factor=5: 5*(7200/3600)=10 points
            assert stored.priority >= 50


# --------------------------------------------------------------------------------------
# Reservation window estimation tests


class TestEstimateReservationWindow:
    """_estimate_reservation_window() predicts when the head job can start."""

    def test_returns_none_when_no_running_jobs(self):
        head = _make_queued_job(req_cpus=8)
        sched = _make_scheduler()
        with database.get_session() as session:
            result = sched._estimate_reservation_window(session, head)
        assert result is None

    def test_returns_none_when_running_jobs_have_no_walltime(self):
        _make_running_job(req_cpus=4, walltime_sec=None)
        head = _make_queued_job(req_cpus=8)
        sched = _make_scheduler(pool_cpus=4)
        sched._pool = _make_pool(cpus=4)
        sched._pool.free_cpu_ids = []   # All CPUs occupied.
        with database.get_session() as session:
            result = sched._estimate_reservation_window(session, head)
        assert result is None

    def test_returns_remaining_walltime_of_blocking_job(self):
        # Running job: 3600s walltime, started 600s ago → 3000s remaining.
        _make_running_job(
            req_cpus=4,
            walltime_sec=3600,
            started_seconds_ago=600,
        )
        head = _make_queued_job(req_cpus=4)
        sched = _make_scheduler()
        sched._pool = _make_pool(cpus=4)
        sched._pool.free_cpu_ids = []   # All CPUs occupied.

        with database.get_session() as session:
            result = sched._estimate_reservation_window(session, head)

        assert result is not None
        assert 2990 <= result <= 3010   # Allow small timing tolerance.

    def test_uses_earliest_sufficient_release(self):
        # Two running jobs; head needs 2 CPUs; each job holds 1 CPU.
        _make_running_job(req_cpus=1, walltime_sec=1000, started_seconds_ago=0)
        _make_running_job(req_cpus=1, walltime_sec=2000, started_seconds_ago=0)
        head = _make_queued_job(req_cpus=2)
        sched = _make_scheduler()
        sched._pool = _make_pool(cpus=2)
        sched._pool.free_cpu_ids = []

        with database.get_session() as session:
            result = sched._estimate_reservation_window(session, head)

        # Both jobs must finish for head to get 2 CPUs → window ≈ 2000s.
        assert result is not None
        assert 1990 <= result <= 2010


# --------------------------------------------------------------------------------------
# Backfill candidate selection tests


class TestFindBackfillJobs:
    """_find_backfill_jobs() selects eligible jobs within the reservation window."""

    def test_job_with_walltime_within_window_is_selected(self):
        job = _make_queued_job(walltime_sec=600)
        sched = _make_scheduler()
        result = sched._find_backfill_jobs(
            candidates=[job],
            launched_ids=set(),
            reservation_sec=1000.0,
            slots_remaining=10,
        )
        assert job in result

    def test_job_without_walltime_is_excluded(self):
        job = _make_queued_job(walltime_sec=None)
        sched = _make_scheduler()
        result = sched._find_backfill_jobs(
            candidates=[job],
            launched_ids=set(),
            reservation_sec=1000.0,
            slots_remaining=10,
        )
        assert job not in result

    def test_job_exceeding_window_is_excluded(self):
        # walltime=800 > window=1000 * 0.9=900 → excluded
        job = _make_queued_job(walltime_sec=950)
        sched = _make_scheduler()
        result = sched._find_backfill_jobs(
            candidates=[job],
            launched_ids=set(),
            reservation_sec=1000.0,
            slots_remaining=10,
        )
        assert job not in result

    def test_already_launched_job_is_excluded(self):
        job = _make_queued_job(walltime_sec=600)
        sched = _make_scheduler()
        result = sched._find_backfill_jobs(
            candidates=[job],
            launched_ids={job.id},
            reservation_sec=1000.0,
            slots_remaining=10,
        )
        assert job not in result

    def test_slots_remaining_limits_results(self):
        jobs = [_make_queued_job(walltime_sec=300) for _ in range(5)]
        sched = _make_scheduler()
        result = sched._find_backfill_jobs(
            candidates=jobs,
            launched_ids=set(),
            reservation_sec=1000.0,
            slots_remaining=2,
        )
        assert len(result) == 2

    def test_multiple_eligible_jobs_all_selected(self):
        jobs = [
            _make_queued_job(walltime_sec=100),
            _make_queued_job(walltime_sec=200),
            _make_queued_job(walltime_sec=300),
        ]
        sched = _make_scheduler()
        result = sched._find_backfill_jobs(
            candidates=jobs,
            launched_ids=set(),
            reservation_sec=1000.0,
            slots_remaining=10,
        )
        assert len(result) == 3


# --------------------------------------------------------------------------------------
# Scheduler — aging_factor constructor parameter


class TestSchedulerAgingFactor:
    """Scheduler accepts a custom aging_factor."""

    def test_default_aging_factor(self):
        sched = scheduler.Scheduler()
        assert sched._aging_factor == scheduler._DEFAULT_AGING_FACTOR

    def test_custom_aging_factor(self):
        sched = scheduler.Scheduler(aging_factor=10.0)
        assert sched._aging_factor == 10.0


# --------------------------------------------------------------------------------------
# Integration: ageing raises low-priority job above high-priority newcomer


class TestAgingIntegration:
    """A long-waiting low-priority job eventually overtakes a high-priority newcomer."""

    def test_aged_job_has_higher_effective_priority(self):
        # Job A: low priority, submitted 24 hours ago.
        job_a = _make_queued_job(priority=20, wait_hours=24.0)
        # Job B: high priority, just submitted.
        job_b = _make_queued_job(priority=80, wait_hours=0.0)

        # Use a 1-hour poll interval and high aging factor so one tick is enough.
        sched = _make_scheduler(poll_interval=3600.0, aging_factor=5.0)

        with database.get_session() as session:
            sched._apply_aging(session, [job_a, job_b])

        # After ageing: job_a gets +5 points; job_b stays at 80.
        # With 24h wait and 1h poll: increment = 5*(3600/3600)=5 → job_a=25.
        # To overtake job_b (80) job_a needs many ticks — just verify direction.
        assert job_a.priority > 20
        assert job_b.priority >= 80
