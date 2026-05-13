from __future__ import annotations

import asyncio
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


def _make_scheduler(**kwargs) -> scheduler.Scheduler:
    """Return a Scheduler with the given kwargs."""

    return scheduler.Scheduler(**kwargs)


# --------------------------------------------------------------------------------------
# _effective_priority tests


class TestEffectivePriority:
    """_effective_priority() computes aging in-memory without mutating the DB."""

    def test_returns_base_priority_when_no_wait(self):
        job = _make_queued_job(priority=50, wait_hours=0.0)
        sched = _make_scheduler(aging_factor=5.0)
        ep = sched._effective_priority(job)
        assert abs(ep - 50.0) < 0.1

    def test_priority_increases_with_wait_time(self):
        job = _make_queued_job(priority=50, wait_hours=1.0)
        sched = _make_scheduler(aging_factor=5.0)
        assert sched._effective_priority(job) > 50.0

    def test_priority_capped_at_100(self):
        job = _make_queued_job(priority=50, wait_hours=1000.0)
        sched = _make_scheduler(aging_factor=100.0)
        assert sched._effective_priority(job) == 100.0

    def test_higher_aging_factor_gives_higher_priority(self):
        job_a = _make_queued_job(priority=50, wait_hours=2.0)
        job_b = _make_queued_job(priority=50, wait_hours=2.0)
        sched_slow = _make_scheduler(aging_factor=1.0)
        sched_fast = _make_scheduler(aging_factor=10.0)
        assert sched_fast._effective_priority(job_b) > sched_slow._effective_priority(job_a)

    def test_db_priority_not_mutated(self):
        """Base priority in the DB must remain unchanged after calling _effective_priority."""
        job = _make_queued_job(priority=50, wait_hours=2.0)
        sched = _make_scheduler(aging_factor=5.0)
        sched._effective_priority(job)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
        assert stored.priority == 50

    def test_linear_growth_not_quadratic(self):
        """Effective priority must grow linearly with wait time (not O(N²))."""
        job1 = _make_queued_job(priority=0, wait_hours=1.0)
        job2 = _make_queued_job(priority=0, wait_hours=2.0)
        sched = _make_scheduler(aging_factor=5.0)
        ep1 = sched._effective_priority(job1)
        ep2 = sched._effective_priority(job2)
        # Linear: ep(2h) / ep(1h) should be very close to 2.
        assert abs(ep2 / ep1 - 2.0) < 0.05

    def test_multiple_calls_do_not_compound(self):
        """Calling _effective_priority twice must not accumulate additional aging."""
        job = _make_queued_job(priority=50, wait_hours=1.0)
        sched = _make_scheduler(aging_factor=5.0)
        ep1 = sched._effective_priority(job)
        ep2 = sched._effective_priority(job)
        assert abs(ep1 - ep2) < 0.01


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
        # Set used_cpus=4 so free_cpus=0 in DB.
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4
            row.used_cpus = 4
        sched = _make_scheduler()
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
        # Set used_cpus=4 so free_cpus=0 in DB.
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4
            row.used_cpus = 4

        sched = _make_scheduler()
        with database.get_session() as session:
            result = sched._estimate_reservation_window(session, head)

        assert result is not None
        assert 2990 <= result <= 3010   # Allow small timing tolerance.

    def test_uses_earliest_sufficient_release(self):
        # Two running jobs; head needs 2 CPUs; each job holds 1 CPU.
        _make_running_job(req_cpus=1, walltime_sec=1000, started_seconds_ago=0)
        _make_running_job(req_cpus=1, walltime_sec=2000, started_seconds_ago=0)
        head = _make_queued_job(req_cpus=2)
        # Set used_cpus=2 so free_cpus=0 in DB.
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 2
            row.used_cpus = 2

        sched = _make_scheduler()
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
# Scheduler tick allocation tests


class TestTickAllocation:
    """One scheduler tick must not assign the same CPU/GPU IDs twice."""

    def test_same_tick_cpu_allocations_are_disjoint(self, monkeypatch):
        launched: list[str] = []

        def _fake_start_job(job, resource_pool):
            launched.append(job.id)

        monkeypatch.setattr(scheduler.runner, "start_job", _fake_start_job)

        job_a = _make_queued_job(req_cpus=2, req_gpus=0)
        job_b = _make_queued_job(req_cpus=2, req_gpus=0)
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4
            row.total_gpus = 0
            row.total_mem_mb = 4096

        sched = _make_scheduler(max_workers=2)
        asyncio.run(sched._tick())

        with database.get_session() as session:
            stored_a = session.get(models.Job, job_a.id)
            stored_b = session.get(models.Job, job_b.id)
            cpus_a = set(json.loads(stored_a.assigned_cpus))
            cpus_b = set(json.loads(stored_b.assigned_cpus))

        assert len(launched) == 2
        assert cpus_a.isdisjoint(cpus_b)

    def test_same_tick_gpu_allocations_are_disjoint(self, monkeypatch):
        launched: list[str] = []

        def _fake_start_job(job, resource_pool):
            launched.append(job.id)

        monkeypatch.setattr(scheduler.runner, "start_job", _fake_start_job)

        job_a = _make_queued_job(req_cpus=1, req_gpus=1)
        job_b = _make_queued_job(req_cpus=1, req_gpus=1)
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 2
            row.total_gpus = 2
            row.total_mem_mb = 4096

        sched = _make_scheduler(max_workers=2)
        asyncio.run(sched._tick())

        with database.get_session() as session:
            stored_a = session.get(models.Job, job_a.id)
            stored_b = session.get(models.Job, job_b.id)
            gpus_a = set(json.loads(stored_a.assigned_gpus))
            gpus_b = set(json.loads(stored_b.assigned_gpus))

        assert len(launched) == 2
        assert gpus_a.isdisjoint(gpus_b)

    def test_launched_job_is_marked_running_in_scheduler_transaction(self, monkeypatch):
        launched: list[str] = []

        def _fake_start_job(job, resource_pool):
            launched.append(job.id)

        monkeypatch.setattr(scheduler.runner, "start_job", _fake_start_job)

        job = _make_queued_job(req_cpus=1)
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 1
            row.total_mem_mb = 1024

        sched = _make_scheduler(max_workers=1)
        asyncio.run(sched._tick())

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.RUNNING

        assert launched == [job.id]

    def test_running_reservation_prevents_second_tick_relaunch(self, monkeypatch):
        launched: list[str] = []

        def _fake_start_job(job, resource_pool):
            launched.append(job.id)

        monkeypatch.setattr(scheduler.runner, "start_job", _fake_start_job)

        job = _make_queued_job(req_cpus=1)
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 1
            row.total_mem_mb = 1024

        sched = _make_scheduler(max_workers=1)
        asyncio.run(sched._tick())
        asyncio.run(sched._tick())

        assert launched == [job.id]


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

    def test_aged_job_sorts_above_newer_high_priority_job(self):
        # Job A: low base priority, submitted 24 hours ago.
        job_a = _make_queued_job(priority=20, wait_hours=24.0)
        # Job B: high priority, just submitted.
        job_b = _make_queued_job(priority=80, wait_hours=0.0)

        sched = _make_scheduler(aging_factor=5.0)

        # job_a effective: min(100, 20 + 5*24) = 100.0
        # job_b effective: min(100, 80 + ~0) = 80.0
        # → job_a sorts first despite lower base priority.
        candidates = [job_a, job_b]
        candidates.sort(
            key=lambda j: (-sched._effective_priority(j), j.submitted_at or 0)
        )
        assert candidates[0].id == job_a.id

    def test_base_priority_unchanged_after_sort(self):
        """Sorting by effective priority must not alter the stored base priority."""
        job_a = _make_queued_job(priority=20, wait_hours=24.0)
        job_b = _make_queued_job(priority=80, wait_hours=0.0)

        sched = _make_scheduler(aging_factor=5.0)
        sched._effective_priority(job_a)
        sched._effective_priority(job_b)

        with database.get_session() as session:
            stored_a = session.get(models.Job, job_a.id)
            stored_b = session.get(models.Job, job_b.id)
        assert stored_a.priority == 20
        assert stored_b.priority == 80


# --------------------------------------------------------------------------------------
# EASY Backfill — single reservation guarantee


class TestEasyBackfillSingleReservation:
    """_tick() must honour EASY Backfill's single-reservation guarantee.

    Classical EASY Backfill sets exactly one reservation for the head job
    (the highest-priority blocked job) and allows backfill only within that
    window.  After handling the first blocked head the loop must stop so that
    jobs ineligible under the head's window are never launched.
    """

    def test_breaks_at_first_blocked_head_when_reservation_unknown(self, monkeypatch):
        """
        When the first blocked head has no estimable reservation (running job
        has no walltime), the scheduler must break immediately rather than
        falling through to try backfill for a subsequent blocked head.

        Layout
        ------
        Resources : 6 CPUs total, 2 free (R1 uses 3, R2 uses 1).
        R1        : 3 CPUs, walltime=None  → makes head[0]'s window = None.
        R2        : 1 CPU,  walltime=60s   → after R2, free=3 ≥ head[1]'s need.
        head[0]   : needs 4 CPUs (blocked). Reservation = None → break (fixed).
        head[1]   : needs 3 CPUs (blocked). Reservation ≈ 60s.
        bf        : needs 2 CPUs, walltime=50s (fits head[1]'s window).

        Before the fix: continue past head[0] → backfill bf for head[1].
        After  the fix: break at head[0]      → bf NOT launched.
        """
        launched: list[str] = []

        def _fake_start(job, pool):
            launched.append(job.id)

        monkeypatch.setattr(scheduler.runner, "start_job", _fake_start)

        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 6
            row.used_cpus = 4        # 2 CPUs free
            row.total_gpus = 0
            row.total_mem_mb = 65536

        now = datetime.datetime.now(datetime.timezone.utc)
        r1 = models.Job(
            user="alice", script_path="/tmp/r.sh",
            status=models.JobStatus.RUNNING,
            req_cpus=3, req_gpus=0, req_mem_mb=512,
            walltime_sec=None,       # No walltime → head[0] reservation = None.
            started_at=now,
            assigned_cpus=json.dumps([0, 1, 2]),
            assigned_gpus=json.dumps([]),
        )
        r2 = models.Job(
            user="alice", script_path="/tmp/r.sh",
            status=models.JobStatus.RUNNING,
            req_cpus=1, req_gpus=0, req_mem_mb=512,
            walltime_sec=60,
            started_at=now,
            assigned_cpus=json.dumps([3]),
            assigned_gpus=json.dumps([]),
        )
        with database.get_session() as session:
            session.add(r1)
            session.add(r2)

        # head[0]: 4 CPUs needed, 2 free → blocked.  Window = None.
        head0 = _make_queued_job(priority=90, req_cpus=4)
        # head[1]: 3 CPUs needed, 2 free → blocked.  Window ≈ 60s (via R2).
        head1 = _make_queued_job(priority=70, req_cpus=3)
        # bf: 2 CPUs, walltime=50s < 60*0.9=54s → eligible for head[1]'s window.
        bf = _make_queued_job(priority=60, req_cpus=2, walltime_sec=50)

        sched = _make_scheduler(max_workers=10)
        asyncio.run(sched._tick())

        # The loop must have broken at head[0]; bf must NOT be launched.
        assert bf.id not in launched

    def test_backfill_is_launched_when_head_has_estimable_window(self, monkeypatch):
        """
        When the first blocked head DOES have an estimable reservation, a
        backfill job that fits within the window IS launched.

        Layout
        ------
        Resources : 4 CPUs total, 2 free (R uses 2).
        R         : 2 CPUs, walltime=120s → after R, 4 CPUs free.
        head[0]   : needs 4 CPUs (blocked). Window ≈ 120*0.9 = 108s.
        bf        : needs 2 CPUs (fits NOW), walltime=100s < 108s → eligible.
        """
        launched: list[str] = []

        def _fake_start(job, pool):
            launched.append(job.id)

        monkeypatch.setattr(scheduler.runner, "start_job", _fake_start)

        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4
            row.used_cpus = 2        # 2 CPUs free
            row.total_gpus = 0
            row.total_mem_mb = 65536

        r = models.Job(
            user="alice", script_path="/tmp/r.sh",
            status=models.JobStatus.RUNNING,
            req_cpus=2, req_gpus=0, req_mem_mb=512,
            walltime_sec=120,
            started_at=datetime.datetime.now(datetime.timezone.utc),
            assigned_cpus=json.dumps([0, 1]),
            assigned_gpus=json.dumps([]),
        )
        with database.get_session() as session:
            session.add(r)

        # head[0]: 4 CPUs needed, 2 free → blocked.  Window ≈ 108s.
        head0 = _make_queued_job(priority=90, req_cpus=4)
        # bf: 2 CPUs, walltime=100s < 108s → eligible backfill.
        bf = _make_queued_job(priority=50, req_cpus=2, walltime_sec=100)

        sched = _make_scheduler(max_workers=10)
        asyncio.run(sched._tick())

        assert bf.id in launched
