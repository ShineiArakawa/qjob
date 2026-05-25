from __future__ import annotations

import asyncio
import json

import pytest

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.scheduler as scheduler

# --------------------------------------------------------------------------------------
# Helpers


def _make_job(
    user:         str = "alice",
    req_cpus:     int = 1,
    req_gpus:     int = 0,
    req_mem_mb:   int = 512,
    priority:     int = 50,
    status:       models.JobStatus = models.JobStatus.QUEUED,
) -> models.Job:
    """Return an unsaved Job with the given resource requirements."""

    return models.Job(
        user=user,
        script_path="/tmp/job.sh",
        status=status,
        req_cpus=req_cpus,
        req_gpus=req_gpus,
        req_mem_mb=req_mem_mb,
        priority=priority,
    )


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


# --------------------------------------------------------------------------------------
# ResourcePool — properties


class TestResourcePoolProperties:
    """free_cpus, free_gpus, and free_mem_mb reflect current allocation state."""

    def test_free_cpus_initial(self):
        pool = _make_pool(cpus=8)
        assert pool.free_cpus == 8

    def test_free_gpus_initial(self):
        pool = _make_pool(gpus=2)
        assert pool.free_gpus == 2

    def test_free_mem_mb_initial(self):
        pool = _make_pool(mem_mb=16384)
        assert pool.free_mem_mb == 16384

    def test_free_mem_mb_after_partial_use(self):
        pool = _make_pool(mem_mb=16384)
        pool.used_mem_mb = 4096
        assert pool.free_mem_mb == 12288


# --------------------------------------------------------------------------------------
# ResourcePool.from_resource_row


class TestResourcePoolFromRow:
    """from_resource_row() maps DB columns to pool fields correctly."""

    def test_total_cpus(self):
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 16
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            pool = scheduler.ResourcePool.from_resource_row(row)
        assert pool.total_cpus == 16

    def test_all_cpus_free_initially(self):
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            pool = scheduler.ResourcePool.from_resource_row(row)
        assert pool.free_cpu_ids == [0, 1, 2, 3]

    def test_all_gpus_free_initially(self):
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_gpus = 2
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            pool = scheduler.ResourcePool.from_resource_row(row)
        assert pool.free_gpu_ids == [0, 1]

    def test_configured_gpu_ids_free_initially(self):
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.set_configured_gpu_ids([2, 5])
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            pool = scheduler.ResourcePool.from_resource_row(row)
        assert pool.free_gpu_ids == [2, 5]


# --------------------------------------------------------------------------------------
# ResourcePool.can_fit


class TestCanFit:
    """can_fit() returns True only when all resource dimensions are satisfiable."""

    def test_fits_when_resources_sufficient(self):
        pool = _make_pool(cpus=4, gpus=1, mem_mb=8192)
        job = _make_job(req_cpus=2, req_gpus=1, req_mem_mb=4096)
        assert pool.can_fit(job) is True

    def test_does_not_fit_insufficient_cpus(self):
        pool = _make_pool(cpus=2)
        job = _make_job(req_cpus=4)
        assert pool.can_fit(job) is False

    def test_does_not_fit_insufficient_gpus(self):
        pool = _make_pool(gpus=0)
        job = _make_job(req_gpus=1)
        assert pool.can_fit(job) is False

    def test_does_not_fit_insufficient_memory(self):
        pool = _make_pool(mem_mb=1024)
        job = _make_job(req_mem_mb=2048)
        assert pool.can_fit(job) is False

    def test_fits_exact_match(self):
        pool = _make_pool(cpus=4, gpus=2, mem_mb=8192)
        job = _make_job(req_cpus=4, req_gpus=2, req_mem_mb=8192)
        assert pool.can_fit(job) is True

    def test_does_not_fit_when_pool_exhausted(self):
        pool = _make_pool(cpus=2)
        job = _make_job(req_cpus=2)
        pool.allocate(job)
        assert pool.can_fit(_make_job(req_cpus=1)) is False


# --------------------------------------------------------------------------------------
# ResourcePool.allocate


class TestAllocate:
    """allocate() reserves resources and returns the assigned IDs."""

    def test_returns_correct_cpu_ids(self):
        pool = _make_pool(cpus=4)
        job = _make_job(req_cpus=2)
        cpu_ids, _ = pool.allocate(job)
        assert cpu_ids == [0, 1]

    def test_returns_correct_gpu_ids(self):
        pool = _make_pool(gpus=2)
        job = _make_job(req_gpus=1)
        _, gpu_ids = pool.allocate(job)
        assert gpu_ids == [0]

    def test_reduces_free_cpus(self):
        pool = _make_pool(cpus=4)
        job = _make_job(req_cpus=3)
        pool.allocate(job)
        assert pool.free_cpus == 1

    def test_reduces_free_mem(self):
        pool = _make_pool(mem_mb=8192)
        job = _make_job(req_mem_mb=2048)
        pool.allocate(job)
        assert pool.free_mem_mb == 6144

    def test_sequential_allocations_return_different_ids(self):
        pool = _make_pool(cpus=4)
        job_a = _make_job(req_cpus=2)
        job_b = _make_job(req_cpus=2)
        ids_a, _ = pool.allocate(job_a)
        ids_b, _ = pool.allocate(job_b)
        assert set(ids_a).isdisjoint(set(ids_b))

    def test_raises_when_insufficient_resources(self):
        pool = _make_pool(cpus=1)
        job = _make_job(req_cpus=4)
        with pytest.raises(RuntimeError, match="Not enough resources"):
            pool.allocate(job)


# --------------------------------------------------------------------------------------
# ResourcePool.release


class TestRelease:
    """release() returns resources to the free pool."""

    def test_cpu_ids_restored_after_release(self):
        pool = _make_pool(cpus=4)
        job = _make_job(req_cpus=2)
        cpu_ids, _ = pool.allocate(job)
        job.assigned_cpus = json.dumps(cpu_ids)
        job.assigned_gpus = json.dumps([])
        pool.release(job)
        assert pool.free_cpus == 4

    def test_gpu_ids_restored_after_release(self):
        pool = _make_pool(gpus=2)
        job = _make_job(req_gpus=2)
        _, gpu_ids = pool.allocate(job)
        job.assigned_cpus = json.dumps([])
        job.assigned_gpus = json.dumps(gpu_ids)
        pool.release(job)
        assert pool.free_gpus == 2

    def test_memory_restored_after_release(self):
        pool = _make_pool(mem_mb=8192)
        job = _make_job(req_mem_mb=4096)
        pool.allocate(job)
        job.assigned_cpus = json.dumps([])
        job.assigned_gpus = json.dumps([])
        pool.release(job)
        assert pool.free_mem_mb == 8192

    def test_release_noop_when_assigned_fields_none(self):
        pool = _make_pool(cpus=4)
        job = _make_job(req_cpus=2)
        pool.allocate(job)
        job.assigned_cpus = None
        job.assigned_gpus = None
        free_before = pool.free_cpus
        pool.release(job)           # Must not raise.
        assert pool.free_cpus == free_before

    def test_free_ids_sorted_after_release(self):
        pool = _make_pool(cpus=4)
        job = _make_job(req_cpus=4)
        pool.allocate(job)
        job.assigned_cpus = json.dumps([3, 1, 0, 2])
        job.assigned_gpus = json.dumps([])
        pool.release(job)
        assert pool.free_cpu_ids == [0, 1, 2, 3]


# --------------------------------------------------------------------------------------
# ResourcePool.sync_from_db


class TestSyncFromDb:
    """sync_from_db() updates the pool from the resources table."""

    def test_total_cpus_updated(self):
        pool = _make_pool(cpus=4)
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 16
        with database.get_session() as session:
            pool.sync_from_db(session)
        assert pool.total_cpus == 16

    def test_running_jobs_excluded_from_free_ids(self):
        pool = _make_pool(cpus=4)
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4

        # Add a running job occupying CPUs 0 and 1.
        job = _make_job(req_cpus=2, status=models.JobStatus.RUNNING)
        job.assigned_cpus = json.dumps([0, 1])
        job.assigned_gpus = json.dumps([])
        with database.get_session() as session:
            session.add(job)

        with database.get_session() as session:
            pool.sync_from_db(session)

        assert 0 not in pool.free_cpu_ids
        assert 1 not in pool.free_cpu_ids
        assert pool.free_cpus == 2


# --------------------------------------------------------------------------------------
# Scheduler._fetch_queued ordering


class TestFetchQueuedOrdering:
    """_fetch_queued() returns jobs ordered by priority DESC then submitted_at ASC."""

    def test_higher_priority_first(self):
        low = _make_job(priority=20)
        high = _make_job(priority=80)
        with database.get_session() as session:
            session.add(low)
            session.add(high)

        sched = scheduler.Scheduler()
        with database.get_session() as session:
            result = sched._fetch_queued(session)

        assert result[0].priority > result[1].priority

    def test_same_priority_earlier_submission_first(self):
        import datetime
        early = _make_job(priority=50)
        late = _make_job(priority=50)
        early.submitted_at = datetime.datetime(2025, 1, 1, 0, 0, 0,
                                               tzinfo=datetime.timezone.utc)
        late.submitted_at = datetime.datetime(2025, 1, 2, 0, 0, 0,
                                              tzinfo=datetime.timezone.utc)
        with database.get_session() as session:
            session.add(early)
            session.add(late)

        sched = scheduler.Scheduler()
        with database.get_session() as session:
            result = sched._fetch_queued(session)

        assert result[0].submitted_at < result[1].submitted_at

    def test_only_queued_jobs_returned(self):
        queued = _make_job(status=models.JobStatus.QUEUED)
        running = _make_job(status=models.JobStatus.RUNNING)
        done = _make_job(status=models.JobStatus.DONE)
        with database.get_session() as session:
            session.add(queued)
            session.add(running)
            session.add(done)

        sched = scheduler.Scheduler()
        with database.get_session() as session:
            result = sched._fetch_queued(session)

        assert len(result) == 1
        assert result[0].status == models.JobStatus.QUEUED


# --------------------------------------------------------------------------------------
# Scheduler._recover_interrupted_jobs


class TestCountRunning:
    """_count_running() counts both RUNNING and CANCELLING jobs."""

    def test_counts_running_job(self):
        job = _make_job(status=models.JobStatus.RUNNING)
        with database.get_session() as session:
            session.add(job)
        sched = scheduler.Scheduler()
        with database.get_session() as session:
            assert sched._count_running(session) >= 1

    def test_counts_cancelling_job(self):
        job = _make_job(status=models.JobStatus.CANCELLING)
        with database.get_session() as session:
            session.add(job)
        sched = scheduler.Scheduler()
        with database.get_session() as session:
            assert sched._count_running(session) >= 1

    def test_does_not_count_queued_job(self):
        job = _make_job(status=models.JobStatus.QUEUED)
        with database.get_session() as session:
            session.add(job)
        sched = scheduler.Scheduler()
        with database.get_session() as session:
            count_before = sched._count_running(session)
        # Add a second QUEUED job and confirm count unchanged.
        job2 = _make_job(status=models.JobStatus.QUEUED)
        with database.get_session() as session:
            session.add(job2)
        with database.get_session() as session:
            assert sched._count_running(session) == count_before


# --------------------------------------------------------------------------------------
# Scheduler._recover_interrupted_jobs


class TestRecoverInterruptedJobs:
    """Orphaned RUNNING and CANCELLING jobs are reset to FAILED on scheduler startup."""

    def test_orphaned_jobs_set_to_failed(self):
        orphan = _make_job(status=models.JobStatus.RUNNING)
        orphan.assigned_cpus = json.dumps([0])
        orphan.assigned_gpus = json.dumps([])
        with database.get_session() as session:
            session.add(orphan)

        sched = scheduler.Scheduler()
        sched._recover_interrupted_jobs()

        with database.get_session() as session:
            recovered = session.get(models.Job, orphan.id)
            assert recovered.status == models.JobStatus.FAILED

    def test_orphaned_jobs_release_resources_in_db(self):
        # Set used_cpus=2 to simulate the orphan occupying 2 cores.
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_cpus = 4
            row.used_cpus = 2

        orphan = _make_job(req_cpus=2, status=models.JobStatus.RUNNING)
        orphan.assigned_cpus = json.dumps([0, 1])
        orphan.assigned_gpus = json.dumps([])
        with database.get_session() as session:
            session.add(orphan)

        sched = scheduler.Scheduler()
        sched._recover_interrupted_jobs()

        # used_cpus should be 0 after recovery.
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            assert row.used_cpus == 0

    def test_cancelling_orphan_set_to_failed(self):
        orphan = _make_job(status=models.JobStatus.CANCELLING)
        orphan.assigned_cpus = json.dumps([0])
        orphan.assigned_gpus = json.dumps([])
        with database.get_session() as session:
            session.add(orphan)

        sched = scheduler.Scheduler()
        sched._recover_interrupted_jobs()

        with database.get_session() as session:
            recovered = session.get(models.Job, orphan.id)
            assert recovered.status == models.JobStatus.FAILED

    def test_cancelling_orphan_releases_resources(self):
        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            row.total_gpus = 2
            row.used_gpus = 1

        orphan = _make_job(req_gpus=1, status=models.JobStatus.CANCELLING)
        orphan.assigned_cpus = json.dumps([])
        orphan.assigned_gpus = json.dumps([0])
        with database.get_session() as session:
            session.add(orphan)

        sched = scheduler.Scheduler()
        sched._recover_interrupted_jobs()

        with database.get_session() as session:
            row = session.get(models.Resource, 1)
            assert row.used_gpus == 0


# --------------------------------------------------------------------------------------
# Scheduler.start / stop — integration


class TestSchedulerStartStop:
    """Scheduler starts, runs at least one tick, then stops cleanly."""

    def test_stop_halts_loop(self):
        """stop() causes start() to return within a reasonable time."""

        sched = scheduler.Scheduler(poll_interval=0.05)

        async def _run():
            task = asyncio.create_task(sched.start())
            await asyncio.sleep(0.15)   # Allow a couple of ticks.
            sched.stop()
            await asyncio.wait_for(task, timeout=1.0)

        asyncio.run(_run())
        assert sched._running is False
