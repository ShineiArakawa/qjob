from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import signal
import threading

import sqlalchemy
import sqlalchemy.orm

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.runner as runner

# --------------------------------------------------------------------------------------
# Module logger

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants

_DEFAULT_POLL_INTERVAL:    float = 2.0   # Seconds between scheduler ticks.
_DEFAULT_MAX_WORKERS:      int = 64     # Upper bound on concurrently running jobs.
_DEFAULT_AGING_FACTOR:     float = 5.0    # Priority points added per hour of waiting.
_BACKFILL_WALLTIME_MARGIN: float = 0.9    # Backfill job's walltime / reserve window.

# PostgreSQL session-level advisory lock key (must be a unique bigint per application).
# Derived from ascii("qjob") — prevents two scheduler processes from running at once.
_ADVISORY_LOCK_KEY: int = 0x716A6F62


# --------------------------------------------------------------------------------------
# Resource pool


@dataclasses.dataclass
class ResourcePool:
    """
    Tracks which CPU cores and GPUs are currently free.

    The pool is initialised from the ``resources`` table and updated in
    memory as jobs start and finish.  It is intentionally kept separate
    from the DB so that allocation decisions can be made without a round
    trip to the database.

    Attributes
    ----------
    total_cpus : int
        Total number of CPU cores available on this server.
    total_gpus : int
        Total number of GPU devices available on this server.
    total_mem_mb : int
        Total memory available in megabytes.
    free_cpu_ids : list[int]
        Indices of currently unallocated CPU cores.
    free_gpu_ids : list[int]
        Indices of currently unallocated GPU devices.
    used_mem_mb : int
        Memory currently allocated to running jobs.
    """

    total_cpus:   int
    total_gpus:   int
    total_mem_mb: int
    free_cpu_ids: list[int] = dataclasses.field(default_factory=list)
    free_gpu_ids: list[int] = dataclasses.field(default_factory=list)
    used_mem_mb:  int = 0

    # -- Properties --------------------------------------------------------------------

    @property
    def free_mem_mb(self) -> int:
        """Remaining unallocated memory in megabytes."""
        return self.total_mem_mb - self.used_mem_mb

    @property
    def free_cpus(self) -> int:
        """Number of unallocated CPU cores."""
        return len(self.free_cpu_ids)

    @property
    def free_gpus(self) -> int:
        """Number of unallocated GPU devices."""
        return len(self.free_gpu_ids)

    # -- Factory -----------------------------------------------------------------------

    @classmethod
    def from_resource_row(cls, row: models.Resource) -> ResourcePool:
        """
        Build a ResourcePool from the ``resources`` DB row.

        All cores and GPUs are marked as free initially.

        Parameters
        ----------
        row : models.Resource
            The single resource configuration row (id=1).

        Returns
        -------
        ResourcePool
            A fully initialised pool with all resources free.
        """

        return cls(
            total_cpus=row.total_cpus,
            total_gpus=row.total_gpus,
            total_mem_mb=row.total_mem_mb,
            free_cpu_ids=list(range(row.total_cpus)),
            free_gpu_ids=list(range(row.total_gpus)),
        )

    # -- Allocation --------------------------------------------------------------------

    def can_fit(self, job: models.Job) -> bool:
        """
        Return True if the pool has enough free resources for *job*.

        Parameters
        ----------
        job : models.Job
            The job whose resource requests are checked.

        Returns
        -------
        bool
            True when all of CPUs, GPUs, and memory can be satisfied.
        """

        return (
            job.req_cpus <= self.free_cpus
            and job.req_gpus <= self.free_gpus
            and job.req_mem_mb <= self.free_mem_mb
        )

    def allocate(self, job: models.Job) -> tuple[list[int], list[int]]:
        """
        Reserve resources for *job* and return the assigned IDs.

        Parameters
        ----------
        job : models.Job
            The job to allocate resources for.  ``can_fit()`` must be True
            before calling this method.

        Returns
        -------
        tuple[list[int], list[int]]
            A pair ``(cpu_ids, gpu_ids)`` of the assigned resource indices.

        Raises
        ------
        RuntimeError
            If the pool does not have enough resources (caller should have
            called ``can_fit()`` first).
        """

        if not self.can_fit(job):
            raise RuntimeError(
                f"Not enough resources to allocate job {job.id!r}."
            )

        cpu_ids = [self.free_cpu_ids.pop(0) for _ in range(job.req_cpus)]
        gpu_ids = [self.free_gpu_ids.pop(0) for _ in range(job.req_gpus)]
        self.used_mem_mb += job.req_mem_mb

        return cpu_ids, gpu_ids

    def release(self, job: models.Job) -> None:
        """
        Return the resources held by *job* back to the free pool.

        Reads the assigned CPU/GPU IDs from the job's ``assigned_cpus`` and
        ``assigned_gpus`` JSON fields.  Safe to call even if those fields are
        None (no-op in that case).

        Parameters
        ----------
        job : models.Job
            A job whose status has transitioned to DONE, FAILED, or CANCELLED.

        Returns
        -------
        None
        """

        if job.assigned_cpus:
            self.free_cpu_ids.extend(json.loads(job.assigned_cpus))
            self.free_cpu_ids.sort()

        if job.assigned_gpus:
            self.free_gpu_ids.extend(json.loads(job.assigned_gpus))
            self.free_gpu_ids.sort()

        self.used_mem_mb = max(0, self.used_mem_mb - job.req_mem_mb)

    def sync_from_db(self, session: sqlalchemy.orm.Session) -> None:
        """
        Reload total resource limits from the DB and adjust free counts.

        Called when an administrator updates the ``resources`` table at
        runtime so that the in-memory pool reflects the new limits without
        restarting the scheduler.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            An open DB session used to read the resources row.

        Returns
        -------
        None
        """

        row: models.Resource | None = session.get(models.Resource, 1)
        if row is None:
            return

        # Recalculate free IDs based on which ones are currently occupied.
        occupied_cpus: set[int] = set()
        occupied_gpus: set[int] = set()

        running_jobs = (
            session.query(models.Job)
            .filter(models.Job.status.in_([
                models.JobStatus.RUNNING,
                models.JobStatus.CANCELLING,
            ]))
            .all()
        )
        for rj in running_jobs:
            if rj.assigned_cpus:
                occupied_cpus.update(json.loads(rj.assigned_cpus))
            if rj.assigned_gpus:
                occupied_gpus.update(json.loads(rj.assigned_gpus))

        self.total_cpus = row.total_cpus
        self.total_gpus = row.total_gpus
        self.total_mem_mb = row.total_mem_mb
        self.free_cpu_ids = sorted(
            set(range(row.total_cpus)) - occupied_cpus
        )
        self.free_gpu_ids = sorted(
            set(range(row.total_gpus)) - occupied_gpus
        )


# --------------------------------------------------------------------------------------
# Scheduler


class Scheduler:
    """
    Polls the database and dispatches queued jobs to the runner.

    The scheduler runs as an ``asyncio`` coroutine.  It performs one
    *tick* every ``poll_interval`` seconds: it queries for QUEUED jobs
    ordered by priority (descending) then submission time (ascending),
    and starts as many as the current resource pool allows — FIFO within
    each priority band.

    Parameters
    ----------
    poll_interval : float
        Seconds to wait between scheduling ticks.
    max_workers : int
        Maximum number of concurrently running jobs regardless of available
        resources.
    """

    def __init__(
        self,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        aging_factor: float = _DEFAULT_AGING_FACTOR,
        install_signal_handlers: bool = False
    ) -> None:
        self._poll_interval = poll_interval
        self._max_workers = max_workers
        self._aging_factor = aging_factor
        self._install_signal_handlers_enabled = install_signal_handlers
        self._running: bool = False
        self._lock_conn: sqlalchemy.engine.Connection | None = None

    # -- Public API --------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initialise the resource pool and enter the scheduling loop.

        Blocks until ``stop()`` is called or the process receives SIGTERM /
        SIGINT.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self._ensure_resource_row()
        self._acquire_lock()
        self._running = True
        if self._install_signal_handlers_enabled:
            self._install_signal_handlers()

        logger.info(
            "Scheduler started — poll_interval=%.1fs  max_workers=%d",
            self._poll_interval,
            self._max_workers,
        )

        # Recover any jobs that were RUNNING when the process last stopped.
        self._recover_interrupted_jobs()

        try:
            while self._running:
                try:
                    await self._tick()
                except Exception:
                    logger.exception("Unhandled error in scheduler tick.")
                await asyncio.sleep(self._poll_interval)

            await runner.shutdown_active_jobs()
            try:
                with database.get_session_for_update() as session:
                    self._release_finished_jobs(session)
            except Exception:
                logger.exception("Failed to release resources during scheduler shutdown.")
        finally:
            self._release_lock()

        logger.info("Scheduler stopped.")

    def stop(self) -> None:
        """
        Request a graceful shutdown after the current tick completes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        logger.info("Scheduler stop requested.")
        self._running = False

    # -- Tick --------------------------------------------------------------------------

    async def _tick(self) -> None:
        """
        Execute one scheduling cycle using EASY Backfill + priority ageing.

        All resource allocation decisions are made inside a single
        ``get_session_for_update()`` context so that concurrent scheduler
        processes (or uvicorn workers) cannot double-allocate resources.

        Steps
        -----
        1. Lock the resources row and release resources held by finished jobs.
        2. Sort the queue by effective priority DESC, submitted_at ASC.
           Effective priority is computed in-memory (base priority + ageing);
           the DB value is never mutated.
        3. Try to run the head job; fall back to EASY Backfill when blocked.
        4. Stop at the first blocked head (single reservation guarantee).
        5. Repeat until the worker slot limit is reached.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with database.get_session_for_update() as session:
            running_count = self._count_running(session)
            slots = self._max_workers - running_count
            if slots <= 0:
                return

            # Release resources for finished jobs and update DB counts.
            self._release_finished_jobs(session)

            candidates = self._fetch_queued(session)
            if not candidates:
                return

            candidates.sort(
                key=lambda j: (-self._effective_priority(j), j.submitted_at or 0)
            )

            dispatched = 0
            launched_ids: set[str] = set()
            reserved_cpu_ids: set[int] = set()
            reserved_gpu_ids: set[int] = set()

            for head_idx, head in enumerate(candidates):
                if dispatched >= slots:
                    break
                if head.id in launched_ids:
                    continue

                resource_row = session.get(models.Resource, 1)
                if resource_row is None:
                    break

                if resource_row.can_fit(head):
                    cpu_ids, gpu_ids = self._allocate_db(
                        session,
                        resource_row,
                        head,
                        reserved_cpu_ids=reserved_cpu_ids,
                        reserved_gpu_ids=reserved_gpu_ids,
                    )
                    self._launch(session, head, cpu_ids, gpu_ids)
                    launched_ids.add(head.id)
                    reserved_cpu_ids.update(cpu_ids)
                    reserved_gpu_ids.update(gpu_ids)
                    dispatched += 1
                else:
                    reservation_sec = self._estimate_reservation_window(session, head)
                    if reservation_sec is not None:
                        backfill = self._find_backfill_jobs(
                            candidates=candidates[head_idx + 1:],
                            launched_ids=launched_ids,
                            reservation_sec=reservation_sec,
                            slots_remaining=slots - dispatched,
                        )
                        for bf_job in backfill:
                            resource_row = session.get(models.Resource, 1)
                            if resource_row is None:
                                break
                            if resource_row.can_fit(bf_job):
                                cpu_ids, gpu_ids = self._allocate_db(
                                    session,
                                    resource_row,
                                    bf_job,
                                    reserved_cpu_ids=reserved_cpu_ids,
                                    reserved_gpu_ids=reserved_gpu_ids,
                                )
                                self._launch(session, bf_job, cpu_ids, gpu_ids)
                                launched_ids.add(bf_job.id)
                                reserved_cpu_ids.update(cpu_ids)
                                reserved_gpu_ids.update(gpu_ids)
                                dispatched += 1
                                logger.info(
                                    "Backfilled job %s (walltime=%ss, window=%ss).",
                                    bf_job.id, bf_job.walltime_sec, reservation_sec,
                                )
                    break  # EASY Backfill: stop at the first blocked head

            if dispatched:
                logger.debug("Tick dispatched %d job(s).", dispatched)

    # -- Helpers -----------------------------------------------------------------------

    def _count_running(self, session: sqlalchemy.orm.Session) -> int:
        """Return the number of jobs currently occupying a worker slot."""

        return (
            session.query(models.Job)
            .filter(models.Job.status.in_([
                models.JobStatus.RUNNING,
                models.JobStatus.CANCELLING,
            ]))
            .count()
        )

    def _release_finished_jobs(self, session: sqlalchemy.orm.Session) -> None:
        """
        Find jobs that have transitioned to a terminal state since the last
        tick and return their resources to the DB resource row.

        This replaces the former in-memory ``ResourcePool.release()`` call.
        The resources row is already locked by ``get_session_for_update()``
        when this method is called from ``_tick()``.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            An open DB session with the resources row locked.

        Returns
        -------
        None
        """

        terminal = (
            session.query(models.Job)
            .filter(
                models.Job.status.in_([
                    models.JobStatus.DONE,
                    models.JobStatus.FAILED,
                    models.JobStatus.CANCELLED,
                ]),
                models.Job.assigned_cpus.isnot(None),
            )
            .all()
        )
        if not terminal:
            return

        resource_row = session.get(models.Resource, 1)
        if resource_row is None:
            return

        for job in terminal:
            cpu_ids: list[int] = json.loads(job.assigned_cpus) if job.assigned_cpus else []
            gpu_ids: list[int] = json.loads(job.assigned_gpus) if job.assigned_gpus else []
            resource_row.used_cpus = max(0, resource_row.used_cpus - len(cpu_ids))
            resource_row.used_gpus = max(0, resource_row.used_gpus - len(gpu_ids))
            resource_row.used_mem_mb = max(0, resource_row.used_mem_mb - job.req_mem_mb)
            # Clear assigned fields so we don't release the same resources twice.
            job.assigned_cpus = None
            job.assigned_gpus = None

    def _fetch_queued(
        self, session: sqlalchemy.orm.Session
    ) -> list[models.Job]:
        """
        Return QUEUED jobs ordered by priority DESC then submitted_at ASC.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            An open DB session.

        Returns
        -------
        list[models.Job]
            Candidate jobs in scheduling order.
        """

        return (
            session.query(models.Job)
            .filter(models.Job.status == models.JobStatus.QUEUED)
            .order_by(
                models.Job.priority.desc(),
                models.Job.submitted_at.asc(),
            )
            .all()
        )

    def _effective_priority(self, job: models.Job) -> float:
        """
        Return the effective priority of *job* after ageing.

        Ageing is computed in-memory from the job's base priority (the value
        stored in the DB) and its total waiting time.  The result is never
        written back to the database so the base priority is preserved across
        ticks and the effective value always reflects the true elapsed wait.

        Parameters
        ----------
        job : models.Job
            The queued job.

        Returns
        -------
        float
            Effective priority capped at 100.0.
        """

        import datetime

        if job.submitted_at is None:
            return float(job.priority)

        now = datetime.datetime.now(datetime.timezone.utc)
        submitted = job.submitted_at
        if submitted.tzinfo is None:
            submitted = submitted.replace(tzinfo=datetime.timezone.utc)

        wait_hours = (now - submitted).total_seconds() / 3600.0
        return min(100.0, job.priority + self._aging_factor * wait_hours)

    def _estimate_reservation_window(
        self,
        session: sqlalchemy.orm.Session,
        head:    models.Job,
    ) -> float | None:
        """
        Estimate the number of seconds until *head* can start.

        Looks at all currently RUNNING jobs and finds the earliest time at
        which enough resources will be free to satisfy ``head``'s requirements,
        assuming every running job finishes exactly at its walltime.

        Jobs without a walltime are ignored — their finish time cannot be
        predicted.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            Open DB session.
        head : models.Job
            The head job that is currently blocked.

        Returns
        -------
        float | None
            Estimated seconds until head can start, or ``None`` if the window
            cannot be determined (e.g. all blocking jobs lack walltimes).
        """

        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)

        running_jobs = (
            session.query(models.Job)
            .filter(models.Job.status == models.JobStatus.RUNNING)
            .all()
        )

        # Build a list of (finish_time_sec_from_now, cpus, gpus, mem_mb)
        # for every running job that has a walltime.
        finish_events: list[tuple[float, int, int, int]] = []
        for rj in running_jobs:
            if rj.walltime_sec is None or rj.started_at is None:
                continue
            started = rj.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=datetime.timezone.utc)
            remaining = rj.walltime_sec - (now - started).total_seconds()
            remaining = max(0.0, remaining)
            cpu_ids: list[int] = json.loads(rj.assigned_cpus) if rj.assigned_cpus else []
            gpu_ids: list[int] = json.loads(rj.assigned_gpus) if rj.assigned_gpus else []
            finish_events.append((remaining, len(cpu_ids), len(gpu_ids), rj.req_mem_mb))

        if not finish_events:
            return None

        # Simulate resource release in chronological order and find the
        # earliest point at which head's requirements can be satisfied.
        finish_events.sort(key=lambda e: e[0])

        resource_row = session.get(models.Resource, 1)
        sim_free_cpus = resource_row.free_cpus if resource_row else 0
        sim_free_gpus = resource_row.free_gpus if resource_row else 0
        sim_free_mem = resource_row.free_mem_mb if resource_row else 0

        for finish_sec, cpus, gpus, mem in finish_events:
            sim_free_cpus += cpus
            sim_free_gpus += gpus
            sim_free_mem += mem
            if (
                head.req_cpus <= sim_free_cpus
                and head.req_gpus <= sim_free_gpus
                and head.req_mem_mb <= sim_free_mem
            ):
                return finish_sec

        return None

    def _find_backfill_jobs(
        self,
        candidates:      list[models.Job],
        launched_ids:    set[str],
        reservation_sec: float,
        slots_remaining: int,
    ) -> list[models.Job]:
        """
        Return jobs from *candidates* eligible for backfilling.

        A job is eligible when:
        - It has a walltime set (jobs without walltime are excluded).
        - Its walltime fits within ``reservation_sec * _BACKFILL_WALLTIME_MARGIN``
          so it will finish before the head job's resources are needed.
        - It has not already been launched this tick.

        Parameters
        ----------
        candidates : list[models.Job]
            Remaining queued jobs after the blocked head job.
        launched_ids : set[str]
            IDs already dispatched this tick (to avoid double-launch).
        reservation_sec : float
            Estimated seconds until the head job's resources are available.
        slots_remaining : int
            Maximum number of additional jobs that can be started this tick.

        Returns
        -------
        list[models.Job]
            Backfill candidates in queue order, limited to *slots_remaining*.
        """

        window = reservation_sec * _BACKFILL_WALLTIME_MARGIN
        result: list[models.Job] = []

        for job in candidates:
            if len(result) >= slots_remaining:
                break
            if job.id in launched_ids:
                continue
            if job.walltime_sec is None:
                # Cannot guarantee this job finishes before the window closes.
                continue
            if job.walltime_sec <= window:
                result.append(job)

        return result

    def _allocate_db(
        self,
        session:          sqlalchemy.orm.Session,
        resource_row:     models.Resource,
        job:              models.Job,
        reserved_cpu_ids: set[int] | None = None,
        reserved_gpu_ids: set[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        """
        Assign CPU and GPU IDs to *job* and update the DB resource counts.

        CPU IDs are chosen from the set that are not currently in use by any
        RUNNING job.  The resource row's ``used_*`` counters are incremented
        atomically within the locked transaction started by
        ``get_session_for_update()``.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            Open DB session (resources row is already locked).
        resource_row : models.Resource
            The locked resource row to update.
        job : models.Job
            The job to allocate resources for.
        reserved_cpu_ids : set[int] | None
            CPU IDs already assigned earlier in the current scheduler tick.
        reserved_gpu_ids : set[int] | None
            GPU IDs already assigned earlier in the current scheduler tick.

        Returns
        -------
        tuple[list[int], list[int]]
            ``(cpu_ids, gpu_ids)`` of the assigned resource indices.
        """

        # Determine which CPU/GPU IDs are already in use, including jobs
        # launched earlier in this tick that are not RUNNING in the DB yet.
        occupied_cpus: set[int] = set(reserved_cpu_ids or set())
        occupied_gpus: set[int] = set(reserved_gpu_ids or set())
        running_jobs = (
            session.query(models.Job)
            .filter(models.Job.status.in_([
                models.JobStatus.RUNNING,
                models.JobStatus.CANCELLING,
            ]))
            .all()
        )
        for rj in running_jobs:
            if rj.assigned_cpus:
                occupied_cpus.update(json.loads(rj.assigned_cpus))
            if rj.assigned_gpus:
                occupied_gpus.update(json.loads(rj.assigned_gpus))

        free_cpus = [
            i for i in range(resource_row.total_cpus) if i not in occupied_cpus
        ]
        free_gpus = [
            i for i in range(resource_row.total_gpus) if i not in occupied_gpus
        ]

        cpu_ids = free_cpus[:job.req_cpus]
        gpu_ids = free_gpus[:job.req_gpus]

        # Update counters atomically.
        resource_row.used_cpus += len(cpu_ids)
        resource_row.used_gpus += len(gpu_ids)
        resource_row.used_mem_mb += job.req_mem_mb

        return cpu_ids, gpu_ids

    def _launch(
        self,
        session:  sqlalchemy.orm.Session,
        job:      models.Job,
        cpu_ids:  list[int],
        gpu_ids:  list[int],
    ) -> None:
        """
        Persist the resource assignment, reserve the job by moving it out of
        QUEUED, and hand it off to the runner.

        When the runner fails to start the process the DB resource counters
        are decremented so the allocation is rolled back.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            An open DB session used to persist the assignment.
        job : models.Job
            The job to launch.
        cpu_ids : list[int]
            CPU core indices returned by ``_allocate_db()``.
        gpu_ids : list[int]
            GPU device indices returned by ``_allocate_db()``.

        Returns
        -------
        None
        """

        job.assigned_cpus = json.dumps(cpu_ids)
        job.assigned_gpus = json.dumps(gpu_ids)
        job.status = models.JobStatus.RUNNING

        try:
            runner.start_job(job, None)
            logger.info(
                "Launched job %s (user=%s name=%s cpus=%s gpus=%s).",
                job.id, job.user, job.name, cpu_ids, gpu_ids,
            )
        except Exception:
            logger.exception("Failed to launch job %s.", job.id)
            job.status = models.JobStatus.FAILED
            job.assigned_cpus = None
            job.assigned_gpus = None
            # Roll back the resource counters.
            resource_row = session.get(models.Resource, 1)
            if resource_row is not None:
                resource_row.used_cpus = max(0, resource_row.used_cpus - len(cpu_ids))
                resource_row.used_gpus = max(0, resource_row.used_gpus - len(gpu_ids))
                resource_row.used_mem_mb = max(0, resource_row.used_mem_mb - job.req_mem_mb)

    # -- Startup helpers ---------------------------------------------------------------

    @staticmethod
    def _ensure_resource_row() -> None:
        """
        Verify that the resources row (id=1) exists in the DB.

        Raises
        ------
        RuntimeError
            If no resource row is found.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with database.get_session() as session:
            row: models.Resource | None = session.get(models.Resource, 1)
            if row is None:
                raise RuntimeError(
                    "No resource row found. Run 'alembic upgrade head' first."
                )

    def _recover_interrupted_jobs(self) -> None:
        """
        Mark any RUNNING jobs left over from a previous crash as FAILED and
        return their resources to the DB resource row.

        When the scheduler process exits unexpectedly jobs that were RUNNING
        in the DB are orphaned — their subprocesses are gone but the status
        still shows RUNNING.  This method resets them to FAILED and
        decrements ``resources.used_*`` so the counts are consistent.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with database.get_session_for_update() as session:
            orphans = (
                session.query(models.Job)
                .filter(models.Job.status.in_([
                    models.JobStatus.RUNNING,
                    models.JobStatus.CANCELLING,
                ]))
                .all()
            )
            if not orphans:
                return

            resource_row = session.get(models.Resource, 1)

            for job in orphans:
                logger.warning(
                    "Recovering orphaned job %s (user=%s name=%s) -> FAILED.",
                    job.id, job.user, job.name,
                )
                if resource_row is not None:
                    cpu_ids = json.loads(job.assigned_cpus) if job.assigned_cpus else []
                    gpu_ids = json.loads(job.assigned_gpus) if job.assigned_gpus else []
                    resource_row.used_cpus = max(0, resource_row.used_cpus - len(cpu_ids))
                    resource_row.used_gpus = max(0, resource_row.used_gpus - len(gpu_ids))
                    resource_row.used_mem_mb = max(0, resource_row.used_mem_mb - job.req_mem_mb)
                job.status = models.JobStatus.FAILED
                job.assigned_cpus = None
                job.assigned_gpus = None

    def _acquire_lock(self) -> None:
        """
        Acquire a PostgreSQL session-level advisory lock.

        The lock is held for the lifetime of ``_lock_conn`` and is released
        automatically when that connection closes — including on process crash.

        Raises
        ------
        RuntimeError
            If another scheduler process already holds the lock.
        """

        engine = database.get_engine()
        conn = engine.connect()
        try:
            acquired: bool = conn.execute(
                sqlalchemy.text(f"SELECT pg_try_advisory_lock({_ADVISORY_LOCK_KEY})")
            ).scalar()
        except Exception:
            conn.close()
            raise

        if not acquired:
            conn.close()
            raise RuntimeError(
                "Another scheduler process is already running "
                "(pg_try_advisory_lock failed). "
                "Only one scheduler instance may run at a time."
            )

        self._lock_conn = conn
        logger.info("Acquired scheduler advisory lock.")

    def _release_lock(self) -> None:
        """Close the lock-holding connection, releasing the advisory lock."""

        if self._lock_conn is not None:
            try:
                self._lock_conn.close()
            except Exception:
                logger.debug("Error closing advisory lock connection.", exc_info=True)
            finally:
                self._lock_conn = None
            logger.info("Released scheduler advisory lock.")

    def _install_signal_handlers(self) -> None:
        """
        Register SIGTERM and SIGINT handlers to trigger a graceful shutdown.

        Signal handlers can only be registered from the main thread.
        When running inside a test or a worker thread this method silently
        skips registration instead of raising RuntimeError.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if threading.current_thread() is not threading.main_thread():
            logger.debug(
                "Skipping signal handler registration: not running in main thread."
            )
            return

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop)
