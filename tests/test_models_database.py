from __future__ import annotations

import datetime

import pytest

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.parser as parser

# --------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def default_directives() -> parser.JobDirectives:
    """
    Return a JobDirectives instance with all fields set to non-default values.

    Parameters
    ----------
    None

    Returns
    -------
    parser.JobDirectives
        A fully populated directives object for use in tests.
    """

    return parser.JobDirectives(
        name="test-job",
        cpus=4,
        gpus=1,
        mem_mb=8192,
        walltime_sec=3600,
        priority=80,
        env_keys=["CUDA_VISIBLE_DEVICES"],
        script_path="/home/user/train.sh",
    )


# --------------------------------------------------------------------------------------
# models.Job — construction tests


class TestJobFromDirectives:
    """Job.from_directives() maps every JobDirectives field correctly."""

    def test_user_is_set(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.user == "alice"

    def test_name_is_set(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.name == "test-job"

    def test_script_path_is_set(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.script_path == "/home/user/train.sh"

    def test_req_cpus(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.req_cpus == 4

    def test_req_gpus(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.req_gpus == 1

    def test_req_mem_mb(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.req_mem_mb == 8192

    def test_walltime_sec(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.walltime_sec == 3600

    def test_priority(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.priority == 80

    def test_initial_status_is_queued(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.status == models.JobStatus.QUEUED

    def test_runtime_fields_are_none(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.pid is None
        assert job.exit_code is None
        assert job.started_at is None
        assert job.finished_at is None
        assert job.assigned_cpus is None
        assert job.assigned_gpus is None

    def test_name_none_when_directive_omitted(self):
        directives = parser.JobDirectives(script_path="/tmp/job.sh")
        job = models.Job.from_directives(directives, user="bob")
        assert job.name is None

    def test_walltime_none_when_directive_omitted(self):
        directives = parser.JobDirectives(script_path="/tmp/job.sh")
        job = models.Job.from_directives(directives, user="bob")
        assert job.walltime_sec is None


# --------------------------------------------------------------------------------------
# models.Job — UUID primary key tests


class TestJobId:
    """Each Job receives a unique UUID string as its primary key."""

    def test_id_is_assigned(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        assert job.id is not None
        assert len(job.id) == 36  # Standard UUID4 string length.

    def test_ids_are_unique(self, default_directives):
        job_a = models.Job.from_directives(default_directives, user="alice")
        job_b = models.Job.from_directives(default_directives, user="alice")
        assert job_a.id != job_b.id


# --------------------------------------------------------------------------------------
# models.JobStatus enum tests


class TestJobStatus:
    """JobStatus values match the documented string literals."""

    def test_queued_value(self):
        assert models.JobStatus.QUEUED == "queued"

    def test_running_value(self):
        assert models.JobStatus.RUNNING == "running"

    def test_done_value(self):
        assert models.JobStatus.DONE == "done"

    def test_failed_value(self):
        assert models.JobStatus.FAILED == "failed"

    def test_cancelled_value(self):
        assert models.JobStatus.CANCELLED == "cancelled"


# --------------------------------------------------------------------------------------
# database.init_db tests


class TestInitDb:
    """init_db() creates tables and returns an engine."""

    def test_returns_engine(self):
        engine = database.get_engine()
        assert engine is not None

    def test_jobs_table_exists(self):
        engine = database.get_engine()
        inspector = sqlalchemy.inspect(engine)
        assert "jobs" in inspector.get_table_names()

    def test_resources_table_exists(self):
        engine = database.get_engine()
        inspector = sqlalchemy.inspect(engine)
        assert "resources" in inspector.get_table_names()

    def test_reinit_with_same_url_is_idempotent(self):
        # Calling init_db() again with the same URL must not raise.
        database.init_db("sqlite:///:memory:")

    def test_reinit_with_different_url_raises(self):
        with pytest.raises(RuntimeError, match="already called"):
            database.init_db("sqlite:////tmp/other.db")


# --------------------------------------------------------------------------------------
# database.get_engine tests


class TestGetEngine:
    """get_engine() raises when init_db() has not been called."""

    def test_raises_before_init(self):
        database.reset_db()
        with pytest.raises(RuntimeError, match="not been initialised"):
            database.get_engine()


# --------------------------------------------------------------------------------------
# database.get_session — CRUD tests


class TestGetSession:
    """get_session() provides a usable transactional session."""

    def test_add_and_retrieve_job(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        with database.get_session() as session:
            session.add(job)

        with database.get_session() as session:
            retrieved = session.get(models.Job, job.id)
            assert retrieved is not None
            assert retrieved.user == "alice"
            assert retrieved.req_cpus == 4

    def test_update_job_status(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        with database.get_session() as session:
            session.add(job)

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.status = models.JobStatus.RUNNING
            stored.pid = 12345
            stored.started_at = datetime.datetime.now(datetime.timezone.utc)

        with database.get_session() as session:
            updated = session.get(models.Job, job.id)
            assert updated.status == models.JobStatus.RUNNING
            assert updated.pid == 12345
            assert updated.started_at is not None

    def test_rollback_on_exception(self, default_directives):
        job = models.Job.from_directives(default_directives, user="alice")
        with database.get_session() as session:
            session.add(job)

        original_name = job.name

        try:
            with database.get_session() as session:
                stored = session.get(models.Job, job.id)
                stored.name = "modified"
                raise RuntimeError("Simulated error to trigger rollback.")
        except RuntimeError:
            pass

        with database.get_session() as session:
            unchanged = session.get(models.Job, job.id)
            assert unchanged.name == original_name

    def test_multiple_jobs_queryable(self, default_directives):
        jobs = [
            models.Job.from_directives(default_directives, user=f"user{i}")
            for i in range(3)
        ]
        with database.get_session() as session:
            for j in jobs:
                session.add(j)

        with database.get_session() as session:
            results = session.query(models.Job).all()
            assert len(results) == 3

    def test_raises_before_init(self):
        database.reset_db()
        with pytest.raises(RuntimeError, match="not been initialised"):
            with database.get_session() as _session:
                pass


# --------------------------------------------------------------------------------------
# Resource default row tests


class TestDefaultResource:
    """A single Resource row (id=1) is inserted automatically on init_db()."""

    def test_default_resource_row_exists(self):
        with database.get_session() as session:
            resource = session.get(models.Resource, 1)
            assert resource is not None

    def test_default_resource_values(self):
        with database.get_session() as session:
            resource = session.get(models.Resource, 1)
            assert resource.total_cpus >= 1
            assert resource.total_gpus >= 0
            assert resource.total_mem_mb >= 1

    def test_resource_is_updatable(self):
        with database.get_session() as session:
            resource = session.get(models.Resource, 1)
            resource.total_cpus = 32
            resource.total_gpus = 4
            resource.total_mem_mb = 65536

        with database.get_session() as session:
            updated = session.get(models.Resource, 1)
            assert updated.total_cpus == 32
            assert updated.total_gpus == 4
            assert updated.total_mem_mb == 65536


# --------------------------------------------------------------------------------------
# Import guard for sqlalchemy.inspect used in TestInitDb

# autopep8: off
import sqlalchemy

# autopep8: on
