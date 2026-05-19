from __future__ import annotations

import asyncio
import datetime
import getpass
import json
import os
import pathlib
import textwrap

import pytest

import qjob.core.database as database
import qjob.core.models as models
import qjob.core.runner as runner
import qjob.core.scheduler as scheduler
from tests.conftest import as_utc

# Use the real OS user so preexec_fn can resolve it via pwd.getpwnam().
_REAL_USER = getpass.getuser()

# --------------------------------------------------------------------------------------
# Helpers


def _make_job(
    req_cpus:   int = 1,
    req_gpus:   int = 0,
    req_mem_mb: int = 512,
    walltime_sec: int | None = None,
    script_path:  str = "",
) -> models.Job:
    """Return a persisted QUEUED job."""

    job = models.Job(
        user=_REAL_USER,
        script_path=script_path,
        status=models.JobStatus.QUEUED,
        req_cpus=req_cpus,
        req_gpus=req_gpus,
        req_mem_mb=req_mem_mb,
        walltime_sec=walltime_sec,
        assigned_cpus=json.dumps(list(range(req_cpus))),
        assigned_gpus=json.dumps(list(range(req_gpus))),
    )
    with database.get_session() as session:
        session.add(job)
    return job


def _make_script(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    """Write a shell script and make it executable."""

    p = tmp_path / "job.sh"
    p.write_text(textwrap.dedent(content))
    p.chmod(0o755)
    return p


def _make_pool() -> scheduler.ResourcePool:
    """Return a ResourcePool with ample free resources."""

    return scheduler.ResourcePool(
        total_cpus=8,
        total_gpus=0,
        total_mem_mb=16384,
        free_cpu_ids=list(range(8)),
        free_gpu_ids=[],
    )


# --------------------------------------------------------------------------------------
# _build_env


class TestBuildEnv:
    """_build_env() constructs the correct environment mapping."""

    def test_cuda_visible_devices_set_from_gpus(self):
        job = _make_job(req_gpus=0)
        job.assigned_gpus = json.dumps([0, 1])
        env = runner._build_env(job)
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_cuda_visible_devices_empty_when_no_gpus(self):
        job = _make_job(req_gpus=0)
        job.assigned_gpus = json.dumps([])
        env = runner._build_env(job)
        assert env["CUDA_VISIBLE_DEVICES"] == ""

    def test_qjob_job_id_set(self):
        job = _make_job()
        env = runner._build_env(job)
        assert env["QJOB_JOB_ID"] == job.id

    def test_qjob_user_set(self):
        job = _make_job()
        env = runner._build_env(job)
        assert env["QJOB_USER"] == _REAL_USER

    def test_inherits_existing_env(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_VAR", "hello")
        job = _make_job()
        env = runner._build_env(job)
        assert env.get("MY_CUSTOM_VAR") == "hello"


# --------------------------------------------------------------------------------------
# _log_paths


class TestLogPaths:
    """_log_paths() creates predictable per-job log filenames."""

    def test_stdout_log_is_next_to_script(self, tmp_path):
        script = tmp_path / "job.sh"
        job = _make_job(script_path=str(script))
        stdout_path, _ = runner._log_paths(job)
        assert stdout_path == tmp_path / f"job.sh.{job.id}.stdout.log"

    def test_stderr_log_is_next_to_script(self, tmp_path):
        script = tmp_path / "job.sh"
        job = _make_job(script_path=str(script))
        _, stderr_path = runner._log_paths(job)
        assert stderr_path == tmp_path / f"job.sh.{job.id}.stderr.log"


# --------------------------------------------------------------------------------------
# _mark_running / _mark_finished / _mark_failed


class TestDbTransitions:
    """DB helper functions update job state correctly."""

    def test_mark_running_sets_status(self, tmp_path):
        job = _make_job(script_path="/tmp/x.sh")
        runner._mark_running(job, pid=1234,
                             stdout_path=tmp_path / "out.log",
                             stderr_path=tmp_path / "err.log")
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.RUNNING

    def test_mark_running_sets_pid(self, tmp_path):
        job = _make_job()
        runner._mark_running(job, pid=9999,
                             stdout_path=tmp_path / "out.log",
                             stderr_path=tmp_path / "err.log")
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.pid == 9999

    def test_mark_running_sets_log_paths(self, tmp_path):
        job = _make_job()
        runner._mark_running(job, pid=1,
                             stdout_path=tmp_path / "out.log",
                             stderr_path=tmp_path / "err.log")
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.log_stdout == str(tmp_path / "out.log")
            assert stored.log_stderr == str(tmp_path / "err.log")

    def test_mark_running_sets_started_at(self, tmp_path):
        job = _make_job()
        before = datetime.datetime.now(datetime.timezone.utc)
        runner._mark_running(job, pid=1,
                             stdout_path=tmp_path / "out.log",
                             stderr_path=tmp_path / "err.log")
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert as_utc(stored.started_at) >= before

    def test_mark_finished_done_on_exit_0(self):
        job = _make_job()
        pool = _make_pool()
        runner._mark_finished(job, exit_code=0, resource_pool=pool)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.DONE
            assert stored.exit_code == 0

    def test_mark_finished_failed_on_nonzero_exit(self):
        job = _make_job()
        pool = _make_pool()
        runner._mark_finished(job, exit_code=1, resource_pool=pool)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.FAILED

    def test_mark_finished_sets_finished_at(self):
        job = _make_job()
        pool = _make_pool()
        before = datetime.datetime.now(datetime.timezone.utc)
        runner._mark_finished(job, exit_code=0, resource_pool=pool)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert as_utc(stored.finished_at) >= before

    def test_mark_finished_releases_pool_resources(self):
        pool = _make_pool()
        job = _make_job(req_cpus=2)
        job.assigned_cpus = json.dumps([0, 1])
        job.assigned_gpus = json.dumps([])
        pool.free_cpu_ids = [2, 3, 4, 5, 6, 7]   # 0,1 simulated as occupied
        runner._mark_finished(job, exit_code=0, resource_pool=pool)
        assert pool.free_cpus == 8

    def test_mark_finished_cancelling_sets_cancelled(self):
        job = _make_job()
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.status = models.JobStatus.CANCELLING
        pool = _make_pool()
        runner._mark_finished(job, exit_code=-15, resource_pool=pool)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.CANCELLED
            assert stored.exit_code == -15
            assert stored.finished_at is not None

    def test_mark_finished_cancelled_preserves_cancelled(self):
        job = _make_job()
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.status = models.JobStatus.CANCELLED
        pool = _make_pool()
        runner._mark_finished(job, exit_code=-9, resource_pool=pool)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.CANCELLED

    def test_mark_failed_sets_status(self):
        job = _make_job()
        pool = _make_pool()
        runner._mark_failed(job, resource_pool=pool)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.FAILED

    def test_mark_failed_releases_pool_resources(self):
        pool = _make_pool()
        job = _make_job(req_cpus=2)
        job.assigned_cpus = json.dumps([0, 1])
        job.assigned_gpus = json.dumps([])
        pool.free_cpu_ids = [2, 3, 4, 5, 6, 7]
        runner._mark_failed(job, resource_pool=pool)
        assert pool.free_cpus == 8


# --------------------------------------------------------------------------------------
# End-to-end subprocess tests
#
# These tests actually spawn real subprocesses, so they are skipped when the
# environment variable QJOB_SKIP_SUBPROCESS_TESTS is set to any non-empty value.

_skip_subprocess = pytest.mark.skipif(
    os.environ.get("QJOB_SKIP_SUBPROCESS_TESTS", "") != "",
    reason="Subprocess tests skipped (QJOB_SKIP_SUBPROCESS_TESTS is set).",
)


class TestRunJobEndToEnd:
    """_run_job() drives the full lifecycle of a real subprocess."""

    @_skip_subprocess
    def test_successful_job_becomes_done(self, tmp_path, monkeypatch):
        script = _make_script(tmp_path, """\
            #!/bin/bash
            echo "hello from job"
            exit 0
        """)
        job = _make_job(script_path=str(script))
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.DONE
            assert stored.exit_code == 0
            assert stored.pid is not None
            assert stored.started_at is not None
            assert stored.finished_at is not None

    @_skip_subprocess
    def test_failing_job_becomes_failed(self, tmp_path, monkeypatch):
        script = _make_script(tmp_path, """\
            #!/bin/bash
            exit 1
        """)
        job = _make_job(script_path=str(script))
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.FAILED
            assert stored.exit_code == 1

    @_skip_subprocess
    def test_stdout_is_captured(self, tmp_path, monkeypatch):
        script = _make_script(tmp_path, """\
            #!/bin/bash
            echo "captured output"
        """)
        job = _make_job(script_path=str(script))
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.log_stdout == str(tmp_path / f"job.sh.{job.id}.stdout.log")
            log_content = pathlib.Path(stored.log_stdout).read_text()
            assert "captured output" in log_content

    @_skip_subprocess
    def test_stderr_is_captured(self, tmp_path, monkeypatch):
        script = _make_script(tmp_path, """\
            #!/bin/bash
            echo "error output" >&2
        """)
        job = _make_job(script_path=str(script))
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.log_stderr == str(tmp_path / f"job.sh.{job.id}.stderr.log")
            log_content = pathlib.Path(stored.log_stderr).read_text()
            assert "error output" in log_content

    @_skip_subprocess
    def test_walltime_exceeded_becomes_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "_SIGTERM_GRACE_SEC", 0.5)
        script = _make_script(tmp_path, """\
            #!/bin/bash
            sleep 60
        """)
        job = _make_job(script_path=str(script), walltime_sec=1)
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.FAILED

    @_skip_subprocess
    def test_resources_released_after_completion(self, tmp_path, monkeypatch):
        script = _make_script(tmp_path, "#!/bin/bash\nexit 0\n")
        job = _make_job(req_cpus=2, script_path=str(script))
        job.assigned_cpus = json.dumps([0, 1])
        job.assigned_gpus = json.dumps([])

        pool = _make_pool()
        pool.free_cpu_ids = [2, 3, 4, 5, 6, 7]  # 0,1 simulated as occupied

        asyncio.run(runner._run_job(job, pool))

        assert pool.free_cpus == 8

    @_skip_subprocess
    def test_env_vars_available_in_script(self, tmp_path, monkeypatch):
        """QJOB_JOB_ID must be accessible from within the script."""
        script = _make_script(tmp_path, """\
            #!/bin/bash
            echo "JOB_ID=${QJOB_JOB_ID}"
        """)
        job = _make_job(script_path=str(script))
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            log_content = pathlib.Path(stored.log_stdout).read_text()
            assert f"JOB_ID={job.id}" in log_content

    @_skip_subprocess
    def test_job_runs_in_configured_workdir(self, tmp_path, monkeypatch):
        script_dir = tmp_path / "scripts"
        workdir = tmp_path / "submit-dir"
        script_dir.mkdir()
        workdir.mkdir()
        script = _make_script(script_dir, """\
            #!/bin/bash
            pwd
        """)
        job = _make_job(script_path=str(script))
        job.workdir = str(workdir)
        pool = _make_pool()

        asyncio.run(runner._run_job(job, pool))

        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            log_content = pathlib.Path(stored.log_stdout).read_text().strip()
            assert log_content == str(workdir)

    @_skip_subprocess
    def test_shutdown_active_jobs_terminates_running_process(self, tmp_path, monkeypatch):
        script = _make_script(tmp_path, """\
            #!/bin/bash
            sleep 60
        """)
        job = _make_job(script_path=str(script))
        pool = _make_pool()

        async def _run():
            runner.start_job(job, pool)
            for _ in range(50):
                if job.id in runner._active_processes:
                    break
                await asyncio.sleep(0.05)
            assert job.id in runner._active_processes
            await runner.shutdown_active_jobs(grace_sec=0.2)

        asyncio.run(_run())

        assert job.id not in runner._active_processes
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.FAILED
            assert stored.exit_code != 0
