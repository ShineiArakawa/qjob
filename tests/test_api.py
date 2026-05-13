from __future__ import annotations

import json
import os
import pathlib
import textwrap

import pytest
from fastapi.testclient import TestClient

import qjob.api.crud as crud
import qjob.api.server as server
import qjob.cli.service as service
import qjob.core.database as database
import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def app():
    """
    Return a FastAPI application wired to the already-initialised test DB.

    The ``isolated_db`` fixture in conftest.py has already called
    ``database.init_db()`` with ``QJOB_TEST_DB_URL``.  We pass the same URL
    here so ``create_app()`` reuses it.

    Parameters
    ----------
    None

    Yields
    ------
    fastapi.FastAPI
        Configured application instance backed by the in-memory DB.
    """

    return server.create_app(db_url=os.environ["QJOB_DB_URL"])


@pytest.fixture
def client(app):
    """
    Return a synchronous TestClient for the FastAPI application.

    The ``with`` block triggers the lifespan context manager, which starts
    (and later stops) the scheduler.

    Parameters
    ----------
    app : fastapi.FastAPI
        The application under test.

    Yields
    ------
    fastapi.testclient.TestClient
        A configured test client.
    """

    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def make_script(tmp_path: pathlib.Path):
    """
    Factory fixture that writes a shell script to a temporary file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory (function-scoped).

    Returns
    -------
    callable
        A function ``make(content: str) -> pathlib.Path`` that writes the
        script and returns its path.
    """

    def _make(content: str) -> pathlib.Path:
        p = tmp_path / "job.sh"
        p.write_text(textwrap.dedent(content))
        return p

    return _make


def _persist_job(
    status:    models.JobStatus = models.JobStatus.QUEUED,
    user:      str = "alice",
    req_cpus:  int = 1,
    req_gpus:  int = 0,
    priority:  int = 50,
    name:      str | None = "test-job",
) -> models.Job:
    """Insert a job row directly into the DB and return the instance."""

    job = models.Job(
        user=user,
        name=name,
        script_path="/tmp/job.sh",
        status=status,
        req_cpus=req_cpus,
        req_gpus=req_gpus,
        req_mem_mb=512,
        priority=priority,
    )
    with database.get_session() as session:
        session.add(job)
    return job


# --------------------------------------------------------------------------------------
# Health check


class TestHealth:
    """GET /health returns 200 with status ok."""

    def test_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# --------------------------------------------------------------------------------------
# POST /jobs


class TestSubmitJob:
    """POST /jobs creates a QUEUED job and returns 201."""

    def test_returns_201(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")
        response = client.post("/jobs", json={"script_path": str(script)})
        assert response.status_code == 201

    def test_response_contains_id(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")
        response = client.post("/jobs", json={"script_path": str(script)})
        assert "id" in response.json()

    def test_status_is_queued(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")
        response = client.post("/jobs", json={"script_path": str(script)})
        assert response.json()["status"] == "queued"

    def test_directives_are_reflected(self, client, make_script):
        script = make_script("""\
            #!/bin/bash
            #QJOB --name my-job --cpus 4 --gpus 1
            python x.py
        """)
        response = client.post(
            "/jobs", json={"script_path": str(script), "user": "alice"}
        )
        data = response.json()
        assert data["name"] == "my-job"
        assert data["req_cpus"] == 4
        assert data["req_gpus"] == 1
        assert data["user"] == "alice"

    def test_missing_script_returns_404(self, client):
        response = client.post(
            "/jobs", json={"script_path": "/nonexistent/job.sh"}
        )
        assert response.status_code == 404

    def test_invalid_directive_returns_422(self, client, make_script):
        script = make_script("#QJOB --cpus 0\npython x.py\n")
        response = client.post("/jobs", json={"script_path": str(script)})
        assert response.status_code == 422


# --------------------------------------------------------------------------------------
# GET /jobs


class TestListJobs:
    """GET /jobs returns filtered job lists."""

    def test_returns_200(self, client):
        response = client.get("/jobs")
        assert response.status_code == 200

    def test_empty_initially(self, client):
        data = client.get("/jobs").json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_returns_submitted_job(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")
        client.post("/jobs", json={"script_path": str(script), "user": "alice"})
        data = client.get("/jobs").json()
        assert data["total"] == 1

    def test_filter_by_user(self, client):
        _persist_job(user="alice")
        _persist_job(user="bob")
        data = client.get("/jobs", params={"user": "alice"}).json()
        assert all(j["user"] == "alice" for j in data["jobs"])

    def test_filter_by_status(self, client):
        _persist_job(status=models.JobStatus.QUEUED)
        _persist_job(status=models.JobStatus.DONE)
        data = client.get("/jobs", params={"status": "queued"}).json()
        assert all(j["status"] == "queued" for j in data["jobs"])

    def test_invalid_status_returns_400(self, client):
        response = client.get("/jobs", params={"status": "invalid"})
        assert response.status_code == 400

    def test_total_matches_jobs_length(self, client):
        _persist_job()
        _persist_job()
        data = client.get("/jobs").json()
        assert data["total"] == len(data["jobs"])

    def test_limit_restricts_returned_jobs_but_not_total(self, client):
        _persist_job(name="job-a")
        _persist_job(name="job-b")
        _persist_job(name="job-c")

        data = client.get("/jobs", params={"limit": 2}).json()

        assert data["total"] == 3
        assert len(data["jobs"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_offset_skips_matching_jobs(self, client):
        _persist_job(name="oldest")
        _persist_job(name="middle")
        _persist_job(name="newest")

        first_page = client.get("/jobs", params={"limit": 1, "offset": 0}).json()
        second_page = client.get("/jobs", params={"limit": 1, "offset": 1}).json()

        assert first_page["total"] == 3
        assert second_page["total"] == 3
        assert first_page["jobs"][0]["id"] != second_page["jobs"][0]["id"]
        assert second_page["offset"] == 1

    def test_invalid_limit_returns_422(self, client):
        response = client.get("/jobs", params={"limit": 0})
        assert response.status_code == 422

    def test_invalid_offset_returns_422(self, client):
        response = client.get("/jobs", params={"offset": -1})
        assert response.status_code == 422


# --------------------------------------------------------------------------------------
# GET /jobs/{job_id}


class TestGetJob:
    """GET /jobs/{job_id} returns a single job or 404."""

    def test_returns_200_for_existing_job(self, client):
        job = _persist_job()
        response = client.get(f"/jobs/{job.id}")
        assert response.status_code == 200

    def test_returns_correct_id(self, client):
        job = _persist_job()
        data = client.get(f"/jobs/{job.id}").json()
        assert data["id"] == job.id

    def test_returns_404_for_unknown_id(self, client):
        response = client.get("/jobs/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_all_fields_present(self, client):
        job = _persist_job(name="test", req_cpus=4, priority=80)
        data = client.get(f"/jobs/{job.id}").json()
        assert data["name"] == "test"
        assert data["req_cpus"] == 4
        assert data["priority"] == 80


# --------------------------------------------------------------------------------------
# DELETE /jobs/{job_id}


class TestCancelJob:
    """DELETE /jobs/{job_id} cancels eligible jobs."""

    def test_cancels_queued_job(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.QUEUED)
        response = client.delete(f"/jobs/{job.id}", params={"user": "alice"})
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancels_running_job(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.RUNNING)
        response = client.delete(f"/jobs/{job.id}", params={"user": "alice"})
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_returns_404_for_unknown_id(self, client):
        response = client.delete(
            "/jobs/00000000-0000-0000-0000-000000000000",
            params={"user": "alice"},
        )
        assert response.status_code == 404

    def test_returns_403_for_wrong_user(self, client):
        job = _persist_job(user="alice")
        response = client.delete(f"/jobs/{job.id}", params={"user": "bob"})
        assert response.status_code == 403

    def test_root_can_cancel_any_job(self, client):
        job = _persist_job(user="alice")
        response = client.delete(f"/jobs/{job.id}", params={"user": "root"})
        assert response.status_code == 200

    def test_returns_409_for_done_job(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.DONE)
        response = client.delete(f"/jobs/{job.id}", params={"user": "alice"})
        assert response.status_code == 409

    def test_returns_409_for_already_cancelled(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.CANCELLED)
        response = client.delete(f"/jobs/{job.id}", params={"user": "alice"})
        assert response.status_code == 409

    def test_persists_cancelled_status(self, client):
        job = _persist_job(user="alice")
        client.delete(f"/jobs/{job.id}", params={"user": "alice"})
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            assert stored.status == models.JobStatus.CANCELLED


# --------------------------------------------------------------------------------------
# GET /jobs/{job_id}/log


class TestGetLog:
    """GET /jobs/{job_id}/log returns log content."""

    def test_returns_200(self, client):
        job = _persist_job()
        response = client.get(f"/jobs/{job.id}/log")
        assert response.status_code == 200

    def test_not_available_message_when_no_log_path(self, client):
        job = _persist_job()
        data = client.get(f"/jobs/{job.id}/log").json()
        assert "not yet available" in data["content"].lower()

    def test_returns_log_content(self, client, tmp_path):
        log_file = tmp_path / "stdout.log"
        log_file.write_text("hello from job\n")
        job = _persist_job()
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.log_stdout = str(log_file)

        data = client.get(f"/jobs/{job.id}/log").json()
        assert "hello from job" in data["content"]

    def test_returns_stderr_content(self, client, tmp_path):
        log_file = tmp_path / "stderr.log"
        log_file.write_text("error output\n")
        job = _persist_job()
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.log_stderr = str(log_file)

        data = client.get(
            f"/jobs/{job.id}/log", params={"stream": "stderr"}
        ).json()
        assert "error output" in data["content"]

    def test_invalid_stream_returns_400(self, client):
        job = _persist_job()
        response = client.get(f"/jobs/{job.id}/log", params={"stream": "invalid"})
        assert response.status_code == 400

    def test_log_response_is_limited_to_tail(self, client, tmp_path):
        log_file = tmp_path / "stdout.log"
        log_file.write_text("old content\n" + ("x" * 128) + "new content\n")
        job = _persist_job()
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.log_stdout = str(log_file)

        data = client.get(
            f"/jobs/{job.id}/log", params={"max_bytes": 32}
        ).json()

        assert "log truncated" in data["content"]
        assert "new content" in data["content"]
        assert "old content" not in data["content"]

    def test_invalid_max_bytes_returns_422(self, client):
        job = _persist_job()
        response = client.get(f"/jobs/{job.id}/log", params={"max_bytes": 0})
        assert response.status_code == 422

    def test_response_contains_stream_field(self, client):
        job = _persist_job()
        data = client.get(f"/jobs/{job.id}/log").json()
        assert data["stream"] == "stdout"
        assert data["job_id"] == job.id


# --------------------------------------------------------------------------------------
# GET /resources


class TestGetResources:
    """GET /resources returns resource configuration."""

    def test_returns_200(self, client):
        response = client.get("/resources")
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.get("/resources").json()
        for field in ("total_cpus", "total_gpus", "total_mem_mb",
                      "used_cpus", "used_gpus", "used_mem_mb"):
            assert field in data

    def test_used_counts_running_jobs(self, client):
        job = _persist_job(req_cpus=4, status=models.JobStatus.RUNNING)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.assigned_cpus = json.dumps([0, 1, 2, 3])
            stored.assigned_gpus = json.dumps([])
            resource = session.get(models.Resource, 1)
            resource.used_cpus = 4

        data = client.get("/resources").json()
        assert data["used_cpus"] == 4


# --------------------------------------------------------------------------------------
# PUT /resources


class TestUpdateResources:
    """PUT /resources updates resource limits."""

    def test_returns_200(self, client):
        response = client.put("/resources", json={"total_cpus": 16})
        assert response.status_code == 200

    def test_updates_cpus(self, client):
        client.put("/resources", json={"total_cpus": 32})
        data = client.get("/resources").json()
        assert data["total_cpus"] == 32

    def test_updates_gpus(self, client):
        client.put("/resources", json={"total_gpus": 4})
        data = client.get("/resources").json()
        assert data["total_gpus"] == 4

    def test_updates_mem(self, client):
        client.put("/resources", json={"total_mem_mb": 65536})
        data = client.get("/resources").json()
        assert data["total_mem_mb"] == 65536

    def test_empty_body_returns_422(self, client):
        # Pydantic model_validator rejects all-None bodies.
        response = client.put("/resources", json={})
        assert response.status_code == 422

    def test_zero_cpus_returns_422(self, client):
        response = client.put("/resources", json={"total_cpus": 0})
        assert response.status_code == 422

    def test_negative_gpus_returns_422(self, client):
        response = client.put("/resources", json={"total_gpus": -1})
        assert response.status_code == 422

    def test_zero_mem_returns_422(self, client):
        response = client.put("/resources", json={"total_mem_mb": 0})
        assert response.status_code == 422

    def test_partial_update_preserves_other_fields(self, client):
        client.put("/resources", json={"total_cpus": 8, "total_gpus": 2})
        client.put("/resources", json={"total_cpus": 16})
        data = client.get("/resources").json()
        assert data["total_cpus"] == 16
        assert data["total_gpus"] == 2

    def test_crud_rejects_invalid_resource_limits(self):
        with pytest.raises(ValueError, match="total_cpus"):
            crud.set_resources(total_cpus=0)
        with pytest.raises(ValueError, match="total_gpus"):
            crud.set_resources(total_gpus=-1)
        with pytest.raises(ValueError, match="total_mem_mb"):
            crud.set_resources(total_mem_mb=0)


# --------------------------------------------------------------------------------------
# service.py HTTP client — integration tests
#
# These tests replace service._async_client with a factory that returns an
# AsyncClient backed by httpx.ASGITransport so requests go through the
# FastAPI app in-process without binding a real TCP port.


class TestServiceHttpClient:
    """service.py async functions correctly parse API responses."""

    @pytest.fixture(autouse=True)
    def patch_async_client(self, client):
        """
        Replace service._async_client with a factory backed by ASGITransport.

        Parameters
        ----------
        client : starlette.testclient.TestClient
            The test client whose app is used as the ASGI transport target.

        Yields
        ------
        None
        """

        import httpx

        original = service._async_client

        def _mock():
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app),
                base_url="http://testserver",
            )

        service._async_client = _mock
        yield
        service._async_client = original

    def test_list_jobs_parses_response(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")
        client.post("/jobs", json={"script_path": str(script), "user": "alice"})

        jobs = service.list_jobs(user="alice")
        assert len(jobs) == 1
        assert jobs[0].user == "alice"
        assert jobs[0].status == "queued"

    def test_get_job_returns_none_for_404(self):
        result = service.get_job("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_cancel_job_raises_permission_error(self):
        job = _persist_job(user="alice")
        with pytest.raises(PermissionError):
            service.cancel_job(job.id, user="bob")

    def test_get_resources_returns_resource_info(self):
        info = service.get_resources()
        assert isinstance(info, service.ResourceInfo)
        assert info.total_cpus >= 0
