from __future__ import annotations

import datetime
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
def client(app, alice_token):
    """
    Return a synchronous TestClient authenticated as 'alice'.

    Parameters
    ----------
    app : fastapi.FastAPI
        The application under test.
    alice_token : str
        Raw API token for alice (injected from conftest fixture).

    Yields
    ------
    fastapi.testclient.TestClient
        A configured test client with alice's Authorization header.
    """

    headers = {"Authorization": f"Bearer {alice_token}"}
    with TestClient(app, headers=headers) as tc:
        yield tc


@pytest.fixture
def admin_client(app, root_token):
    """
    Return a synchronous TestClient authenticated as 'root' (admin).

    Yields
    ------
    fastapi.testclient.TestClient
        A configured test client with root's Authorization header.
    """

    headers = {"Authorization": f"Bearer {root_token}"}
    with TestClient(app, headers=headers) as tc:
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
    submitted_at: datetime.datetime | None = None,
    started_at: datetime.datetime | None = None,
    finished_at: datetime.datetime | None = None,
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
    if submitted_at is not None:
        job.submitted_at = submitted_at
    if started_at is not None:
        job.started_at = started_at
    if finished_at is not None:
        job.finished_at = finished_at
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
        crud.set_resources(total_cpus=4, gpu_ids=[0])
        script = make_script("""\
            #!/bin/bash
            #QJOB --name my-job --cpus 4 --gpus 1
            python x.py
        """)
        response = client.post("/jobs", json={"script_path": str(script)})
        data = response.json()
        assert data["name"] == "my-job"
        assert data["req_cpus"] == 4
        assert data["req_gpus"] == 1
        assert data["user"] == "alice"

    def test_workdir_is_stored(self, client, make_script, tmp_path):
        script = make_script("#!/bin/bash\npython x.py\n")
        workdir = tmp_path / "submit-dir"
        workdir.mkdir()

        response = client.post(
            "/jobs",
            json={"script_path": str(script), "workdir": str(workdir)},
        )

        assert response.status_code == 201
        assert response.json()["workdir"] == str(workdir)

    def test_default_workdir_is_script_parent(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")

        response = client.post("/jobs", json={"script_path": str(script)})

        assert response.status_code == 201
        assert response.json()["workdir"] == str(script.parent)

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
        client.post("/jobs", json={"script_path": str(script)})
        data = client.get("/jobs").json()
        assert data["total"] == 1

    def test_filter_by_user(self, client):
        _persist_job(user="alice")
        _persist_job(user="bob")
        data = client.get("/jobs", params={"user": "alice"}).json()
        assert all(j["user"] == "alice" for j in data["jobs"])

    def test_filter_by_state(self, client):
        _persist_job(status=models.JobStatus.QUEUED)
        _persist_job(status=models.JobStatus.DONE)
        data = client.get("/jobs", params={"state": "queued"}).json()
        assert all(j["status"] == "queued" for j in data["jobs"])

    def test_filter_by_multiple_states(self, client):
        _persist_job(status=models.JobStatus.QUEUED)
        _persist_job(status=models.JobStatus.RUNNING)
        _persist_job(status=models.JobStatus.DONE)

        data = client.get("/jobs", params={"state": "queued,running"}).json()

        assert data["total"] == 2
        assert {j["status"] for j in data["jobs"]} == {"queued", "running"}

    def test_invalid_state_returns_400(self, client):
        response = client.get("/jobs", params={"state": "invalid"})
        assert response.status_code == 400

    def test_filter_by_since(self, client):
        cutoff = datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc)
        _persist_job(
            name="old",
            submitted_at=cutoff - datetime.timedelta(days=1),
        )
        _persist_job(
            name="new",
            submitted_at=cutoff + datetime.timedelta(hours=1),
        )

        data = client.get("/jobs", params={"since": cutoff.isoformat()}).json()

        assert data["total"] == 1
        assert data["jobs"][0]["name"] == "new"

    def test_sort_by_priority(self, client):
        _persist_job(name="low", priority=10)
        _persist_job(name="high", priority=90)

        data = client.get("/jobs", params={"sort": "priority"}).json()

        assert [j["name"] for j in data["jobs"]] == ["high", "low"]

    def test_sort_by_user(self, admin_client):
        now = datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc)
        _persist_job(user="bob", name="bob-job", submitted_at=now)
        _persist_job(user="alice", name="alice-job", submitted_at=now)

        data = admin_client.get("/jobs", params={"sort": "user"}).json()

        assert [j["user"] for j in data["jobs"]] == ["alice", "bob"]

    def test_invalid_sort_returns_400(self, client):
        response = client.get("/jobs", params={"sort": "invalid"})
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
        response = client.delete(f"/jobs/{job.id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancels_running_job_without_pid(self, client):
        # No PID means no live process — cancel completes immediately.
        job = _persist_job(user="alice", status=models.JobStatus.RUNNING)
        response = client.delete(f"/jobs/{job.id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancels_running_job_with_pid_becomes_cancelling(self, client):
        # PID present — SIGTERM sent, status becomes CANCELLING until runner confirms exit.
        job = _persist_job(user="alice", status=models.JobStatus.RUNNING)
        with database.get_session() as session:
            stored = session.get(models.Job, job.id)
            stored.pid = 99999999  # Non-existent PID; ProcessLookupError is silently caught.
        response = client.delete(f"/jobs/{job.id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelling"

    def test_returns_409_for_cancelling_job(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.CANCELLING)
        response = client.delete(f"/jobs/{job.id}")
        assert response.status_code == 409

    def test_returns_404_for_unknown_id(self, client):
        response = client.delete("/jobs/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_returns_403_for_wrong_user(self, client, bob_token):
        job = _persist_job(user="alice")
        bob_headers = {"Authorization": f"Bearer {bob_token}"}
        response = client.delete(f"/jobs/{job.id}", headers=bob_headers)
        assert response.status_code == 403

    def test_root_can_cancel_any_job(self, admin_client):
        job = _persist_job(user="alice")
        response = admin_client.delete(f"/jobs/{job.id}")
        assert response.status_code == 200

    def test_returns_409_for_done_job(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.DONE)
        response = client.delete(f"/jobs/{job.id}")
        assert response.status_code == 409

    def test_returns_409_for_already_cancelled(self, client):
        job = _persist_job(user="alice", status=models.JobStatus.CANCELLED)
        response = client.delete(f"/jobs/{job.id}")
        assert response.status_code == 409

    def test_persists_cancelled_status(self, client):
        job = _persist_job(user="alice")
        client.delete(f"/jobs/{job.id}")
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
    """PUT /resources updates resource limits (admin only)."""

    def test_returns_200(self, admin_client):
        response = admin_client.put("/resources", json={"total_cpus": 16})
        assert response.status_code == 200

    def test_non_admin_returns_403(self, client):
        response = client.put("/resources", json={"total_cpus": 16})
        assert response.status_code == 403

    def test_updates_cpus(self, admin_client):
        admin_client.put("/resources", json={"total_cpus": 32})
        data = admin_client.get("/resources").json()
        assert data["total_cpus"] == 32

    def test_updates_gpus(self, admin_client):
        admin_client.put("/resources", json={"total_gpus": 4})
        data = admin_client.get("/resources").json()
        assert data["total_gpus"] == 4
        assert data["gpu_ids"] == [0, 1, 2, 3]

    def test_updates_gpu_ids(self, admin_client):
        response = admin_client.put("/resources", json={"gpu_ids": [1, 3, 7]})
        assert response.status_code == 200
        data = response.json()
        assert data["total_gpus"] == 3
        assert data["gpu_ids"] == [1, 3, 7]

    def test_duplicate_gpu_ids_returns_422(self, admin_client):
        response = admin_client.put("/resources", json={"gpu_ids": [1, 1]})
        assert response.status_code == 422

    def test_updates_mem(self, admin_client):
        admin_client.put("/resources", json={"total_mem_mb": 65536})
        data = admin_client.get("/resources").json()
        assert data["total_mem_mb"] == 65536

    def test_empty_body_returns_422(self, admin_client):
        # Pydantic model_validator rejects all-None bodies.
        response = admin_client.put("/resources", json={})
        assert response.status_code == 422

    def test_zero_cpus_returns_422(self, admin_client):
        response = admin_client.put("/resources", json={"total_cpus": 0})
        assert response.status_code == 422

    def test_negative_gpus_returns_422(self, admin_client):
        response = admin_client.put("/resources", json={"total_gpus": -1})
        assert response.status_code == 422

    def test_zero_mem_returns_422(self, admin_client):
        response = admin_client.put("/resources", json={"total_mem_mb": 0})
        assert response.status_code == 422

    def test_partial_update_preserves_other_fields(self, admin_client):
        admin_client.put("/resources", json={"total_cpus": 8, "total_gpus": 2})
        admin_client.put("/resources", json={"total_cpus": 16})
        data = admin_client.get("/resources").json()
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
    def patch_async_client(self, client, alice_token):
        """
        Replace service._async_client with a factory backed by ASGITransport.

        The mock client includes alice's Bearer token so all service calls are
        authenticated as alice.

        Parameters
        ----------
        client : starlette.testclient.TestClient
            The test client whose app is used as the ASGI transport target.
        alice_token : str
            Raw API token for alice.

        Yields
        ------
        None
        """

        import httpx

        original = service._async_client
        auth_headers = {"Authorization": f"Bearer {alice_token}"}

        def _mock():
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app),
                base_url="http://testserver",
                headers=auth_headers,
            )

        service._async_client = _mock
        yield
        service._async_client = original

    def test_list_jobs_parses_response(self, client, make_script):
        script = make_script("#!/bin/bash\npython x.py\n")
        client.post("/jobs", json={"script_path": str(script)})

        jobs = service.list_jobs(user="alice")
        assert len(jobs) == 1
        assert jobs[0].user == "alice"
        assert jobs[0].status == "queued"

    def test_list_jobs_filters_by_states(self):
        _persist_job(user="alice", status=models.JobStatus.QUEUED)
        _persist_job(user="alice", status=models.JobStatus.DONE)

        jobs = service.list_jobs(states=["queued"])

        assert len(jobs) == 1
        assert jobs[0].status == "queued"

    def test_list_jobs_filters_by_since(self):
        cutoff = datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc)
        _persist_job(
            user="alice",
            status=models.JobStatus.QUEUED,
            submitted_at=cutoff - datetime.timedelta(days=1),
        )
        _persist_job(
            user="alice",
            status=models.JobStatus.QUEUED,
            submitted_at=cutoff + datetime.timedelta(hours=1),
        )

        jobs = service.list_jobs(since=cutoff)

        assert len(jobs) == 1

    def test_list_jobs_sorts_by_priority(self):
        _persist_job(user="alice", name="low", priority=10)
        _persist_job(user="alice", name="high", priority=90)

        jobs = service.list_jobs(sort="priority")

        assert [j.name for j in jobs] == ["high", "low"]

    def test_get_job_returns_none_for_404(self):
        result = service.get_job("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_cancel_job_raises_permission_error(self):
        # alice (authenticated) tries to cancel a job owned by bob — expects 403.
        job = _persist_job(user="bob")
        with pytest.raises(PermissionError):
            service.cancel_job(job.id)

    def test_get_resources_returns_resource_info(self):
        info = service.get_resources()
        assert isinstance(info, service.ResourceInfo)
        assert info.total_cpus >= 0
