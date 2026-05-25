# qjob

qjob is a lightweight job scheduler for single-node lab-scale GPU servers. It allows users to submit ordinary shell scripts annotated with `#QJOB` directives, and dispatches them while accounting for CPU cores, GPU devices, memory, walltime, user ownership, and job priority.

The project is intended for small to medium shared GPU servers used in research laboratories, where a full HPC scheduler may be unnecessary but unmanaged execution can easily lead to resource conflicts.

## Main Features

- Shell-script based submission using leading `#QJOB` comments.
- Per-job requests for CPU cores, GPU count, memory, walltime, and priority.
- Priority aging and EASY-style backfilling.
- PostgreSQL-backed persistent job and resource state.
- FastAPI-based REST API with Bearer-token authentication.
- systemd service integration with `qjob admin up` / `qjob admin down` for single-command startup.
- Root-managed execution with privilege dropping to the submitting OS user.
- CPU affinity via `taskset` and GPU assignment through `CUDA_VISIBLE_DEVICES`.
- Terminal dashboard for local resource and queue monitoring.

## Requirements

- Linux, tested primarily for Ubuntu-like environments.
- Python 3.12 or later.
- PostgreSQL.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- `taskset` from `util-linux` if CPU affinity is required.
- Root privileges for the API server and scheduler when jobs must run as their submitting OS users.

## Installation

```bash
git clone <repository-url>
cd qjob
uv sync
```

For system-wide use, install the package in the environment from which `qjob` will be invoked by administrators and users.

## PostgreSQL Setup

Create a PostgreSQL database and user for qjob:

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
sudo -u postgres psql
```

Inside `psql`:

```sql
CREATE USER qjob WITH PASSWORD 'your_password';
CREATE DATABASE qjob OWNER qjob;
\q
```

Set the database URL before running any qjob command that touches the database:

```bash
export QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob"
```

qjob currently creates missing tables automatically on first database initialisation through SQLAlchemy `create_all`. This is sufficient for a fresh deployment. For schema evolution on an existing deployment, Alembic is configured, but migration version files should be generated and reviewed before relying on `alembic upgrade head` in production.

## Initial Administrative Setup

The API server and scheduler should normally be run as root. This allows qjob to start each job under the submitting OS user's UID and GID.

### 1. Bootstrap an admin token

Before the API server is available, create a token directly in the database:

```bash
sudo QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob" \
  qjob admin create-token <admin_username>
```

This command must be run as root. It creates a token for an existing OS user, stores only the SHA-256 hash in the database, writes the raw token to:

```text
~/.config/qjob/token
```

under the target user's home directory, and prints the token once.

Administrative users are determined by either:

- the `QJOB_ADMIN_USERS` environment variable, a comma-separated list of usernames; or
- membership in the OS group `qjob_admin`.

If `QJOB_ADMIN_USERS` is not set, `root` is treated as the default admin user.

### 2. Install and start as systemd services

The recommended way to run qjob in production is as a pair of systemd services. This provides automatic restart on failure and OS-boot startup.

Generate the unit files and enable both services:

```bash
sudo qjob admin install \
  --svc-env-file /absolute/path/to/.env
```

`--svc-env-file` embeds the path to your `.env` file into the unit files so that `QJOB_DB_URL` and other variables are loaded automatically. The path must be absolute.

Additional options mirror those of the manual commands:

```bash
sudo qjob admin install \
  --svc-env-file /absolute/path/to/.env \
  --host 127.0.0.1 --port 8000 --log-level info --workers 1 \
  --poll-interval 2.0 --max-workers 64
```

By default both services are enabled to start on boot (`--no-enable` to opt out). After installation, start them immediately:

```bash
sudo qjob admin up
```

To stop both services:

```bash
sudo qjob admin down
```

To remove the unit files entirely:

```bash
sudo qjob admin uninstall
```

If the project directory is moved, re-run `qjob admin install` so that the unit files are updated with the new binary path.

**Note on server binding:** By default, the server listens on `127.0.0.1:8000`. This is appropriate for single-node deployments where users access qjob on the same host. If binding to a non-local interface, place the service behind an appropriate network security boundary.

### Starting services manually (development / one-off)

For development or debugging, the server and scheduler can be started directly without systemd:

```bash
sudo QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob" \
  qjob admin serve

sudo QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob" \
  qjob admin scheduler
```

Only one scheduler should be active at a time. The implementation uses a PostgreSQL advisory lock so that a second scheduler process exits instead of dispatching jobs concurrently.

### 3. Configure node resources

Set the total resources managed by qjob:

```bash
qjob admin set-resources --cpus 32 --gpus 4 --mem 1T
```

`--gpus` keeps the legacy count-based behavior and manages GPU IDs
`0..N-1`. To manage a non-contiguous or explicit device set, use
`--gpu-ids`:

```bash
qjob admin set-resources --gpu-ids 0,2,5
```

A maximum per-job walltime can also be configured:

```bash
qjob admin set-resources --max-walltime 24:00:00
```

If a maximum walltime is configured, submitted jobs without an explicit walltime inherit that maximum. Jobs requesting a longer walltime are rejected.

## User Management

All API operations require a Bearer token. Users normally store their token at:

```text
~/.config/qjob/token
```

The token file should be readable only by the user:

```bash
chmod 600 ~/.config/qjob/token
```

A different path can be specified with `QJOB_TOKEN_PATH`.

### Issue a token through the API

After the API server is running, an admin can create a token for a user:

```bash
qjob admin init-token --username alice
```

If `--username` is omitted, a token is created for the current OS user and saved automatically to `~/.config/qjob/token`. If another username is given, the raw token is printed once and should be delivered to that user securely.

The target username must exist as an OS user. A user can have only one active token; existing tokens must be revoked manually in the database before a new one can be issued.

## Writing Job Scripts

A job script is an ordinary shell script. qjob reads `#QJOB` directives only from the leading contiguous comment block, after an optional shebang. Parsing stops at the first blank line or non-comment line.

Example:

```bash
#!/usr/bin/env bash
#QJOB --name train-vit
#QJOB --cpus 8 --gpus 1
#QJOB --mem 32G --walltime 08:00:00 --priority high

set -euo pipefail
python train.py
```

### Directive Reference

| Directive | Meaning | Default |
| --- | --- | --- |
| `--name <NAME>` | Human-readable job name. If omitted, a random readable name is generated. | generated |
| `--cpus <N>` | Number of CPU cores requested. Must be greater than zero. | `1` |
| `--gpus <N>` | Number of GPU devices requested. Must be zero or greater. | `0` |
| `--mem <SIZE>` | Memory request. Accepts values such as `512M`, `8G`, or `1T`. A unitless value is interpreted as bytes. | `1G` |
| `--walltime <HH:MM:SS>` | Maximum wall-clock runtime. `MM:SS` is also accepted. | unlimited, or admin maximum if configured |
| `--priority <LEVEL\|N>` | Scheduling priority. `low`, `normal`, `high`, or integer `0` to `100`. | `normal` (`50`) |
| `--env <KEY[,KEY...]>` | Parsed as a list of environment variable names. The current runner still starts from the scheduler environment; this option is therefore metadata rather than a strict allow-list. | none |

Priority labels map to the following values:

| Label | Value |
| --- | ---: |
| `low` | 20 |
| `normal` | 50 |
| `high` | 80 |

## Runtime Environment

qjob starts each job using `bash <script_path>`. If CPU cores are assigned, the command is wrapped with:

```bash
taskset -c <cpu-list> bash <script_path>
```

The working directory is the submitter's current directory as sent by the CLI. If no working directory is supplied to the API, qjob uses the script's parent directory.

The following variables are injected into the job environment:

| Variable | Value |
| --- | --- |
| `QJOB_JOB_ID` | 12-character lowercase hexadecimal job ID. |
| `QJOB_JOB_NAME` | Job name. |
| `QJOB_USER` | Submitting username. |
| `CUDA_VISIBLE_DEVICES` | Comma-separated assigned GPU device IDs, or an empty string for CPU-only jobs. |

The runner begins from the scheduler process environment. Therefore, environment variables available to the scheduler may also be visible to jobs unless explicitly controlled by deployment policy.

## Command Line Usage

### Submit a job

```bash
qjob submit train.sh
```

The CLI validates the script locally and then sends its absolute path and the current working directory to the API server. The script must be accessible on the server where the scheduler runs.

### Show job status

```bash
qjob status
qjob status <job_id>
qjob status --status running
qjob status --all
```

By default, `qjob status` lists the latest 20 jobs for the current OS user. Use `--all` to fetch all matching jobs for the current user.

Additional filters:

```bash
qjob status --user alice
qjob status --all-users
qjob status --all-users --all
```

Administrators can use `--all-users` to view jobs across users, or `--user USER` to inspect a specific user. Non-admin users are restricted by server-side authorization policy for most operations.

### Cancel jobs

```bash
qjob cancel <job_id>
qjob cancel <job_id_1> <job_id_2>
qjob cancel --all
```

Queued jobs move directly to `cancelled`. Running jobs move to `cancelling`; qjob sends SIGTERM to the job process group and escalates to SIGKILL if the process does not exit within the grace period. Resources are not released until the job reaches a terminal state and the scheduler reconciles resource usage.

### Read logs

```bash
qjob log <job_id>
qjob log <job_id> --stderr
```

The API returns a bounded tail of the selected log stream. The default maximum is 1 MiB, and the server-side upper bound is 16 MiB.

### Show resources

```bash
qjob resources
```

This prints configured totals, current usage, free resources, configured GPU IDs, and the configured maximum walltime.

### Open the dashboard

```bash
qjob dash
qjob dash --refresh 1.0
```

The dashboard reads the local database directly and is intended for use on the qjob host.

## Administrative Commands

| Command | Purpose |
| --- | --- |
| `qjob admin up` | Start the API server and scheduler via systemd. Requires root. |
| `qjob admin down` | Stop both services via systemd. Requires root. |
| `qjob admin install` | Generate systemd unit files, reload the daemon, and optionally enable services. Requires root. |
| `qjob admin uninstall` | Stop, disable, and remove the systemd unit files. Requires root. |
| `qjob admin create-token <username>` | Create a token directly in the database. Requires root. Used for bootstrap or recovery. |
| `qjob admin init-token [--username USER]` | Create a token through the API. Requires an admin token. |
| `qjob admin set-resources` | Update total CPUs, GPU count or GPU IDs, memory, or maximum walltime. Requires admin privileges. |
| `qjob admin list-jobs` | List jobs across all users. Requires admin privileges. |
| `qjob admin serve` | Start the FastAPI server directly (without systemd). Requires root. |
| `qjob admin scheduler` | Start the scheduler directly (without systemd). Requires root. |

## Job Lifecycle

qjob uses the following job states:

| State | Meaning |
| --- | --- |
| `queued` | The job has been accepted and is waiting for resources. |
| `running` | The scheduler has assigned resources and started the subprocess. |
| `cancelling` | Cancellation has been requested for a running job; termination is in progress. |
| `done` | The process exited with code `0`. |
| `failed` | The process exited with a non-zero code, exceeded walltime and was killed, or failed to spawn. |
| `cancelled` | The job was cancelled before completion. |

## Scheduling Model

The scheduler periodically performs the following operations:

1. Reconciles finished jobs and releases their assigned resources from the resource counters.
2. Fetches queued jobs.
3. Sorts candidates by effective priority and submission time.
4. Dispatches jobs that fit within the available CPU, GPU, and memory counters.
5. Applies EASY-style backfilling when the head job is blocked and a safe reservation window can be estimated from running jobs with known walltime.

Effective priority is computed as:

```text
effective_priority = min(100, base_priority + aging_factor * waiting_hours)
```

The current aging factor is 5 priority points per waiting hour.

Resource accounting is stored in the database. The scheduler locks the single `resources` row with `SELECT ... FOR UPDATE` while allocating jobs, which prevents concurrent resource updates from observing stale counters.

## REST API Summary

The CLI uses the REST API internally. All endpoints require a Bearer token.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/auth/token` | Create a token for an OS user. Admin only. |
| `POST` | `/jobs` | Submit a job. |
| `GET` | `/jobs` | List jobs with optional filters. |
| `GET` | `/jobs/{job_id}` | Get a job record. |
| `DELETE` | `/jobs/{job_id}` | Cancel a job. |
| `GET` | `/jobs/{job_id}/log` | Read stdout or stderr log tail. |
| `GET` | `/resources` | Read configured and used resources. |
| `PUT` | `/resources` | Update resource limits. Admin only. |

Example request:

```bash
curl -H "Authorization: Bearer $(cat ~/.config/qjob/token)" \
  http://127.0.0.1:8000/jobs
```

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `QJOB_DB_URL` | PostgreSQL SQLAlchemy URL. Required for the server, scheduler, dashboard, and direct DB admin commands. | none |
| `QJOB_API_URL` | Base URL used by the CLI. | `http://127.0.0.1:8000` |
| `QJOB_TOKEN_PATH` | Path to the local API token file. | `~/.config/qjob/token` |
| `QJOB_ADMIN_USERS` | Comma-separated admin usernames. | `root` |
| `QJOB_DB_POOL_ENABLED` | Enable SQLAlchemy connection pooling. | `false` |
| `QJOB_DB_POOL_SIZE` | Pool size when pooling is enabled. | `5` |
| `QJOB_DB_MAX_OVERFLOW` | Maximum overflow connections when pooling is enabled. | `5` |

The CLI also accepts `--env-file`, defaulting to `.env`, and loads it if present.

## Operational Notes and Limitations

- qjob is a single-node scheduler. It does not currently schedule across multiple machines.
- Memory is accounted for by request, but not enforced by cgroups or containers. Jobs can still exceed their requested memory unless the deployment adds an external limit.
- GPU assignment is implemented through `CUDA_VISIBLE_DEVICES`; qjob does not enforce GPU memory limits.
- CPU affinity is applied with `taskset`, but CPU time is not hard-limited.
- Job logs are written next to the submitted script using `<script>.out` and `<script>.err`. In shared deployments, consider changing this policy to a central log directory.
- The API server and scheduler are expected to run as root for multi-user execution. Non-root execution cannot safely drop privileges to arbitrary submitting users.
- The submitted script path must be meaningful on the server host. qjob does not upload script contents.
- Each user currently has at most one active API token.
- Alembic is configured, but this repository currently relies on automatic table creation for fresh deployments unless migration version files are added.

## Minimal Example

Create `train.sh`:

```bash
#!/usr/bin/env bash
#QJOB --name example
#QJOB --cpus 2 --gpus 1
#QJOB --mem 8G --walltime 00:30:00 --priority normal

set -euo pipefail

echo "job id: ${QJOB_JOB_ID}"
echo "visible GPUs: ${CUDA_VISIBLE_DEVICES}"
python train.py
```

Submit and monitor:

```bash
qjob submit train.sh
qjob status
qjob log <job_id>
```
