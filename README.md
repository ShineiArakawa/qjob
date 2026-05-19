# qjob

A lightweight job scheduler designed for **lab-scale GPU servers**. Add `#QJOB` directives to any shell script, submit it to the queue, and qjob automatically dispatches jobs while managing CPU, GPU, and memory resources across users.

## Features

- Submit jobs by adding `#QJOB` comments to any shell script — no special job script format required
- Per-job resource requests for CPU cores, GPUs, and memory
- EASY Backfill scheduling with Priority Aging for flexible priority control
- REST API server built on FastAPI + uvicorn
- PostgreSQL backend for persistent job management
- Live TUI dashboard (`qjob dash`)

---

## Requirements

- **Ubuntu 22.04 or later**
- Python 3.12 or later
- PostgreSQL 14 or later
- [uv](https://docs.astral.sh/uv/)

---

## Installation

```bash
git clone <repository-url>
cd qjob
uv sync
```

---

## PostgreSQL Setup

### 1. Install PostgreSQL

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

### 2. Open a psql session

Connect to PostgreSQL as the `postgres` superuser:

```bash
sudo -u postgres psql
```

### 3. Create the database and user

Run the following inside `psql`:

```sql
CREATE USER qjob WITH PASSWORD 'your_password';
CREATE DATABASE qjob OWNER qjob;
\q
```

### 4. Create tables

Tables are created automatically on the first run of `qjob admin serve` or `qjob admin scheduler` using `CREATE TABLE IF NOT EXISTS`. No manual migration is needed for a fresh deployment.

To apply schema changes to an existing database, use Alembic:

```bash
export QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob"
alembic upgrade head
```

---

## Initial Setup

This section walks through the one-time setup an administrator performs before any user can submit jobs.

### 1. Set the database URL

```bash
export QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob"
```

### 2. Bootstrap the admin token

Create an API token for the admin user **directly in the database** (no server required). This must be run as root and is typically done once on first deployment.

```bash
sudo qjob admin create-token <admin_username>
```

The token is saved to `~/.config/qjob/token` under the admin's home directory and also printed to stdout.

### 3. Start the API server

```bash
sudo qjob admin serve
```

Listens on `127.0.0.1:8000` by default.

### 4. Start the scheduler (separate terminal)

```bash
sudo qjob admin scheduler
```

The API server and scheduler run as independent processes. The scheduler uses a PostgreSQL Advisory Lock to ensure only one instance runs at a time.

### 5. Configure available resources

```bash
qjob admin set-resources --cpus 32 --gpus 4 --mem 65536
```

---

## User Management

API tokens are required to use qjob. Tokens are issued by an administrator.

### Adding a new user

Run as an administrator (your token must be stored at `~/.config/qjob/token`):

```bash
qjob admin init-token --username alice
```

- If `--username` is omitted, a token is created for the **current user** and saved to `~/.config/qjob/token` automatically.
- If `--username` names another user, the token is **printed to stdout** — distribute it to that user manually.

The user saves the token:

```bash
mkdir -p ~/.config/qjob
echo "<token>" > ~/.config/qjob/token
chmod 600 ~/.config/qjob/token
```

### Bootstrapping without a running server

If the API server is not yet available (e.g. first deployment), use `create-token` to write the token directly to the database. This requires root:

```bash
sudo qjob admin create-token alice
```

The token is saved to `alice`'s `~/.config/qjob/token` automatically.

---

## Quick Start (user)

Once an administrator has issued your token:

```bash
# Submit a job
qjob submit train.sh

# Check your jobs
qjob status

# View logs
qjob log <job_id>
```

---

## Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `QJOB_DB_URL` | PostgreSQL connection URL | — (required) |
| `QJOB_API_URL` | API server URL used by the CLI | `http://127.0.0.1:8000` |
| `QJOB_DB_POOL_ENABLED` | Enable connection pooling (`1` / `true`) | `false` (NullPool) |
| `QJOB_DB_POOL_SIZE` | Pool size (when `QJOB_DB_POOL_ENABLED=1`) | `5` |
| `QJOB_DB_MAX_OVERFLOW` | Pool overflow limit | `5` |

---

## Writing Job Scripts

Place `#QJOB` directives in the leading comment block of your shell script.

```bash
#!/usr/bin/env bash
#QJOB --name my-training-job
#QJOB --cpus 4 --gpus 1
#QJOB --mem 8G --walltime 01:00:00 --priority high

set -euo pipefail

python train.py
```

### Directive Reference

| Directive | Description | Default |
| --------- | ----------- | ------- |
| `--name <NAME>` | Human-readable job name | — |
| `--cpus <N>` | Number of CPU cores | `1` |
| `--gpus <N>` | Number of GPUs | `0` |
| `--mem <SIZE>` | Memory (e.g. `4G`, `2048M`) | `1G` |
| `--walltime <HH:MM:SS>` | Maximum wall-clock time | unlimited |
| `--priority <LEVEL\|N>` | Priority: `low`=20, `normal`=50, `high`=80, or an integer 0–100 | `normal` (50) |
| `--env <KEY>` | Environment variable name to forward to the job | — |

### Environment Variables Injected into Jobs

| Variable | Value |
| -------- | ----- |
| `QJOB_JOB_ID` | Job ID (12-character lowercase hex string) |
| `QJOB_JOB_NAME` | Job name |
| `QJOB_USER` | Submitting user's username |
| `CUDA_VISIBLE_DEVICES` | Comma-separated list of assigned GPU indices |

---

## CLI Reference

### `qjob submit <script>`

Submit a shell script to the job queue.

```bash
qjob submit train.sh
```

### `qjob status [JOB_ID]`

Show job status. Without arguments, lists the latest 10 jobs for the current user.

```bash
qjob status                        # Current user's jobs (latest 10)
qjob status abc123def456           # Detailed view of a specific job
qjob status --all                  # All users
qjob status --status running       # Filter by status
qjob status --all-jobs             # No count limit
```

Status values: `queued` / `running` / `done` / `failed` / `cancelled`

### `qjob cancel <JOB_ID>`

Cancel a queued or running job.

```bash
qjob cancel abc123def456
qjob cancel --all                  # Cancel all your queued and running jobs
```

### `qjob log <JOB_ID>`

Print a job's stdout. Use `--stderr` to print stderr instead.

```bash
qjob log abc123def456
qjob log abc123def456 --stderr
```

### `qjob resources`

Show current resource availability.

```bash
qjob resources
```

### `qjob dash`

Open the live TUI dashboard.

```bash
qjob dash
qjob dash --refresh 5.0    # Refresh interval in seconds
```

### `qjob admin` — Administrative commands

All `admin` subcommands require root or admin privileges.

#### `qjob admin serve`

Start the API server.

```bash
qjob admin serve                                 # Default settings
qjob admin serve --host 0.0.0.0 --port 8080     # Bind to all interfaces
qjob admin serve --workers 4                     # Multiple worker processes
qjob admin serve --log-level debug               # Verbose logging
```

#### `qjob admin scheduler`

Start the scheduler process. Only one instance may run at a time.

```bash
qjob admin scheduler
qjob admin scheduler --poll-interval 5.0         # Poll every 5 seconds
qjob admin scheduler --max-workers 128           # Max concurrent jobs
```

#### `qjob admin init-token`

Create an API token for a user **via the API server** (server must be running, admin token required).

```bash
qjob admin init-token                            # Token for current user
qjob admin init-token --username alice           # Token for alice (printed to stdout)
```

#### `qjob admin create-token <username>`

Create an API token **directly in the database** (root only, no server required). Use for initial bootstrapping.

```bash
sudo qjob admin create-token alice
```

#### `qjob admin set-resources`

Update resource limits. `--mem` is in megabytes.

```bash
qjob admin set-resources --cpus 32 --gpus 4 --mem 65536
qjob admin set-resources --max-walltime 08:00:00
```

#### `qjob admin list-jobs`

List all jobs from all users.

```bash
qjob admin list-jobs
qjob admin list-jobs --status queued
```

---

## Shell Completion

Bash, Zsh, and Fish are supported. Once enabled, job IDs and status values can be tab-completed.

```bash
# Bash
qjob --install-completion bash
source ~/.bashrc

# Zsh
qjob --install-completion zsh
source ~/.zshrc
```

---

## Scheduling Algorithm

- **EASY Backfill**: When the head job (highest-priority blocked job) cannot run, qjob sets a reservation for it and allows smaller jobs to run ahead, provided they finish before the reservation window closes.
- **Priority Aging**: The effective priority of a waiting job increases linearly over time at a rate of `aging_factor` priority points per hour. Aging is computed in-memory at sort time; the base priority stored in the database is never modified.
