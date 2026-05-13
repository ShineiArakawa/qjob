# qjob

A lightweight job scheduler for research servers. Write `#QJOB` directives in shell scripts, submit them to a queue, and let qjob dispatch them automatically while managing CPU, GPU, and memory resources.

## Features

- Submit jobs by adding `#QJOB` comments to any shell script
- Resource management for CPU cores, GPUs, and memory
- EASY Backfill scheduling with Priority Aging
- REST API built on FastAPI + uvicorn
- PostgreSQL backend with multi-process support
- Live TUI dashboard (`qjob dash`)
- API server (`qjob serve`) and scheduler (`qjob scheduler`) run as independent processes

---

## Architecture

```text
┌─────────────────────┐     HTTP      ┌───────────────────┐
│   qjob CLI / User   │ ────────────► │  qjob serve       │
│   (submit / status) │               │  (FastAPI/uvicorn) │
└─────────────────────┘               └────────┬──────────┘
                                               │ SQLAlchemy
                                               ▼
                                      ┌────────────────────┐
                                      │     PostgreSQL      │
                                      └────────┬───────────┘
                                               │ SQLAlchemy
                                      ┌────────▼───────────┐
                                      │  qjob scheduler    │
                                      │  (EASY Backfill)   │
                                      └────────────────────┘
```

The API server and scheduler are separate processes. `qjob serve` can run with multiple workers. The scheduler uses a PostgreSQL Advisory Lock to ensure only one scheduler instance runs at a time.

---

## Requirements

- Python 3.12 or later
- PostgreSQL 14 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

---

## Installation

### Using uv (recommended)

```bash
git clone <repository-url>
cd qjob
uv sync
```

### Using pip

```bash
git clone <repository-url>
cd qjob
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## PostgreSQL Setup

### Create the database and user

```sql
-- Run in psql
CREATE USER qjob WITH PASSWORD 'your_password';
CREATE DATABASE qjob OWNER qjob;
```

### Create tables

Tables are created automatically on the first run of `qjob serve` or `qjob scheduler` using `CREATE TABLE IF NOT EXISTS`. No manual migration is needed for a fresh deployment.

To apply schema changes to an existing database, use Alembic:

```bash
export QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob"
alembic upgrade head
```

---

## Quick Start

### 1. Set the database URL

```bash
export QJOB_DB_URL="postgresql+psycopg://qjob:your_password@localhost:5432/qjob"
```

### 2. Start the API server

```bash
qjob serve
```

Listens on `127.0.0.1:8000` by default.

### 3. Start the scheduler (separate terminal)

```bash
qjob scheduler
```

### 4. Configure available resources

```bash
qjob admin set-resources --cpus 32 --gpus 4 --mem 65536
```

### 5. Submit a job

```bash
qjob submit examples/train.sh
```

### 6. Check status

```bash
qjob status
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

### `qjob serve`

Start the API server.

```bash
qjob serve                                 # Default settings
qjob serve --host 0.0.0.0 --port 8080     # Bind to all interfaces
qjob serve --workers 4                    # Multiple worker processes
qjob serve --log-level debug              # Verbose logging
```

### `qjob scheduler`

Start the scheduler process.

```bash
qjob scheduler
qjob scheduler --poll-interval 5.0        # Poll every 5 seconds
qjob scheduler --max-workers 128          # Max concurrent jobs
```

Only one scheduler may run at a time. A second invocation exits immediately with an error.

### `qjob dash`

Open the live TUI dashboard (connects directly to the database).

```bash
qjob dash
qjob dash --refresh 5.0    # Refresh interval in seconds
```

### `qjob admin set-resources`

Update resource limits.

```bash
qjob admin set-resources --cpus 32 --gpus 4 --mem 65536
```

`--mem` is in megabytes.

### `qjob admin list-jobs`

List all jobs from all users (admin view).

```bash
qjob admin list-jobs
qjob admin list-jobs --status queued
```

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

## Multi-Process Deployment

To scale with multiple uvicorn workers, use `--workers` and run the scheduler as a separate process.

```bash
# Terminal 1: API server with 4 workers
QJOB_DB_URL="..." qjob serve --workers 4

# Terminal 2: Scheduler
QJOB_DB_URL="..." qjob scheduler
```

> When using `--workers 2` or more, `QJOB_DB_URL` must be set as an environment variable.

---

## Shell Completion

Bash, Zsh, and Fish are supported.

```bash
# Bash
qjob --install-completion bash
source ~/.bashrc

# Zsh
qjob --install-completion zsh
source ~/.zshrc
```

Once enabled, job IDs and status values can be tab-completed.

---

## Scheduling Algorithm

- **EASY Backfill**: When the head job (highest-priority blocked job) cannot run, qjob sets a reservation for it and allows smaller jobs to run ahead, provided they finish before the reservation window closes.
- **Priority Aging**: The effective priority of a waiting job increases linearly over time at a rate of `aging_factor` priority points per hour. Aging is computed in-memory at sort time; the base priority stored in the database is never modified.

---

## Development and Testing

### Create a test database

```sql
CREATE USER qjob WITH PASSWORD 'your_test_password';
CREATE DATABASE qjob_test OWNER qjob;
```

### Run the test suite

```bash
QJOB_TEST_DB_URL="postgresql+psycopg://qjob:your_test_password@localhost:5432/qjob_test" \
  python -m pytest tests/ -v
```

Tests create the schema automatically and delete all data after each test. Tables are never dropped.

### Development server with hot reload

```bash
QJOB_DB_URL="..." qjob serve --reload
```

> `--reload` and `--workers` cannot be used together.

---

## Project Layout

```text
qjob/
├── qjob/
│   ├── api/           # FastAPI routers, schemas, CRUD
│   ├── cli/           # Typer CLI, dashboard, service layer
│   ├── core/          # Database, models, scheduler, runner, parser
│   └── migrations/    # Alembic configuration (versions/ excluded from distribution)
├── examples/          # Sample job scripts
├── tests/             # pytest test suite
├── alembic.ini        # Alembic configuration
└── pyproject.toml
```
