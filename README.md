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
- Multi-GPU jobs are placed on a block of consecutive GPU IDs, chosen by best fit.
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

Multi-GPU jobs are placed only on consecutive GPU IDs (see [GPU placement](#gpu-placement)). When the configured IDs have gaps, the largest multi-GPU job is therefore limited by the longest run of consecutive IDs, and a job that requests more GPUs than that is rejected when it is submitted. With `0,2,5`, for example, no two IDs are consecutive, so a job cannot request two or more GPUs; with `0,1,2,5`, a job can request up to three.

GPU IDs are treated in ascending order, not in the order in which they were listed: placement ignores the order of the list, and both `CUDA_VISIBLE_DEVICES` and the free GPU IDs that qjob reports are always ascending. The exception is the `GPU IDs` line of `qjob resources`, which shows the configured IDs in the order in which they were entered.

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
| `--gpus <N>` | Number of GPU devices requested. Must be zero or greater. Two or more GPUs are placed on consecutive GPU IDs (see [GPU placement](#gpu-placement)). | `0` |
| `--mem <SIZE>` | Memory request. Accepts values such as `512M`, `8G`, or `1T`. A unitless value is interpreted as bytes. | `1G` |
| `--walltime <HH:MM:SS>` | Maximum wall-clock runtime. `MM:SS` is also accepted. | unlimited, or admin maximum if configured |
| `--priority <LEVEL\|N>` | Scheduling priority. `low`, `normal`, `high`, or integer `0` to `100`. | `normal` (`50`) |
| `--env <KEY[,KEY...]>` | Parsed as a list of environment variable names. The current runner still starts from the scheduler environment; this option is therefore metadata rather than a strict allow-list. | none |

Options given to `qjob submit` (see below) override the same directives in the script.

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

Because the script is run as `bash <script_path>` — neither a login nor an interactive shell — no `/etc/profile`, `~/.profile` or `~/.bashrc` is read by the job itself. The environment qjob builds is therefore the complete environment the script sees.

The following variables are injected into the job environment:

| Variable | Value |
| --- | --- |
| `QJOB_JOB_ID` | 12-character lowercase hexadecimal job ID. |
| `QJOB_JOB_NAME` | Job name. |
| `QJOB_USER` | Submitting username. |
| `CUDA_VISIBLE_DEVICES` | Comma-separated assigned GPU device IDs, or an empty string for CPU-only jobs. |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID`, unless the scheduler's own environment already sets it, in which case that value is kept. With `PCI_BUS_ID`, the device numbers in `CUDA_VISIBLE_DEVICES` match the numbering shown by `nvidia-smi`. |
| `HOME` | Home directory of the submitting user, from the password database. |
| `USER`, `LOGNAME` | Submitting username. |
| `SHELL` | Login shell of the submitting user. |
| `PATH` | The submitting user's login PATH (see below), not the minimal PATH systemd gives the scheduler. |

`PATH` is resolved per user, in this order, and cached for the lifetime of the scheduler process:

1. `QJOB_JOB_PATH` from the scheduler environment, when set. Use it to pin one PATH for every job.
2. The user's own login shell, probed once with `runuser -l <user> -c ...` (or `su -l`, or the shell directly when the scheduler is not root). This applies `/etc/profile` and `~/.profile`, so conda, pyenv and `~/.local/bin` entries are picked up.
3. `ENV_PATH` (`ENV_SUPATH` for root) from `/etc/login.defs`.
4. A built-in fallback, `/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin`.

`~/.local/bin` and `~/bin` are prepended when they exist and are not already present.

The runner otherwise begins from the scheduler process environment, so variables set in the unit file (or in `--env-file`) reach jobs. Variables that describe the scheduler itself are removed first, because inheriting them breaks or leaks into jobs:

- systemd service state: `INVOCATION_ID`, `JOURNAL_STREAM`, `NOTIFY_SOCKET`, `LISTEN_*`, `WATCHDOG_*`, `*_DIRECTORY`.
- The session that started the scheduler: `XDG_*`, `DBUS_SESSION_BUS_ADDRESS`, `SSH_*`, `SUDO_*`, `MAIL`, `PWD`, `OLDPWD`.
- The scheduler's own Python interpreter: `VIRTUAL_ENV`, `PYTHONPATH`, `PYTHONHOME`, `CONDA_*`, `LD_PRELOAD`. Left in place, these hijack `python` inside every job.
- Scheduler configuration and credentials: `QJOB_DB_URL`, `QJOB_DB_POOL_*`, `QJOB_ADMIN_USERS`, `QJOB_TOKEN_PATH`, `QJOB_JOB_PATH`.

## Command Line Usage

### Submit a job

```bash
qjob submit train.sh
```

The CLI validates the script locally and then sends its absolute path and the current working directory to the API server. The script must be accessible on the server where the scheduler runs.

Directives can be overridden from the command line without editing the script. `qjob submit` accepts `--name`, `--cpus`, `--gpus`, `--mem`, `--walltime`, `--priority`, and `--env`, with the same values as the `#QJOB` directives. Options given on the command line take precedence over the script's directives; directives that are not overridden keep the script's values.

```bash
qjob submit train.sh --cpus 4 --mem 8G --priority high
```

### Show job status

```bash
qjob status
qjob status <job_id>
qjob status --state running
qjob status --state queued,running
qjob status --since 24h
qjob status --since 2026-05-01
qjob status --limit 50
qjob status --sort priority
qjob status --all
```

By default, `qjob status` lists the latest 20 jobs for the current OS user sorted by submission time. Use `--since` to show jobs submitted after a relative duration (`24h`, `7d`) or ISO date/time, `--limit N` to show a different number of recent jobs, `--sort submitted|started|finished|priority|user` to change ordering, or `--all` to fetch all matching jobs for the current user.

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

This prints configured totals, current usage, free resources, configured GPU IDs, the GPU IDs that are free right now, the size of the largest block of consecutive free GPU IDs, and the configured maximum walltime.

The free IDs are on the `Free GPU IDs` line (`—` when none is free) and the largest block is on the `GPU block` line. A job that requests GPUs needs a free block at least that large: if `GPU block` is smaller than the request, the job waits even when the `GPUs` row shows enough free devices.

- The `FREE` column is computed from the scheduler's usage counters, whereas `Free GPU IDs` and `GPU block` are computed from the GPUs assigned to running (and cancelling) jobs. The `GPUs` row and the two GPU lines can therefore disagree for a short time, for example right after a job ends and before the scheduler has reconciled resource usage.
- An older qjob server does not return the free GPU fields. Against such a server, the `Free GPU IDs` and `GPU block` lines are not printed.

### Open the dashboard

```bash
qjob dash
qjob dash --refresh 1.0
```

The Resources pane has a `Free IDs` row that lists the GPU IDs that are free right now (`—` when none is free), followed by `max block N`, the size of the largest block of consecutive free IDs. These are the same values as in `qjob resources`, including the caveat that they can briefly disagree with the counter-based GPU free count. The row is not shown when no GPUs are configured or when the server does not report free GPU IDs.

The dashboard gets its data from the REST API, like the other commands, not from the database, so it needs your API token and uses `QJOB_API_URL`. `qjob dash` still connects to the database once at startup, so `QJOB_DB_URL` must be set where it runs, which in practice means the qjob host.

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
4. Dispatches jobs that fit within the available CPU, GPU, and memory counters and can be given a block of consecutive free GPU IDs (see [GPU placement](#gpu-placement)).
5. Applies EASY-style backfilling when the head job is blocked and a safe reservation window can be estimated from running jobs with known walltime.

Effective priority is computed as:

```text
effective_priority = min(100, base_priority + aging_factor * waiting_hours)
```

The current aging factor is 5 priority points per waiting hour.

Resource accounting is stored in the database. The scheduler locks the single `resources` row with `SELECT ... FOR UPDATE` while allocating jobs, which prevents concurrent resource updates from observing stale counters.

### GPU placement

A job that requests GPUs is placed on a block of consecutive GPU IDs, and every ID in the block must be one of the configured GPU IDs. For example, GPUs 2 and 3 form a valid block for a two-GPU job, but GPUs 1 and 3 do not, even when both are free. The block is chosen as follows:

- Among the runs of consecutive free IDs that are long enough, the shortest one is used (best fit), so that longer runs stay available for larger jobs. A one-GPU job therefore takes an isolated free GPU in preference to a GPU inside a longer run.
- If several runs are equally short, the one with the lowest starting ID is used.
- The job gets the first GPUs of that run, in ascending order.

For example, with GPUs 0 to 3 configured and GPU 2 busy, the free runs are `0,1` and `3`: a one-GPU job is placed on GPU 3 and a two-GPU job on GPUs 0 and 1.

Because a block is required, a job can wait even though enough GPUs are free in number. With GPUs 0 to 3 configured and GPUs 1 and 2 busy, GPUs 0 and 3 are free but not consecutive, so a two-GPU job waits. Such a job holds up the jobs behind it like any other blocked head job: only EASY-style backfilling can start them, and a backfilled job needs a block of consecutive free GPU IDs of its own.

Backfilling depends on an estimate of when the blocked job can start, which comes from the walltimes of the running jobs. When no estimate is possible, for example because the running jobs have no walltime, nothing behind the blocked job is started and the whole queue waits. Setting a maximum walltime with `qjob admin set-resources --max-walltime` is therefore recommended: jobs submitted without a walltime then inherit it.

A queued job that the current configuration can never satisfy is skipped instead of blocking the queue. This can happen when the administrator changes the node resources after the job was submitted, for example with a smaller `--cpus` or `--mem`, or with a `--gpu-ids` list that has shorter runs of consecutive IDs, because submission only rejects requests that the configuration at that moment cannot satisfy. The skipped job stays `queued` and starts once the configuration allows it. The scheduler logs a WARNING for it the first time it sees the job (a restarted scheduler logs it again), with a message that begins `Skipping job <job_id>`. The WARNING goes to the scheduler's standard error output. When the scheduler runs under systemd (the `qjob-scheduler.service` unit created by `qjob admin install`), read it with:

```bash
journalctl -u qjob-scheduler
```

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
| `QJOB_JOB_PATH` | PATH given to every job, bypassing the per-user login-shell probe. | none |
| `QJOB_DB_POOL_ENABLED` | Enable SQLAlchemy connection pooling. | `false` |
| `QJOB_DB_POOL_SIZE` | Pool size when pooling is enabled. | `5` |
| `QJOB_DB_MAX_OVERFLOW` | Maximum overflow connections when pooling is enabled. | `5` |

The CLI also accepts `--env-file`, defaulting to `.env`, and loads it if present.

## Operational Notes and Limitations

- qjob is a single-node scheduler. It does not currently schedule across multiple machines.
- Memory is accounted for by request, but not enforced by cgroups or containers. Jobs can still exceed their requested memory unless the deployment adds an external limit.
- GPU assignment is implemented through `CUDA_VISIBLE_DEVICES`; qjob does not enforce GPU memory limits.
- GPU placement does not take topology such as NVLink or NUMA into account. Consecutive GPU IDs are used only as a stand-in for topological closeness and do not guarantee that the GPUs are interconnected.
- The reservation window used for backfilling is estimated from CPU, GPU, and memory counts and ignores the requirement for consecutive GPU IDs, so it is a lower bound on when the blocked job can actually start. Backfilling therefore stays on the conservative side: it never uses a window longer than the real one.
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
