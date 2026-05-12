from __future__ import annotations

import dataclasses
import pathlib
import re
import shlex

# --------------------------------------------------------------------------------------
# Data class for job directives


@dataclasses.dataclass
class JobDirectives:
    # autopep8: off
    name:         str | None                    = None
    cpus:         int                           = 1
    gpus:         int                           = 0
    mem_mb:       int                           = 1024        # 1 GB
    walltime_sec: int | None                    = None        # infinit
    priority:     int                           = 50          # 0–100, default 50
    env_keys:     list[str]                     = dataclasses.field(default_factory=list)
    script_path:  str | None                    = None        # Set by parse
    # autopep8: on


# --------------------------------------------------------------------------------------
# Constants and regex patterns

_DIRECTIVE_RE = re.compile(r"^#QJOB\s+(.+)$")
_PRIORITY_MAP = {"low": 20, "normal": 50, "high": 80}
_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Public API

def parse_script(path: str | pathlib.Path) -> JobDirectives:
    """
    Read a shell script, parse its #QJOB directives, and return the result.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the target script.

    Returns
    -------
    JobDirectives
        Parsed result. Fields without directives keep their default values.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DirectiveParseError
        If a directive value is invalid.
    """

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    raw_lines = _extract_directive_lines(path)
    args = _lines_to_argv(raw_lines)
    directives = _parse_argv(args)
    directives.script_path = str(path.resolve())

    return directives

# --------------------------------------------------------------------------------------
# Private helper functions


def _extract_directive_lines(path: pathlib.Path) -> list[str]:
    """Extract only #QJOB lines from the leading contiguous comment block."""

    lines = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            # Skip the shebang.
            if line.startswith("#!"):
                continue
            # Stop the leading block at the first blank or non-comment line.
            if not line.startswith("#"):
                break
            m = _DIRECTIVE_RE.match(line)
            if m:
                lines.append(m.group(1))

    return lines


def _lines_to_argv(lines: list[str]) -> list[str]:
    """
    Join argument fragments from multiple lines and split them with shlex.

    Example: ["--name train", "--cpus 4"] -> ["--name", "train", "--cpus", "4"]
    """

    combined = " ".join(lines)
    return shlex.split(combined)


def _parse_argv(argv: list[str]) -> JobDirectives:
    """Convert a list of key-value pairs into JobDirectives."""

    d = JobDirectives()
    it = iter(argv)

    for token in it:
        if not token.startswith("--"):
            raise DirectiveParseError(f"Unexpected token: {token!r}")

        key = token.lstrip("-")
        val = next(it, None)
        if val is None or val.startswith("--"):
            raise DirectiveParseError(f"--{key} requires a value")

        match key:
            case "name":
                d.name = val

            case "cpus":
                d.cpus = _parse_positive_int(key, val)

            case "gpus":
                d.gpus = _parse_non_negative_int(key, val)

            case "mem":
                d.mem_mb = _parse_mem(val)

            case "walltime":
                d.walltime_sec = _parse_walltime(val)

            case "priority":
                d.priority = _parse_priority(val)

            case "env":
                d.env_keys = [k.strip() for k in val.split(",") if k.strip()]

            case _:
                raise DirectiveParseError(f"Unknown directive: --{key}")

    return d


def _parse_positive_int(key: str, val: str) -> int:
    try:
        n = int(val)
    except ValueError:
        raise DirectiveParseError(f"--{key} must be an integer, got {val!r}")
    if n <= 0:
        raise DirectiveParseError(f"--{key} must be > 0, got {n}")
    return n


def _parse_non_negative_int(key: str, val: str) -> int:
    try:
        n = int(val)
    except ValueError:
        raise DirectiveParseError(f"--{key} must be an integer, got {val!r}")
    if n < 0:
        raise DirectiveParseError(f"--{key} must be >= 0, got {n}")
    return n


def _parse_mem(val: str) -> int:
    """Convert values such as '16G', '512M', and '1024' bytes to megabytes."""

    m = _MEM_RE.match(val.strip())
    if not m:
        raise DirectiveParseError(f"Invalid --mem value: {val!r}  (e.g. 8G, 512M, 2048)")
    amount = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) else ""
    multipliers = {"": 1/1024/1024, "K": 1/1024, "M": 1, "G": 1024, "T": 1024**2}
    return max(1, int(amount * multipliers[unit]))


def _parse_walltime(val: str) -> int:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""

    parts = val.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        raise DirectiveParseError(f"Invalid --walltime: {val!r}  (e.g. 02:30:00)")

    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, *parts
    else:
        raise DirectiveParseError(f"Invalid --walltime format: {val!r}")

    if not (0 <= m < 60 and 0 <= s < 60):
        raise DirectiveParseError(f"Invalid --walltime value: {val!r}. Minutes and seconds must be 0–59.")

    return h * 3600 + m * 60 + s


def _parse_priority(val: str) -> int:
    """Accept 'low', 'normal', 'high', or an integer from 0 to 100."""

    if val in _PRIORITY_MAP:
        return _PRIORITY_MAP[val]
    try:
        n = int(val)
    except ValueError:
        raise DirectiveParseError(
            f"--priority must be low/normal/high or 0–100, got {val!r}"
        )
    if not 0 <= n <= 100:
        raise DirectiveParseError(f"--priority must be 0–100, got {n}")
    return n

# --------------------------------------------------------------------------------------
# Custom exception for directive parsing errors


class DirectiveParseError(ValueError):
    pass
