from __future__ import annotations

import collections.abc

# --------------------------------------------------------------------------------------
# Contiguous GPU blocks
#
# A contiguous block is a run of consecutive integer GPU IDs such as 2, 3, 4.
# Multi-GPU jobs are only ever placed on a single contiguous block.


def contiguous_runs(ids: collections.abc.Iterable[int]) -> list[list[int]]:
    """
    Split *ids* into maximal runs of consecutive integers.

    Parameters
    ----------
    ids : Iterable[int]
        GPU IDs in any order.  Duplicates are ignored.

    Returns
    -------
    list[list[int]]
        Runs in ascending order, each sorted ascending.
    """

    runs: list[list[int]] = []
    for gpu_id in sorted(set(ids)):
        if runs and gpu_id == runs[-1][-1] + 1:
            runs[-1].append(gpu_id)
        else:
            runs.append([gpu_id])
    return runs


def find_contiguous_block(
    free_ids: collections.abc.Iterable[int],
    count:    int,
) -> list[int] | None:
    """
    Choose *count* consecutive GPU IDs from *free_ids*.

    The block is taken from the shortest run that is long enough (best fit),
    so that longer runs stay available for larger jobs.  Ties are broken by
    the lowest starting ID, and the block starts at the beginning of the run.

    Parameters
    ----------
    free_ids : Iterable[int]
        Currently free GPU IDs in any order.
    count : int
        Number of GPUs requested.

    Returns
    -------
    list[int] | None
        The chosen GPU IDs in ascending order, ``[]`` when *count* is zero or
        negative, or ``None`` when no run of *count* consecutive IDs is free.
    """

    if count <= 0:
        return []

    candidates = [run for run in contiguous_runs(free_ids) if len(run) >= count]
    if not candidates:
        return None

    best = min(candidates, key=lambda run: (len(run), run[0]))
    return best[:count]


def max_contiguous_run(ids: collections.abc.Iterable[int]) -> int:
    """
    Return the length of the longest run of consecutive integers in *ids*.

    Parameters
    ----------
    ids : Iterable[int]
        GPU IDs in any order.

    Returns
    -------
    int
        Length of the longest run, or 0 when *ids* is empty.
    """

    return max((len(run) for run in contiguous_runs(ids)), default=0)
