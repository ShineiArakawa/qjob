from __future__ import annotations

import itertools

import qjob.core.gpu_blocks as gpu_blocks

# --------------------------------------------------------------------------------------
# Helpers


def _all_subsets(universe: range) -> list[tuple[int, ...]]:
    """Return every subset of *universe*, including the empty one, as a tuple."""

    return [
        subset
        for size in range(len(universe) + 1)
        for subset in itertools.combinations(universe, size)
    ]


# --------------------------------------------------------------------------------------
# contiguous_runs


class TestContiguousRuns:
    """contiguous_runs() splits GPU IDs into maximal runs of consecutive integers."""

    def test_empty_input_gives_no_runs(self):
        assert gpu_blocks.contiguous_runs([]) == []

    def test_unsorted_input_with_duplicate_is_normalised(self):
        runs = gpu_blocks.contiguous_runs([3, 1, 2, 7, 5, 6, 1])
        assert runs == [[1, 2, 3], [5, 6, 7]]

    def test_isolated_ids_form_single_element_runs(self):
        assert gpu_blocks.contiguous_runs([0, 2, 4]) == [[0], [2], [4]]


# --------------------------------------------------------------------------------------
# find_contiguous_block


class TestFindContiguousBlock:
    """find_contiguous_block() picks the best-fit run of consecutive free GPU IDs."""

    def test_zero_count_returns_empty_list(self):
        assert gpu_blocks.find_contiguous_block([0, 1], 0) == []

    def test_negative_count_returns_empty_list(self):
        assert gpu_blocks.find_contiguous_block([0, 1], -1) == []

    def test_zero_count_with_no_free_ids_returns_empty_list(self):
        assert gpu_blocks.find_contiguous_block([], 0) == []

    def test_no_free_ids_returns_none(self):
        assert gpu_blocks.find_contiguous_block([], 1) is None

    def test_block_starts_at_beginning_of_run(self):
        assert gpu_blocks.find_contiguous_block([0, 1, 2, 3], 2) == [0, 1]

    def test_prefers_shortest_run_that_fits(self):
        assert gpu_blocks.find_contiguous_block([0, 1, 2, 4, 5], 2) == [4, 5]

    def test_single_gpu_prefers_isolated_gpu(self):
        assert gpu_blocks.find_contiguous_block([0, 2, 3, 4, 5], 1) == [0]

    def test_single_gpu_prefers_isolated_gpu_over_lowest_id(self):
        assert gpu_blocks.find_contiguous_block([0, 1, 3], 1) == [3]

    def test_equal_length_runs_pick_lowest_starting_id(self):
        assert gpu_blocks.find_contiguous_block([0, 1, 3, 4], 2) == [0, 1]

    def test_unsorted_input_returns_ascending_block(self):
        assert gpu_blocks.find_contiguous_block([5, 4, 3], 3) == [3, 4, 5]

    def test_no_run_long_enough_returns_none(self):
        assert gpu_blocks.find_contiguous_block([0, 2, 5], 2) is None

    def test_count_exceeding_free_ids_returns_none(self):
        assert gpu_blocks.find_contiguous_block([0, 1], 3) is None

    def test_duplicate_and_unsorted_input(self):
        assert gpu_blocks.find_contiguous_block([3, 3, 2], 2) == [2, 3]

    def test_accepts_set(self):
        assert gpu_blocks.find_contiguous_block({0, 1, 2, 3}, 2) == [0, 1]


# --------------------------------------------------------------------------------------
# max_contiguous_run


class TestMaxContiguousRun:
    """max_contiguous_run() returns the length of the longest run of consecutive IDs."""

    def test_empty_input_gives_zero(self):
        assert gpu_blocks.max_contiguous_run([]) == 0

    def test_isolated_ids_give_one(self):
        assert gpu_blocks.max_contiguous_run([0, 2, 5]) == 1

    def test_longest_run_wins(self):
        assert gpu_blocks.max_contiguous_run([0, 1, 3, 4, 5]) == 3


# --------------------------------------------------------------------------------------
# Exhaustive check over every subset of GPUs 0-5


class TestExhaustiveSubsets:
    """find_contiguous_block() and max_contiguous_run() agree on every subset of 0-5."""

    def test_block_found_exactly_when_a_long_enough_run_exists(self):
        for subset in _all_subsets(range(6)):
            for count in range(1, 7):
                found = gpu_blocks.find_contiguous_block(subset, count) is not None
                expected = count <= gpu_blocks.max_contiguous_run(subset)
                assert found == expected, f"subset={subset}, count={count}"

    def test_found_block_is_consecutive_free_ids_of_requested_length(self):
        for subset in _all_subsets(range(6)):
            for count in range(1, 7):
                block = gpu_blocks.find_contiguous_block(subset, count)
                if block is None:
                    continue
                context = f"subset={subset}, count={count}, block={block}"
                assert len(block) == count, context
                assert all(b - a == 1 for a, b in itertools.pairwise(block)), context
                assert set(block) <= set(subset), context
