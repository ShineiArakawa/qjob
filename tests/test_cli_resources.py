from __future__ import annotations

import qjob.cli.main as cli_main
import qjob.cli.service as service

# --------------------------------------------------------------------------------------
# Helpers

_RESOURCE_PAYLOAD: dict = {
    "total_cpus": 32,
    "total_gpus": 4,
    "total_mem_mb": 131072,
    "max_walltime_sec": None,
    "gpu_ids": [0, 1, 2, 3],
    "used_cpus": 8,
    "used_gpus": 2,
    "used_mem_mb": 32768,
}


def _make_resource_info(**kwargs) -> service.ResourceInfo:
    """Return a ResourceInfo with sensible defaults overridden by *kwargs*."""

    defaults = dict(
        total_cpus=32,
        total_gpus=4,
        total_mem_mb=131072,
        max_walltime_sec=None,
        used_cpus=8,
        used_gpus=2,
        used_mem_mb=32768,
        gpu_ids=[0, 1, 2, 3],
    )
    defaults.update(kwargs)
    return service.ResourceInfo(**defaults)


# --------------------------------------------------------------------------------------
# _print_resources


class TestPrintResources:
    """_print_resources() shows the free GPU IDs and the largest free block."""

    def test_prints_free_gpu_ids_and_gpu_block(self, capsys):
        info = _make_resource_info(free_gpu_ids=[0, 2, 3], max_contiguous_free_gpus=2)

        cli_main._print_resources(info)

        lines = capsys.readouterr().out.splitlines()
        assert "Free GPU IDs       0,2,3" in lines
        assert "GPU block              2  (largest run of consecutive free GPU IDs)" in lines

    def test_free_gpu_lines_follow_the_gpu_ids_line(self, capsys):
        info = _make_resource_info(free_gpu_ids=[0, 2, 3], max_contiguous_free_gpus=2)

        cli_main._print_resources(info)

        lines = capsys.readouterr().out.splitlines()
        gpu_ids_at = next(i for i, ln in enumerate(lines) if ln.startswith("GPU IDs"))
        assert lines[gpu_ids_at + 1].startswith("Free GPU IDs")
        assert lines[gpu_ids_at + 2].startswith("GPU block")

    def test_no_free_gpu_ids_prints_dash_and_zero_block(self, capsys):
        info = _make_resource_info(free_gpu_ids=[], max_contiguous_free_gpus=0)

        cli_main._print_resources(info)

        lines = capsys.readouterr().out.splitlines()
        assert "Free GPU IDs           —" in lines
        assert "GPU block              0  (largest run of consecutive free GPU IDs)" in lines

    def test_omits_free_gpu_lines_when_fields_are_none(self, capsys):
        info = _make_resource_info()

        cli_main._print_resources(info)

        out = capsys.readouterr().out
        assert "Free GPU IDs" not in out
        assert "GPU block" not in out
        assert "GPU IDs" in out

    def test_older_server_payload_prints_without_free_gpu_lines(self, capsys):
        info = service._parse_resource(dict(_RESOURCE_PAYLOAD))

        cli_main._print_resources(info)

        out = capsys.readouterr().out
        assert "Free GPU IDs" not in out
        assert "GPU block" not in out
        assert "Memory(MB)" in out


# --------------------------------------------------------------------------------------
# service._parse_resource


class TestParseResource:
    """_parse_resource() reads the free GPU fields only when the server sends them."""

    def test_missing_keys_parse_as_none(self):
        info = service._parse_resource(dict(_RESOURCE_PAYLOAD))

        assert info.free_gpu_ids is None
        assert info.max_contiguous_free_gpus is None

    def test_present_keys_are_parsed(self):
        data = {**_RESOURCE_PAYLOAD, "free_gpu_ids": [0, 2, 3], "max_contiguous_free_gpus": 2}

        info = service._parse_resource(data)

        assert info.free_gpu_ids == [0, 2, 3]
        assert info.max_contiguous_free_gpus == 2

    def test_empty_and_zero_are_not_confused_with_missing(self):
        data = {**_RESOURCE_PAYLOAD, "free_gpu_ids": [], "max_contiguous_free_gpus": 0}

        info = service._parse_resource(data)

        assert info.free_gpu_ids == []
        assert info.max_contiguous_free_gpus == 0
