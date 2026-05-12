from __future__ import annotations

import pathlib
import textwrap

import pytest

import qjob.core.parser as parser

# --------------------------------------------------------------------------------------
# Fixtures


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
        A function ``make(content: str) -> pathlib.Path`` that dedents *content*,
        writes it to ``tmp_path/job.sh``, and returns the path.
    """

    def _make(content: str) -> pathlib.Path:
        p = tmp_path / "job.sh"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _make


# --------------------------------------------------------------------------------------
# Happy-path tests


class TestFullDirectives:
    """All supported directives are present and valid."""

    def test_name(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --name train-resnet
            python train.py
        """)
        assert parser.parse_script(p).name == "train-resnet"

    def test_cpus(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --cpus 8
            python train.py
        """)
        assert parser.parse_script(p).cpus == 8

    def test_gpus(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --gpus 2
            python train.py
        """)
        assert parser.parse_script(p).gpus == 2

    def test_walltime_hhmmss(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --walltime 02:30:00
            python train.py
        """)
        assert parser.parse_script(p).walltime_sec == 9000

    def test_walltime_mmss(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --walltime 50:00
            python train.py
        """)
        assert parser.parse_script(p).walltime_sec == 3000

    def test_priority_word_low(self, make_script):
        p = make_script("#QJOB --priority low\npython x.py\n")
        assert parser.parse_script(p).priority == 20

    def test_priority_word_normal(self, make_script):
        p = make_script("#QJOB --priority normal\npython x.py\n")
        assert parser.parse_script(p).priority == 50

    def test_priority_word_high(self, make_script):
        p = make_script("#QJOB --priority high\npython x.py\n")
        assert parser.parse_script(p).priority == 80

    def test_priority_numeric(self, make_script):
        p = make_script("#QJOB --priority 75\npython x.py\n")
        assert parser.parse_script(p).priority == 75

    def test_env_keys(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --env CUDA_VISIBLE_DEVICES,HOME,PATH
            python train.py
        """)
        assert parser.parse_script(p).env_keys == ["CUDA_VISIBLE_DEVICES", "HOME", "PATH"]

    def test_script_path_is_set(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        d = parser.parse_script(p)
        assert d.script_path == str(p.resolve())

    def test_all_directives_combined(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --name train-resnet
            #QJOB --cpus 4
            #QJOB --gpus 2
            #QJOB --mem 16G
            #QJOB --walltime 02:30:00
            #QJOB --priority high
            #QJOB --env CUDA_VISIBLE_DEVICES,HOME
            python train.py
        """)
        d = parser.parse_script(p)
        assert d.name == "train-resnet"
        assert d.cpus == 4
        assert d.gpus == 2
        assert d.mem_mb == 16384
        assert d.walltime_sec == 9000
        assert d.priority == 80
        assert d.env_keys == ["CUDA_VISIBLE_DEVICES", "HOME"]


# --------------------------------------------------------------------------------------
# Default value tests


class TestDefaults:
    """Fields omitted from directives must fall back to documented defaults."""

    def test_cpus_default(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).cpus == 1

    def test_gpus_default(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).gpus == 0

    def test_mem_mb_default(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 1024

    def test_priority_default(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).priority == 50

    def test_walltime_default_is_none(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).walltime_sec is None

    def test_name_default_is_none(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).name is None

    def test_env_keys_default_is_empty(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert parser.parse_script(p).env_keys == []


# --------------------------------------------------------------------------------------
# Memory unit conversion tests


class TestMemParsing:
    """_parse_mem must convert all supported unit suffixes to megabytes."""

    def test_megabytes(self, make_script):
        p = make_script("#QJOB --mem 512M\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 512

    def test_gigabytes(self, make_script):
        p = make_script("#QJOB --mem 4G\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 4096

    def test_gigabytes_decimal(self, make_script):
        p = make_script("#QJOB --mem 1.5G\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 1536

    def test_kilobytes(self, make_script):
        p = make_script("#QJOB --mem 2048K\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 2

    def test_terabytes(self, make_script):
        p = make_script("#QJOB --mem 1T\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 1024 ** 2

    def test_case_insensitive(self, make_script):
        p = make_script("#QJOB --mem 8g\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 8192

    def test_with_b_suffix(self, make_script):
        p = make_script("#QJOB --mem 4GB\npython x.py\n")
        assert parser.parse_script(p).mem_mb == 4096


# --------------------------------------------------------------------------------------
# Directive block boundary tests


class TestDirectiveBlockBoundary:
    """#QJOB lines after a non-comment line must be silently ignored."""

    def test_stops_at_non_comment_line(self, make_script):
        p = make_script("""\
            #QJOB --cpus 4
            echo hello
            #QJOB --gpus 99
            python x.py
        """)
        d = parser.parse_script(p)
        assert d.cpus == 4
        assert d.gpus == 0  # Ignored because the block ended at 'echo hello'.

    def test_stops_at_blank_line(self, make_script):
        p = make_script("""\
            #QJOB --cpus 4

            #QJOB --gpus 99
            python x.py
        """)
        d = parser.parse_script(p)
        assert d.cpus == 4
        assert d.gpus == 0  # Blank line terminates the leading comment block.

    def test_shebang_is_skipped(self, make_script):
        p = make_script("""\
            #!/bin/bash
            #QJOB --cpus 4
            python x.py
        """)
        assert parser.parse_script(p).cpus == 4

    def test_non_qjob_comments_are_ignored(self, make_script):
        p = make_script("""\
            # This is a regular comment.
            #QJOB --cpus 4
            python x.py
        """)
        assert parser.parse_script(p).cpus == 4


# --------------------------------------------------------------------------------------
# Error tests: file system


class TestFileErrors:
    """parser.parse_script raises FileNotFoundError for missing files."""

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Script not found"):
            parser.parse_script("/nonexistent/path/job.sh")

    def test_path_as_string(self, make_script):
        p = make_script("#QJOB --cpus 2\npython x.py\n")
        d = parser.parse_script(str(p))  # Accepts str as well as Path.
        assert d.cpus == 2


# --------------------------------------------------------------------------------------
# Error tests: invalid directive values


class TestInvalidValues:
    """parser.DirectiveParseError is raised for any malformed directive."""

    def test_cpus_zero(self, make_script):
        p = make_script("#QJOB --cpus 0\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--cpus"):
            parser.parse_script(p)

    def test_cpus_negative(self, make_script):
        p = make_script("#QJOB --cpus -1\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--cpus"):
            parser.parse_script(p)

    def test_cpus_non_integer(self, make_script):
        p = make_script("#QJOB --cpus two\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--cpus"):
            parser.parse_script(p)

    def test_gpus_negative(self, make_script):
        p = make_script("#QJOB --gpus -1\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--gpus"):
            parser.parse_script(p)

    def test_mem_unknown_unit(self, make_script):
        p = make_script("#QJOB --mem 16X\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--mem"):
            parser.parse_script(p)

    def test_mem_empty_string(self, make_script):
        p = make_script('#QJOB --mem ""\npython x.py\n')
        with pytest.raises(parser.DirectiveParseError, match="--mem"):
            parser.parse_script(p)

    def test_walltime_bad_format(self, make_script):
        p = make_script("#QJOB --walltime 2hours\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--walltime"):
            parser.parse_script(p)

    def test_walltime_invalid_minutes(self, make_script):
        p = make_script("#QJOB --walltime 01:99:00\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError):
            parser.parse_script(p)

    def test_walltime_invalid_seconds(self, make_script):
        p = make_script("#QJOB --walltime 01:00:99\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError):
            parser.parse_script(p)

    def test_priority_out_of_range(self, make_script):
        p = make_script("#QJOB --priority 101\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--priority"):
            parser.parse_script(p)

    def test_priority_negative(self, make_script):
        p = make_script("#QJOB --priority -1\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--priority"):
            parser.parse_script(p)

    def test_priority_unknown_word(self, make_script):
        p = make_script("#QJOB --priority urgent\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--priority"):
            parser.parse_script(p)

    def test_unknown_directive(self, make_script):
        p = make_script("#QJOB --unknown value\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="Unknown directive"):
            parser.parse_script(p)

    def test_directive_missing_value(self, make_script):
        p = make_script("#QJOB --cpus\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="--cpus requires a value"):
            parser.parse_script(p)

    def test_bare_token_without_dashes(self, make_script):
        p = make_script("#QJOB cpus 4\npython x.py\n")
        with pytest.raises(parser.DirectiveParseError, match="Unexpected token"):
            parser.parse_script(p)


# --------------------------------------------------------------------------------------
# Return type test


class TestReturnType:
    """parser.parse_script must always return a parser.JobDirectives instance."""

    def test_returns_job_directives(self, make_script):
        p = make_script("#!/bin/bash\npython x.py\n")
        assert isinstance(parser.parse_script(p), parser.JobDirectives)
