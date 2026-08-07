# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "pd_transfer"
    / "nixl_p2d1"
    / "nixl_p2d1_microbench.py"
)
_SPEC = importlib.util.spec_from_file_location("nixl_p2d1_microbench", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def test_packed_chunks_preserve_descriptor_order_and_exact_bytes() -> None:
    chunks = _BENCH._packed_chunks(
        indices=[0, 2, 1],
        total_bytes=10,
        descriptor_bytes=4,
        staging_bytes=8,
    )

    assert chunks == [([0, 2], 6), ([1], 4)]
    assert [index for chunk, _ in chunks for index in chunk] == [0, 2, 1]
    assert sum(used_bytes for _, used_bytes in chunks) == 10


def test_packed_chunks_reject_descriptor_larger_than_staging() -> None:
    with pytest.raises(ValueError, match="descriptor needs 8 bytes"):
        _BENCH._packed_chunks(
            indices=[0],
            total_bytes=8,
            descriptor_bytes=8,
            staging_bytes=4,
        )


def test_parse_sizes_keeps_all_as_one_descriptor() -> None:
    assert _BENCH._parse_sizes("16KiB,all", 64 << 10) == [16 << 10, 64 << 10]
