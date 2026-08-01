# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tests.pd_transfer.nixl_pack_scatter_bench import (
    SUPPORTED_PATTERNS,
    _build_mapping,
    _chunk_ranges,
    _count_forward_ranges,
    _descriptor_indices,
)


@pytest.mark.parametrize("pattern", SUPPORTED_PATTERNS)
def test_mapping_is_a_valid_pair_preserving_workload(pattern):
    workload = _build_mapping(
        pattern=pattern,
        request_blocks=17,
        physical_blocks=64,
        seed=1234,
        mixed_run_length=4,
    )

    assert workload.pattern == pattern
    assert workload.request_blocks == 17
    assert len(workload.local_block_ids) == 17
    assert len(workload.remote_block_ids) == 17
    assert len(set(workload.local_block_ids)) == 17
    assert len(set(workload.remote_block_ids)) == 17
    assert min(workload.local_block_ids) >= 0
    assert max(workload.local_block_ids) < 64
    assert min(workload.remote_block_ids) >= 0
    assert max(workload.remote_block_ids) < 64
    assert workload.forward_range_count == _count_forward_ranges(
        workload.local_block_ids, workload.remote_block_ids
    )


def test_forward_and_reverse_have_expected_forward_only_ranges():
    forward = _build_mapping("forward", 8, 64, 1, 4)
    reverse = _build_mapping("reverse", 8, 64, 1, 4)

    assert forward.forward_range_count == 1
    assert reverse.forward_range_count == 8


def test_descriptor_indices_are_region_major():
    indices = _descriptor_indices(
        block_ids=(3, 1), region_count=3, physical_blocks=10
    )

    np.testing.assert_array_equal(indices, [3, 1, 13, 11, 23, 21])
    assert indices.dtype == np.int32


@pytest.mark.parametrize(
    ("request_blocks", "blocks_per_chunk", "expected"),
    [
        (1, 8, [(0, 1)]),
        (8, 8, [(0, 8)]),
        (9, 8, [(0, 8), (8, 9)]),
        (17, 8, [(0, 8), (8, 16), (16, 17)]),
    ],
)
def test_chunk_ranges_cover_each_block_once(
    request_blocks, blocks_per_chunk, expected
):
    ranges = _chunk_ranges(request_blocks, blocks_per_chunk)

    assert ranges == expected
    flattened = [
        block for start, end in ranges for block in range(start, end)
    ]
    assert flattened == list(range(request_blocks))


def test_invalid_mapping_and_chunk_arguments_fail():
    with pytest.raises(ValueError, match="request_blocks"):
        _build_mapping("forward", 65, 64, 1, 4)
    with pytest.raises(ValueError, match="unsupported"):
        _build_mapping("unknown", 4, 64, 1, 4)
    with pytest.raises(ValueError, match="positive"):
        _chunk_ranges(4, 0)
