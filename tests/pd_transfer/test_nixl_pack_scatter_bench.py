# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tests.pd_transfer.nixl_pack_scatter_bench import (
    SUPPORTED_PATTERNS,
    _base_sample,
    _build_controlled_mapping,
    _build_mapping,
    _chunk_ranges,
    _count_forward_ranges,
    _descriptor_indices,
    _summaries,
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


@pytest.mark.parametrize(
    ("request_blocks", "runs_per_region"),
    [(1, 1), (8, 1), (8, 2), (8, 3), (8, 8), (17, 4)],
)
def test_controlled_mapping_has_exact_forward_run_count(
    request_blocks, runs_per_region
):
    workload = _build_controlled_mapping(
        request_blocks=request_blocks,
        physical_blocks=64,
        runs_per_region=runs_per_region,
    )

    assert workload.pattern == "controlled"
    assert workload.runs_per_region == runs_per_region
    assert workload.forward_range_count == runs_per_region
    assert len(workload.local_block_ids) == request_blocks
    assert len(workload.remote_block_ids) == request_blocks
    assert len(set(workload.local_block_ids)) == request_blocks
    assert len(set(workload.remote_block_ids)) == request_blocks
    assert min(workload.local_block_ids) >= 0
    assert max(workload.local_block_ids) < 64
    assert min(workload.remote_block_ids) >= 0
    assert max(workload.remote_block_ids) < 64


def test_controlled_mapping_changes_ranges_not_bytes_or_descriptors():
    compact = _build_controlled_mapping(8, 64, 1)
    fragmented = _build_controlled_mapping(8, 64, 8)

    compact_sample = _base_sample(compact, "direct", 72, 32 * 1024)
    fragmented_sample = _base_sample(fragmented, "direct", 72, 32 * 1024)

    assert compact_sample["total_bytes"] == fragmented_sample["total_bytes"]
    assert (
        compact_sample["direct_descriptor_count"]
        == fragmented_sample["direct_descriptor_count"]
        == 72 * 8
    )
    assert compact_sample["estimated_direct_backend_ranges"] == 72
    assert fragmented_sample["estimated_direct_backend_ranges"] == 72 * 8


def test_summaries_keep_controlled_run_counts_separate():
    samples = []
    for runs_per_region in (1, 2):
        workload = _build_controlled_mapping(4, 64, runs_per_region)
        sample = _base_sample(workload, "direct", 72, 32 * 1024)
        sample.update(
            {
                "chunks": 1,
                "effective_gbps": 1.0,
                "source_index_build_ns": 1,
                "destination_index_build_ns": 1,
                "pack_control_wait_ns": 0,
                "pack_gpu_ns": 0,
                "make_xfer_ns": 1,
                "post_xfer_ns": 1,
                "poll_xfer_ns": 1,
                "transfer_total_ns": 2,
                "scatter_gpu_ns": 0,
                "data_path_ns": 2,
                "wall_ns": 3,
            }
        )
        samples.append(sample)

    summaries = _summaries(samples)

    assert [summary["runs_per_region"] for summary in summaries] == [1, 2]
    assert all(summary["sample_count"] == 1 for summary in summaries)


def test_descriptor_indices_are_region_major():
    indices = _descriptor_indices(block_ids=(3, 1), region_count=3, physical_blocks=10)

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
def test_chunk_ranges_cover_each_block_once(request_blocks, blocks_per_chunk, expected):
    ranges = _chunk_ranges(request_blocks, blocks_per_chunk)

    assert ranges == expected
    flattened = [block for start, end in ranges for block in range(start, end)]
    assert flattened == list(range(request_blocks))


def test_invalid_mapping_and_chunk_arguments_fail():
    with pytest.raises(ValueError, match="request_blocks"):
        _build_mapping("forward", 65, 64, 1, 4)
    with pytest.raises(ValueError, match="unsupported"):
        _build_mapping("unknown", 4, 64, 1, 4)
    with pytest.raises(ValueError, match="runs_per_region"):
        _build_controlled_mapping(4, 64, 5)
    with pytest.raises(ValueError, match="physical_blocks"):
        _build_controlled_mapping(4, 4, 4)
    with pytest.raises(ValueError, match="positive"):
        _chunk_ranges(4, 0)
