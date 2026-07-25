# SPDX-License-Identifier: Apache-2.0

import json

from tests.pd_transfer.parse_pd_trace import build_timelines


def test_transfer_profile_is_merged_into_request_timeline(tmp_path):
    records = [
        {
            "event": "pull_decode_kv_allocated",
            "request_id": "req",
            "perf_ns": 1_000_000,
        },
        {
            "event": "pull_transfer_profile",
            "request_id": "req-0",
            "perf_ns": 9_000_000,
            "kv_load_start_perf_ns": 2_000_000,
            "connector_finished_perf_ns": 8_000_000,
            "desc_build_ms": 0.25,
            "submit_to_done_observed_ms": 4.5,
            "total_bytes": 8192,
            "poll_rounds": 3,
            "paired_forward_run_count": 1,
            "paired_reverse_run_count": 2,
            "forward_only_range_count": 3,
            "reverse_only_range_count": 5,
            "theoretical_merged_range_count": 4,
            "block_pair_stats_ms": 0.125,
            "reverse_block_pair_canonicalization_enabled": True,
            "canonicalized_reverse_run_count": 2,
            "canonicalized_reverse_block_count": 16,
        },
        {
            "event": "decode_remote_kv_ready",
            "request_id": "req",
            "perf_ns": 10_000_000,
        },
        {
            "event": "decode_remote_kv_schedulable",
            "request_id": "req",
            "perf_ns": 11_000_000,
        },
    ]
    for record in records:
        record["ts_ns"] = record["perf_ns"]
    trace_path = tmp_path / "decode.trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    case_records = [
        {
            "event": "bench_case_start",
            "case_id": "case-8k-c1",
            "ts_ns": 0,
            "mode": "pull",
            "sleep_ms": 0,
            "input_len": 8192,
            "output_len": 1,
            "concurrency": 1,
            "num_prompts": 20,
        },
        {
            "event": "bench_case_end",
            "case_id": "case-8k-c1",
            "ts_ns": 12_000_000,
        },
    ]
    (tmp_path / "case.trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in case_records),
        encoding="utf-8",
    )

    rows = build_timelines(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == "req"
    assert row["desc_build_ms"] == 0.25
    assert row["submit_to_done_observed_ms"] == 4.5
    assert row["total_bytes"] == 8192
    assert row["poll_rounds"] == 3
    assert row["paired_forward_run_count"] == 1
    assert row["paired_reverse_run_count"] == 2
    assert row["forward_only_range_count"] == 3
    assert row["reverse_only_range_count"] == 5
    assert row["theoretical_merged_range_count"] == 4
    assert row["block_pair_stats_ms"] == 0.125
    assert row["reverse_block_pair_canonicalization_enabled"] is True
    assert row["canonicalized_reverse_run_count"] == 2
    assert row["canonicalized_reverse_block_count"] == 16
    assert row["case_id"] == "case-8k-c1"
    assert row["input_len"] == 8192
    assert row["concurrency"] == 1
    assert row["kv_alloc_to_load_ms"] == 1.0
    assert row["kv_load_to_remote_ready_ms"] == 8.0
    assert row["connector_finished_to_remote_ready_ms"] == 2.0
    assert row["remote_ready_to_schedulable_ms"] == 1.0
