# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.pd_transfer.compare_pd_generalization_perf import (
    METRICS, _summarize_diagnostic_traces, analyze_results)


def _write_result(
    root,
    *,
    repetition,
    variant,
    request_rate,
    completed=100,
    input_lens=None,
):
    path = (
        root
        / "performance"
        / variant
        / f"rps-{request_rate}"
        / f"rep-{repetition}"
    )
    path.mkdir(parents=True)
    multiplier = 1.1 if variant == "on" else 1.0
    data = {
        "phase": "performance",
        "variant": variant,
        "repetition": str(repetition),
        "configured_request_rate": str(request_rate),
        "num_prompts": 100,
        "completed": completed,
        "errors": [""] * 100,
        "input_lens": input_lens or [128] * 100,
        "output_lens": [32] * 100,
    }
    for metric, higher_is_better in METRICS:
        baseline = 100.0
        data[metric] = (
            baseline * multiplier
            if higher_is_better
            else baseline / multiplier
        )
    (path / "result.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def test_analyze_results_pairs_variants_and_orients_improvement(tmp_path):
    for repetition in (1, 2):
        _write_result(
            tmp_path,
            repetition=repetition,
            variant="off",
            request_rate="2",
        )
        _write_result(
            tmp_path,
            repetition=repetition,
            variant="on",
            request_rate="2",
        )

    pairwise, summary, problems = analyze_results(tmp_path)

    assert problems == []
    throughput = next(
        row for row in pairwise
        if row["metric"] == "request_throughput"
    )
    latency = next(
        row for row in pairwise if row["metric"] == "p99_ttft_ms"
    )
    assert throughput["improvement_pct"] == pytest.approx(10.0)
    assert round(latency["improvement_pct"], 8) == round(
        (1 - 1 / 1.1) * 100, 8)
    assert {
        row["paired_repetitions"] for row in summary
    } == {2}


def test_analyze_results_reports_missing_variant(tmp_path):
    _write_result(
        tmp_path,
        repetition=1,
        variant="off",
        request_rate="1",
    )

    pairwise, _summary, problems = analyze_results(tmp_path)

    assert pairwise == []
    assert any("missing on result" in problem for problem in problems)


def test_analyze_results_reports_incomplete_requests(tmp_path):
    _write_result(
        tmp_path,
        repetition=1,
        variant="off",
        request_rate="1",
        completed=99,
    )
    _write_result(
        tmp_path,
        repetition=1,
        variant="on",
        request_rate="1",
    )

    _pairwise, _summary, problems = analyze_results(tmp_path)

    assert any("completed=99" in problem for problem in problems)


def test_analyze_results_reports_workload_length_mismatch(tmp_path):
    _write_result(
        tmp_path,
        repetition=1,
        variant="off",
        request_rate="1",
    )
    _write_result(
        tmp_path,
        repetition=1,
        variant="on",
        request_rate="1",
        input_lens=[256] * 100,
    )

    _pairwise, _summary, problems = analyze_results(tmp_path)

    assert any("input_lens differs" in problem for problem in problems)


def _write_diagnostic_trace(root, variant, profiles):
    case_dir = root / "diagnostic" / variant / "rps-1" / "rep-1"
    case_dir.mkdir(parents=True)
    case_events = [
        {
            "event": "bench_case_start",
            "ts_ns": 100,
            "case_id": "formal-case",
        },
        {
            "event": "bench_case_end",
            "ts_ns": 200,
            "case_id": "formal-case",
        },
    ]
    (case_dir / "case.trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in case_events),
        encoding="utf-8",
    )
    (case_dir / "decode.trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in profiles),
        encoding="utf-8",
    )


def test_diagnostic_summary_reports_reverse_and_block_fractions(tmp_path):
    common_profiles = [
        {
            "event": "pull_transfer_profile",
            "ts_ns": 120,
            "request_id": "healthcheck",
            "num_local_blocks": 100,
            "num_remote_blocks": 100,
            "paired_forward_run_count": 0,
            "paired_reverse_run_count": 1,
            "canonicalized_reverse_block_count": 100,
            "forward_only_range_count": 100,
            "reverse_canonicalized_range_count": 10,
            "reordered_optimal_range_count": 5,
            "additional_reorderable_edge_count": 5,
            "generalized_reordered_block_count": 90,
        },
        {
            "event": "pull_transfer_profile",
            "ts_ns": 150,
            "request_id": "formal-case-0",
            "num_local_blocks": 10,
            "num_remote_blocks": 10,
            "paired_forward_run_count": 0,
            "paired_reverse_run_count": 1,
            "forward_only_range_count": 10,
            "reverse_canonicalized_range_count": 4,
            "reordered_optimal_range_count": 2,
            "additional_reorderable_edge_count": 2,
            "generalized_reordered_block_count": 8,
        },
        {
            "event": "pull_transfer_profile",
            "ts_ns": 160,
            "request_id": "formal-case-1",
            "num_local_blocks": 6,
            "num_remote_blocks": 6,
            "paired_forward_run_count": 1,
            "paired_reverse_run_count": 0,
            "forward_only_range_count": 3,
            "reverse_canonicalized_range_count": 3,
            "reordered_optimal_range_count": 3,
            "additional_reorderable_edge_count": 0,
            "generalized_reordered_block_count": 0,
        },
    ]
    _write_diagnostic_trace(
        tmp_path,
        "off",
        [
            {
                **row,
                "canonicalized_reverse_block_count": 0,
            }
            for row in common_profiles
        ],
    )
    _write_diagnostic_trace(
        tmp_path,
        "on",
        [
            common_profiles[0],
            {
                **common_profiles[1],
                "canonicalized_reverse_block_count": 8,
            },
            {
                **common_profiles[2],
                "canonicalized_reverse_block_count": 0,
            },
        ],
    )

    summaries = _summarize_diagnostic_traces(tmp_path)

    off = next(row for row in summaries if row["variant"] == "off")
    on = next(row for row in summaries if row["variant"] == "on")
    assert off["profile_count"] == 2
    assert off["canonicalized_block_fraction"] == 0
    assert on["profile_count"] == 2
    assert on["reverse_request_fraction"] == pytest.approx(0.5)
    assert on["canonicalized_request_fraction"] == pytest.approx(0.5)
    assert on["total_local_block_count"] == 16
    assert on["canonicalized_block_count"] == 8
    assert on["canonicalized_block_fraction"] == pytest.approx(0.5)
    assert on["reorder_opportunity_request_count"] == 1
    assert on["reorder_opportunity_request_fraction"] == pytest.approx(0.5)
    assert on["total_forward_only_range_count"] == 13
    assert on["total_reverse_canonicalized_range_count"] == 7
    assert on["total_reordered_optimal_range_count"] == 5
    assert on["total_additional_reorderable_edge_count"] == 2
    assert on["additional_reorderable_range_fraction"] == pytest.approx(2 / 7)
    assert on[
        "p90_request_additional_reorderable_range_fraction"
    ] == pytest.approx(0.5)
    assert on[
        "p50_request_canonicalized_block_fraction"
    ] == pytest.approx(0.0)
    assert on[
        "p90_request_canonicalized_block_fraction"
    ] == pytest.approx(0.8)
