# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.pd_transfer.compare_pd_generalization_perf import (
    METRICS, analyze_results)


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
