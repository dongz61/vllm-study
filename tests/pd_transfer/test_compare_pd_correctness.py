# SPDX-License-Identifier: Apache-2.0

import csv
import json

from tests.pd_transfer.compare_pd_correctness import (
    BENCHMARK_REQUEST_SUFFIX, TRANSFER_SHAPE_FIELDS, compare_runs)
from tests.pd_transfer.parse_pd_trace import normalize_request_id

CASE_ID = "pull-sleep-0-input-2048-output-32-concurrency-1"


def test_benchmark_request_id_survives_engine_rank_normalization():
    request_id = f"{CASE_ID}-17{BENCHMARK_REQUEST_SUFFIX}"

    assert normalize_request_id(f"cmpl-{request_id}-3") == request_id


def _write_run(root, *, optimized, text_override=None, byte_override=None):
    result_dir = root / "pull" / "sleep-0"
    result_dir.mkdir(parents=True)
    result = {
        "num_prompts": 2,
        "input_lens": [2048, 2048],
        "output_lens": [32, 32],
        "generated_texts": ["forward output", "reverse output"],
        "errors": ["", ""],
    }
    if text_override is not None:
        result["generated_texts"][1] = text_override
    (result_dir / f"{CASE_ID}.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    rows = []
    for index in range(2):
        row = {
            "request_id": f"{CASE_ID}-{index}{BENCHMARK_REQUEST_SUFFIX}",
            "paired_forward_run_count": 1 if index == 0 else 0,
            "paired_reverse_run_count": 0 if index == 0 else 1,
            "canonicalized_reverse_block_count":
            128 if optimized and index == 1 else 0,
        }
        row.update({
            field: 128 if "blocks" in field else 64
            for field in TRANSFER_SHAPE_FIELDS
        })
        row["num_handles"] = 1
        row["total_bytes"] = 4096
        rows.append(row)
    if byte_override is not None:
        rows[1]["total_bytes"] = byte_override

    fieldnames = sorted({key for row in rows for key in row})
    with (root / "pd_request_timeline_ms.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compare_runs_passes_for_identical_outputs_and_shapes(tmp_path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"
    _write_run(baseline, optimized=False)
    _write_run(optimized, optimized=True)

    mismatches, coverage = compare_runs(baseline, optimized)

    assert mismatches == []
    assert coverage[0].forward_request_count == 1
    assert coverage[0].reverse_request_count == 1
    assert coverage[0].canonicalized_request_count == 1
    assert coverage[0].canonicalized_block_count == 128


def test_compare_runs_reports_output_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"
    _write_run(baseline, optimized=False)
    _write_run(optimized, optimized=True, text_override="different")

    mismatches, _ = compare_runs(baseline, optimized)

    assert any("generated_texts differs" in item for item in mismatches)


def test_compare_runs_reports_transfer_shape_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"
    _write_run(baseline, optimized=False)
    _write_run(optimized, optimized=True, byte_override=8192)

    mismatches, _ = compare_runs(baseline, optimized)

    assert any("total_bytes differs" in item for item in mismatches)
