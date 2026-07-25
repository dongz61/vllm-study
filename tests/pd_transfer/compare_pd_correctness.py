#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare deterministic PD benchmark outputs and transfer shapes."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSFER_SHAPE_FIELDS = (
    "num_local_blocks",
    "num_remote_blocks",
    "num_local_descs",
    "num_remote_descs",
    "num_handles",
    "total_bytes",
)

BENCHMARK_REQUEST_SUFFIX = "-pdreq"

DETAILED_RESULT_FIELDS = (
    "generated_texts",
    "input_lens",
    "output_lens",
    "errors",
)


class ComparisonInputError(ValueError):
    """Raised when a run is missing required correctness artifacts."""


@dataclass(frozen=True)
class CaseCoverage:
    case_id: str
    request_count: int
    forward_request_count: int
    reverse_request_count: int
    canonicalized_request_count: int
    canonicalized_block_count: int


def _load_benchmark_cases(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("pull-sleep-*.json")):
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if "num_prompts" not in data:
            continue

        case_id = path.stem
        if case_id in cases:
            raise ComparisonInputError(
                f"Duplicate benchmark case {case_id!r} under {root}")
        missing = [
            field for field in DETAILED_RESULT_FIELDS if field not in data
        ]
        if missing:
            raise ComparisonInputError(
                f"{path} is missing detailed fields {missing}; rerun with "
                "--save-detailed")

        num_prompts = int(data["num_prompts"])
        for field in DETAILED_RESULT_FIELDS:
            values = data[field]
            if not isinstance(values, list) or len(values) != num_prompts:
                actual_count = (len(values)
                                if isinstance(values, list) else "non-list")
                raise ComparisonInputError(
                    f"{path}: {field} has {actual_count} entries, "
                    f"expected {num_prompts}")

        cases[case_id] = data

    if not cases:
        raise ComparisonInputError(
            f"No detailed pull benchmark result JSON files found under {root}")
    return cases


def _load_timeline_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "pd_request_timeline_ms.csv"
    if not path.is_file():
        raise ComparisonInputError(
            f"{path} does not exist; run parse_pd_trace.py first")

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            request_id = row.get("request_id", "")
            if not request_id:
                continue
            if request_id in rows:
                raise ComparisonInputError(
                    f"Duplicate request ID {request_id!r} in {path}")
            rows[request_id] = row
    return rows


def _as_int(row: dict[str, str], field: str, request_id: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise ComparisonInputError(
            f"Request {request_id!r} is missing trace field {field!r}")
    try:
        parsed = float(value)
        integer = int(parsed)
    except ValueError as error:
        raise ComparisonInputError(
            f"Request {request_id!r} has invalid {field}={value!r}") from error
    if parsed != integer:
        raise ComparisonInputError(
            f"Request {request_id!r} has non-integral {field}={value!r}")
    return integer


def compare_runs(
    baseline_root: Path,
    optimized_root: Path,
    *,
    check_coverage: bool = True,
) -> tuple[list[str], list[CaseCoverage]]:
    """Return mismatch descriptions and optimized coverage per case."""
    baseline_cases = _load_benchmark_cases(baseline_root)
    optimized_cases = _load_benchmark_cases(optimized_root)
    baseline_rows = _load_timeline_rows(baseline_root)
    optimized_rows = _load_timeline_rows(optimized_root)

    mismatches: list[str] = []
    baseline_case_ids = set(baseline_cases)
    optimized_case_ids = set(optimized_cases)
    if baseline_case_ids != optimized_case_ids:
        missing_optimized = sorted(baseline_case_ids - optimized_case_ids)
        missing_baseline = sorted(optimized_case_ids - baseline_case_ids)
        if missing_optimized:
            mismatches.append(
                f"Cases missing from optimized run: {missing_optimized}")
        if missing_baseline:
            mismatches.append(
                f"Cases missing from baseline run: {missing_baseline}")

    coverage: list[CaseCoverage] = []
    for case_id in sorted(baseline_case_ids & optimized_case_ids):
        baseline = baseline_cases[case_id]
        optimized = optimized_cases[case_id]
        baseline_count = int(baseline["num_prompts"])
        optimized_count = int(optimized["num_prompts"])
        if baseline_count != optimized_count:
            mismatches.append(
                f"{case_id}: num_prompts differs: baseline={baseline_count}, "
                f"optimized={optimized_count}")
            continue

        forward_requests = 0
        reverse_requests = 0
        canonicalized_requests = 0
        canonicalized_blocks = 0

        for index in range(baseline_count):
            request_id = f"{case_id}-{index}{BENCHMARK_REQUEST_SUFFIX}"
            for field in ("input_lens", "output_lens", "generated_texts",
                          "errors"):
                baseline_value = baseline[field][index]
                optimized_value = optimized[field][index]
                if baseline_value != optimized_value:
                    mismatches.append(
                        f"{request_id}: {field} differs: "
                        f"baseline={baseline_value!r}, "
                        f"optimized={optimized_value!r}")

            if baseline["errors"][index] or optimized["errors"][index]:
                mismatches.append(
                    f"{request_id}: request error is not empty: "
                    f"baseline={baseline['errors'][index]!r}, "
                    f"optimized={optimized['errors'][index]!r}")

            baseline_row = baseline_rows.get(request_id)
            optimized_row = optimized_rows.get(request_id)
            if baseline_row is None:
                mismatches.append(
                    f"{request_id}: missing baseline transfer trace")
                continue
            if optimized_row is None:
                mismatches.append(
                    f"{request_id}: missing optimized transfer trace")
                continue

            for field in TRANSFER_SHAPE_FIELDS:
                baseline_value = _as_int(baseline_row, field, request_id)
                optimized_value = _as_int(optimized_row, field, request_id)
                if baseline_value != optimized_value:
                    mismatches.append(
                        f"{request_id}: {field} differs: "
                        f"baseline={baseline_value}, "
                        f"optimized={optimized_value}")

            baseline_canonicalized = _as_int(
                baseline_row, "canonicalized_reverse_block_count", request_id)
            if baseline_canonicalized:
                mismatches.append(
                    f"{request_id}: baseline unexpectedly canonicalized "
                    f"{baseline_canonicalized} reverse blocks")

            paired_forward = _as_int(optimized_row,
                                     "paired_forward_run_count", request_id)
            paired_reverse = _as_int(optimized_row,
                                     "paired_reverse_run_count", request_id)
            request_canonicalized_blocks = _as_int(
                optimized_row, "canonicalized_reverse_block_count",
                request_id)
            forward_requests += paired_forward > 0
            reverse_requests += paired_reverse > 0
            canonicalized_requests += request_canonicalized_blocks > 0
            canonicalized_blocks += request_canonicalized_blocks

        case_coverage = CaseCoverage(
            case_id=case_id,
            request_count=baseline_count,
            forward_request_count=forward_requests,
            reverse_request_count=reverse_requests,
            canonicalized_request_count=canonicalized_requests,
            canonicalized_block_count=canonicalized_blocks,
        )
        coverage.append(case_coverage)
        if check_coverage:
            if forward_requests == 0:
                mismatches.append(
                    f"{case_id}: no paired-forward request was covered")
            if reverse_requests == 0:
                mismatches.append(
                    f"{case_id}: no paired-reverse request was covered")
            if canonicalized_requests == 0:
                mismatches.append(
                    f"{case_id}: optimized run did not canonicalize any "
                    "paired-reverse request")

    return mismatches, coverage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OFF/ON PD correctness artifacts.")
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("optimized_root", type=Path)
    parser.add_argument(
        "--skip-coverage-checks",
        action="store_true",
        help="Do not require forward, reverse, and canonicalized requests in "
        "every case.",
    )
    args = parser.parse_args()

    try:
        mismatches, coverage = compare_runs(
            args.baseline_root,
            args.optimized_root,
            check_coverage=not args.skip_coverage_checks,
        )
    except (ComparisonInputError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2

    for item in coverage:
        print(
            f"{item.case_id}: requests={item.request_count}, "
            f"forward={item.forward_request_count}, "
            f"reverse={item.reverse_request_count}, "
            f"canonicalized_requests={item.canonicalized_request_count}, "
            f"canonicalized_blocks={item.canonicalized_block_count}")

    if mismatches:
        max_reported = 50
        for mismatch in mismatches[:max_reported]:
            print(f"MISMATCH: {mismatch}")
        if len(mismatches) > max_reported:
            print(f"... {len(mismatches) - max_reported} more mismatches")
        print(f"FAIL: found {len(mismatches)} correctness mismatch(es)")
        return 1

    print("PASS: outputs and transfer shapes are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
