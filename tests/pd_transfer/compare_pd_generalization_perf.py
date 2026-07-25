#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate paired OFF/ON BurstGPT generalization benchmark results."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRICS: tuple[tuple[str, bool], ...] = (
    ("request_throughput", True),
    ("output_throughput", True),
    ("total_token_throughput", True),
    ("p50_ttft_ms", False),
    ("p90_ttft_ms", False),
    ("p99_ttft_ms", False),
    ("p50_tpot_ms", False),
    ("p90_tpot_ms", False),
    ("p99_tpot_ms", False),
    ("p50_itl_ms", False),
    ("p90_itl_ms", False),
    ("p99_itl_ms", False),
    ("p50_e2el_ms", False),
    ("p90_e2el_ms", False),
    ("p99_e2el_ms", False),
)
METRIC_DIRECTIONS = dict(METRICS)
CONSOLE_METRICS = {
    "request_throughput",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_e2el_ms",
}


class PerformanceInputError(ValueError):
    """Raised when performance artifacts cannot be paired safely."""


@dataclass(frozen=True)
class ResultRecord:
    path: Path
    variant: str
    repetition: int
    request_rate: str
    num_prompts: int
    data: dict[str, Any]


def _as_float(data: dict[str, Any], field: str, path: Path) -> float:
    value = data.get(field)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PerformanceInputError(
            f"{path}: missing or invalid metric {field}={value!r}") from error
    if not math.isfinite(result):
        raise PerformanceInputError(
            f"{path}: non-finite metric {field}={value!r}")
    return result


def _rate_sort_key(rate: str) -> tuple[int, float]:
    if rate == "inf":
        return (1, math.inf)
    return (0, float(rate))


def load_performance_results(
    root: Path,
) -> tuple[dict[tuple[int, str, str], ResultRecord], list[str]]:
    records: dict[tuple[int, str, str], ResultRecord] = {}
    problems: list[str] = []
    for path in sorted(root.rglob("result.json")):
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if data.get("phase") != "performance":
            continue

        variant = str(data.get("variant", ""))
        if variant not in {"off", "on"}:
            raise PerformanceInputError(
                f"{path}: invalid variant metadata {variant!r}")
        try:
            repetition = int(data["repetition"])
            num_prompts = int(data["num_prompts"])
        except (KeyError, TypeError, ValueError) as error:
            raise PerformanceInputError(
                f"{path}: invalid repetition or num_prompts metadata") from error
        request_rate = str(data.get("configured_request_rate", ""))
        if not request_rate:
            raise PerformanceInputError(
                f"{path}: missing configured_request_rate metadata")

        key = (repetition, request_rate, variant)
        if key in records:
            raise PerformanceInputError(
                f"Duplicate performance result for {key}: "
                f"{records[key].path} and {path}")
        for metric, _ in METRICS:
            _as_float(data, metric, path)

        completed = int(data.get("completed", -1))
        if completed != num_prompts:
            problems.append(
                f"{path}: completed={completed}, expected {num_prompts}")
        errors = data.get("errors")
        if isinstance(errors, list):
            error_count = sum(bool(error) for error in errors)
            if error_count:
                problems.append(
                    f"{path}: {error_count} request error(s) were recorded")

        records[key] = ResultRecord(
            path=path,
            variant=variant,
            repetition=repetition,
            request_rate=request_rate,
            num_prompts=num_prompts,
            data=data,
        )

    if not records:
        raise PerformanceInputError(
            f"No performance result.json files found under {root}")
    return records, problems


def analyze_results(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records, problems = load_performance_results(root)
    pair_keys = {(repetition, rate)
                 for repetition, rate, _variant in records}
    pairwise: list[dict[str, Any]] = []

    for repetition, request_rate in sorted(
            pair_keys, key=lambda item: (_rate_sort_key(item[1]), item[0])):
        off = records.get((repetition, request_rate, "off"))
        on = records.get((repetition, request_rate, "on"))
        if off is None or on is None:
            missing = "off" if off is None else "on"
            problems.append(
                f"repetition={repetition}, request_rate={request_rate}: "
                f"missing {missing} result")
            continue
        if off.num_prompts != on.num_prompts:
            problems.append(
                f"repetition={repetition}, request_rate={request_rate}: "
                f"num_prompts differs: off={off.num_prompts}, "
                f"on={on.num_prompts}")
            continue
        for field in ("input_lens", "output_lens"):
            off_values = off.data.get(field)
            on_values = on.data.get(field)
            if off_values is not None and on_values is not None:
                if off_values != on_values:
                    problems.append(
                        f"repetition={repetition}, "
                        f"request_rate={request_rate}: {field} differs "
                        "between OFF and ON")

        for metric, higher_is_better in METRICS:
            off_value = _as_float(off.data, metric, off.path)
            on_value = _as_float(on.data, metric, on.path)
            delta_pct = ((on_value - off_value) / off_value * 100
                         if off_value else None)
            improvement_pct = (
                delta_pct if higher_is_better or delta_pct is None
                else -delta_pct
            )
            pairwise.append({
                "request_rate": request_rate,
                "repetition": repetition,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "off": off_value,
                "on": on_value,
                "on_minus_off_pct": delta_pct,
                "improvement_pct": improvement_pct,
            })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise:
        grouped[(row["request_rate"], row["metric"])].append(row)

    summary: list[dict[str, Any]] = []
    for (request_rate, metric), rows in sorted(
            grouped.items(),
            key=lambda item: (
                _rate_sort_key(item[0][0]),
                [name for name, _ in METRICS].index(item[0][1]),
            )):
        improvements = [
            row["improvement_pct"] for row in rows
            if row["improvement_pct"] is not None
        ]
        summary.append({
            "request_rate": request_rate,
            "metric": metric,
            "higher_is_better": METRIC_DIRECTIONS[metric],
            "paired_repetitions": len(rows),
            "median_off": statistics.median(row["off"] for row in rows),
            "median_on": statistics.median(row["on"] for row in rows),
            "median_improvement_pct": (
                statistics.median(improvements) if improvements else None
            ),
            "min_improvement_pct": (
                min(improvements) if improvements else None
            ),
            "max_improvement_pct": (
                max(improvements) if improvements else None
            ),
        })
    return pairwise, summary, problems


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_diagnostic_traces(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in ("off", "on"):
        profiles: list[dict[str, Any]] = []
        for path in sorted((root / "diagnostic" / variant).rglob(
                "decode.trace.jsonl")):
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    record = json.loads(line)
                    if record.get("event") != "pull_transfer_profile":
                        continue
                    if "warmup-" in str(record.get("request_id", "")):
                        continue
                    profiles.append(record)
        if not profiles:
            continue
        summaries.append({
            "variant": variant,
            "profile_count": len(profiles),
            "forward_request_count": sum(
                int(item.get("paired_forward_run_count", 0)) > 0
                for item in profiles
            ),
            "reverse_request_count": sum(
                int(item.get("paired_reverse_run_count", 0)) > 0
                for item in profiles
            ),
            "canonicalized_request_count": sum(
                int(item.get("canonicalized_reverse_block_count", 0)) > 0
                for item in profiles
            ),
            "canonicalized_block_count": sum(
                int(item.get("canonicalized_reverse_block_count", 0))
                for item in profiles
            ),
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate paired BurstGPT OFF/ON performance results.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    try:
        pairwise, summary, problems = analyze_results(args.root)
        diagnostic = _summarize_diagnostic_traces(args.root)
    except (PerformanceInputError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2

    _write_csv(args.root / "generalization_pairwise.csv", pairwise)
    _write_csv(args.root / "generalization_summary.csv", summary)
    _write_csv(args.root / "diagnostic_trace_summary.csv", diagnostic)

    for row in summary:
        if row["metric"] not in CONSOLE_METRICS:
            continue
        improvement = row["median_improvement_pct"]
        improvement_text = (
            f"{improvement:+.2f}%" if improvement is not None else "n/a"
        )
        print(
            f"rps={row['request_rate']} {row['metric']}: "
            f"off={row['median_off']:.3f}, on={row['median_on']:.3f}, "
            f"median improvement={improvement_text}, "
            f"pairs={row['paired_repetitions']}")

    for row in diagnostic:
        print(
            f"diagnostic {row['variant']}: profiles={row['profile_count']}, "
            f"forward={row['forward_request_count']}, "
            f"reverse={row['reverse_request_count']}, "
            f"canonicalized_requests="
            f"{row['canonicalized_request_count']}, "
            f"canonicalized_blocks={row['canonicalized_block_count']}")

    diagnostic_root = args.root / "diagnostic"
    if diagnostic_root.exists():
        off = next((row for row in diagnostic
                    if row["variant"] == "off"), None)
        on = next((row for row in diagnostic
                   if row["variant"] == "on"), None)
        if off is None:
            problems.append(
                "diagnostic OFF run has no transfer profile records")
        if on is None:
            problems.append(
                "diagnostic ON run has no transfer profile records")
        if off and off["canonicalized_block_count"]:
            problems.append(
                "diagnostic OFF run unexpectedly canonicalized reverse blocks")
        if on and not on["canonicalized_block_count"]:
            problems.append(
                "diagnostic ON run did not canonicalize any reverse blocks")

    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        print(f"FAIL: found {len(problems)} artifact or pairing problem(s)")
        return 1

    print("PASS: all performance results are complete and paired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
