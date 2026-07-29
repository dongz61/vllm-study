#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate and filter a Mooncake FAST'25 JSONL trace for vLLM 0.11.

JSONL output preserves timestamps, lengths, request order, and prefix hash IDs
for the native Mooncake benchmark loader. CSV output retains the older
BurstGPT-compatible length-only conversion for historical experiments.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


class MooncakeConversionError(ValueError):
    """Raised when a Mooncake trace row is malformed."""


def _integer(value: Any, *, field: str, line_number: int) -> int:
    if isinstance(value, bool):
        raise MooncakeConversionError(
            f"line {line_number}: {field} must be an integer, "
            f"got {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise MooncakeConversionError(
            f"line {line_number}: {field} must be an integer, "
            f"got {value!r}") from error
    if converted != value:
        raise MooncakeConversionError(
            f"line {line_number}: {field} must be an integer, "
            f"got {value!r}")
    return converted


def _timestamp(value: Any, *, line_number: int) -> int | float:
    if value is None or isinstance(value, (dict, list, bool)):
        raise MooncakeConversionError(
            f"line {line_number}: timestamp must be numeric, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise MooncakeConversionError(
            f"line {line_number}: timestamp must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(converted) or converted < 0:
        raise MooncakeConversionError(
            f"line {line_number}: timestamp must be non-negative")
    return value if isinstance(value, (int, float)) else converted


def _hash_ids(value: Any, *, line_number: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise MooncakeConversionError(
            f"line {line_number}: hash_ids must be a non-empty list")
    converted: list[int] = []
    for hash_id in value:
        converted_hash_id = _integer(
            hash_id, field="hash_id", line_number=line_number)
        if converted_hash_id < 0:
            raise MooncakeConversionError(
                f"line {line_number}: hash_id must be non-negative")
        converted.append(converted_hash_id)
    return converted


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _length_summary(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
    }


def _prefix_reuse_summary(
    rows: list[tuple[int | float, int, int, list[int]]],
    block_size: int,
) -> dict[str, int | float]:
    seen_hash_ids: set[int] = set()
    reused_tokens = 0
    requests_with_reuse = 0
    total_input_tokens = 0
    for _timestamp_value, input_length, _output_length, hash_ids in rows:
        matched_blocks = 0
        for hash_id in hash_ids:
            if hash_id not in seen_hash_ids:
                break
            matched_blocks += 1
        request_reused_tokens = min(
            matched_blocks * block_size, input_length)
        reused_tokens += request_reused_tokens
        requests_with_reuse += request_reused_tokens > 0
        total_input_tokens += input_length
        seen_hash_ids.update(hash_ids)
    return {
        "reused_tokens": reused_tokens,
        "total_input_tokens": total_input_tokens,
        "ratio": reused_tokens / total_input_tokens,
        "requests_with_reuse": requests_with_reuse,
    }


def convert_trace(
    source_path: Path,
    output_path: Path,
    *,
    max_total_tokens: int,
    block_size: int = 512,
    force: bool = False,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    """Convert one Mooncake JSONL trace and return its conversion summary."""
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not source_path.is_file():
        raise FileNotFoundError(f"Mooncake trace not found: {source_path}")

    resolved_stats_path = (
        stats_path
        if stats_path is not None
        else output_path.with_suffix(".stats.json")
    )
    for path in (output_path, resolved_stats_path):
        if path.exists() and not force:
            raise FileExistsError(
                f"output already exists (pass --force to replace it): {path}")

    preserve_hash_ids = output_path.suffix.lower() == ".jsonl"
    if not preserve_hash_ids and output_path.suffix.lower() != ".csv":
        raise ValueError("output path must end in .jsonl or .csv")

    accepted: list[tuple[int | float, int, int, list[int]]] = []
    source_rows = 0
    filtered_non_positive_length = 0
    filtered_over_context_limit = 0
    previous_timestamp: int | float | None = None
    with source_path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            source_rows += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise MooncakeConversionError(
                    f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise MooncakeConversionError(
                    f"line {line_number}: expected a JSON object")

            timestamp = _timestamp(
                record.get("timestamp"), line_number=line_number)
            if (previous_timestamp is not None
                    and timestamp < previous_timestamp):
                raise MooncakeConversionError(
                    f"line {line_number}: timestamps must be non-decreasing")
            previous_timestamp = timestamp
            input_length = _integer(
                record.get("input_length"),
                field="input_length",
                line_number=line_number,
            )
            output_length = _integer(
                record.get("output_length"),
                field="output_length",
                line_number=line_number,
            )
            hash_ids = _hash_ids(
                record.get("hash_ids"), line_number=line_number)
            expected_hash_ids = math.ceil(input_length / block_size)
            if len(hash_ids) != expected_hash_ids:
                raise MooncakeConversionError(
                    f"line {line_number}: input_length={input_length} "
                    f"requires {expected_hash_ids} hash IDs at block size "
                    f"{block_size}, got {len(hash_ids)}")
            if input_length <= 0 or output_length <= 0:
                filtered_non_positive_length += 1
                continue
            if input_length + output_length > max_total_tokens:
                filtered_over_context_limit += 1
                continue
            accepted.append(
                (timestamp, input_length, output_length, hash_ids))

    if not accepted:
        raise MooncakeConversionError(
            "no requests remain after validation and context-length filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        if preserve_hash_ids:
            for timestamp, input_length, output_length, hash_ids in accepted:
                output.write(json.dumps({
                    "timestamp": timestamp,
                    "input_length": input_length,
                    "output_length": output_length,
                    "hash_ids": hash_ids,
                }, separators=(",", ":")) + "\n")
        else:
            writer = csv.writer(output)
            writer.writerow(
                ["Timestamp", "Model", "Request tokens", "Response tokens"])
            for timestamp, input_length, output_length, _hash_ids_value in accepted:
                writer.writerow(
                    [timestamp, "GPT-4", input_length, output_length])

    input_lengths = [row[1] for row in accepted]
    output_lengths = [row[2] for row in accepted]
    total_lengths = [row[1] + row[2] for row in accepted]
    summary: dict[str, Any] = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "format": (
            "mooncake-fast25-jsonl"
            if preserve_hash_ids
            else "burstgpt-v1.1-compatible-length-trace"
        ),
        "source_rows": source_rows,
        "written_rows": len(accepted),
        "filtered_non_positive_length": filtered_non_positive_length,
        "filtered_over_context_limit": filtered_over_context_limit,
        "max_total_tokens": max_total_tokens,
        "block_size": block_size,
        "input_length": _length_summary(input_lengths),
        "output_length": _length_summary(output_lengths),
        "total_length": _length_summary(total_lengths),
        "infinite_capacity_prefix_reuse": _prefix_reuse_summary(
            accepted, block_size),
        "trace_duration_ms": accepted[-1][0] - accepted[0][0],
        "timestamp_preserved": True,
        "request_order_preserved": True,
        "hash_ids_preserved": preserve_hash_ids,
        "hash_ids_replayed": preserve_hash_ids,
    }
    resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_stats_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and filter a Mooncake FAST'25 trace. JSONL preserves "
            "prefix hashes; CSV is the legacy length-only format."
        ))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=40960,
        help="Drop requests whose input plus output exceeds this value.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Defaults to <output stem>.stats.json.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Number of prompt tokens represented by each hash ID.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing CSV and statistics file.",
    )
    args = parser.parse_args()

    try:
        summary = convert_trace(
            args.source,
            args.output,
            max_total_tokens=args.max_total_tokens,
            block_size=args.block_size,
            force=args.force,
            stats_path=args.stats_path,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        MooncakeConversionError,
        OSError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}")
        return 2

    print(
        f"Converted {summary['written_rows']}/{summary['source_rows']} "
        f"Mooncake requests to {args.output}")
    print(
        "Filtered non-positive lengths: "
        f"{summary['filtered_non_positive_length']}")
    print(
        "Filtered over context limit: "
        f"{summary['filtered_over_context_limit']}")
    for name in ("input_length", "output_length", "total_length"):
        values = summary[name]
        print(
            f"{name}: mean={values['mean']:.1f}, p50={values['p50']}, "
            f"p90={values['p90']}, p99={values['p99']}, "
            f"max={values['max']}")
    reuse = summary["infinite_capacity_prefix_reuse"]
    print(
        "infinite_capacity_prefix_reuse: "
        f"ratio={reuse['ratio']:.4f}, "
        f"requests={reuse['requests_with_reuse']}/"
        f"{summary['written_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
