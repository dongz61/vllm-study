#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Convert a Mooncake FAST'25 JSONL trace for the vLLM 0.11 loader.

The vLLM 0.11 BurstGPT loader consumes request lengths from a positional CSV
schema.  This converter preserves Mooncake request lengths and timestamps while
emitting that compatible schema.  Prefix hash IDs are intentionally not
replayed: the resulting workload is a length-trace benchmark with synthetic,
request-unique prompt contents.
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


def _timestamp(value: Any, *, line_number: int) -> int | float | str:
    if value is None or isinstance(value, (dict, list, bool)):
        raise MooncakeConversionError(
            f"line {line_number}: timestamp must be a scalar, got {value!r}")
    return value


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


def convert_trace(
    source_path: Path,
    output_path: Path,
    *,
    max_total_tokens: int,
    force: bool = False,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    """Convert one Mooncake JSONL trace and return its conversion summary."""
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive")
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

    accepted: list[tuple[int | float | str, int, int]] = []
    source_rows = 0
    filtered_non_positive_length = 0
    filtered_over_context_limit = 0
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
            if input_length <= 0 or output_length <= 0:
                filtered_non_positive_length += 1
                continue
            if input_length + output_length > max_total_tokens:
                filtered_over_context_limit += 1
                continue
            accepted.append((timestamp, input_length, output_length))

    if not accepted:
        raise MooncakeConversionError(
            "no requests remain after validation and context-length filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["Timestamp", "Model", "Request tokens", "Response tokens"])
        for timestamp, input_length, output_length in accepted:
            writer.writerow(
                [timestamp, "GPT-4", input_length, output_length])

    input_lengths = [row[1] for row in accepted]
    output_lengths = [row[2] for row in accepted]
    total_lengths = [row[1] + row[2] for row in accepted]
    summary: dict[str, Any] = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "format": "burstgpt-v1.1-compatible-length-trace",
        "source_rows": source_rows,
        "written_rows": len(accepted),
        "filtered_non_positive_length": filtered_non_positive_length,
        "filtered_over_context_limit": filtered_over_context_limit,
        "max_total_tokens": max_total_tokens,
        "input_length": _length_summary(input_lengths),
        "output_length": _length_summary(output_lengths),
        "total_length": _length_summary(total_lengths),
        "timestamp_preserved": True,
        "hash_ids_replayed": False,
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
            "Convert a Mooncake FAST'25 JSONL trace into the positional CSV "
            "schema consumed by the vLLM 0.11 BurstGPT loader."
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
