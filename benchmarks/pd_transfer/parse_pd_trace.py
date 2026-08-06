# SPDX-License-Identifier: Apache-2.0
"""Join PD JSONL events and report end-to-end KV transfer wait time."""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

_ENGINE_SUFFIX = re.compile(r"-\d+-[0-9a-fA-F]{8}$")
_TP_SUFFIX = re.compile(r"-\d+$")


def normalize_request_id(value: Any) -> str:
    request_id = str(value)
    if not request_id.startswith("cmpl-"):
        return request_id
    request_id = request_id.removeprefix("cmpl-")
    try:
        UUID(request_id)
    except ValueError:
        request_id = _ENGINE_SUFFIX.sub("", request_id)
        return _TP_SUFFIX.sub("", request_id)
    return request_id


def iter_records(root: Path):
    for path in root.rglob("*.trace.jsonl"):
        with path.open(encoding="utf-8-sig") as trace_file:
            for line in trace_file:
                if line.strip():
                    record = json.loads(line)
                    if "request_id" in record:
                        record["request_id"] = normalize_request_id(
                            record["request_id"]
                        )
                    record["trace_file"] = str(path)
                    yield record


def _first(records: list[dict[str, Any]], event: str):
    matches = [record for record in records if record.get("event") == event]
    return min(matches, key=lambda record: record["ts_ns"]) if matches else None


def _last(records: list[dict[str, Any]], event: str):
    matches = [record for record in records if record.get("event") == event]
    return max(matches, key=lambda record: record["ts_ns"]) if matches else None


def _delta_ms(start: dict[str, Any] | None, end: dict[str, Any] | None):
    if start is None or end is None:
        return ""
    clock = (
        "perf_ns"
        if start.get("host") == end.get("host")
        and "perf_ns" in start
        and "perf_ns" in end
        else "ts_ns"
    )
    return round((end[clock] - start[clock]) / 1_000_000, 3)


def _aggregate_numeric_field(
    records: list[dict[str, Any]], field: str, operation: str
) -> float | int | str:
    values = [record[field] for record in records if field in record]
    if not values:
        return ""
    if operation == "sum":
        value = sum(values)
    elif operation == "max":
        value = max(values)
    else:
        raise ValueError(f"unsupported aggregation: {operation}")
    return round(value, 3) if isinstance(value, float) else value


def build_rows(root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_records(root):
        if request_id := record.get("request_id"):
            if request_id.startswith("warmup-"):
                continue
            grouped[request_id].append(record)

    rows = []
    for request_id, records in grouped.items():
        received = _first(records, "proxy_request_received")
        prefill_done = _last(records, "prefill_compute_done")
        request_prepare_start = _first(records, "transfer_request_prepare_start")
        request_prepare_done = _last(records, "transfer_request_prepare_done")
        submit = _first(records, "transfer_submit")
        physical_done = _last(records, "transfer_physical_done")
        sleep_start = _last(records, "transfer_injected_sleep_start")
        sleep_end = _last(records, "transfer_injected_sleep_end")
        reported_done = _last(records, "transfer_reported_done")
        schedulable = _last(records, "transfer_schedulable")
        first_chunk = _first(records, "proxy_first_response_chunk")
        submit_records = [
            record for record in records if record.get("event") == "transfer_submit"
        ]
        submit_records.sort(key=lambda record: record["ts_ns"])
        rank_done_records = [
            record
            for record in records
            if record.get("event") == "transfer_rank_done_observed"
        ]
        rank_done_records.sort(key=lambda record: record["ts_ns"])
        first_submit = submit_records[0] if submit_records else None
        last_submit = submit_records[-1] if submit_records else None
        last_rank_done = rank_done_records[-1] if rank_done_records else None
        hosts = sorted({str(record.get("host", "")) for record in records})
        failed = any(record.get("event") == "transfer_failed" for record in records)
        full_wait_ms = "" if failed else _delta_ms(prefill_done, reported_done)
        proxy_ttft_ms = _delta_ms(received, first_chunk)
        physical_to_reported_ms = _delta_ms(physical_done, reported_done)
        sleep_start_to_end_ms = _delta_ms(sleep_start, sleep_end)
        # Request-level exposed injection is the interval from completion on
        # all TP ranks to expiration of all rank-local sleep gates.
        observed_sleep_ms = _delta_ms(physical_done, sleep_end)
        configured_sleep_ms: float | str = ""
        if physical_done is not None:
            configured_sleep_ms = float(physical_done.get("injected_sleep_ms", 0))
            if configured_sleep_ms == 0 and physical_to_reported_ms != "":
                observed_sleep_ms = 0.0
        report_excluding_sleep_ms: float | str = ""
        if physical_to_reported_ms != "" and observed_sleep_ms != "":
            report_excluding_sleep_ms = round(
                max(0.0, physical_to_reported_ms - observed_sleep_ms), 3
            )
        wait_fraction = ""
        if full_wait_ms != "" and proxy_ttft_ms != "" and proxy_ttft_ms > 0:
            wait_fraction = round(full_wait_ms / proxy_ttft_ms, 6)

        row = {
            "request_id": request_id,
            "input_len": received.get("input_len", "") if received else "",
            "output_len": received.get("output_len", "") if received else "",
            "prefill_tp_size": (
                prefill_done.get("tp_size", "") if prefill_done else ""
            ),
            "decode_tp_size": (
                submit_records[0].get("local_tp_size", "")
                if submit_records
                else ""
            ),
            "num_transfer_submits": len(submit_records),
            "num_rank_done_observed": len(rank_done_records),
            "transfer_remote_rank_order": ";".join(
                str(record.get("remote_rank", "")) for record in submit_records
            ),
            "hosts": ";".join(hosts),
            "cross_host": len(hosts) > 1,
            "full_interval_clock": (
                "monotonic"
                if prefill_done
                and reported_done
                and prefill_done.get("host") == reported_done.get("host")
                else "wall"
            ),
            "transfer_failed": failed,
            "prefill_to_request_prepare_start_ms": _delta_ms(
                prefill_done, request_prepare_start
            ),
            "request_prepare_start_to_first_submit_ms": _delta_ms(
                request_prepare_start, submit
            ),
            "request_prepare_total_ms": (
                request_prepare_done.get("request_prepare_total_ms", "")
                if request_prepare_done
                else ""
            ),
            "read_plan_ms": (
                request_prepare_done.get("read_plan_ms", "")
                if request_prepare_done
                else ""
            ),
            "rank_submit_loop_ms": (
                request_prepare_done.get("rank_submit_loop_ms", "")
                if request_prepare_done
                else ""
            ),
            "prefill_to_submit_ms": _delta_ms(prefill_done, submit),
            "first_to_last_submit_ms": _delta_ms(first_submit, last_submit),
            "first_rank_preprocess_ms": (
                first_submit.get("rank_preprocess_ms", "") if first_submit else ""
            ),
            "first_rank_remote_desc_ms": (
                first_submit.get("remote_desc_ms", "") if first_submit else ""
            ),
            "first_rank_local_desc_ms": (
                first_submit.get("local_desc_ms", "") if first_submit else ""
            ),
            "first_rank_make_prepped_xfer_ms": (
                first_submit.get("make_prepped_xfer_ms", "")
                if first_submit
                else ""
            ),
            "first_rank_transfer_call_ms": (
                first_submit.get("transfer_call_ms", "") if first_submit else ""
            ),
            "first_rank_prepare_total_ms": (
                first_submit.get("rank_prepare_total_ms", "")
                if first_submit
                else ""
            ),
            "all_rank_preprocess_sum_ms": _aggregate_numeric_field(
                submit_records, "rank_preprocess_ms", "sum"
            ),
            "all_rank_remote_desc_sum_ms": _aggregate_numeric_field(
                submit_records, "remote_desc_ms", "sum"
            ),
            "all_rank_local_desc_sum_ms": _aggregate_numeric_field(
                submit_records, "local_desc_ms", "sum"
            ),
            "all_rank_make_prepped_xfer_sum_ms": _aggregate_numeric_field(
                submit_records, "make_prepped_xfer_ms", "sum"
            ),
            "all_rank_transfer_call_sum_ms": _aggregate_numeric_field(
                submit_records, "transfer_call_ms", "sum"
            ),
            "all_rank_prepare_sum_ms": _aggregate_numeric_field(
                submit_records, "rank_prepare_total_ms", "sum"
            ),
            "max_rank_submit_to_done_observed_ms": _aggregate_numeric_field(
                rank_done_records, "submit_to_done_observed_ms", "max"
            ),
            "first_submit_to_last_rank_done_observed_ms": _delta_ms(
                first_submit, last_rank_done
            ),
            "last_submit_to_last_rank_done_observed_ms": _delta_ms(
                last_submit, last_rank_done
            ),
            "last_rank_done_observed_to_physical_done_ms": _delta_ms(
                last_rank_done, physical_done
            ),
            "max_nixl_xfer_duration_ms": _aggregate_numeric_field(
                rank_done_records, "nixl_xfer_duration_ms", "max"
            ),
            "max_cuda_ipc_kernel_ms": _aggregate_numeric_field(
                rank_done_records, "kernel_ms", "max"
            ),
            "max_nixl_post_duration_ms": _aggregate_numeric_field(
                rank_done_records, "nixl_post_duration_ms", "max"
            ),
            "total_bytes_transferred": _aggregate_numeric_field(
                rank_done_records, "bytes_transferred", "sum"
            ),
            "total_descriptors": _aggregate_numeric_field(
                submit_records, "num_descriptors", "sum"
            ),
            "submit_to_physical_done_ms": _delta_ms(submit, physical_done),
            "configured_transfer_sleep_ms": configured_sleep_ms,
            "physical_done_to_sleep_start_ms": _delta_ms(physical_done, sleep_start),
            "injected_sleep_observed_ms": observed_sleep_ms,
            "sleep_start_to_sleep_end_ms": sleep_start_to_end_ms,
            "sleep_end_to_reported_ms": (
                _delta_ms(sleep_end, reported_done)
                if sleep_end is not None
                else physical_to_reported_ms
            ),
            "physical_done_to_reported_excluding_sleep_ms": (report_excluding_sleep_ms),
            "physical_done_to_reported_ms": physical_to_reported_ms,
            "prefill_to_reported_ms": full_wait_ms,
            "prefill_to_reported_over_ttft": wait_fraction,
            "reported_to_schedulable_ms": _delta_ms(reported_done, schedulable),
            "reported_to_first_chunk_ms": _delta_ms(reported_done, first_chunk),
            "proxy_ttft_ms": proxy_ttft_ms,
        }
        rows.append(row)
    return sorted(rows, key=lambda row: row["request_id"])


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = root / "pd_request_timeline_ms.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    values = [
        float(row["prefill_to_reported_ms"])
        for row in rows
        if row["prefill_to_reported_ms"] != ""
    ]
    fractions = [
        float(row["prefill_to_reported_over_ttft"])
        for row in rows
        if row["prefill_to_reported_over_ttft"] != ""
    ]
    configured_sleep_values = sorted(
        {
            float(row["configured_transfer_sleep_ms"])
            for row in rows
            if row["configured_transfer_sleep_ms"] != ""
        }
    )
    observed_sleep_values = [
        float(row["injected_sleep_observed_ms"])
        for row in rows
        if row["injected_sleep_observed_ms"] != ""
    ]
    summary = {
        "requests_seen": len(rows),
        "requests_with_complete_interval": len(values),
        "cross_host_requests": sum(bool(row["cross_host"]) for row in rows),
        "clock_note": (
            "Same-host intervals use perf_counter_ns; cross-host intervals use "
            "time_ns and require verified PTP/NTP synchronization."
        ),
        "configured_transfer_sleep_ms": configured_sleep_values,
    }
    if values:
        summary["prefill_to_reported_ms"] = {
            "mean": round(statistics.fmean(values), 3),
            "p50": round(percentile(values, 0.50), 3),
            "p90": round(percentile(values, 0.90), 3),
            "p99": round(percentile(values, 0.99), 3),
            "max": round(max(values), 3),
        }
    if fractions:
        summary["prefill_to_reported_over_ttft"] = {
            "mean": round(statistics.fmean(fractions), 6),
            "p50": round(percentile(fractions, 0.50), 6),
            "p90": round(percentile(fractions, 0.90), 6),
            "p99": round(percentile(fractions, 0.99), 6),
            "max": round(max(fractions), 6),
        }
    if observed_sleep_values:
        summary["injected_sleep_observed_ms"] = {
            "mean": round(statistics.fmean(observed_sleep_values), 3),
            "p50": round(percentile(observed_sleep_values, 0.50), 3),
            "p90": round(percentile(observed_sleep_values, 0.90), 3),
            "p99": round(percentile(observed_sleep_values, 0.99), 3),
            "max": round(max(observed_sleep_values), 3),
        }
    summary_path = root / "pd_transfer_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {csv_path} and {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    if not args.result_dir.is_dir():
        parser.error(f"not a directory: {args.result_dir}")
    write_outputs(args.result_dir, build_rows(args.result_dir))


if __name__ == "__main__":
    main()
