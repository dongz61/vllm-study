#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

REQUEST_EVENTS = [
    "proxy_request_received",
    "proxy_prefill_request_start",
    "proxy_prefill_request_end",
    "proxy_decode_request_start",
    "proxy_first_response_chunk",
    "proxy_decode_request_end",
    "pull_prefill_finished",
    "pull_decode_kv_allocated",
    "pull_transfer_start",
    "pull_transfer_sleep_start",
    "pull_transfer_sleep_end",
    "pull_transfer_end",
    "pull_kv_recv_done",
    "pull_transfer_profile",
    "decode_remote_kv_ready",
    "decode_remote_kv_schedulable",
]

LAST_EVENTS = {
    "proxy_prefill_request_end",
    "proxy_first_response_chunk",
    "proxy_decode_request_end",
    "pull_transfer_end",
    "pull_kv_recv_done",
    "pull_transfer_profile",
    "decode_remote_kv_ready",
    "decode_remote_kv_schedulable",
}

PROFILE_BASE_FIELDS = {
    "_trace_file",
    "ts_ns",
    "perf_ns",
    "pid",
    "host",
    "role",
    "event",
    "request_id",
    "request_ids",
}

REQUEST_ID_SUFFIX_RE = re.compile(r"-\d+$")
BENCH_RESULT_RE = re.compile(
    r"pull-sleep-(?P<sleep_ms>\d+(?:\.\d+)?)-input-"
    r"(?P<input_len>\d+)-output-(?P<output_len>\d+)-concurrency-"
    r"(?P<concurrency>\d+)\.json$")


def resolve_result_root(argument: Path) -> Path:
    """Resolve a result directory or a bare run ID below the current directory."""
    if argument.is_dir():
        return argument.resolve()
    if argument.parent != Path("."):
        raise ValueError(f"Not a directory: {argument}")

    matches = sorted(
        path.resolve()
        for path in Path.cwd().rglob(argument.name)
        if path.is_dir() and (path / "run_manifest.json").is_file()
    )
    if not matches:
        raise ValueError(
            f"No result run named {argument.name!r} found below {Path.cwd()}"
        )
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple result runs named {argument.name!r} found:\n  {formatted}"
        )
    return matches[0]


def normalize_request_id(request_id: Any) -> str:
    request_id = str(request_id)
    # Proxy events contain the original UUID, whose final UUID component may be
    # all digits. Never treat that component as an engine rank suffix.
    if not request_id.startswith("cmpl-"):
        return request_id

    # Engine events add ``cmpl-`` and may append a numeric TP-rank suffix. A
    # valid UUID immediately after removing the prefix has no rank suffix.
    engine_request_id = request_id.removeprefix("cmpl-")
    try:
        UUID(engine_request_id)
    except ValueError:
        return REQUEST_ID_SUFFIX_RE.sub("", engine_request_id)
    return engine_request_id


def req_ids(record: dict[str, Any]) -> list[str]:
    if "request_id" in record:
        return [normalize_request_id(record["request_id"])]
    if "request_ids" in record:
        return [normalize_request_id(req_id) for req_id in record["request_ids"]]
    return []


def to_ms(delta_ns: int | float | None) -> float | str:
    if delta_ns is None:
        return ""
    return round(delta_ns / 1_000_000, 3)


def as_float(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def metric_value(data: dict[str, Any], *keys: str) -> float | str:
    for key in keys:
        value = as_float(data.get(key))
        if value != "":
            return value
    return ""


def percentile_metric(data: dict[str, Any], metric: str, pct: int) -> float | str:
    return metric_value(data, f"p{pct}_{metric}_ms", f"P{pct}_{metric}_ms",
                        f"{metric}_p{pct}_ms", f"{metric}_P{pct}_ms")


def iter_trace_records(root: Path):
    for path in root.rglob("*.trace.jsonl"):
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["_trace_file"] = str(path)
                    yield record


def parse_case_intervals(root: Path) -> list[dict[str, Any]]:
    intervals = []
    for path in root.rglob("case.trace.jsonl"):
        starts: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                case_id = record.get("case_id")
                if not case_id:
                    continue
                if record.get("event") == "bench_case_start":
                    starts[case_id] = record
                elif record.get("event") == "bench_case_end":
                    start = starts.pop(case_id, None)
                    if start is None:
                        continue
                    intervals.append({
                        "start_ns": start["ts_ns"],
                        "end_ns": record["ts_ns"],
                        "case_id": case_id,
                        "mode": start.get("mode", "pull"),
                        "sleep_ms": start.get("sleep_ms", ""),
                        "input_len": start.get("input_len", ""),
                        "output_len": start.get("output_len", ""),
                        "concurrency": start.get("concurrency", ""),
                        "num_prompts": start.get("num_prompts", ""),
                        "workload": start.get("dataset", ""),
                        "phase": start.get("phase", ""),
                        "variant": start.get("variant", ""),
                        "repetition": start.get("repetition", ""),
                        "configured_request_rate": start.get(
                            "request_rate",
                            start.get("configured_request_rate", ""),
                        ),
                        "injected_transfer_delay_ms": start.get(
                            "injected_transfer_delay_ms", ""),
                    })
    intervals.sort(key=lambda interval: interval["start_ns"])
    return intervals


def parse_benchmark_results(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("*.json"):
        match = BENCH_RESULT_RE.search(path.name)
        if match is None:
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        rows.append({
            "case_id": path.stem,
            "mode": "pull",
            "sleep_ms": float(match.group("sleep_ms")),
            "input_len": int(match.group("input_len")),
            "output_len": int(match.group("output_len")),
            "concurrency": int(match.group("concurrency")),
            "result_file": str(path),
            "completed": data.get("completed", data.get("num_completed_requests", "")),
            "request_throughput": metric_value(data, "request_throughput",
                                                "requests_per_second"),
            "output_throughput": metric_value(data, "output_throughput"),
            "total_token_throughput": metric_value(data, "total_token_throughput",
                                                   "tokens_per_second"),
            "mean_ttft_ms": metric_value(data, "mean_ttft_ms"),
            "p99_ttft_ms": percentile_metric(data, "ttft", 99),
            "mean_tpot_ms": metric_value(data, "mean_tpot_ms"),
            "p99_tpot_ms": percentile_metric(data, "tpot", 99),
            "mean_itl_ms": metric_value(data, "mean_itl_ms"),
            "p99_itl_ms": percentile_metric(data, "itl", 99),
            "mean_e2el_ms": metric_value(data, "mean_e2el_ms"),
            "p99_e2el_ms": percentile_metric(data, "e2el", 99),
        })
    rows.sort(key=lambda r: (r["sleep_ms"], r["input_len"], r["output_len"],
                             r["concurrency"]))
    return rows


def build_timelines(root: Path) -> list[dict[str, Any]]:
    timelines: dict[str, dict[str, Any]] = defaultdict(dict)
    case_intervals = parse_case_intervals(root)
    for record in iter_trace_records(root):
        event = record.get("event")
        if event not in REQUEST_EVENTS:
            continue
        for req_id in req_ids(record):
            row = timelines[req_id]
            row["request_id"] = req_id
            row.setdefault("mode", record.get("mode", "pull"))
            row.setdefault("trace_file", record.get("_trace_file", ""))
            ts_ns = record.get("ts_ns")
            if ts_ns is not None:
                row["_first_ts_ns"] = min(row.get("_first_ts_ns", ts_ns),
                                          ts_ns)
            key = f"{event}_perf_ns"
            if event in LAST_EVENTS or key not in row:
                row[key] = record.get("perf_ns")
            if event == "pull_transfer_end":
                row["pull_transfer_end_injected_sleep_ms"] = record.get(
                    "injected_sleep_ms", "")
            elif event == "pull_transfer_profile":
                for field, value in record.items():
                    if field not in PROFILE_BASE_FIELDS:
                        row[field] = value
    rows = []
    for row in timelines.values():
        first_ts_ns = row.pop("_first_ts_ns", None)
        if first_ts_ns is not None:
            case = next((interval for interval in case_intervals
                         if interval["start_ns"] <= first_ts_ns <=
                         interval["end_ns"]), None)
            if case is not None:
                for field in (
                    "case_id",
                    "mode",
                    "sleep_ms",
                    "input_len",
                    "output_len",
                    "concurrency",
                    "num_prompts",
                    "workload",
                    "phase",
                    "variant",
                    "repetition",
                    "configured_request_rate",
                    "injected_transfer_delay_ms",
                ):
                    row[field] = case[field]

                match = re.fullmatch(
                    rf"{re.escape(str(case['case_id']))}-(\d+)-pdreq",
                    str(row.get("request_id", "")),
                )
                if match is not None:
                    row["case_request_index"] = int(match.group(1))

        def delta(start: str, end: str):
            s = row.get(f"{start}_perf_ns")
            e = row.get(f"{end}_perf_ns")
            return None if s is None or e is None else e - s

        row["proxy_prefill_rpc_ms"] = to_ms(delta("proxy_prefill_request_start",
                                                  "proxy_prefill_request_end"))
        row["proxy_decode_rpc_ms"] = to_ms(delta("proxy_decode_request_start",
                                                 "proxy_decode_request_end"))
        row["proxy_ttft_ms"] = to_ms(delta("proxy_request_received",
                                           "proxy_first_response_chunk"))
        row["proxy_e2e_ms"] = to_ms(delta("proxy_request_received",
                                           "proxy_decode_request_end"))
        row["transfer_wall_ms"] = to_ms(delta("pull_transfer_start",
                                              "pull_transfer_end"))
        row["prefill_to_transfer_end_ms"] = to_ms(delta(
            "pull_prefill_finished", "pull_transfer_end"))
        row["decode_start_to_transfer_end_ms"] = to_ms(delta(
            "proxy_decode_request_start", "pull_transfer_end"))
        row["kv_alloc_to_load_ms"] = to_ms(delta(
            "pull_decode_kv_allocated", "kv_load_start"))
        row["kv_load_to_remote_ready_ms"] = to_ms(delta(
            "kv_load_start", "decode_remote_kv_ready"))
        row["connector_finished_to_remote_ready_ms"] = to_ms(delta(
            "connector_finished", "decode_remote_kv_ready"))
        row["remote_ready_to_schedulable_ms"] = to_ms(delta(
            "decode_remote_kv_ready", "decode_remote_kv_schedulable"))
        row["prefill_to_remote_kv_ready_ms"] = to_ms(delta(
            "pull_prefill_finished", "decode_remote_kv_ready"))
        row["kv_ready_to_first_chunk_ms"] = to_ms(delta(
            "decode_remote_kv_ready", "proxy_first_response_chunk"))

        # The packed path pipelines prepare/submit calls across chunks, so its
        # per-call timers overlap and must not be stacked.  These four slices
        # instead use monotonic request landmarks and form a non-overlapping
        # decomposition of kv_load_start -> connector_finished.
        issue_starts = [
            value for event in ("xfer_prepare_start", "xfer_submit_start")
            if (value := row.get(f"{event}_perf_ns")) is not None
        ]
        issue_ends = [
            value for event in ("xfer_prepare_end", "xfer_submit_end")
            if (value := row.get(f"{event}_perf_ns")) is not None
        ]
        kv_load_start = row.get("kv_load_start_perf_ns")
        done_observed = row.get("xfer_done_observed_perf_ns")
        connector_finished = row.get("connector_finished_perf_ns")
        if (kv_load_start is not None and issue_starts and issue_ends
                and done_observed is not None
                and connector_finished is not None):
            first_issue = min(issue_starts)
            last_issue = max(issue_ends)
            slices_ns = (
                first_issue - kv_load_start,
                last_issue - first_issue,
                done_observed - last_issue,
                connector_finished - done_observed,
            )
            if all(value >= 0 for value in slices_ns):
                fields = (
                    "transfer_before_issue_ms",
                    "transfer_issue_span_ms",
                    "transfer_post_issue_wait_ms",
                    "transfer_connector_finalize_ms",
                )
                for field, value in zip(fields, slices_ns):
                    row[field] = to_ms(value)
                row["transfer_accounted_wall_ms"] = to_ms(sum(slices_ns))
                visible_wall = as_float(
                    row.get("kv_load_to_connector_finished_ms"))
                if visible_wall != "":
                    row["transfer_critical_path_residual_ms"] = round(
                        visible_wall - sum(slices_ns) / 1_000_000, 6)
        rows.append(row)
    rows.sort(key=lambda r: r.get("request_id", ""))
    return rows


def diff(row: dict[str, Any], baseline: dict[str, Any], key: str) -> float | str:
    cur = as_float(row.get(key))
    base = as_float(baseline.get(key))
    if cur == "" or base == "":
        return ""
    return round(cur - base, 6)


def summarize_sleep_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["input_len"], row["output_len"], row["concurrency"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        baseline = next((item for item in items if item["sleep_ms"] == 0), None)
        if baseline is None:
            continue
        for item in sorted(items, key=lambda r: r["sleep_ms"]):
            out.append({
                "mode": "pull",
                "input_len": key[0],
                "output_len": key[1],
                "concurrency": key[2],
                "sleep_ms": item["sleep_ms"],
                "delta_p99_ttft_ms": diff(item, baseline, "p99_ttft_ms"),
                "delta_p99_tpot_ms": diff(item, baseline, "p99_tpot_ms"),
                "delta_p99_e2el_ms": diff(item, baseline, "p99_e2el_ms"),
                "delta_request_throughput": diff(item, baseline,
                                                 "request_throughput"),
                "delta_output_throughput": diff(item, baseline,
                                                "output_throughput"),
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Result directory, or a bare run ID searched below the current directory",
    )
    args = parser.parse_args()
    try:
        root = resolve_result_root(args.root)
    except ValueError as error:
        parser.error(str(error))
    bench_rows = parse_benchmark_results(root)
    timeline_rows = build_timelines(root)
    sensitivity_rows = summarize_sleep_sensitivity(bench_rows)
    write_csv(root / "serve_summary.csv", bench_rows)
    write_csv(root / "sleep_sensitivity_summary.csv", sensitivity_rows)
    write_csv(root / "pd_request_timeline_ms.csv", timeline_rows)
    print(f"Wrote summaries under {root}")


if __name__ == "__main__":
    main()
