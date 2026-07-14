#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUEST_EVENTS = [
    "proxy_request_received",
    "proxy_prefill_request_start",
    "proxy_prefill_request_end",
    "proxy_decode_request_start",
    "proxy_first_response_chunk",
    "proxy_decode_request_end",
    "pull_prefill_finished",
    "pull_transfer_start",
    "pull_transfer_sleep_start",
    "pull_transfer_sleep_end",
    "pull_transfer_end",
    "pull_kv_recv_done",
    "decode_remote_kv_ready",
]

LAST_EVENTS = {
    "proxy_prefill_request_end",
    "proxy_first_response_chunk",
    "proxy_decode_request_end",
    "pull_transfer_end",
    "pull_kv_recv_done",
    "decode_remote_kv_ready",
}

REQUEST_ID_SUFFIX_RE = re.compile(r"-\d+$")
BENCH_RESULT_RE = re.compile(
    r"pull-sleep-(?P<sleep_ms>\d+(?:\.\d+)?)-input-"
    r"(?P<input_len>\d+)-output-(?P<output_len>\d+)-concurrency-"
    r"(?P<concurrency>\d+)\.json$")


def normalize_request_id(request_id: Any) -> str:
    return REQUEST_ID_SUFFIX_RE.sub("", str(request_id))


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
    for record in iter_trace_records(root):
        event = record.get("event")
        if event not in REQUEST_EVENTS:
            continue
        for req_id in req_ids(record):
            row = timelines[req_id]
            row["request_id"] = req_id
            row.setdefault("mode", record.get("mode", "pull"))
            row.setdefault("trace_file", record.get("_trace_file", ""))
            key = f"{event}_perf_ns"
            if event in LAST_EVENTS or key not in row:
                row[key] = record.get("perf_ns")
            if event == "pull_transfer_end":
                row["pull_transfer_end_injected_sleep_ms"] = record.get(
                    "injected_sleep_ms", "")
    rows = []
    for row in timelines.values():

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
        row["transfer_wall_ms"] = to_ms(delta("pull_transfer_start",
                                              "pull_transfer_end"))
        row["prefill_to_transfer_end_ms"] = to_ms(delta(
            "pull_prefill_finished", "pull_transfer_end"))
        row["decode_start_to_transfer_end_ms"] = to_ms(delta(
            "proxy_decode_request_start", "pull_transfer_end"))
        row["kv_ready_to_first_chunk_ms"] = to_ms(delta(
            "decode_remote_kv_ready", "proxy_first_response_chunk"))
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
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    bench_rows = parse_benchmark_results(root)
    timeline_rows = build_timelines(root)
    sensitivity_rows = summarize_sleep_sensitivity(bench_rows)
    write_csv(root / "serve_summary.csv", bench_rows)
    write_csv(root / "sleep_sensitivity_summary.csv", sensitivity_rows)
    write_csv(root / "pd_request_timeline_ms.csv", timeline_rows)
    print(f"Wrote summaries under {root}")


if __name__ == "__main__":
    main()
