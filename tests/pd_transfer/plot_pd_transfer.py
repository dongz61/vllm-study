#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot sleep sweeps and request-level KV transfer profiles."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("mean_ttft_ms", "Mean TTFT", "ms"),
    ("p99_ttft_ms", "P99 TTFT", "ms"),
    ("request_throughput", "Request throughput", "req/s"),
)

TRANSFER_PHASES = (
    ("handshake_wait_ms", "Handshake wait"),
    ("desc_build_ms", "Descriptor build"),
    ("xfer_prepare_ms", "NIXL prepare"),
    ("xfer_submit_ms", "NIXL submit"),
    ("submit_to_done_observed_ms", "Submitted to DONE observed"),
    ("done_to_connector_finished_ms", "DONE to connector finish"),
)


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def value_after(parts: list[str], key: str) -> str | None:
    try:
        return parts[parts.index(key) + 1]
    except (ValueError, IndexError):
        return None


def parse_case_name(path: Path) -> dict[str, int | float] | None:
    """Parse pull-sleep-<ms>-input-<n>-output-<n>-concurrency-<n>.json."""
    parts = path.stem.split("-")
    if len(parts) < 8 or parts[:2] != ["pull", "sleep"]:
        return None
    values = {key: value_after(parts, key)
              for key in ("sleep", "input", "output", "concurrency")}
    if any(value is None for value in values.values()):
        return None
    try:
        return {
            "sleep_ms": float(values["sleep"]),
            "input_len": int(values["input"]),
            "output_len": int(values["output"]),
            "concurrency": int(values["concurrency"]),
        }
    except ValueError:
        return None


def run_id(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    return root.name if not parts or parts[0] == "pull" else parts[0]


def load_benchmark_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("*.json"):
        case = parse_case_name(path)
        if case is None:
            continue
        with path.open(encoding="utf-8-sig") as result_file:
            result = json.load(result_file)
        row: dict[str, Any] = {
            "run_id": run_id(root, path), "result_file": str(path), **case
        }
        for metric, _, _ in METRICS:
            row[metric] = number(result.get(metric))
        rows.append(row)
    return sorted(rows, key=lambda row: (
        row["run_id"], row["input_len"], row["output_len"],
        row["concurrency"], row["sleep_ms"]))


def load_timeline_rows(root: Path) -> list[dict[str, Any]]:
    """Load parser output, retaining only requests with a transfer profile."""
    rows = []
    for path in root.rglob("pd_request_timeline_ms.csv"):
        with path.open(newline="", encoding="utf-8-sig") as timeline_file:
            for row in csv.DictReader(timeline_file):
                if row.get("transfer_skipped", "").lower() == "true":
                    continue
                if number(row.get("kv_load_to_connector_finished_ms")) is None:
                    continue
                row["run_id"] = run_id(root, path)
                rows.append(row)
    return rows


def grouped(rows: list[dict[str, Any]], fields: tuple[str, ...]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups.items()


def token_label(value: int) -> str:
    return f"{value // 1024}k" if value % 1024 == 0 else str(value)


def save_sleep_sweep(rows: list[dict[str, Any]], title: str,
                     destination: Path) -> None:
    rows = sorted(rows, key=lambda row: row["sleep_ms"])
    x_values = [row["sleep_ms"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    figure.suptitle(title, fontsize=13)
    for axis, (metric, label, unit) in zip(axes, METRICS):
        values = [row[metric] for row in rows]
        axis.plot(x_values, values, marker="o", linewidth=2, color="#1f77b4")
        for x_value, value in zip(x_values, values):
            if value is not None:
                axis.annotate(f"{value:.2f}", (x_value, value), xytext=(0, 6),
                              textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"{value:g}" for value in x_values])
        axis.set_title(label)
        axis.set_xlabel("Injected sleep (ms)")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_sleep_sweeps(rows: list[dict[str, Any]], output_dir: Path) -> int:
    count = 0
    fields = ("run_id", "input_len", "output_len", "concurrency")
    for key, items in grouped(rows, fields):
        if len({row["sleep_ms"] for row in items}) < 2:
            continue
        run, input_len, output_len, concurrency = key
        title = (f"Sleep sweep: input={token_label(input_len)}, output={output_len}, "
                 f"concurrency={concurrency}")
        name = (f"input-{input_len}-output-{output_len}-"
                f"concurrency-{concurrency}.png")
        save_sleep_sweep(items, title,
                         output_dir / run / "01_sleep_sweep" / name)
        count += 1
    return count


def save_pie(labels: list[str], values: list[float], title: str,
             destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.4), layout="constrained")
    _, _, autotexts = axis.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        textprops={"fontsize": 9},
    )
    for text in autotexts:
        text.set_color("white")
    axis.set_title(title, fontsize=12)
    axis.axis("equal")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def case_label(rows: list[dict[str, Any]]) -> str:
    row = rows[0]
    return (f"sleep={row.get('sleep_ms', '?')} ms, "
            f"input={row.get('input_len', '?')}, "
            f"output={row.get('output_len', '?')}, "
            f"concurrency={row.get('concurrency', '?')}")


def create_transfer_pies(rows: list[dict[str, Any]], output_dir: Path) -> int:
    """Create mean-per-request transfer breakdown and request-share pies."""
    count = 0
    fields = ("run_id", "case_id")
    for key, items in grouped(rows, fields):
        run, case_id = key
        total_values = [number(row.get("kv_load_to_connector_finished_ms"))
                        for row in items]
        total_values = [value for value in total_values if value is not None]
        total_ms = mean(total_values)
        if total_ms is None or total_ms <= 0:
            continue

        labels = []
        values = []
        for field, label in TRANSFER_PHASES:
            phase_ms = mean([value for row in items
                             if (value := number(row.get(field))) is not None])
            if phase_ms is not None and phase_ms > 0:
                labels.append(label)
                values.append(phase_ms)
        other_ms = max(total_ms - sum(values), 0)
        if other_ms > 0:
            labels.append("Other connector work")
            values.append(other_ms)
        if values:
            save_pie(
                labels, values,
                f"KV transfer breakdown (mean/request)\n{case_label(items)}",
                output_dir / run / "02_kv_transfer_breakdown" /
                f"{case_id}.png",
            )
            count += 1

        paired = []
        for row in items:
            transfer_ms = number(row.get("kv_load_to_connector_finished_ms"))
            request_ms = number(row.get("proxy_e2e_ms"))
            if (transfer_ms is not None and request_ms is not None
                    and request_ms >= transfer_ms):
                paired.append((transfer_ms, request_ms))
        if not paired:
            continue
        transfer_ms = mean([pair[0] for pair in paired])
        request_ms = mean([pair[1] for pair in paired])
        assert transfer_ms is not None and request_ms is not None
        save_pie(
            ["KV load to connector finish", "Other request time"],
            [transfer_ms, request_ms - transfer_ms],
            f"KV transfer share of end-to-end request time (mean/request)\n"
            f"{case_label(items)}",
            output_dir / run / "03_kv_transfer_request_share" / f"{case_id}.png",
        )
        count += 1
    return count


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["run_id", "input_len", "output_len", "concurrency", "sleep_ms",
              "result_file"]
    fields.extend(metric for metric, _, _ in METRICS)
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="One result run or a directory containing result runs")
    parser.add_argument("--output-dir", type=Path, help="Default: <root>/plots")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    benchmark_rows = load_benchmark_rows(root)
    if not benchmark_rows:
        parser.error(f"No PD benchmark JSON files found below {root}")
    output_dir = (args.output_dir or root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "plot_summary.csv", benchmark_rows)
    sleep_count = create_sleep_sweeps(benchmark_rows, output_dir)
    timeline_rows = load_timeline_rows(root)
    pie_count = create_transfer_pies(timeline_rows, output_dir)
    print(f"Read {len(benchmark_rows)} cases; wrote {sleep_count} sleep sweeps and "
          f"{pie_count} transfer pie charts under {output_dir}")
    if not timeline_rows:
        print("No request timelines found; run parse_pd_trace.py first to create "
              "pd_request_timeline_ms.csv before plotting transfer pies.")
    print(f"Wrote {output_dir / 'plot_summary.csv'}")


if __name__ == "__main__":
    main()
