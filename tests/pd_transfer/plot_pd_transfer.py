#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create charts from PD-transfer benchmark JSON results."""

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


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("*.json"):
        case = parse_case_name(path)
        if case is None:
            continue
        with path.open(encoding="utf-8-sig") as result_file:
            result = json.load(result_file)
        row: dict[str, Any] = {"run_id": run_id(root, path), "result_file": str(path), **case}
        for metric, _, _ in METRICS:
            row[metric] = number(result.get(metric))
        rows.append(row)
    return sorted(rows, key=lambda row: (
        row["run_id"], row["input_len"], row["output_len"],
        row["concurrency"], row["sleep_ms"]))


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["sleep_ms"] == 0:
            key = (row["run_id"], row["input_len"], row["output_len"], row["concurrency"])
            baselines[key] = row
    for row in rows:
        key = (row["run_id"], row["input_len"], row["output_len"], row["concurrency"])
        baseline = baselines.get(key)
        for metric, _, _ in METRICS:
            value = row[metric]
            base = None if baseline is None else baseline[metric]
            row[f"baseline_{metric}"] = base
            row[f"delta_{metric}"] = None if value is None or base is None else value - base
            row[f"pct_change_{metric}"] = (
                None if value is None or base in (None, 0) else (value - base) / base * 100)


def grouped(rows: list[dict[str, Any]], fields: tuple[str, ...]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups.items()


def token_label(value: int) -> str:
    return f"{value // 1024}k" if value % 1024 == 0 else str(value)


def set_x_axis(axis: plt.Axes, values: list[float], kind: str) -> None:
    if kind in {"input", "concurrency"}:
        axis.set_xscale("log", base=2)
    axis.set_xticks(values)
    labels = [token_label(int(value)) for value in values] if kind == "input" else [f"{value:g}" for value in values]
    axis.set_xticklabels(labels)


def save_values_chart(rows: list[dict[str, Any]], x_field: str, x_label: str,
                      x_kind: str, title: str, destination: Path) -> None:
    rows = sorted(rows, key=lambda row: row[x_field])
    x_values = [row[x_field] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    figure.suptitle(title, fontsize=13)
    for axis, (metric, label, unit) in zip(axes, METRICS):
        values = [row[metric] for row in rows]
        axis.plot(x_values, values, marker="o", linewidth=2, color="#1f77b4")
        for x_value, value in zip(x_values, values):
            axis.annotate(f"{value:.2f}", (x_value, value), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8)
        set_x_axis(axis, x_values, x_kind)
        axis.set_title(label)
        axis.set_xlabel(x_label)
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_proportional_chart(rows: list[dict[str, Any]], title: str, destination: Path) -> None:
    rows = sorted(rows, key=lambda row: row["input_len"])
    x_values = [row["input_len"] for row in rows]
    metrics = (
        ("delta_mean_ttft_ms", "Mean TTFT change", "ms"),
        ("delta_p99_ttft_ms", "P99 TTFT change", "ms"),
        ("pct_change_request_throughput", "Request throughput change", "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    figure.suptitle(title, fontsize=13)
    for axis, (metric, label, unit) in zip(axes, metrics):
        values = [row[metric] for row in rows]
        axis.axhline(0, color="#444444", linewidth=1)
        axis.plot(x_values, values, marker="o", linewidth=2, color="#d62728")
        for x_value, value in zip(x_values, values):
            axis.annotate(f"{value:+.2f}", (x_value, value), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8)
        set_x_axis(axis, x_values, "input")
        axis.set_title(label)
        axis.set_xlabel("Input length (tokens)")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def chart_name(*parts: Any) -> str:
    return "-".join(str(part).replace(".", "p") for part in parts) + ".png"


def chart_title(kind: str, values: dict[str, Any]) -> str:
    input_value = token_label(values["input_len"]) if "input_len" in values else "varied"
    sleep_value = f"{values['sleep_ms']:g}" if "sleep_ms" in values else "varied"
    concurrency = values.get("concurrency", "varied")
    return (f"{kind}: input={input_value}, output={values['output_len']}, "
            f"sleep={sleep_value} ms, concurrency={concurrency}")


def create_standard_charts(rows: list[dict[str, Any]], output_dir: Path) -> int:
    definitions = (
        (("run_id", "input_len", "output_len", "concurrency"), "sleep_ms", "Injected sleep (ms)", "sleep", "01_sleep_sweep", "Sleep sweep"),
        (("run_id", "input_len", "output_len", "sleep_ms"), "concurrency", "Concurrency", "concurrency", "02_concurrency_sweep", "Concurrency sweep"),
        (("run_id", "output_len", "sleep_ms", "concurrency"), "input_len", "Input length (tokens)", "input", "03_input_sweep", "Input sweep"),
    )
    count = 0
    for fields, x_field, x_label, x_kind, directory, kind in definitions:
        for key, items in grouped(rows, fields):
            if len({row[x_field] for row in items}) < 2:
                continue
            values = dict(zip(fields, key))
            file_name = chart_name(*[f"{field}-{values[field]}" for field in fields])
            save_values_chart(items, x_field, x_label, x_kind, chart_title(kind, values),
                              output_dir / values["run_id"] / directory / file_name)
            count += 1
    return count


def create_proportional_charts(rows: list[dict[str, Any]], output_dir: Path) -> int:
    candidates = []
    for row in rows:
        if row["sleep_ms"] > 0 and row["baseline_mean_ttft_ms"] is not None:
            copied = dict(row)
            copied["sleep_per_token"] = row["sleep_ms"] / row["input_len"]
            candidates.append(copied)
    count = 0
    fields = ("run_id", "output_len", "concurrency", "sleep_per_token")
    for key, items in grouped(candidates, fields):
        if len({row["input_len"] for row in items}) < 2:
            continue
        run, output, concurrency, ratio = key
        title = (f"Proportional extrapolation: {ratio * 1024:.2f} ms per 1k input tokens, "
                 f"output={output}, concurrency={concurrency}")
        file_name = chart_name("output", output, "concurrency", concurrency,
                               "ms-per-1k", round(ratio * 1024, 6))
        save_proportional_chart(items, title,
                                output_dir / run / "04_proportional_extrapolation" / file_name)
        count += 1
    return count


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["run_id", "input_len", "output_len", "concurrency", "sleep_ms", "result_file"]
    for metric, _, _ in METRICS:
        fields.extend((metric, f"baseline_{metric}", f"delta_{metric}", f"pct_change_{metric}"))
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="One result run or a directory containing result runs")
    parser.add_argument("--output-dir", type=Path, help="Default: <root>/plots")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    rows = load_rows(root)
    if not rows:
        parser.error(f"No PD benchmark JSON files found below {root}")
    output_dir = (args.output_dir or root / "plots").resolve()
    add_baseline_deltas(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "plot_summary.csv", rows)
    standard_count = create_standard_charts(rows, output_dir)
    proportional_count = create_proportional_charts(rows, output_dir)
    print(f"Read {len(rows)} cases; wrote {standard_count + proportional_count} PNG files under {output_dir}")
    print(f"Wrote {output_dir / 'plot_summary.csv'}")


if __name__ == "__main__":
    main()
