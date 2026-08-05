#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot report-focused PD transfer results and optional trace diagnostics."""

import argparse
import csv
import json
import statistics
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

TRANSFER_CRITICAL_PATH_PHASES = (
    ("transfer_before_issue_ms", "Before first issue"),
    ("transfer_issue_span_ms", "Issue window"),
    ("transfer_post_issue_wait_ms", "Last issue to DONE"),
    ("transfer_connector_finalize_ms", "Connector finalization"),
)

PACKED_DIAGNOSTICS = (
    ("kv_load_to_connector_finished_ms", "Visible transfer wall"),
    ("packed_source_handler_ms", "Source handler (overlapping)"),
    ("packed_pack_gpu_ms", "Pack GPU (summed kernels)"),
    ("packed_scatter_gpu_ms", "Scatter GPU (summed kernels)"),
    ("packed_source_stream_wait_gpu_ms", "Source stream wait GPU"),
)

VARIANT_COLORS = {"off": "#4c78a8", "on": "#f58518"}
VARIANT_LABELS = {"off": "off / direct", "on": "on / auto"}


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


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [value for row in rows
            if (value := number(row.get(field))) is not None]


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
    relative_path = path.parent if path.name == "pd_request_timeline_ms.csv" else path
    parts = relative_path.relative_to(root).parts
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


def load_generalization_delay_rows(root: Path) -> list[dict[str, Any]]:
    """Load mixed-length generalization results with injected delay metadata."""
    rows = []
    for path in root.rglob("result.json"):
        with path.open(encoding="utf-8-sig") as result_file:
            result = json.load(result_file)
        delay_ms = number(result.get("injected_transfer_delay_ms"))
        if delay_ms is None:
            continue
        row: dict[str, Any] = {
            "run_id": root.name,
            "result_file": str(path),
            "phase": str(result.get("phase", "unknown")),
            "workload": str(result.get("workload", root.name)),
            "variant": str(result.get("variant", "unknown")),
            "repetition": int(result.get("repetition", 0)),
            "configured_request_rate": str(
                result.get("configured_request_rate", "unknown")),
            "transfer_delay_ms": delay_ms,
            "num_prompts": int(result.get("num_prompts", 0)),
            "completed": int(result.get("completed", 0)),
        }
        for metric, _, _ in METRICS:
            row[metric] = number(result.get(metric))
        rows.append(row)
    return sorted(rows, key=lambda row: (
        row["phase"], row["workload"], row["variant"],
        row["configured_request_rate"], row["transfer_delay_ms"],
        row["repetition"]))


def load_timeline_rows(root: Path) -> list[dict[str, Any]]:
    """Load formal benchmark requests with a transfer profile.

    Warm-up requests have no case ID and must not appear in benchmark charts.
    """
    rows = []
    for path in root.rglob("pd_request_timeline_ms.csv"):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as timeline_file:
            for row in csv.DictReader(timeline_file):
                if not row.get("case_id"):
                    continue
                if row.get("transfer_skipped", "").lower() == "true":
                    continue
                if number(row.get("kv_load_to_connector_finished_ms")) is None:
                    continue
                row["run_id"] = run_id(root, path)
                rows.append(row)
    return rows


def load_request_latency_rows(root: Path) -> list[dict[str, Any]]:
    """Load one end-to-end latency sample for every traced proxy request."""
    rows = []
    for path in root.rglob("pd_request_timeline_ms.csv"):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as timeline_file:
            for row in csv.DictReader(timeline_file):
                if (not row.get("case_id")
                        or number(row.get("proxy_e2e_ms")) is None):
                    continue
                row["run_id"] = run_id(root, path)
                rows.append(row)
    return rows


def grouped(rows: list[dict[str, Any]], fields: tuple[str, ...]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups.items()


def rate_sort_key(value: Any) -> tuple[int, float | str]:
    parsed = number(value)
    return (0, parsed) if parsed is not None else (1, str(value))


def steady_state_rows(rows: list[dict[str, Any]],
                      include_cold_handshake: bool) -> list[dict[str, Any]]:
    if include_cold_handshake:
        return list(rows)
    return [row for row in rows
            if row.get("handshake_cached", "").lower() != "false"]


def repetition_center_and_error(values: list[float]
                                ) -> tuple[float, float, float] | None:
    """Return repetition median and asymmetric min/max error distances."""
    if not values:
        return None
    center = statistics.median(values)
    return center, center - min(values), max(values) - center


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


def filename_slug(value: Any) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-"
                   for character in str(value)).strip("-") or "unknown"


def save_generalization_delay_sweep(rows: list[dict[str, Any]], title: str,
                                    destination: Path) -> None:
    by_delay = {
        key[0]: items
        for key, items in grouped(rows, ("transfer_delay_ms", ))
    }
    delay_values = sorted(by_delay)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    figure.suptitle(title, fontsize=13)
    for axis, (metric, label, unit) in zip(axes, METRICS):
        medians = []
        minimum_errors = []
        maximum_errors = []
        repetition_counts = []
        for delay_ms in delay_values:
            values = [
                value for row in by_delay[delay_ms]
                if (value := number(row.get(metric))) is not None
            ]
            if not values:
                medians.append(float("nan"))
                minimum_errors.append(0.0)
                maximum_errors.append(0.0)
                repetition_counts.append(0)
                continue
            median_value = statistics.median(values)
            medians.append(median_value)
            minimum_errors.append(median_value - min(values))
            maximum_errors.append(max(values) - median_value)
            repetition_counts.append(len(values))
        axis.errorbar(
            delay_values,
            medians,
            yerr=[minimum_errors, maximum_errors],
            marker="o",
            linewidth=2,
            capsize=4,
            color="#1f77b4",
        )
        for delay_ms, value, repetitions in zip(
                delay_values, medians, repetition_counts):
            if repetitions:
                suffix = f" (n={repetitions})" if repetitions > 1 else ""
                axis.annotate(
                    f"{value:.2f}{suffix}",
                    (delay_ms, value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        axis.set_xticks(delay_values)
        axis.set_xticklabels([f"{value:g}" for value in delay_values])
        axis.set_title(label)
        axis.set_xlabel("Injected transfer completion delay (ms)")
        axis.set_ylabel(unit)
        axis.ticklabel_format(style="plain", axis="y", useOffset=False)
        axis.grid(True, alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_generalization_delay_sweeps(
        rows: list[dict[str, Any]], output_dir: Path) -> int:
    count = 0
    fields = ("run_id", "phase", "workload", "variant",
              "configured_request_rate")
    for key, items in grouped(rows, fields):
        if len({row["transfer_delay_ms"] for row in items}) < 2:
            continue
        run, phase, workload, variant, request_rate = key
        trace_note = ", trace enabled" if phase == "diagnostic" else ""
        title = (
            f"Transfer-delay sweep ({phase}{trace_note})\n"
            f"{workload}, variant={variant}, rate={request_rate}"
        )
        name = (
            f"{filename_slug(phase)}-{filename_slug(workload)}-"
            f"{filename_slug(variant)}-rps-{filename_slug(request_rate)}.png"
        )
        save_generalization_delay_sweep(
            items,
            title,
            output_dir / run / "01_generalization_delay_sweep" / name,
        )
        count += 1
    return count


def save_request_latency_scatter(rows: list[dict[str, Any]], title: str,
                                 destination: Path) -> None:
    rows = sorted(rows, key=lambda row: number(
        row.get("proxy_request_received_perf_ns")) or float("inf"))
    x_values = list(range(1, len(rows) + 1))
    y_values = [number(row["proxy_e2e_ms"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.scatter(x_values, y_values, s=26, alpha=0.8, color="#1f77b4")
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Request arrival order within case")
    axis.set_ylabel("End-to-end request latency (ms)")
    axis.grid(True, alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_request_latency_scatters(rows: list[dict[str, Any]],
                                    output_dir: Path) -> int:
    """Plot raw per-request end-to-end latency for every benchmark case."""
    count = 0
    for (run, case_id), items in grouped(rows, ("run_id", "case_id")):
        row = items[0]
        if row.get("phase"):
            title = ("Raw end-to-end request latency\n"
                     f"{row.get('workload')}, {row.get('phase')}, "
                     f"variant={row.get('variant')}, "
                     f"rate={row.get('configured_request_rate')}, "
                     f"rep={row.get('repetition')}, requests={len(items)}")
        else:
            title = ("Raw end-to-end request latency\n"
                     f"sleep={row['sleep_ms']} ms, input={row['input_len']}, "
                     f"output={row['output_len']}, "
                     f"concurrency={row['concurrency']}, requests={len(items)}")
        save_request_latency_scatter(
            items, title,
            output_dir / run / "04_request_e2e_scatter" / f"{case_id}.png")
        count += 1
    return count


def save_transfer_latency_scatter(rows: list[dict[str, Any]], title: str,
                                  destination: Path) -> None:
    rows = sorted(rows, key=lambda row: number(
        row.get("kv_load_start_perf_ns")) or float("inf"))
    x_values = list(range(1, len(rows) + 1))
    y_values = [number(row["kv_load_to_connector_finished_ms"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.scatter(x_values, y_values, s=26, alpha=0.8, color="#d62728")
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("KV load start order within case")
    axis.set_ylabel("KV load to connector finish (ms)")
    axis.grid(True, alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_transfer_latency_scatters(rows: list[dict[str, Any]],
                                     output_dir: Path,
                                     include_cold_handshake: bool = False) -> int:
    """Plot raw steady-state transfer latency per case."""
    count = 0
    for (run, case_id), items in grouped(rows, ("run_id", "case_id")):
        cold_excluded = 0
        if not include_cold_handshake:
            cold_excluded = sum(
                row.get("handshake_cached", "").lower() == "false"
                for row in items)
            items = [row for row in items
                     if row.get("handshake_cached", "").lower() != "false"]
        if not items:
            continue
        row = items[0]
        cold = sum(row.get("handshake_cached", "").lower() == "false"
                   for row in items)
        if row.get("phase"):
            title = ("Raw KV transfer latency\n"
                     f"{row.get('workload')}, {row.get('phase')}, "
                     f"variant={row.get('variant')}, "
                     f"rate={row.get('configured_request_rate')}, "
                     f"rep={row.get('repetition')}\n"
                     f"profiles={len(items)}, cold handshake={cold}, "
                     f"cold excluded={cold_excluded}")
        else:
            title = ("Raw KV transfer latency\n"
                     f"input={row['input_len']}, output={row['output_len']}, "
                     f"sleep={row['sleep_ms']} ms, "
                     f"concurrency={row['concurrency']}\n"
                     f"profiles={len(items)}, cold handshake={cold}, "
                     f"cold excluded={cold_excluded}")
        save_transfer_latency_scatter(
            items, title,
            output_dir / run / "05_kv_transfer_scatter" / f"{case_id}.png")
        count += 1
    return count


def save_pie(labels: list[str], values: list[float], title: str,
             destination: Path) -> None:
    total = sum(values)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    wedges, _, autotexts = axis.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        textprops={"fontsize": 9},
    )
    for text in autotexts:
        text.set_color("white")
    axis.set_title(title, fontsize=12)
    axis.axis("equal")
    legend_labels = [
        f"{label}: {value:.3f} ms ({value / total * 100:.1f}%)"
        for label, value in zip(labels, values)
    ]
    axis.legend(wedges, legend_labels, title="Mean per request",
                loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def case_label(rows: list[dict[str, Any]], cold_excluded: int = 0) -> str:
    row = rows[0]
    cached = sum(row.get("handshake_cached", "").lower() == "true"
                 for row in rows)
    cold = sum(row.get("handshake_cached", "").lower() == "false"
               for row in rows)
    if row.get("phase"):
        configuration = (
            f"{row.get('workload')}, {row.get('phase')}, "
            f"variant={row.get('variant')}, "
            f"rate={row.get('configured_request_rate')}, "
            f"rep={row.get('repetition')}"
        )
    else:
        configuration = (
            f"sleep={row.get('sleep_ms', '?')} ms, "
            f"input={row.get('input_len', '?')}, "
            f"output={row.get('output_len', '?')}, "
            f"concurrency={row.get('concurrency', '?')}"
        )
    return (f"{configuration}\nprofiles={len(rows)} "
            f"(cold handshake={cold}, cached={cached}, "
            f"cold excluded={cold_excluded})")


def create_transfer_pies(rows: list[dict[str, Any]], output_dir: Path,
                         include_cold_handshake: bool = False) -> int:
    """Create mean-per-request transfer breakdown and request-share pies."""
    count = 0
    fields = ("run_id", "case_id")
    for key, items in grouped(rows, fields):
        run, case_id = key
        cold_excluded = 0
        if not include_cold_handshake:
            cold_excluded = sum(
                row.get("handshake_cached", "").lower() == "false"
                for row in items)
            items = [row for row in items
                     if row.get("handshake_cached", "").lower() != "false"]
        if not items:
            continue
        total_values = [number(row.get("kv_load_to_connector_finished_ms"))
                        for row in items]
        total_values = [value for value in total_values if value is not None]
        total_ms = mean(total_values)
        if total_ms is None or total_ms <= 0:
            continue

        labels = []
        values = []
        critical_path_available = all(
            all(number(row.get(field)) is not None
                for field, _ in TRANSFER_CRITICAL_PATH_PHASES)
            for row in items
        )
        phases = (TRANSFER_CRITICAL_PATH_PHASES if critical_path_available
                  else TRANSFER_PHASES)
        for field, label in phases:
            # A missing phase means this request did not execute it. Include
            # it as zero so phase and total averages use the same population.
            phase_ms = mean([
                value if (value := number(row.get(field))) is not None else 0.0
                for row in items
            ])
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
                f"KV transfer breakdown (mean/request, steady state"
                f"{', non-overlapping landmarks' if critical_path_available else ''})\n"
                f"{case_label(items, cold_excluded)}",
                output_dir / run / "02_kv_transfer_breakdown" /
                f"{case_id}.png",
            )
            count += 1

        paired = []
        for row in items:
            transfer_ms = number(row.get("kv_load_to_connector_finished_ms"))
            ttft_ms = number(row.get("proxy_ttft_ms"))
            if (transfer_ms is not None and ttft_ms is not None
                    and ttft_ms >= transfer_ms):
                paired.append((transfer_ms, ttft_ms))
        if not paired:
            continue
        transfer_ms = mean([pair[0] for pair in paired])
        ttft_ms = mean([pair[1] for pair in paired])
        assert transfer_ms is not None and ttft_ms is not None
        save_pie(
            ["KV load to connector finish", "Other TTFT-path time"],
            [transfer_ms, ttft_ms - transfer_ms],
            f"KV transfer share of time to first token "
            f"(mean/request, steady state)\n{case_label(items, cold_excluded)}",
            output_dir / run / "03_kv_transfer_ttft_share" / f"{case_id}.png",
        )
        count += 1
    return count


def build_motivation_points(
        rows: list[dict[str, Any]],
        include_cold_handshake: bool = False,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build repetition-level paired-delay and delay-zero summaries."""
    rows = steady_state_rows(rows, include_cold_handshake)
    required = (
        "run_id", "phase", "workload", "variant", "repetition",
        "configured_request_rate", "injected_transfer_delay_ms",
        "case_request_index",
    )
    rows = [row for row in rows
            if all(row.get(field, "") != "" for field in required)
            and number(row.get("proxy_ttft_ms")) is not None]
    repetition_fields = (
        "run_id", "phase", "workload", "variant",
        "configured_request_rate", "repetition",
    )
    delay_points = []
    baseline_points = []
    for key, items in grouped(rows, repetition_fields):
        by_delay: dict[float, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in items:
            delay_ms = number(row.get("injected_transfer_delay_ms"))
            if delay_ms is None:
                continue
            by_delay[delay_ms][str(row["case_request_index"])] = row
        baseline = by_delay.get(0.0)
        if not baseline:
            continue
        metadata = dict(zip(repetition_fields, key))
        for delay_ms, delayed in sorted(by_delay.items()):
            paired_ids = sorted(set(baseline) & set(delayed))
            deltas = [
                number(delayed[request_id]["proxy_ttft_ms"])
                - number(baseline[request_id]["proxy_ttft_ms"])
                for request_id in paired_ids
            ]
            deltas = [value for value in deltas if value is not None]
            if deltas:
                delay_points.append({
                    **metadata,
                    "transfer_delay_ms": delay_ms,
                    "paired_request_count": len(deltas),
                    "paired_delta_ttft_p50_ms": percentile(deltas, 50),
                })

        baseline_paired = [
            row for row in baseline.values()
            if number(row.get("kv_load_to_connector_finished_ms")) is not None
            and number(row.get("proxy_ttft_ms")) is not None
        ]
        transfer_values = numeric_values(
            baseline_paired, "kv_load_to_connector_finished_ms")
        ttft_values = numeric_values(baseline_paired, "proxy_ttft_ms")
        if transfer_values and ttft_values:
            baseline_points.append({
                **metadata,
                "request_count": len(baseline_paired),
                "transfer_p50_ms": percentile(transfer_values, 50),
                "ttft_p50_ms": percentile(ttft_values, 50),
                "aggregate_transfer_ttft_share_pct": (
                    sum(transfer_values) / sum(ttft_values) * 100
                    if sum(ttft_values) > 0 else None
                ),
            })
    return delay_points, baseline_points


def _aggregate_repetitions(rows: list[dict[str, Any]], field: str
                           ) -> tuple[float, float, float] | None:
    return repetition_center_and_error(numeric_values(rows, field))


def save_motivation_paired_delay_figure(
        delay_points: list[dict[str, Any]], title: str,
        destination: Path) -> None:
    """Save one three-load paired delta-TTFT figure."""
    rates = sorted({row["configured_request_rate"] for row in delay_points},
                   key=rate_sort_key)
    if len(rates) > 3:
        raise ValueError("Paired delay figure supports at most three loads")

    figure, axes = plt.subplots(
        1, len(rates), figsize=(5.2 * len(rates), 4.9),
        layout="constrained", squeeze=False, sharey=True,
    )
    figure.suptitle(title, fontsize=14)

    global_values = numeric_values(delay_points, "paired_delta_ttft_p50_ms")
    all_delays = numeric_values(delay_points, "transfer_delay_ms")
    y_min = min([0.0, *global_values]) if global_values else 0.0
    y_max = max([0.0, *global_values, *all_delays]) if all_delays else 1.0
    padding = max((y_max - y_min) * 0.1, 5.0)

    for axis, rate in zip(axes[0], rates):
        rate_rows = [row for row in delay_points
                     if row["configured_request_rate"] == rate]
        delay_values = sorted({row["transfer_delay_ms"] for row in rate_rows})
        centers = []
        lower_errors = []
        upper_errors = []
        for delay_ms in delay_values:
            aggregate = _aggregate_repetitions(
                [row for row in rate_rows
                 if row["transfer_delay_ms"] == delay_ms],
                "paired_delta_ttft_p50_ms",
            )
            assert aggregate is not None
            center, lower, upper = aggregate
            centers.append(center)
            lower_errors.append(lower)
            upper_errors.append(upper)
        axis.errorbar(
            delay_values, centers, yerr=[lower_errors, upper_errors],
            marker="o", linewidth=2.2, capsize=5, color="#e45756",
            label="Measured paired P50",
        )
        identity_limit = max(delay_values) if delay_values else 0
        axis.plot([0, identity_limit], [0, identity_limit], "--",
                  color="#777777", linewidth=1.5, label="1:1 propagation")
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Arrival multiplier = {rate}")
        axis.set_xlabel("Injected transfer delay (ms)")
        axis.set_ylabel("Paired delta TTFT P50 (ms)")
        axis.set_xticks(delay_values)
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, loc="upper left")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_motivation_delay0_load_figure(
        baseline_points: list[dict[str, Any]], title: str,
        destination: Path) -> None:
    """Save transfer and TTFT latency trends at zero delay."""
    rates = sorted({row["configured_request_rate"]
                    for row in baseline_points}, key=rate_sort_key)
    x_values = list(range(len(rates)))
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4.6), layout="constrained", squeeze=False,
    )
    figure.suptitle(title, fontsize=14)
    panels = (
        ("transfer_p50_ms", "#4c78a8", "Transfer P50 (ms)",
         "Request-visible transfer latency"),
        ("ttft_p50_ms", "#f58518", "TTFT P50 (ms)", "Time to first token"),
    )
    for axis, (field, color, ylabel, panel_title) in zip(axes[0], panels):
        centers = []
        lower_errors = []
        upper_errors = []
        for rate in rates:
            aggregate = _aggregate_repetitions(
                [row for row in baseline_points
                 if row["configured_request_rate"] == rate], field)
            assert aggregate is not None
            center, lower, upper = aggregate
            centers.append(center)
            lower_errors.append(lower)
            upper_errors.append(upper)
        axis.errorbar(x_values, centers,
                      yerr=[lower_errors, upper_errors], marker="o",
                      linewidth=2.2, capsize=5, color=color)
        axis.set_ylabel(ylabel)
        axis.set_title(panel_title)
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(rate) for rate in rates])
        axis.set_xlabel("Recorded arrival-rate multiplier")
        axis.grid(True, alpha=0.25)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_motivation_transfer_share_figure(
        baseline_points: list[dict[str, Any]], title: str,
        destination: Path) -> None:
    """Save aggregate request-visible transfer share across load."""
    rates = sorted({row["configured_request_rate"]
                    for row in baseline_points}, key=rate_sort_key)
    x_values = list(range(len(rates)))
    centers = []
    lower_errors = []
    upper_errors = []
    for rate in rates:
        aggregate = _aggregate_repetitions(
            [row for row in baseline_points
             if row["configured_request_rate"] == rate],
            "aggregate_transfer_ttft_share_pct",
        )
        assert aggregate is not None
        center, lower, upper = aggregate
        centers.append(center)
        lower_errors.append(lower)
        upper_errors.append(upper)
    figure, axis = plt.subplots(figsize=(6.5, 4.6), layout="constrained")
    axis.errorbar(
        x_values, centers, yerr=[lower_errors, upper_errors], marker="o",
        linewidth=2.2, capsize=5, color="#54a24b",
    )
    for x_value, center in zip(x_values, centers):
        axis.annotate(f"{center:.2f}%", (x_value, center), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=9)
    axis.set_xticks(x_values)
    axis.set_xticklabels([str(rate) for rate in rates])
    axis.set_xlabel("Recorded arrival-rate multiplier")
    axis.set_ylabel("sum(transfer) / sum(TTFT) (%)")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_motivation_core_analysis(
        rows: list[dict[str, Any]], output_dir: Path,
        include_cold_handshake: bool = False) -> int:
    delay_points, baseline_points = build_motivation_points(
        rows, include_cold_handshake)
    if len({row["transfer_delay_ms"] for row in delay_points}) < 2:
        return 0
    write_dict_rows(output_dir / "motivation_paired_delta_ttft.csv",
                    delay_points)
    write_dict_rows(output_dir / "motivation_delay0_baseline.csv",
                    baseline_points)
    count = 0
    scenario_fields = ("run_id", "phase", "workload", "variant")
    for (run, phase, workload, variant), items in grouped(
            delay_points, scenario_fields):
        scenario_baseline = [
            row for row in baseline_points
            if row["run_id"] == run and row["phase"] == phase
            and row["workload"] == workload and row["variant"] == variant
        ]
        if not scenario_baseline:
            continue
        slug = f"{filename_slug(phase)}-{filename_slug(workload)}-{variant}"
        base = output_dir / run / "01_motivation_core"
        save_motivation_paired_delay_figure(
            items,
            f"KV transfer latency is on the TTFT critical path\n{workload} ({phase})",
            base / f"{slug}-paired-delta-ttft.png",
        )
        save_motivation_delay0_load_figure(
            scenario_baseline,
            f"Delay = 0 baseline across load\n{workload} ({phase})",
            base / f"{slug}-delay0-load-summary.png",
        )
        save_motivation_transfer_share_figure(
            scenario_baseline,
            f"Request-visible KV transfer share of TTFT\n"
            f"{workload} ({phase}, delay=0)",
            base / f"{slug}-delay0-transfer-share.png",
        )
        count += 3
    return count


def summarize_transfer_cases(
        rows: list[dict[str, Any]],
        include_cold_handshake: bool = False) -> list[dict[str, Any]]:
    """Summarize request-visible transfer latency per repetition and path."""
    rows = steady_state_rows(rows, include_cold_handshake)
    case_fields = (
        "run_id", "phase", "workload", "variant", "repetition",
        "configured_request_rate", "case_id",
    )
    summaries = []
    for case_key, case_rows in grouped(rows, case_fields):
        path_groups = [("all", case_rows)]
        path_groups.extend(
            (path, path_rows)
            for (path,), path_rows in grouped(case_rows,
                                               ("selected_transfer_path",))
            if path
        )
        for path, items in path_groups:
            visible = numeric_values(items, "kv_load_to_connector_finished_ms")
            if not visible:
                continue
            summary = dict(zip(case_fields, case_key))
            summary.update({
                "selected_transfer_path": path,
                "request_count": len(visible),
                "transfer_mean_ms": mean(visible),
                "transfer_p50_ms": percentile(visible, 50),
                "transfer_p90_ms": percentile(visible, 90),
                "transfer_p99_ms": percentile(visible, 99),
            })
            selected_paths = [row.get("selected_transfer_path", "")
                              for row in items]
            summary["direct_request_count"] = selected_paths.count("direct")
            summary["packed_request_count"] = selected_paths.count("packed")
            summary["packed_request_fraction"] = (
                selected_paths.count("packed") / len(selected_paths)
                if selected_paths else None
            )
            paired_ttft = [
                (transfer, ttft)
                for row in items
                if (transfer := number(row.get(
                    "kv_load_to_connector_finished_ms"))) is not None
                and (ttft := number(row.get("proxy_ttft_ms"))) is not None
                and ttft > 0
            ]
            if paired_ttft:
                summary["aggregate_transfer_ttft_share_pct"] = (
                    sum(pair[0] for pair in paired_ttft)
                    / sum(pair[1] for pair in paired_ttft) * 100
                )
            bandwidths = [
                total_bytes / transfer_ms / 1_000_000
                for row in items
                if (total_bytes := number(row.get("total_bytes"))) is not None
                and (transfer_ms := number(row.get(
                    "kv_load_to_connector_finished_ms"))) is not None
                and transfer_ms > 0
            ]
            summary["effective_bandwidth_p50_GBps"] = percentile(
                bandwidths, 50)
            for field, _ in TRANSFER_CRITICAL_PATH_PHASES:
                summary[f"mean_{field}"] = mean(numeric_values(items, field))
            for field, _ in PACKED_DIAGNOSTICS[1:]:
                summary[f"p50_{field}"] = percentile(
                    numeric_values(items, field), 50)
                summary[f"p90_{field}"] = percentile(
                    numeric_values(items, field), 90)
            summaries.append(summary)
    return summaries


def pair_transfer_requests(rows: list[dict[str, Any]],
                           include_cold_handshake: bool = False
                           ) -> list[dict[str, Any]]:
    """Match off/on requests by workload position within each repetition."""
    rows = steady_state_rows(rows, include_cold_handshake)
    pair_fields = (
        "run_id", "phase", "workload", "configured_request_rate",
        "repetition", "case_request_index",
    )
    candidates = [row for row in rows
                  if row.get("variant") in {"off", "on"}
                  and row.get("case_request_index", "") != ""]
    pairs = []
    for key, items in grouped(candidates, pair_fields):
        by_variant = {row["variant"]: row for row in items}
        if set(by_variant) != {"off", "on"}:
            continue
        baseline = by_variant["off"]
        optimized = by_variant["on"]
        baseline_ms = number(baseline.get("kv_load_to_connector_finished_ms"))
        optimized_ms = number(optimized.get("kv_load_to_connector_finished_ms"))
        if baseline_ms is None or optimized_ms is None or baseline_ms <= 0:
            continue
        row = dict(zip(pair_fields, key))
        row.update({
            "off_request_id": baseline.get("request_id", ""),
            "on_request_id": optimized.get("request_id", ""),
            "on_selected_transfer_path": optimized.get(
                "selected_transfer_path", "unknown"),
            "off_transfer_ms": baseline_ms,
            "on_transfer_ms": optimized_ms,
            "on_minus_off_transfer_ms": optimized_ms - baseline_ms,
            "transfer_speedup_pct": (baseline_ms - optimized_ms)
            / baseline_ms * 100,
            "off_total_bytes": baseline.get("total_bytes", ""),
            "on_total_bytes": optimized.get("total_bytes", ""),
            "off_forward_ranges": baseline.get("selector_forward_ranges", ""),
            "on_forward_ranges": optimized.get("selector_forward_ranges", ""),
        })
        off_bytes = number(baseline.get("total_bytes"))
        on_bytes = number(optimized.get("total_bytes"))
        row["payload_bytes_match"] = (
            off_bytes is not None and on_bytes is not None
            and off_bytes == on_bytes
        )
        row["transfer_bytes_delta"] = (
            "" if off_bytes is None or on_bytes is None else on_bytes - off_bytes
        )
        pairs.append(row)
    return sorted(pairs, key=lambda row: (
        rate_sort_key(row["configured_request_rate"]),
        number(row["repetition"]) or 0,
        number(row["case_request_index"]) or 0,
    ))


def pair_ttft_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match request TTFT between off/direct and on/auto by workload order."""
    pair_fields = (
        "run_id", "phase", "workload", "configured_request_rate",
        "repetition", "case_request_index",
    )
    candidates = [
        row for row in rows
        if row.get("variant") in {"off", "on"}
        and row.get("case_request_index", "") != ""
        and number(row.get("proxy_ttft_ms")) is not None
    ]
    pairs = []
    for key, items in grouped(candidates, pair_fields):
        by_variant = {row["variant"]: row for row in items}
        if set(by_variant) != {"off", "on"}:
            continue
        off_ttft = number(by_variant["off"].get("proxy_ttft_ms"))
        on_ttft = number(by_variant["on"].get("proxy_ttft_ms"))
        if off_ttft is None or on_ttft is None:
            continue
        pairs.append({
            **dict(zip(pair_fields, key)),
            "off_request_id": by_variant["off"].get("request_id", ""),
            "on_request_id": by_variant["on"].get("request_id", ""),
            "off_ttft_ms": off_ttft,
            "on_ttft_ms": on_ttft,
            "on_minus_off_ttft_ms": on_ttft - off_ttft,
        })
    return sorted(pairs, key=lambda row: (
        rate_sort_key(row["configured_request_rate"]),
        number(row["repetition"]) or 0,
        number(row["case_request_index"]) or 0,
    ))


def summarize_paired_ttft(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize paired request-level TTFT delta once per repetition."""
    fields = (
        "run_id", "phase", "workload", "configured_request_rate",
        "repetition",
    )
    rows = []
    for key, items in grouped(pairs, fields):
        deltas = numeric_values(items, "on_minus_off_ttft_ms")
        if not deltas:
            continue
        rows.append({
            **dict(zip(fields, key)),
            "matched_request_count": len(deltas),
            "paired_delta_ttft_p50_ms": percentile(deltas, 50),
            "paired_delta_ttft_mean_ms": mean(deltas),
            "paired_delta_ttft_p90_ms": percentile(deltas, 90),
        })
    return rows


def summarize_transfer_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "run_id", "phase", "workload", "configured_request_rate",
        "repetition", "on_selected_transfer_path",
    )
    rows = []
    comparable = [row for row in pairs if row.get("payload_bytes_match")]
    for key, items in grouped(comparable, fields):
        deltas = numeric_values(items, "on_minus_off_transfer_ms")
        speedups = numeric_values(items, "transfer_speedup_pct")
        row = dict(zip(fields, key))
        row.update({
            "matched_request_count": len(items),
            "on_minus_off_mean_ms": mean(deltas),
            "on_minus_off_p10_ms": percentile(deltas, 10),
            "on_minus_off_p50_ms": percentile(deltas, 50),
            "on_minus_off_p90_ms": percentile(deltas, 90),
            "transfer_speedup_p50_pct": percentile(speedups, 50),
        })
        rows.append(row)
    return rows


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_transfer_cdf(rows: list[dict[str, Any]], title: str,
                      destination: Path) -> None:
    rates = sorted({row["configured_request_rate"] for row in rows},
                   key=rate_sort_key)
    figure, axes = plt.subplots(
        1, len(rates), figsize=(5.2 * len(rates), 4.6),
        layout="constrained", squeeze=False,
    )
    figure.suptitle(title, fontsize=13)
    for axis, rate in zip(axes[0], rates):
        rate_rows = [row for row in rows
                     if row["configured_request_rate"] == rate]
        for variant in ("off", "on"):
            values = sorted(numeric_values(
                [row for row in rate_rows if row.get("variant") == variant],
                "kv_load_to_connector_finished_ms",
            ))
            if not values:
                continue
            cdf = [(index + 1) / len(values) for index in range(len(values))]
            p50 = percentile(values, 50)
            p90 = percentile(values, 90)
            p99 = percentile(values, 99)
            axis.step(
                values, cdf, where="post", linewidth=2,
                color=VARIANT_COLORS[variant],
                label=(f"{VARIANT_LABELS[variant]} (n={len(values)})\n"
                       f"P50/P90/P99={p50:.1f}/{p90:.1f}/{p99:.1f} ms"),
            )
        axis.set_title(f"arrival multiplier={rate}")
        axis.set_xlabel("KV load to connector finish (ms, log scale)")
        axis.set_ylabel("Request CDF")
        axis.set_xscale("log")
        axis.set_ylim(0, 1.01)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_paired_transfer_delta(pairs: list[dict[str, Any]], title: str,
                               destination: Path) -> None:
    pairs = [row for row in pairs if row.get("payload_bytes_match")]
    rates = sorted({row["configured_request_rate"] for row in pairs},
                   key=rate_sort_key)
    paths = [path for path in ("direct", "packed")
             if any(row["on_selected_transfer_path"] == path for row in pairs)]
    figure, axis = plt.subplots(
        figsize=(max(8.0, len(rates) * 2.4), 5.0), layout="constrained")
    width = 0.32 if len(paths) > 1 else 0.5
    positions = []
    values_by_position = []
    colors = []
    labels = []
    for rate_index, rate in enumerate(rates):
        offsets = ([0.0] if len(paths) == 1 else
                   [-(len(paths) - 1) * width / 2 + index * width
                    for index in range(len(paths))])
        for offset, path in zip(offsets, paths):
            values = [
                row["on_minus_off_transfer_ms"] for row in pairs
                if row["configured_request_rate"] == rate
                and row["on_selected_transfer_path"] == path
            ]
            if not values:
                continue
            positions.append(rate_index + offset)
            values_by_position.append(values)
            colors.append("#72b7b2" if path == "direct" else "#e45756")
            labels.append(path)
    boxes = axis.boxplot(
        values_by_position, positions=positions, widths=width * 0.8,
        showfliers=False, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.8)
    axis.axhline(0, color="black", linewidth=1, alpha=0.7)
    axis.set_xticks(range(len(rates)))
    axis.set_xticklabels([str(rate) for rate in rates])
    axis.set_xlabel("Recorded arrival-rate multiplier")
    axis.set_ylabel("auto - direct visible transfer latency (ms)\nnegative is faster")
    axis.set_title(title + "\nOnly equal-byte request pairs are shown")
    axis.grid(True, axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8)
               for color in ("#72b7b2", "#e45756")[:len(paths)]]
    axis.legend(handles, paths, title="auto selected path", fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_critical_path_breakdown(rows: list[dict[str, Any]], title: str,
                                 destination: Path) -> None:
    rates = sorted({row["configured_request_rate"] for row in rows},
                   key=rate_sort_key)
    figure, axes = plt.subplots(
        1, len(rates), figsize=(5.4 * len(rates), 5.0),
        layout="constrained", squeeze=False,
    )
    figure.suptitle(title, fontsize=13)
    colors = ("#4c78a8", "#f2cf5b", "#e45756", "#72b7b2")
    for axis, rate in zip(axes[0], rates):
        rate_rows = [row for row in rows
                     if row["configured_request_rate"] == rate]
        categories = [
            ("off / direct", [row for row in rate_rows
                              if row.get("variant") == "off"]),
            ("on / direct", [row for row in rate_rows
                             if row.get("variant") == "on"
                             and row.get("selected_transfer_path") == "direct"]),
            ("on / packed", [row for row in rate_rows
                             if row.get("variant") == "on"
                             and row.get("selected_transfer_path") == "packed"]),
        ]
        categories = [(label, items) for label, items in categories if items]
        bottoms = [0.0] * len(categories)
        for (field, label), color in zip(TRANSFER_CRITICAL_PATH_PHASES, colors):
            heights = [mean(numeric_values(items, field)) or 0.0
                       for _, items in categories]
            axis.bar(range(len(categories)), heights, bottom=bottoms,
                     label=label, color=color)
            bottoms = [bottom + height
                       for bottom, height in zip(bottoms, heights)]
        for index, ((_, items), total) in enumerate(zip(categories, bottoms)):
            axis.text(index, total, f"{total:.1f}\nn={len(items)}",
                      ha="center", va="bottom", fontsize=8)
        axis.set_title(f"arrival multiplier={rate}")
        axis.set_xticks(range(len(categories)))
        axis.set_xticklabels([label for label, _ in categories], rotation=15)
        axis.set_ylabel("Mean request-visible transfer wall (ms)")
        axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4,
                  fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_packed_diagnostics(rows: list[dict[str, Any]], title: str,
                            destination: Path) -> None:
    rows = [row for row in rows if row.get("variant") == "on"
            and row.get("selected_transfer_path") == "packed"]
    rates = sorted({row["configured_request_rate"] for row in rows},
                   key=rate_sort_key)
    figure, axis = plt.subplots(figsize=(10, 5.2), layout="constrained")
    x_values = list(range(len(rates)))
    for field, label in PACKED_DIAGNOSTICS:
        medians = []
        low_errors = []
        high_errors = []
        for rate in rates:
            values = numeric_values(
                [row for row in rows
                 if row["configured_request_rate"] == rate], field)
            p50 = percentile(values, 50)
            p10 = percentile(values, 10)
            p90 = percentile(values, 90)
            medians.append(p50 or float("nan"))
            low_errors.append(0.0 if p50 is None or p10 is None else p50 - p10)
            high_errors.append(0.0 if p50 is None or p90 is None else p90 - p50)
        axis.errorbar(x_values, medians, yerr=[low_errors, high_errors],
                      marker="o", linewidth=1.8, capsize=3, label=label)
    axis.set_xticks(x_values)
    axis.set_xticklabels([str(rate) for rate in rates])
    axis.set_yscale("log")
    axis.set_xlabel("Recorded arrival-rate multiplier")
    axis.set_ylabel("P50 with P10-P90 range (ms, symlog)")
    axis.set_title(title + "\nComponent timers overlap and must not be summed")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def build_matched_cohort_bar_rows(
        pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize matched off/on cohorts once per repetition and selected path."""
    fields = (
        "run_id", "phase", "workload", "configured_request_rate",
        "repetition", "on_selected_transfer_path",
    )
    rows = []
    comparable = [
        row for row in pairs
        if row.get("payload_bytes_match")
        and row.get("on_selected_transfer_path") in {"direct", "packed"}
    ]
    for key, items in grouped(comparable, fields):
        off_values = numeric_values(items, "off_transfer_ms")
        on_values = numeric_values(items, "on_transfer_ms")
        if not off_values or not on_values:
            continue
        off_p50 = percentile(off_values, 50)
        on_p50 = percentile(on_values, 50)
        assert off_p50 is not None and on_p50 is not None
        rows.append({
            **dict(zip(fields, key)),
            "matched_request_count": len(items),
            "off_counterpart_transfer_p50_ms": off_p50,
            "on_transfer_p50_ms": on_p50,
            "p50_reduction_ms": off_p50 - on_p50,
            "p50_reduction_pct": (
                (off_p50 - on_p50) / off_p50 * 100
                if off_p50 > 0 else None
            ),
        })
    return rows


def save_matched_cohort_transfer_bars(
        rows: list[dict[str, Any]], title: str, destination: Path) -> None:
    """Save one multi-load figure with two matched path cohorts per panel."""
    rates = sorted({row["configured_request_rate"] for row in rows},
                   key=rate_sort_key)
    figure, axes = plt.subplots(
        1, len(rates), figsize=(5.2 * len(rates), 5.6),
        layout="constrained", squeeze=False, sharey=True,
    )
    figure.suptitle(title, fontsize=14)
    bar_width = 0.34
    off_color = "#bab0ac"
    path_colors = {"direct": "#4c78a8", "packed": "#f58518"}
    paths = ("direct", "packed")

    panel_aggregates = []
    global_top = 0.0
    for rate in rates:
        rate_rows = [row for row in rows
                     if row["configured_request_rate"] == rate]
        path_aggregates = []
        for path in paths:
            path_rows = [row for row in rate_rows
                         if row["on_selected_transfer_path"] == path]
            off_aggregate = _aggregate_repetitions(
                path_rows, "off_counterpart_transfer_p50_ms")
            on_aggregate = _aggregate_repetitions(
                path_rows, "on_transfer_p50_ms")
            if off_aggregate is None or on_aggregate is None:
                path_aggregates.append(None)
                continue
            counts = numeric_values(path_rows, "matched_request_count")
            aggregate = {
                "path": path,
                "off": off_aggregate,
                "on": on_aggregate,
                "median_count": statistics.median(counts) if counts else 0,
                "repetitions": len(path_rows),
            }
            path_aggregates.append(aggregate)
            global_top = max(
                global_top,
                off_aggregate[0] + off_aggregate[2],
                on_aggregate[0] + on_aggregate[2],
            )
        panel_aggregates.append(path_aggregates)

    label_padding = max(global_top * 0.035, 2.0)
    for axis, rate, aggregates in zip(axes[0], rates, panel_aggregates):
        for x_value, (path, aggregate) in enumerate(zip(paths, aggregates)):
            if aggregate is None:
                continue
            off_center, off_low, off_high = aggregate["off"]
            on_center, on_low, on_high = aggregate["on"]
            axis.bar(
                x_value - bar_width / 2, off_center, bar_width,
                yerr=[[off_low], [off_high]], capsize=5,
                color=off_color, edgecolor="#666666", hatch="//",
                label="Off counterpart" if x_value == 0 else None,
            )
            axis.bar(
                x_value + bar_width / 2, on_center, bar_width,
                yerr=[[on_low], [on_high]], capsize=5,
                color=path_colors[path], edgecolor="#444444",
                label=(f"On: {path}"),
            )
            axis.text(x_value - bar_width / 2, off_center + off_high
                      + label_padding * 0.15, f"{off_center:.1f}",
                      ha="center", va="bottom", fontsize=8)
            axis.text(x_value + bar_width / 2, on_center + on_high
                      + label_padding * 0.15, f"{on_center:.1f}",
                      ha="center", va="bottom", fontsize=8)
            reduction = off_center - on_center
            reduction_pct = reduction / off_center * 100 if off_center > 0 else 0
            comparison_top = max(off_center + off_high,
                                 on_center + on_high) + label_padding
            axis.plot(
                [x_value - bar_width / 2, x_value - bar_width / 2,
                 x_value + bar_width / 2, x_value + bar_width / 2],
                [comparison_top - label_padding * 0.2, comparison_top,
                 comparison_top, comparison_top - label_padding * 0.2],
                color="#333333", linewidth=1,
            )
            sign = "−" if reduction >= 0 else "+"
            axis.text(
                x_value, comparison_top + label_padding * 0.12,
                f"{sign}{abs(reduction):.1f} ms ({sign}{abs(reduction_pct):.1f}%)",
                ha="center", va="bottom", fontsize=8,
            )
        tick_labels = []
        for path, aggregate in zip(paths, aggregates):
            if aggregate is None:
                tick_labels.append(f"Auto chose {path}\n(no matched requests)")
            else:
                tick_labels.append(
                    f"Auto chose {path}\n"
                    f"median n={aggregate['median_count']:.0f}/rep, "
                    f"reps={aggregate['repetitions']}"
                )
        axis.set_xticks(range(len(paths)))
        axis.set_xticklabels(tick_labels)
        axis.set_title(f"Arrival multiplier = {rate}")
        axis.set_ylabel("KV load to connector finish P50 (ms)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_ylim(0, global_top + label_padding * 4.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="outside lower center",
                  ncol=3, fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_paired_ttft_delta_figure(
        rows: list[dict[str, Any]], title: str, destination: Path) -> None:
    """Save paired P50 TTFT(auto - direct) across offered loads."""
    rates = sorted({row["configured_request_rate"] for row in rows},
                   key=rate_sort_key)
    centers = []
    lower_errors = []
    upper_errors = []
    matched_counts = []
    repetition_counts = []
    plotted_rates = []
    for rate in rates:
        rate_rows = [row for row in rows
                     if row["configured_request_rate"] == rate]
        aggregate = _aggregate_repetitions(
            rate_rows, "paired_delta_ttft_p50_ms")
        if aggregate is None:
            continue
        center, lower, upper = aggregate
        plotted_rates.append(rate)
        centers.append(center)
        lower_errors.append(lower)
        upper_errors.append(upper)
        counts = numeric_values(rate_rows, "matched_request_count")
        matched_counts.append(statistics.median(counts) if counts else 0)
        repetition_counts.append(len(rate_rows))

    if not plotted_rates:
        return

    figure, axis = plt.subplots(figsize=(7.6, 5.2), layout="constrained")
    axis.set_title(title, fontsize=13, pad=14)
    x_values = list(range(len(plotted_rates)))
    axis.errorbar(
        x_values, centers, yerr=[lower_errors, upper_errors],
        color="#9467bd", marker="o", markersize=7, linewidth=2,
        capsize=6, capthick=1.5,
    )
    axis.axhline(0, color="#333333", linewidth=1.1, linestyle="--")

    value_span = max([
        1.0,
        *(abs(center) + lower + upper for center, lower, upper
          in zip(centers, lower_errors, upper_errors)),
    ])
    label_offset = value_span * 0.035
    for x_value, center, lower, upper in zip(
            x_values, centers, lower_errors, upper_errors):
        if center >= 0:
            y_value = center + upper + label_offset
            vertical_alignment = "bottom"
        else:
            y_value = center - lower - label_offset
            vertical_alignment = "top"
        axis.text(
            x_value, y_value, f"{center:+.1f} ms",
            ha="center", va=vertical_alignment, fontsize=9,
        )

    lower_extreme = min(
        [0.0, *(center - lower for center, lower
                in zip(centers, lower_errors))])
    upper_extreme = max(
        [0.0, *(center + upper for center, upper
                in zip(centers, upper_errors))])
    y_padding = max((upper_extreme - lower_extreme) * 0.1, 5.0)
    axis.set_ylim(lower_extreme - y_padding, upper_extreme + y_padding)

    tick_labels = [
        f"{rate}\nn={count:.0f}/rep\n{repetitions} reps"
        for rate, count, repetitions in zip(
            plotted_rates, matched_counts, repetition_counts)
    ]
    axis.set_xticks(x_values)
    axis.set_xticklabels(tick_labels)
    axis.set_xlabel("Arrival multiplier")
    axis.set_ylabel("Paired TTFT P50 delta (ms)\nauto - direct")
    axis.grid(True, axis="y", alpha=0.25)
    axis.text(
        0.01, 0.98, "Below zero: auto has lower TTFT",
        transform=axis.transAxes, ha="left", va="top",
        fontsize=9, color="#555555",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_paired_transfer_analysis(
        rows: list[dict[str, Any]], output_dir: Path,
        include_cold_handshake: bool = False,
        include_detail_plots: bool = False) -> int:
    rows = steady_state_rows(rows, include_cold_handshake)
    required_metadata = (
        "run_id", "phase", "workload", "variant", "repetition",
        "configured_request_rate", "case_id",
    )
    rows = [row for row in rows
            if row.get("variant") in {"off", "on"}
            and all(field in row for field in required_metadata)]
    if (not rows
            or not {row.get("variant") for row in rows}.issuperset(
                {"off", "on"})
            or not any(row.get("selected_transfer_path")
                       for row in rows if row.get("variant") == "on")):
        return 0
    summaries = summarize_transfer_cases(rows, include_cold_handshake=True)
    pairs = pair_transfer_requests(rows, include_cold_handshake=True)
    pair_summaries = summarize_transfer_pairs(pairs)
    cohort_rows = build_matched_cohort_bar_rows(pairs)
    ttft_pairs = pair_ttft_requests(rows)
    ttft_summaries = summarize_paired_ttft(ttft_pairs)
    write_dict_rows(output_dir / "paired_transfer_summary.csv", summaries)
    write_dict_rows(output_dir / "paired_request_transfer_delta.csv", pairs)
    write_dict_rows(output_dir / "paired_transfer_delta_summary.csv",
                    pair_summaries)
    write_dict_rows(output_dir / "matched_cohort_transfer_p50.csv",
                    cohort_rows)
    write_dict_rows(output_dir / "paired_ttft_request_delta.csv", ttft_pairs)
    write_dict_rows(output_dir / "paired_ttft_delta_summary.csv",
                    ttft_summaries)

    count = 0
    scenario_fields = ("run_id", "phase", "workload")
    for (run, phase, workload), items in grouped(rows, scenario_fields):
        if not {row.get("variant") for row in items}.issuperset({"off", "on"}):
            continue
        slug = f"{filename_slug(phase)}-{filename_slug(workload)}"
        title = f"Paired transfer-mode A/B: {workload} ({phase})"
        base = output_dir / run / "06_paired_transfer_latency"
        scenario_pairs = [row for row in pairs
                          if row["run_id"] == run and row["phase"] == phase
                          and row["workload"] == workload]
        scenario_cohorts = [row for row in cohort_rows
                            if row["run_id"] == run and row["phase"] == phase
                            and row["workload"] == workload]
        scenario_ttft = [row for row in ttft_summaries
                         if row["run_id"] == run and row["phase"] == phase
                         and row["workload"] == workload]
        if scenario_cohorts:
            save_matched_cohort_transfer_bars(
                scenario_cohorts,
                title + "\nMatched request cohorts; bars are repetition medians, "
                "error bars are min-max",
                base / f"{slug}-matched-cohort-bars.png",
            )
            count += 1
        if scenario_ttft:
            save_paired_ttft_delta_figure(
                scenario_ttft,
                f"Paired TTFT: auto vs direct ({workload}, {phase})\n"
                "Center = median of repetition P50s; error bars = min-max",
                base / f"{slug}-paired-delta-ttft.png",
            )
            count += 1
        if not include_detail_plots:
            continue
        save_transfer_cdf(items, title, base / f"{slug}-cdf.png")
        count += 1
        if scenario_pairs:
            save_paired_transfer_delta(
                scenario_pairs,
                title + "\nMatched by request position and repetition",
                base / f"{slug}-paired-delta.png")
            count += 1
        critical_rows = [
            row for row in items
            if all(number(row.get(field)) is not None
                   for field, _ in TRANSFER_CRITICAL_PATH_PHASES)
        ]
        if critical_rows:
            save_critical_path_breakdown(
                critical_rows, title + "\nNon-overlapping trace landmarks",
                base / f"{slug}-critical-path.png")
            count += 1
        if any(row.get("variant") == "on"
               and row.get("selected_transfer_path") == "packed"
               for row in items):
            save_packed_diagnostics(
                items, title + " — packed requests",
                base / f"{slug}-packed-diagnostics.png")
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


def write_generalization_delay_summary(
        path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id", "phase", "workload", "variant", "repetition",
        "configured_request_rate", "transfer_delay_ms", "num_prompts",
        "completed", "result_file",
    ]
    fields.extend(metric for metric, _, _ in METRICS)
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Result directory, or a bare run ID searched below the current directory",
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <root>/plots")
    parser.add_argument("--include-cold-handshake", action="store_true",
                        help="Include requests that performed a cold NIXL handshake")
    parser.add_argument(
        "--diagnostic-detail-plots",
        action="store_true",
        help=("Also generate legacy CDF, pie, raw scatter, and detailed "
              "transfer diagnostic plots"),
    )
    args = parser.parse_args()
    try:
        root = resolve_result_root(args.root)
    except ValueError as error:
        parser.error(str(error))
    benchmark_rows = load_benchmark_rows(root)
    generalization_delay_rows = load_generalization_delay_rows(root)
    timeline_rows = load_timeline_rows(root)
    request_latency_rows = load_request_latency_rows(root)
    if (not benchmark_rows and not generalization_delay_rows
            and not timeline_rows and not request_latency_rows):
        parser.error(
            f"No PD benchmark JSON files or parsed request timelines found below {root}. "
            "Run parse_pd_trace.py first for trace-only diagnostic results."
        )
    output_dir = (args.output_dir or root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "plot_summary.csv", benchmark_rows)
    write_generalization_delay_summary(
        output_dir / "generalization_delay_summary.csv",
        generalization_delay_rows,
    )
    motivation_count = create_motivation_core_analysis(
        timeline_rows, output_dir,
        include_cold_handshake=args.include_cold_handshake)
    paired_transfer_count = create_paired_transfer_analysis(
        timeline_rows, output_dir,
        include_cold_handshake=args.include_cold_handshake,
        include_detail_plots=args.diagnostic_detail_plots)
    sleep_count = 0
    generalization_delay_count = 0
    pie_count = 0
    scatter_count = 0
    transfer_scatter_count = 0
    if args.diagnostic_detail_plots:
        sleep_count = create_sleep_sweeps(benchmark_rows, output_dir)
        generalization_delay_count = create_generalization_delay_sweeps(
            generalization_delay_rows, output_dir)
        pie_count = create_transfer_pies(
            timeline_rows, output_dir,
            include_cold_handshake=args.include_cold_handshake)
        scatter_count = create_request_latency_scatters(
            request_latency_rows, output_dir)
        transfer_scatter_count = create_transfer_latency_scatters(
            timeline_rows, output_dir, args.include_cold_handshake)
    print(f"Read {len(benchmark_rows)} fixed-length cases and "
          f"{len(generalization_delay_rows)} generalization delay cases; wrote "
          f"{motivation_count} motivation core figures and "
          f"{paired_transfer_count} paired transfer figures under {output_dir}")
    if args.diagnostic_detail_plots:
        print(f"Also wrote {sleep_count} fixed-length sleep sweeps, "
              f"{generalization_delay_count} generalization delay sweeps, "
              f"{pie_count} transfer pie charts, {scatter_count} raw request "
              f"scatter plots, and {transfer_scatter_count} raw transfer "
              "scatter plots")
    if not timeline_rows:
        print("No request timelines found; run parse_pd_trace.py first to create "
              "pd_request_timeline_ms.csv before plotting request-level figures.")
    print(f"Wrote {output_dir / 'plot_summary.csv'}")
    print(f"Wrote {output_dir / 'generalization_delay_summary.csv'}")


if __name__ == "__main__":
    main()
