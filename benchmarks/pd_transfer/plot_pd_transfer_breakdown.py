# SPDX-License-Identifier: Apache-2.0
"""Plot complete PD-transfer latency as a share of TTFT and its breakdown."""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_pd_trace import build_rows, percentile


COMPONENT_FIELDS = (
    "prefill_to_request_prepare_start_ms",
    "request_prepare_start_to_first_submit_ms",
    "first_to_last_submit_ms",
    "last_submit_to_physical_done_ms",
    "physical_done_to_reported_ms",
)
COMPONENT_LABELS = (
    "Wait until D starts processing",
    "Rank 0 transfer prep + submit",
    "Rank 1 transfer prep + submit",
    "Wait for all transfers to finish",
    "Completion polling and reporting",
)
COMPONENT_COLORS = ("#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B")


@dataclass
class Case:
    run_dir: Path
    label: str
    manifest: dict[str, Any]
    rows_seen: int
    ttft_ms: list[float]
    full_ms: list[float]
    ratios: list[float]
    e2e_ms: list[float]
    e2e_ratios: list[float]
    benchmark_ttft_ms: float | None
    benchmark_e2e_ms: float | None
    request_throughput: float | None
    components_ms: tuple[list[float], ...]


def _float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field, "")
    if value == "" or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _dataset_name(trace_path: str) -> str:
    name = Path(trace_path).stem
    if name.startswith("mooncake_"):
        name = name[len("mooncake_") :]
    if name.endswith("_500"):
        name = name[:-4]
    if name.endswith("_motivation"):
        name = name[: -len("_motivation")]
    aliases = {
        "synthetic": "Synthetic",
        "toolagent": "ToolAgent",
    }
    return aliases.get(name, name.replace("_", " ").title()) or "Unknown"


def _case_label(manifest: dict[str, Any]) -> str:
    dataset = _dataset_name(str(manifest.get("trace_path", "")))
    prefill_tp = manifest.get("prefill_tp_size", "?")
    decode_tp = manifest.get("decode_tp_size", "?")
    scale = manifest.get("trace_time_scale", "?")
    label = f"{dataset}\nP{prefill_tp}-D{decode_tp}, s={scale}"
    if "transfer_sleep_ms" in manifest:
        label += f", sleep={manifest['transfer_sleep_ms']} ms"
    return label


def _load_benchmark_metrics(
    run_dir: Path,
) -> tuple[float | None, float | None, float | None]:
    benchmark_path = run_dir / "benchmark.json"
    if not benchmark_path.is_file():
        return None, None, None
    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        mean_ttft_ms = float(benchmark["mean_ttft_ms"])
        request_throughput = float(benchmark["request_throughput"])
        ttfts = benchmark.get("ttfts", [])
        itls = benchmark.get("itls", [])
        if not ttfts or len(ttfts) != len(itls):
            mean_e2e_ms = None
        else:
            e2e_s = [
                float(ttft) + sum(float(itl) for itl in request_itls)
                for ttft, request_itls in zip(ttfts, itls, strict=True)
            ]
            mean_e2e_ms = statistics.fmean(e2e_s) * 1000
        return mean_ttft_ms, mean_e2e_ms, request_throughput
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Ignoring invalid benchmark metrics in {benchmark_path}: {error}")
        return None, None, None


def discover_runs(inputs: list[Path]) -> list[Path]:
    runs: set[Path] = set()
    for input_path in inputs:
        path = input_path.resolve()
        if not path.is_dir():
            raise ValueError(f"not a directory: {input_path}")
        if (path / "run_manifest.json").is_file() and (path / "traces").is_dir():
            runs.add(path)
            continue
        for manifest_path in path.rglob("run_manifest.json"):
            run_dir = manifest_path.parent
            if (run_dir / "traces").is_dir():
                runs.add(run_dir)
    return sorted(runs)


def load_case(run_dir: Path) -> Case | None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = build_rows(run_dir / "traces")

    ttft_ms: list[float] = []
    full_ms: list[float] = []
    ratios: list[float] = []
    e2e_ms: list[float] = []
    e2e_ratios: list[float] = []
    component_lists: tuple[list[float], ...] = tuple([] for _ in COMPONENT_FIELDS)
    for row in rows:
        ttft = _float(row, "proxy_ttft_ms")
        e2e = _float(row, "proxy_e2e_ms")
        full = _float(row, "prefill_to_reported_ms")
        components = tuple(_float(row, field) for field in COMPONENT_FIELDS)
        if ttft is None or full is None or ttft <= 0 or full < 0:
            continue
        ttft_ms.append(ttft)
        full_ms.append(full)
        ratios.append(full / ttft)
        if e2e is not None and e2e > 0:
            e2e_ms.append(e2e)
            e2e_ratios.append(full / e2e)
        if all(value is not None and value >= 0 for value in components):
            for values, value in zip(component_lists, components, strict=True):
                assert value is not None
                values.append(value)

    if not full_ms:
        print(f"Skipping {run_dir}: no request has a complete PD interval and TTFT")
        return None
    benchmark_ttft_ms, benchmark_e2e_ms, request_throughput = (
        _load_benchmark_metrics(run_dir)
    )
    return Case(
        run_dir=run_dir,
        label=_case_label(manifest),
        manifest=manifest,
        rows_seen=len(rows),
        ttft_ms=ttft_ms,
        full_ms=full_ms,
        ratios=ratios,
        e2e_ms=e2e_ms,
        e2e_ratios=e2e_ratios,
        benchmark_ttft_ms=benchmark_ttft_ms,
        benchmark_e2e_ms=benchmark_e2e_ms,
        request_throughput=request_throughput,
        components_ms=component_lists,
    )


def make_labels_unique(cases: list[Case]) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.label] = counts.get(case.label, 0) + 1
    for case in cases:
        if counts[case.label] > 1:
            run_id = case.manifest.get("run_id", case.run_dir.name)
            case.label = f"{case.label}, run={run_id}"


def _figure_height(case_count: int) -> float:
    return max(2.8, 1.15 * case_count + 1.5)


def plot_pd_share(
    cases: list[Case], output_path: Path, denominator: str
) -> None:
    if denominator == "TTFT":
        cases = [case for case in cases if case.ratios]
        ratio_lists = [case.ratios for case in cases]
    elif denominator == "E2E":
        cases = [case for case in cases if case.e2e_ratios]
        ratio_lists = [case.e2e_ratios for case in cases]
    else:
        raise ValueError(f"unsupported denominator: {denominator}")
    if not cases:
        print(f"Skipping {output_path}: no complete {denominator} interval")
        return

    labels = [case.label for case in cases]
    pd_shares = [statistics.fmean(ratios) * 100 for ratios in ratio_lists]
    other_shares = [100 - share for share in pd_shares]
    y_positions = list(range(len(cases)))

    fig, ax = plt.subplots(
        figsize=(9.2, _figure_height(len(cases))), layout="constrained"
    )
    ax.barh(y_positions, pd_shares, color="#4C78A8", label="PD Handoff")
    ax.barh(
        y_positions,
        other_shares,
        left=pd_shares,
        color="#D9D9D9",
        label=f"Other {denominator}",
    )
    for y, share in zip(y_positions, pd_shares, strict=True):
        ax.text(
            min(share / 2, 94),
            y,
            f"{share:.1f}%",
            ha="center",
            va="center",
            color="white" if share >= 12 else "black",
            fontweight="bold",
        )
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Mean per-request share of {denominator} (%)")
    ax.set_title(f"PD Handoff / {denominator}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ratio_distribution(cases: list[Case], output_path: Path) -> None:
    labels = [case.label for case in cases]
    values = [[ratio * 100 for ratio in case.ratios] for case in cases]
    fig, ax = plt.subplots(
        figsize=(9.2, _figure_height(len(cases))), layout="constrained"
    )
    box = ax.boxplot(
        values,
        tick_labels=labels,
        vert=False,
        whis=(1, 99),
        showfliers=False,
        showmeans=True,
        patch_artist=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#E45756",
            "markeredgecolor": "#E45756",
            "markersize": 5,
        },
        medianprops={"color": "#222222", "linewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#9ECAE1")
        patch.set_edgecolor("#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Complete PD transfer / TTFT per request (%)")
    ax.set_title("Per-request PD Handoff / TTFT")
    ax.grid(axis="x", alpha=0.25)
    ax.scatter([], [], marker="D", color="#E45756", label="Mean")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _annotate_latency_points(
    ax: Any, x_values: list[float], values: list[float]
) -> None:
    baseline = values[0]
    ax.annotate(
        f"{baseline:.1f} ms\nBaseline",
        (x_values[0], baseline),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
    )
    annotation_styles = (
        {"xytext": (6, -12), "ha": "left", "va": "top"},
        {"xytext": (4, 10), "ha": "left", "va": "bottom"},
        {"xytext": (-4, 8), "ha": "right", "va": "bottom"},
    )
    for index, (x, value) in enumerate(
        zip(x_values[1:], values[1:], strict=True)
    ):
        delta = value - baseline
        percent = delta / baseline * 100
        style = annotation_styles[min(index, len(annotation_styles) - 1)]
        ax.annotate(
            f"{delta:+.1f} ms ({percent:+.1f}%)",
            (x, value),
            xytext=style["xytext"],
            textcoords="offset points",
            ha=style["ha"],
            va=style["va"],
            fontsize=8,
        )


def plot_sleep_injection(cases: list[Case], output_path: Path) -> None:
    sleep_groups: dict[float, list[Case]] = {}
    for case in cases:
        sleep = case.manifest.get("transfer_sleep_ms")
        if (
            sleep is None
            or case.benchmark_ttft_ms is None
            or case.benchmark_e2e_ms is None
            or case.request_throughput is None
        ):
            continue
        sleep_groups.setdefault(float(sleep), []).append(case)
    if not sleep_groups or 0.0 not in sleep_groups:
        print(f"Skipping {output_path}: no complete sleep sweep with a 0 ms case")
        return

    sleep_ms = sorted(sleep_groups)
    ttft_ms = [
        statistics.fmean(
            case.benchmark_ttft_ms
            for case in sleep_groups[sleep]
            if case.benchmark_ttft_ms is not None
        )
        for sleep in sleep_ms
    ]
    e2e_ms = [
        statistics.fmean(
            case.benchmark_e2e_ms
            for case in sleep_groups[sleep]
            if case.benchmark_e2e_ms is not None
        )
        for sleep in sleep_ms
    ]
    throughput = [
        statistics.fmean(
            case.request_throughput
            for case in sleep_groups[sleep]
            if case.request_throughput is not None
        )
        for sleep in sleep_ms
    ]

    fig = plt.figure(figsize=(9.4, 6.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 0.32))
    ttft_ax = fig.add_subplot(grid[0, 0])
    e2e_ax = fig.add_subplot(grid[0, 1])
    throughput_ax = fig.add_subplot(grid[1, :])
    series = (
        (ttft_ax, ttft_ms, "#F58518", "Mean TTFT (ms)"),
        (e2e_ax, e2e_ms, "#4C78A8", "Mean E2E (ms)"),
    )
    x_min = min(sleep_ms) - 10
    x_max = max(sleep_ms) + 10
    for ax, values, color, ylabel in series:
        ax.plot(sleep_ms, values, color=color, marker="o", linewidth=2)
        _annotate_latency_points(ax, sleep_ms, values)
        baseline = values[0]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(baseline, baseline + 310)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Injected Delay (ms)")
        ax.set_xticks(sleep_ms)
        ax.grid(alpha=0.25)

    throughput_ax.plot(
        sleep_ms,
        throughput,
        color="#54A24B",
        marker="D",
        linestyle="--",
        linewidth=2,
    )
    throughput_baseline = throughput[0]
    throughput_delta_pct = (
        (throughput[-1] - throughput_baseline) / throughput_baseline * 100
    )
    throughput_ax.annotate(
        f"{throughput_delta_pct:+.3f}%",
        (sleep_ms[-1], throughput[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    throughput_center = statistics.fmean(throughput)
    throughput_margin = max(
        throughput_center * 0.015,
        (max(throughput) - min(throughput)) * 4,
        1e-6,
    )
    throughput_ax.set_ylim(
        throughput_center - throughput_margin,
        throughput_center + throughput_margin,
    )
    throughput_ax.set_ylabel("Throughput\n(req/s)")
    throughput_ax.set_xlabel("Injected Delay (ms)")
    throughput_ax.set_xlim(x_min, x_max)
    throughput_ax.grid(alpha=0.25)
    throughput_ax.set_xticks(sleep_ms)
    fig.suptitle("Sleep Injection Effect")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_internal_breakdown(cases: list[Case], output_path: Path) -> None:
    cases = [case for case in cases if all(case.components_ms)]
    if not cases:
        print(f"Skipping {output_path}: no detailed five-stage trace found")
        return

    labels = [case.label for case in cases]
    component_means = [
        [statistics.fmean(values) for values in case.components_ms] for case in cases
    ]
    case_totals = [sum(values) for values in component_means]

    y_positions = list(range(len(cases)))
    left = [0.0] * len(cases)
    fig, ax = plt.subplots(
        figsize=(11.2, _figure_height(len(cases))), layout="constrained"
    )
    for index, (component_label, color) in enumerate(
        zip(COMPONENT_LABELS, COMPONENT_COLORS, strict=True)
    ):
        values = [case_means[index] for case_means in component_means]
        ax.barh(
            y_positions,
            values,
            left=left,
            color=color,
            label=component_label,
        )
        for y, start, value, total in zip(
            y_positions, left, values, case_totals, strict=True
        ):
            share = value / total * 100 if total else 0
            if share >= 8:
                ax.text(
                    start + value / 2,
                    y,
                    f"{value:.1f} ms\n{share:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
            else:
                ax.annotate(
                    f"{value:.1f} ms ({share:.1f}%)",
                    (start + value, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color=color,
                    fontweight="bold",
                )
        left = [start + value for start, value in zip(left, values, strict=True)]

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(case_totals) * 1.16)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title("PD Handoff Breakdown")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(cases: list[Case], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        has_detailed_breakdown = all(case.components_ms)
        component_means: list[float | str] = (
            [statistics.fmean(values) for values in case.components_ms]
            if has_detailed_breakdown
            else [""] * len(COMPONENT_FIELDS)
        )
        component_sums = (
            [sum(values) for values in case.components_ms]
            if has_detailed_breakdown
            else []
        )
        components_total = sum(component_sums)

        def component_pct(index: int) -> float | str:
            if not components_total:
                return ""
            return component_sums[index] / components_total * 100

        full_total = sum(case.full_ms)
        ttft_total = sum(case.ttft_ms)
        e2e_ratio_values = case.e2e_ratios
        rows.append(
            {
                "label": case.label.replace("\n", " | "),
                "run_dir": str(case.run_dir),
                "trace_path": case.manifest.get("trace_path", ""),
                "prefill_tp_size": case.manifest.get("prefill_tp_size", ""),
                "decode_tp_size": case.manifest.get("decode_tp_size", ""),
                "trace_time_scale": case.manifest.get("trace_time_scale", ""),
                "transfer_sleep_ms": case.manifest.get("transfer_sleep_ms", 0),
                "requests_seen": case.rows_seen,
                "requests_complete": len(case.full_ms),
                "requests_with_detailed_breakdown": (
                    len(case.components_ms[0]) if has_detailed_breakdown else 0
                ),
                "pd_over_ttft_mean_request_pct": statistics.fmean(case.ratios)
                * 100,
                "pd_over_ttft_p50_request_pct": percentile(case.ratios, 0.50)
                * 100,
                "pd_over_ttft_p90_request_pct": percentile(case.ratios, 0.90)
                * 100,
                "pd_over_ttft_p99_request_pct": percentile(case.ratios, 0.99)
                * 100,
                "pd_over_ttft_aggregate_pct": full_total / ttft_total * 100,
                "pd_over_e2e_mean_request_pct": (
                    statistics.fmean(e2e_ratio_values) * 100
                    if e2e_ratio_values
                    else ""
                ),
                "pd_over_e2e_p50_request_pct": (
                    percentile(e2e_ratio_values, 0.50) * 100
                    if e2e_ratio_values
                    else ""
                ),
                "pd_over_e2e_p90_request_pct": (
                    percentile(e2e_ratio_values, 0.90) * 100
                    if e2e_ratio_values
                    else ""
                ),
                "pd_over_e2e_p99_request_pct": (
                    percentile(e2e_ratio_values, 0.99) * 100
                    if e2e_ratio_values
                    else ""
                ),
                "pd_complete_mean_ms": statistics.fmean(case.full_ms),
                "prefill_to_prepare_start_mean_ms": component_means[0],
                "prepare_start_to_first_submit_mean_ms": component_means[1],
                "first_to_last_submit_mean_ms": component_means[2],
                "last_submit_to_physical_done_mean_ms": component_means[3],
                "physical_done_to_reported_mean_ms": component_means[4],
                "prefill_to_prepare_start_internal_pct": component_pct(0),
                "prepare_start_to_first_submit_internal_pct": component_pct(1),
                "first_to_last_submit_internal_pct": component_pct(2),
                "last_submit_to_physical_done_internal_pct": component_pct(3),
                "physical_done_to_reported_internal_pct": component_pct(4),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="Run directory or a parent directory containing multiple runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: RESULT_DIR/plots/pd_transfer_breakdown)",
    )
    args = parser.parse_args()

    try:
        run_dirs = discover_runs(args.result_dirs)
    except ValueError as error:
        parser.error(str(error))
    if not run_dirs:
        parser.error("no run_manifest.json with a sibling traces directory found")

    cases = [case for run_dir in run_dirs if (case := load_case(run_dir))]
    if not cases:
        parser.error("no complete PD-transfer request found")
    make_labels_unique(cases)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.result_dirs[0] / "plots" / "pd_transfer_breakdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pd_share(cases, output_dir / "pd_share_of_ttft.png", "TTFT")
    plot_pd_share(cases, output_dir / "pd_share_of_e2e.png", "E2E")
    plot_ratio_distribution(cases, output_dir / "pd_ttft_ratio_distribution.png")
    plot_internal_breakdown(cases, output_dir / "pd_internal_breakdown.png")
    plot_sleep_injection(cases, output_dir / "sleep_injection_effect.png")
    write_summary(cases, output_dir / "pd_transfer_plot_summary.csv")

    print(f"Plotted {len(cases)} case(s) from {len(run_dirs)} discovered run(s)")
    print(f"Wrote figures and summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
