#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot the paired latency benefit of reverse-run canonicalization.

Reads ``generalization_summary.csv`` written by ``compare_pd_generalization_perf.py``
and draws one grouped bar chart: median improvement per latency metric, grouped by
request rate, with the min/max across repetitions as the error range.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Metric(NamedTuple):
    key: str
    label: str
    family: str


# Latency metrics only. Throughput is deliberately excluded: these runs are
# open loop at a fixed request rate, so throughput tracks the arrival rate and
# its sub-percent deltas are noise rather than benefit.
METRICS: tuple[Metric, ...] = (
    Metric("p50_ttft_ms", "p50", "TTFT"),
    Metric("p90_ttft_ms", "p90", "TTFT"),
    Metric("p99_ttft_ms", "p99", "TTFT"),
    Metric("p50_e2el_ms", "p50", "End-to-end"),
    Metric("p90_e2el_ms", "p90", "End-to-end"),
    Metric("p99_e2el_ms", "p99", "End-to-end"),
    Metric("p99_tpot_ms", "TPOT p99", "Decode"),
    Metric("p99_itl_ms", "ITL p99", "Decode"),
)

# Ordinal blue ramp (steps 250/400/600) for the ordered request-rate axis.
# Checked against the light chart surface: contrast 2.06 / 3.54 / 7.89 : 1,
# worst pair OKLab dE 15.6 normal and 14.7 under simulated protanopia.
RAMP = ("#86b6ef", "#3987e5", "#184f95")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Direct-label a bar only when it carries information the axis alone does not:
# a large median, or an error range that spans zero.
LABEL_THRESHOLD_PCT = 5.0


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rate_label(rate: float) -> str:
    return f"{rate:g} req/s"


def load_summary(path: Path) -> dict[tuple[float, str], dict[str, float]]:
    """Index median/min/max improvement by (request rate, metric)."""
    rows: dict[tuple[float, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as summary_file:
        for row in csv.DictReader(summary_file):
            rate = number(row.get("request_rate"))
            median = number(row.get("median_improvement_pct"))
            if rate is None or median is None:
                continue
            rows[(rate, row["metric"])] = {
                "median": median,
                "min": number(row.get("min_improvement_pct")) or median,
                "max": number(row.get("max_improvement_pct")) or median,
                "repetitions": number(row.get("paired_repetitions")) or 0.0,
            }
    return rows


def subtitle(run_dir: Path, rows: dict[tuple[float, str], dict[str, float]]) -> str:
    manifest_path = run_dir / "run_manifest.json"
    parts = ["higher is better"]
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("workload"):
            parts.append(str(manifest["workload"]))
        if manifest.get("num_prompts"):
            parts.append(f"{manifest['num_prompts']} requests per case")
    repetitions = {int(entry["repetitions"]) for entry in rows.values()}
    if repetitions:
        count = max(repetitions)
        parts.append(f"{count} paired repetition{'s' if count != 1 else ''}, "
                     "error bars span min-max")
    return "  |  ".join(parts)


def draw(rows: dict[tuple[float, str], dict[str, float]], run_dir: Path,
         destination: Path) -> None:
    rates = sorted({rate for rate, _ in rows})
    if len(rates) > len(RAMP):
        raise SystemExit(
            f"{len(rates)} request rates exceed the {len(RAMP)}-step ordinal ramp")

    figure, axis = plt.subplots(figsize=(11.0, 5.6))
    # Explicit margins: the title band, the family headers below the tick
    # labels and the footnote all live outside the axes.
    figure.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.19)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    positions = list(range(len(METRICS)))
    # Leave a visible gap between adjacent bars rather than drawing borders.
    bar_width = 0.76 / len(rates)
    span = bar_width * len(rates)
    labelled: list[tuple[float, float, float]] = []

    for index, rate in enumerate(rates):
        offset = -span / 2 + bar_width * (index + 0.5)
        centers, values, lower, upper = [], [], [], []
        for position, metric in zip(positions, METRICS):
            entry = rows.get((rate, metric.key))
            if entry is None:
                continue
            centers.append(position + offset)
            values.append(entry["median"])
            lower.append(max(entry["median"] - entry["min"], 0.0))
            upper.append(max(entry["max"] - entry["median"], 0.0))
            spans_zero = entry["min"] <= 0.0 <= entry["max"]
            if abs(entry["median"]) >= LABEL_THRESHOLD_PCT or spans_zero:
                labelled.append((position + offset, entry["median"], entry["max"]))
        axis.bar(centers, values, width=bar_width * 0.88, color=RAMP[index],
                 label=rate_label(rate), zorder=3)
        axis.errorbar(centers, values, yerr=[lower, upper], fmt="none",
                      ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=2.5,
                      capthick=1.0, zorder=4)

    for center, median, top in labelled:
        anchor = max(median, top)
        axis.annotate(f"{median:.1f}", (center, anchor), xytext=(0, 5),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=8, color=TEXT_PRIMARY, zorder=5)

    axis.axhline(0.0, color=BASELINE, linewidth=1.2, zorder=2)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    axis.xaxis.grid(False)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(BASELINE)
    axis.tick_params(axis="both", length=0, colors=TEXT_MUTED, labelsize=9)

    axis.set_xticks(positions)
    axis.set_xticklabels([metric.label for metric in METRICS])
    axis.set_xlim(-0.55, len(METRICS) - 0.45)
    axis.set_ylabel("Improvement over baseline (%)", fontsize=10,
                    color=TEXT_SECONDARY)

    # Family headers under the tick labels, plus hairline separators.
    start = 0
    for position in positions[1:] + [len(METRICS)]:
        if position < len(METRICS) and METRICS[position].family == METRICS[start].family:
            continue
        axis.annotate(METRICS[start].family, ((start + position - 1) / 2, -0.085),
                      xycoords=("data", "axes fraction"), ha="center", va="top",
                      fontsize=10, color=TEXT_SECONDARY, annotation_clip=False)
        if position < len(METRICS):
            axis.axvline(position - 0.5, color=GRIDLINE, linewidth=0.8, zorder=1)
        start = position

    figure.text(0.075, 0.955, "KV transfer reverse-run canonicalization: "
                "latency benefit", fontsize=13, color=TEXT_PRIMARY,
                ha="left", va="top")
    caption = subtitle(run_dir, rows)
    if caption:
        figure.text(0.075, 0.905, caption, fontsize=9, color=TEXT_MUTED,
                    ha="left", va="top")
    figure.text(0.075, 0.035, "Throughput metrics are omitted: this is an "
                "open-loop benchmark at a fixed request rate, so throughput "
                "tracks the arrival rate rather than the optimization.",
                fontsize=8, color=TEXT_MUTED, ha="left", va="bottom")

    legend = axis.legend(title="Request rate", frameon=False, fontsize=9,
                         loc="lower right", bbox_to_anchor=(1.0, 1.005),
                         ncols=len(rates), handlelength=1.1, handleheight=0.9,
                         columnspacing=1.4, borderpad=0.0)
    legend.get_title().set_fontsize(9)
    legend.get_title().set_color(TEXT_SECONDARY)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, facecolor=SURFACE)
    plt.close(figure)


def write_table(rows: dict[tuple[float, str], dict[str, float]],
                destination: Path) -> None:
    """Write the table twin so every plotted value is readable as text."""
    fields = ["request_rate", "metric", "median_improvement_pct",
              "min_improvement_pct", "max_improvement_pct", "paired_repetitions"]
    with destination.open("w", newline="", encoding="utf-8") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=fields)
        writer.writeheader()
        for rate in sorted({rate for rate, _ in rows}):
            for metric in METRICS:
                entry = rows.get((rate, metric.key))
                if entry is None:
                    continue
                writer.writerow({
                    "request_rate": rate,
                    "metric": metric.key,
                    "median_improvement_pct": f"{entry['median']:.4f}",
                    "min_improvement_pct": f"{entry['min']:.4f}",
                    "max_improvement_pct": f"{entry['max']:.4f}",
                    "paired_repetitions": int(entry["repetitions"]),
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path,
                        help="Run directory holding generalization_summary.csv")
    parser.add_argument("--output-dir", type=Path, help="Default: <run_dir>/plots")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "generalization_summary.csv"
    if not summary_path.is_file():
        parser.error(f"Not found: {summary_path}")

    rows = load_summary(summary_path)
    plotted = {key for key in rows if key[1] in {metric.key for metric in METRICS}}
    if not plotted:
        parser.error(f"No plottable latency metrics in {summary_path}")

    output_dir = (args.output_dir or run_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "benefit_overview.png"
    table_path = output_dir / "benefit_overview.csv"
    draw(rows, run_dir, chart_path)
    write_table(rows, table_path)
    print(f"Plotted {len(plotted)} metric/rate cells from {summary_path}")
    print(f"Wrote {chart_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
