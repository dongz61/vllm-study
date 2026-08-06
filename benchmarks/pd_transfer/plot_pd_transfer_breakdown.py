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
    "prefill_to_submit_ms",
    "submit_to_physical_done_ms",
    "physical_done_to_reported_ms",
)
COMPONENT_LABELS = (
    "Prefill done -> first submit",
    "First submit -> all transfers done",
    "Physical done -> reported",
)
COMPONENT_COLORS = ("#4C78A8", "#F58518", "#54A24B")


@dataclass
class Case:
    run_dir: Path
    label: str
    manifest: dict[str, Any]
    rows_seen: int
    ttft_ms: list[float]
    full_ms: list[float]
    ratios: list[float]
    components_ms: tuple[list[float], list[float], list[float]]


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
    return name.replace("_", " ") or "unknown dataset"


def _case_label(manifest: dict[str, Any]) -> str:
    dataset = _dataset_name(str(manifest.get("trace_path", "")))
    prefill_tp = manifest.get("prefill_tp_size", "?")
    decode_tp = manifest.get("decode_tp_size", "?")
    scale = manifest.get("trace_time_scale", "?")
    return f"{dataset}\nP{prefill_tp}-D{decode_tp}, scale={scale}"


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
    component_lists: tuple[list[float], list[float], list[float]] = ([], [], [])
    for row in rows:
        ttft = _float(row, "proxy_ttft_ms")
        full = _float(row, "prefill_to_reported_ms")
        components = tuple(_float(row, field) for field in COMPONENT_FIELDS)
        if (
            ttft is None
            or full is None
            or ttft <= 0
            or full < 0
            or any(value is None or value < 0 for value in components)
        ):
            continue
        ttft_ms.append(ttft)
        full_ms.append(full)
        ratios.append(full / ttft)
        for values, value in zip(component_lists, components, strict=True):
            assert value is not None
            values.append(value)

    if not full_ms:
        print(f"Skipping {run_dir}: no request has a complete PD interval and TTFT")
        return None
    return Case(
        run_dir=run_dir,
        label=_case_label(manifest),
        manifest=manifest,
        rows_seen=len(rows),
        ttft_ms=ttft_ms,
        full_ms=full_ms,
        ratios=ratios,
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


def plot_pd_share(cases: list[Case], output_path: Path) -> None:
    labels = [case.label for case in cases]
    pd_shares = [statistics.fmean(case.ratios) * 100 for case in cases]
    other_shares = [100 - share for share in pd_shares]
    y_positions = list(range(len(cases)))

    fig, ax = plt.subplots(
        figsize=(9.2, _figure_height(len(cases))), layout="constrained"
    )
    ax.barh(y_positions, pd_shares, color="#4C78A8", label="Complete PD transfer")
    ax.barh(
        y_positions,
        other_shares,
        left=pd_shares,
        color="#D9D9D9",
        label="Other TTFT",
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
    ax.set_xlabel("Mean per-request share of TTFT (%)")
    ax.set_title("Complete PD-transfer latency as a share of TTFT")
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
    ax.set_title("Request-level PD-transfer share distribution (whiskers: p1-p99)")
    ax.grid(axis="x", alpha=0.25)
    ax.scatter([], [], marker="D", color="#E45756", label="Mean")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_internal_breakdown(cases: list[Case], output_path: Path) -> None:
    labels = [case.label for case in cases]
    component_shares: list[list[float]] = []
    for case in cases:
        sums = [sum(values) for values in case.components_ms]
        total = sum(sums)
        component_shares.append([value / total * 100 for value in sums])

    y_positions = list(range(len(cases)))
    left = [0.0] * len(cases)
    fig, ax = plt.subplots(
        figsize=(10.2, _figure_height(len(cases))), layout="constrained"
    )
    for index, (component_label, color) in enumerate(
        zip(COMPONENT_LABELS, COMPONENT_COLORS, strict=True)
    ):
        shares = [case_shares[index] for case_shares in component_shares]
        ax.barh(
            y_positions,
            shares,
            left=left,
            color=color,
            label=component_label,
        )
        for y, start, share in zip(y_positions, left, shares, strict=True):
            if share >= 4:
                ax.text(
                    start + share / 2,
                    y,
                    f"{share:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
        left = [start + share for start, share in zip(left, shares, strict=True)]

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of complete PD-transfer latency (%)")
    ax.set_title("Internal breakdown of complete PD-transfer latency")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(cases: list[Case], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        component_means = [
            statistics.fmean(values) for values in case.components_ms
        ]
        component_sums = [sum(values) for values in case.components_ms]
        components_total = sum(component_sums)
        full_total = sum(case.full_ms)
        ttft_total = sum(case.ttft_ms)
        rows.append(
            {
                "label": case.label.replace("\n", " | "),
                "run_dir": str(case.run_dir),
                "trace_path": case.manifest.get("trace_path", ""),
                "prefill_tp_size": case.manifest.get("prefill_tp_size", ""),
                "decode_tp_size": case.manifest.get("decode_tp_size", ""),
                "trace_time_scale": case.manifest.get("trace_time_scale", ""),
                "requests_seen": case.rows_seen,
                "requests_complete": len(case.full_ms),
                "pd_over_ttft_mean_request_pct": statistics.fmean(case.ratios)
                * 100,
                "pd_over_ttft_p50_request_pct": percentile(case.ratios, 0.50)
                * 100,
                "pd_over_ttft_p90_request_pct": percentile(case.ratios, 0.90)
                * 100,
                "pd_over_ttft_p99_request_pct": percentile(case.ratios, 0.99)
                * 100,
                "pd_over_ttft_aggregate_pct": full_total / ttft_total * 100,
                "pd_complete_mean_ms": statistics.fmean(case.full_ms),
                "prefill_to_submit_mean_ms": component_means[0],
                "submit_to_all_physical_done_mean_ms": component_means[1],
                "physical_done_to_reported_mean_ms": component_means[2],
                "prefill_to_submit_internal_pct": component_sums[0]
                / components_total
                * 100,
                "submit_to_all_physical_done_internal_pct": component_sums[1]
                / components_total
                * 100,
                "physical_done_to_reported_internal_pct": component_sums[2]
                / components_total
                * 100,
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

    plot_pd_share(cases, output_dir / "pd_share_of_ttft.png")
    plot_ratio_distribution(cases, output_dir / "pd_ttft_ratio_distribution.png")
    plot_internal_breakdown(cases, output_dir / "pd_internal_breakdown.png")
    write_summary(cases, output_dir / "pd_transfer_plot_summary.csv")

    print(f"Plotted {len(cases)} case(s) from {len(run_dirs)} discovered run(s)")
    print(f"Wrote figures and summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
