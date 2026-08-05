import json

from tests.pd_transfer.plot_pd_transfer import (
    build_motivation_points,
    create_generalization_delay_sweeps,
    create_motivation_core_analysis,
    create_paired_transfer_analysis,
    load_generalization_delay_rows,
)


def _write_result(root, delay_ms, repetition=1):
    case_dir = (
        root / "diagnostic" / "on" / "rps-0p18"
        / f"delay-{delay_ms}" / f"rep-{repetition}"
    )
    case_dir.mkdir(parents=True)
    result = {
        "phase": "diagnostic",
        "variant": "on",
        "repetition": repetition,
        "configured_request_rate": "0.18",
        "injected_transfer_delay_ms": str(delay_ms),
        "workload": "mooncake-toolagent",
        "num_prompts": 50,
        "completed": 50,
        "mean_ttft_ms": 3000 + delay_ms,
        "p99_ttft_ms": 12000 + delay_ms,
        "request_throughput": 1.1,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result), encoding="utf-8")


def test_generalization_delay_sweep_loads_and_plots_result_json(tmp_path):
    _write_result(tmp_path, 10)
    _write_result(tmp_path, 10, repetition=2)
    _write_result(tmp_path, 50)
    _write_result(tmp_path, 50, repetition=2)

    rows = load_generalization_delay_rows(tmp_path)

    assert [(row["transfer_delay_ms"], row["repetition"]) for row in rows] == [
        (10.0, 1),
        (10.0, 2),
        (50.0, 1),
        (50.0, 2),
    ]
    assert all(row["phase"] == "diagnostic" for row in rows)
    assert create_generalization_delay_sweeps(rows, tmp_path / "plots") == 1
    plots = list((tmp_path / "plots").rglob("*.png"))
    assert len(plots) == 1
    assert "01_generalization_delay_sweep" in plots[0].parts


def test_paired_transfer_analysis_uses_trace_latency_and_request_pairs(tmp_path):
    rows = []
    for rate in ("0.08", "0.16"):
        for repetition in (1, 2):
            for request_index in range(4):
                for variant in ("off", "on"):
                    packed = variant == "on" and request_index % 2 == 1
                    transfer_ms = (35.0 if variant == "off" else
                                   (22.0 if packed else 32.0))
                    rows.append({
                        "run_id": "conv-pair",
                        "phase": "diagnostic",
                        "workload": "mooncake-conversation",
                        "variant": variant,
                        "repetition": str(repetition),
                        "configured_request_rate": rate,
                        "case_id": f"case-{variant}-{rate}-{repetition}",
                        "case_request_index": str(request_index),
                        "request_id": f"request-{variant}-{request_index}",
                        "handshake_cached": "True",
                        "selected_transfer_path": (
                            "packed" if packed else "direct"),
                        "kv_load_to_connector_finished_ms": str(transfer_ms),
                        "proxy_ttft_ms": str(
                            300 if variant == "off" else 270 + request_index),
                        "total_bytes": "1000000000",
                        "selector_forward_ranges": "100",
                        "transfer_before_issue_ms": str(transfer_ms * 0.4),
                        "transfer_issue_span_ms": str(transfer_ms * 0.3),
                        "transfer_post_issue_wait_ms": str(transfer_ms * 0.2),
                        "transfer_connector_finalize_ms": str(
                            transfer_ms * 0.1),
                        "packed_source_handler_ms": "8" if packed else "0",
                        "packed_pack_gpu_ms": "6" if packed else "0",
                        "packed_scatter_gpu_ms": "4" if packed else "0",
                        "packed_source_stream_wait_gpu_ms": (
                            "0.05" if packed else "0"),
                    })

    count = create_paired_transfer_analysis(rows, tmp_path)

    assert count == 2
    assert (tmp_path / "paired_transfer_summary.csv").is_file()
    assert (tmp_path / "matched_cohort_transfer_p50.csv").is_file()
    delta_path = tmp_path / "paired_request_transfer_delta.csv"
    assert delta_path.is_file()
    assert len(delta_path.read_text(encoding="utf-8").splitlines()) == 17
    ttft_delta_path = tmp_path / "paired_ttft_request_delta.csv"
    assert ttft_delta_path.is_file()
    assert len(ttft_delta_path.read_text(encoding="utf-8").splitlines()) == 17
    assert (tmp_path / "paired_ttft_delta_summary.csv").is_file()
    plots = list((tmp_path / "conv-pair").rglob("*.png"))
    assert len(plots) == 2
    assert any(plot.name.endswith("-matched-cohort-bars.png") for plot in plots)
    assert any(plot.name.endswith("-paired-delta-ttft.png") for plot in plots)


def test_motivation_core_combines_loads_and_repetition_error_bars(tmp_path):
    rows = []
    for rate in ("0.08", "0.16", "0.24"):
        for repetition in (1, 2):
            for delay_ms in (0, 50, 100, 200):
                for request_index in range(4):
                    rows.append({
                        "run_id": "conversation-motivation",
                        "phase": "diagnostic",
                        "workload": "mooncake-conversation",
                        "variant": "off",
                        "repetition": str(repetition),
                        "configured_request_rate": rate,
                        "injected_transfer_delay_ms": str(delay_ms),
                        "case_id": f"case-{rate}-{delay_ms}-{repetition}",
                        "case_request_index": str(request_index),
                        "handshake_cached": "True",
                        "proxy_ttft_ms": str(
                            1000 + request_index + delay_ms * 1.2
                            + repetition),
                        "kv_load_to_connector_finished_ms": str(
                            80 + request_index + delay_ms),
                    })

    delay_points, baseline_points = build_motivation_points(rows)

    assert len(delay_points) == 3 * 2 * 4
    assert len(baseline_points) == 3 * 2
    delay_100 = [
        row["paired_delta_ttft_p50_ms"] for row in delay_points
        if row["transfer_delay_ms"] == 100
    ]
    assert delay_100 == [120.0] * 6
    assert create_motivation_core_analysis(rows, tmp_path) == 3
    assert (tmp_path / "motivation_paired_delta_ttft.csv").is_file()
    plots = list((tmp_path / "conversation-motivation").rglob("*.png"))
    assert len(plots) == 3
    assert all("01_motivation_core" in plot.parts for plot in plots)
    assert {plot.name.rsplit("-", 3)[-1] for plot in plots} == {
        "ttft.png", "summary.png", "share.png"
    }
