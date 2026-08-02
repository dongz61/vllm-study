import json

from tests.pd_transfer.plot_pd_transfer import (
    create_generalization_delay_sweeps,
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
    _write_result(tmp_path, 50)

    rows = load_generalization_delay_rows(tmp_path)

    assert [row["transfer_delay_ms"] for row in rows] == [10.0, 50.0]
    assert all(row["phase"] == "diagnostic" for row in rows)
    assert create_generalization_delay_sweeps(rows, tmp_path / "plots") == 1
    plots = list((tmp_path / "plots").rglob("*.png"))
    assert len(plots) == 1
    assert "01_generalization_delay_sweep" in plots[0].parts
