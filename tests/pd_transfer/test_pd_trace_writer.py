# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path


def _load_pd_trace_module():
    module_path = (Path(__file__).parents[2] / "vllm" / "distributed" /
                   "kv_transfer" / "pd_trace.py")
    spec = importlib.util.spec_from_file_location("pd_trace_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_writer_reuses_file_and_flushes_lines(tmp_path, monkeypatch):
    pd_trace = _load_pd_trace_module()
    trace_path = tmp_path / "decode.trace.jsonl"
    monkeypatch.setenv("VLLM_PD_TRACE_PATH", str(trace_path))

    pd_trace.trace_event("first", "req", value=1)
    pd_trace.trace_event("second", "req", value=2)

    assert len(pd_trace._trace_files) == 1
    records = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["first", "second"]
    pd_trace._close_trace_files()
    assert pd_trace._trace_files == {}
