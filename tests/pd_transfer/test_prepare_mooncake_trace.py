# SPDX-License-Identifier: Apache-2.0

import csv
import json

import pytest

from tests.pd_transfer.prepare_mooncake_trace import (
    MooncakeConversionError, convert_trace)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_convert_trace_writes_compatible_csv_and_stats(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "converted.csv"
    _write_jsonl(
        source,
        [
            {
                "timestamp": 10,
                "input_length": 1024,
                "output_length": 32,
                "hash_ids": [1, 2],
            },
            {
                "timestamp": 20,
                "input_length": 4000,
                "output_length": 200,
                "hash_ids": [3],
            },
            {
                "timestamp": 30,
                "input_length": 40900,
                "output_length": 100,
                "hash_ids": [],
            },
            {
                "timestamp": 40,
                "input_length": 100,
                "output_length": 0,
                "hash_ids": [],
            },
        ],
    )

    summary = convert_trace(
        source,
        output,
        max_total_tokens=40960,
    )

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["Timestamp", "Model", "Request tokens", "Response tokens"],
        ["10", "GPT-4", "1024", "32"],
        ["20", "GPT-4", "4000", "200"],
    ]
    assert summary["source_rows"] == 4
    assert summary["written_rows"] == 2
    assert summary["filtered_non_positive_length"] == 1
    assert summary["filtered_over_context_limit"] == 1
    assert summary["input_length"]["p50"] == 1024
    assert summary["total_length"]["max"] == 4200
    assert summary["hash_ids_replayed"] is False
    persisted = json.loads(
        output.with_suffix(".stats.json").read_text(encoding="utf-8"))
    assert persisted == summary


def test_convert_trace_rejects_malformed_length(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [{
            "timestamp": 10,
            "input_length": "not-an-integer",
            "output_length": 32,
        }],
    )

    with pytest.raises(MooncakeConversionError, match="line 1.*input_length"):
        convert_trace(
            source,
            tmp_path / "converted.csv",
            max_total_tokens=40960,
        )


def test_convert_trace_does_not_overwrite_without_force(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "converted.csv"
    _write_jsonl(
        source,
        [{
            "timestamp": 10,
            "input_length": 1024,
            "output_length": 32,
        }],
    )
    convert_trace(source, output, max_total_tokens=40960)

    with pytest.raises(FileExistsError):
        convert_trace(source, output, max_total_tokens=40960)

    summary = convert_trace(
        source,
        output,
        max_total_tokens=40960,
        force=True,
    )
    assert summary["written_rows"] == 1
