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
                "hash_ids": list(range(3, 11)),
            },
            {
                "timestamp": 30,
                "input_length": 40900,
                "output_length": 100,
                "hash_ids": list(range(11, 91)),
            },
            {
                "timestamp": 40,
                "input_length": 100,
                "output_length": 0,
                "hash_ids": [91],
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
    assert summary["hash_ids_preserved"] is False
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
            "hash_ids": [1, 2],
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


def test_convert_trace_jsonl_preserves_hashes_order_and_timestamps(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "filtered.jsonl"
    records = [
        {
            "timestamp": 0,
            "input_length": 600,
            "output_length": 32,
            "hash_ids": [10, 11],
        },
        {
            "timestamp": 25,
            "input_length": 700,
            "output_length": 64,
            "hash_ids": [10, 12],
        },
    ]
    _write_jsonl(source, records)

    summary = convert_trace(
        source,
        output,
        max_total_tokens=40960,
    )

    converted = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert converted == records
    assert summary["format"] == "mooncake-fast25-jsonl"
    assert summary["timestamp_preserved"] is True
    assert summary["request_order_preserved"] is True
    assert summary["hash_ids_preserved"] is True
    assert summary["hash_ids_replayed"] is True
    reuse = summary["infinite_capacity_prefix_reuse"]
    assert reuse["reused_tokens"] == 512
    assert reuse["total_input_tokens"] == 1300
    assert reuse["ratio"] == pytest.approx(512 / 1300)
    assert reuse["requests_with_reuse"] == 1


def test_convert_trace_rejects_invalid_hash_count(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [{
            "timestamp": 0,
            "input_length": 600,
            "output_length": 32,
            "hash_ids": [10],
        }],
    )

    with pytest.raises(MooncakeConversionError, match="requires 2 hash IDs"):
        convert_trace(
            source,
            tmp_path / "filtered.jsonl",
            max_total_tokens=40960,
        )
