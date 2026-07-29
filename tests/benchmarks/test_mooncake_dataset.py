# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm.benchmarks.datasets import MooncakeTraceDataset


class FakeTokenizer:
    vocab_size = 128
    all_special_ids = [0, 1, 127]


def _write_trace(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_mooncake_replay_preserves_prefix_tokens_and_timestamps(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_trace(
        path,
        [
            {
                "timestamp": 100,
                "input_length": 600,
                "output_length": 32,
                "hash_ids": [10, 11],
            },
            {
                "timestamp": 125,
                "input_length": 700,
                "output_length": 64,
                "hash_ids": [10, 12],
            },
        ],
    )

    samples = MooncakeTraceDataset(
        dataset_path=str(path),
        block_size=512,
        arrival_rate_scale=2.0,
        token_seed=7,
    ).sample(
        tokenizer=FakeTokenizer(),
        num_requests=2,
        request_id_prefix="trace-",
    )

    assert len(samples[0].prompt) == 600
    assert len(samples[1].prompt) == 700
    assert samples[0].prompt[:512] == samples[1].prompt[:512]
    assert samples[0].prompt[512:] != samples[1].prompt[512:600]
    assert not set(samples[0].prompt) & {0, 1, 127}
    assert samples[0].scheduled_offset_s == 0
    assert samples[1].scheduled_offset_s == pytest.approx(0.0125)
    assert samples[0].expected_output_len == 32
    assert samples[1].expected_output_len == 64
    assert samples[0].request_id == "trace-0"
    assert samples[1].request_id == "trace-1"


def test_mooncake_replay_rejects_oversampling(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_trace(
        path,
        [{
            "timestamp": 0,
            "input_length": 10,
            "output_length": 2,
            "hash_ids": [1],
        }],
    )
    dataset = MooncakeTraceDataset(dataset_path=str(path))

    with pytest.raises(ValueError, match="does not oversample"):
        dataset.sample(tokenizer=FakeTokenizer(), num_requests=2)


def test_mooncake_replay_rejects_non_monotonic_timestamps(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_trace(
        path,
        [
            {
                "timestamp": 10,
                "input_length": 10,
                "output_length": 2,
                "hash_ids": [1],
            },
            {
                "timestamp": 9,
                "input_length": 10,
                "output_length": 2,
                "hash_ids": [2],
            },
        ],
    )

    with pytest.raises(ValueError, match="timestamps must be non-decreasing"):
        MooncakeTraceDataset(dataset_path=str(path))
