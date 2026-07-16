# PD Transfer Latency Experiment

This directory contains a pull-mode benchmark harness for measuring how extra
P-to-D KV transfer latency affects PD-disaggregated serving.

The instrumentation records request-level events from:

- proxy: request arrival, P request start/end, D request start/end, first output
  chunk
- prefiller: prefill completion
- decoder: NIXL READ start, transfer completion, remote KV ready

`VLLM_PD_TRANSFER_SLEEP_MS` injects extra delay on the decoder side after the
NIXL receive transfer has completed and before vLLM marks the remote KV as
ready. This is intentionally scoped to the critical path that can affect TTFT.

## Usage

```bash
cp tests/pd_transfer/config.example.env tests/pd_transfer/config.env
```

Edit at least `MODEL`, `BENCH_TOKENIZER`, GPU ids, lengths, and sleep values.

By default, the runner executes the Cartesian product of `INPUT_LENS`,
`OUTPUT_LENS`, `CONCURRENCIES`, and `TRANSFER_SLEEP_MS_LIST`. To run only
selected cases, set `BENCH_CASES` instead:

```bash
BENCH_CASES="2048,32,1,0 2048,32,1,20 4096,32,1,0 4096,32,1,40"
```

Each entry is `input_len,output_len,concurrency,sleep_ms`. When `BENCH_CASES`
is set, the matrix variables are ignored.

```bash
bash tests/pd_transfer/run_pd_transfer_bench.sh tests/pd_transfer/config.env
python tests/pd_transfer/parse_pd_trace.py results/pd_transfer/<run-id>
```

The parser writes:

- `serve_summary.csv`: one row per `vllm bench serve` case
- `sleep_sensitivity_summary.csv`: delta against `sleep_ms=0`
- `pd_request_timeline_ms.csv`: request-level event timeline from JSONL traces

This v0.11 branch has NIXL pull support only; push-mode comparison is not part
of this harness.
