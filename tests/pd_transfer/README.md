# PD transfer latency experiment

This directory contains a small first-stage benchmark harness for measuring how
extra P-to-D KV transfer latency affects PD-disaggregated serving.

## What is measured

The instrumentation records request-level events from three places:

- proxy: request arrival, P request start/end, D request start/end, first output
  chunk
- prefiller: prefill completion and push transfer start
- decoder: pull transfer start, push registration, transfer completion, remote KV
  ready

`VLLM_PD_TRANSFER_SLEEP_MS` injects extra delay on the decoder side after the
NIXL receive transfer has completed and before vLLM marks the remote KV as
ready. This is intentionally scoped to the critical path that can affect TTFT.

## Usage

Create a config from the example:

```bash
cp tests/pd_transfer/config.example.env tests/pd_transfer/config.env
```

Edit at least:

- `MODEL`
- `BENCH_TOKENIZER`
- `PREFILL_DEVICES`
- `DECODE_DEVICES`
- `INPUT_LENS`
- `OUTPUT_LENS`
- `TRANSFER_SLEEP_MS_LIST`

Run:

```bash
bash tests/pd_transfer/run_pd_transfer_bench.sh tests/pd_transfer/config.env
```

Parse results:

```bash
python tests/pd_transfer/parse_pd_trace.py results/pd_transfer/<run-id>
```

The parser writes:

- `serve_summary.csv`: one row per `vllm bench serve` case
- `sleep_sensitivity_summary.csv`: delta against `sleep_ms=0`
- `pd_request_timeline_ms.csv`: request-level event timeline derived from JSONL
  traces

## Notes

- `MODES="pull push"` runs both NIXL pull and NIXL push.
- The default config uses one GPU for P and one GPU for D.
- The runner is a Linux bash script and is intended to run on the A100 machine,
  not from Windows PowerShell.
