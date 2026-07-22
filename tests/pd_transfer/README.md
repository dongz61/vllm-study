# PD Transfer Latency Experiment

This directory contains a pull-mode benchmark harness for measuring how extra
P-to-D KV transfer latency affects PD-disaggregated serving.

The instrumentation records request-level events from:

- proxy: request arrival, P request start/end, D request start/end, first output
  chunk
- prefiller: prefill completion
- decoder: NIXL READ start, transfer completion, remote KV ready

For real-transfer decomposition, the decoder also emits one
`pull_transfer_profile` record per request. The profile is accumulated in
memory and written only after the connector reports the request complete. It
contains:

- cold/warm handshake status and handshake wait
- descriptor-ID construction, transfer-handle preparation, and submit time
- first poll, poll rounds, `PROC` checks, and the time DONE was observed
- connector completion time
- block, descriptor, handle, and byte counts

`xfer_done_observed_perf_ns` is the time vLLM first observed NIXL return DONE;
it is not a hardware-level physical-completion timestamp. Compare an isolated
NIXL benchmark with the integrated profile before attributing this whole span
to data movement.

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

The timeline CSV merges `pull_transfer_profile` fields with scheduler events
and derives allocation-to-load, load-to-ready, connector-to-scheduler, and
ready-to-schedulable durations. It also assigns each request to the enclosing
benchmark case, adding input length, output length, concurrency, and sleep.

This v0.11 branch has NIXL pull support only; push-mode comparison is not part
of this harness.

## Visualization

After running the trace parser, `plot_pd_transfer.py` writes PNG figures plus
`plot_summary.csv` under `<run-id>/plots` by default:

- `01_sleep_sweep`: fixed input and concurrency, varying injected sleep
- `02_kv_transfer_breakdown`: one pie per benchmark case, showing the
  mean-per-request connector time split across handshake, descriptor build,
  NIXL preparation/submission, observed completion, and other connector work
- `03_kv_transfer_request_share`: one pie per benchmark case, showing the
  mean KV load-to-connector-finish time as a share of mean end-to-end request
  time (from proxy receipt until the Decode RPC ends)

The sleep-sweep charts show mean TTFT, p99 TTFT, and request throughput. The
pie charts require `pd_request_timeline_ms.csv`, so run
`parse_pd_trace.py` first. Use `--output-dir <path>` to place artifacts elsewhere.

```bash
python tests/pd_transfer/plot_pd_transfer.py results/pd_transfer/<run-id>
```
