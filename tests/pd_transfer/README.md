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
- paired-forward, paired-reverse, and fragmented local/remote block runs,
  including forward-only, reverse-only, and bidirectional theoretical range
  counts plus the longest paired runs
- first/last/min/max local and remote block IDs, plus block-pair analysis time

A paired run preserves each local-to-remote block mapping while both block ID
sequences advance by `+1` (forward) or `-1` (reverse). Blocks that cannot join
either kind of run are reported as one-block fragments. The instrumentation
reports the original allocator order even when the experimental paired-reverse
canonicalization described below changes the descriptor submission order.

## Paired-reverse canonicalization experiment

Set this in `config.env` to enable the optimization:

```bash
CANONICALIZE_REVERSE_BLOCK_PAIRS=1
```

The default is `0`. When enabled, the NIXL connector reverses both the local
and remote block pairs inside every run where both sides advance by `-1`.
This preserves each local-to-remote mapping while presenting `+1` contiguous
descriptor IDs to the transport backend. No allocator or block table is
modified.

Each run writes the resolved value to `experiment_config.json`. Decoder
`pull_transfer_profile` records also contain
`reverse_block_pair_canonicalization_enabled`,
`canonicalized_reverse_run_count`, and
`canonicalized_reverse_block_count`.

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
BENCH_CASES="2048,32,1,0,384 4096,32,1,0,192 8192,32,1,0,96 16384,32,1,0,48"
```

Each entry is
`input_len,output_len,concurrency,sleep_ms[,num_prompts]`. The optional fifth
value controls the request count for that case. When it is omitted, the runner
uses `concurrency * NUM_FOLDS`, preserving the previous behavior. When
`BENCH_CASES` is set, the matrix variables are ignored.

The runner saves per-request generated text and timing details in
`<case-dir>/<case-id>.json`. It also gives every formal benchmark request a
deterministic ID derived from the case ID and request index. The proxy adds a
fixed non-numeric suffix before forwarding the ID to the Prefill and Decode
servers. This keeps the logical request ID unambiguous when an engine appends a
numeric TP-rank suffix, and allows benchmark outputs to be joined with transfer
trace records.

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

## Correctness comparison

Run the trace parser for both an optimization-disabled run and an
optimization-enabled run, then compare them:

```bash
python tests/pd_transfer/compare_pd_correctness.py \
  results/pd_transfer/<baseline-run-id> \
  results/pd_transfer/<optimized-run-id>
```

The comparison joins detailed benchmark outputs to transfer profiles by the
deterministic request ID. For every formal request, it checks generated text,
input/output lengths, errors, local/remote block and descriptor counts,
transfer handle count, and total bytes. By default, every case must cover
paired-forward and paired-reverse requests, and the optimized run must
canonicalize at least one paired-reverse request. Use
`--skip-coverage-checks` only for deliberately short diagnostic cases.

## BurstGPT generalization performance

The correctness runner above deliberately controls lengths, prefix caching,
and eager execution. Use the separate generalization runner for paired OFF/ON
performance measurements with mixed BurstGPT request and response lengths:

```bash
cp tests/pd_transfer/generalization_config.example.env \
  tests/pd_transfer/generalization_config.env

wget -O /data/BurstGPT_without_fails_2.csv \
  https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv

bash tests/pd_transfer/run_pd_transfer_generalization_bench.sh \
  tests/pd_transfer/generalization_config.env
```

Use the BurstGPT v1.1 CSV. The vLLM 0.11 loader reads the `Model`, `Request
tokens`, and `Response tokens` fields by column position; the runner rejects
newer incompatible layouts before starting the model servers.
The built-in loader selects rows whose original workload label is `GPT-4`;
this does not restrict the model being benchmarked. It also does not filter
requests against the tested model's context length, so any over-length request
is treated as a failed benchmark artifact and must be removed from the input
CSV before the final run.

Each `repetition x request-rate x variant` case starts fresh P/D servers,
executes the same separate Random warm-up, and then samples the same BurstGPT
requests using a fixed seed. Odd repetitions run OFF then ON; even repetitions
run ON then OFF. Main performance cases leave detailed PD tracing disabled and
do not set model length, scheduler capacity, GPU utilization, eager execution,
or prefix caching options. The optional diagnostic OFF/ON pair enables JSONL
tracing separately.

Aggregate the results with:

```bash
python tests/pd_transfer/compare_pd_generalization_perf.py \
  results/pd_transfer_generalization/<run-id>
```

The analyzer verifies completed request counts, request errors, OFF/ON
input/output length sequences, and pairing. It writes
`generalization_pairwise.csv`, `generalization_summary.csv`, and, when
diagnostic traces exist, `diagnostic_trace_summary.csv`. The diagnostic
summary includes reverse-request fraction, total local/remote blocks,
canonicalized-block fraction, and p50/p90/p99 per-request canonicalized-block
fractions. Diagnostic profiles are restricted to the formal benchmark time
window, excluding health checks and Random warm-up requests.

## Mooncake long-input performance

Use the same isolated OFF/ON runner with a Mooncake FAST'25 length trace to
measure the optimization when long inputs turn over the KV block pool many
times. Download one of the official JSONL traces locally. The synthetic trace
has the longest average input and is the recommended first mechanism test:

```bash
wget -O dataset/synthetic_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl
```

Convert it with the standard-library-only adapter:

```bash
python tests/pd_transfer/prepare_mooncake_trace.py \
  dataset/synthetic_trace.jsonl \
  dataset/mooncake_synthetic_vllm.csv \
  --max-total-tokens 40960
```

The converter writes the positional CSV consumed by the vLLM 0.11 BurstGPT
loader and a neighboring `mooncake_synthetic_vllm.stats.json` file. It rejects
malformed rows, filters requests whose input plus output exceeds the configured
context limit, and prints input/output/total length statistics. Pass `--force`
to replace an existing conversion.

Only timestamps and input/output lengths are preserved. Mooncake `hash_ids`
are intentionally not replayed, so the benchmark uses synthetic request-unique
contents rather than reproducing prefix-cache sharing. The runner also uses its
configured request rate instead of replaying the original timestamps. This
keeps OFF/ON request lengths identical while isolating long-input block
turnover.

Copy both the converted CSV and the repository changes to the server, then
prepare the separate configuration:

```bash
cp tests/pd_transfer/long_context_config.example.env \
  tests/pd_transfer/long_context_config.env

bash tests/pd_transfer/run_pd_transfer_generalization_bench.sh \
  tests/pd_transfer/long_context_config.env

python tests/pd_transfer/compare_pd_generalization_perf.py \
  results/pd_transfer_long_context/<run-id>
```

The example's `0.1 0.2 0.4` request rates are calibration starting points, not
portable final rates. Run a small smoke test on the target model/GPU before the
three-repetition experiment. The long-input result is valid only when the
diagnostic ON case reports substantial `canonicalized_block_fraction`; compare
that fraction with the end-to-end and transfer-path improvement rather than
assuming prompt length alone guarantees paired-reverse allocation.

The runner now accepts the generic `DATASET_PATH`, `WORKLOAD_NAME`,
`WORKLOAD_SLUG`, and `DATASET_SOURCE_FORMAT` settings. Existing configs using
`BURSTGPT_DATASET_PATH` remain supported as a compatibility fallback.

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
- `04_request_e2e_scatter`: one raw scatter plot per benchmark case; every
  point is one request's end-to-end latency, ordered by proxy arrival time
- `05_kv_transfer_scatter`: one raw scatter plot per benchmark case; every
  point is one Decode-side KV load-to-connector-finish latency; cold handshake
  requests are excluded by default

The sleep-sweep charts show mean TTFT, p99 TTFT, and request throughput. The
pie charts require `pd_request_timeline_ms.csv`, so run
`parse_pd_trace.py` first. Use `--output-dir <path>` to place artifacts elsewhere.
By default, pie charts exclude requests with a cold NIXL handshake so they
represent steady-state transfer latency. Pass `--include-cold-handshake` to
include startup cost instead.

```bash
python tests/pd_transfer/plot_pd_transfer.py results/pd_transfer/<run-id>
```
