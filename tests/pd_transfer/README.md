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

## Direct, packed, and auto NIXL paths

The pull runner can switch the connector data path without editing its JSON:

```bash
NIXL_TRANSFER_MODE=direct  # original descriptor READ baseline
# NIXL_TRANSFER_MODE=packed  # force GPU gather / contiguous READ / GPU scatter
# NIXL_TRANSFER_MODE=auto    # choose from exact block and forward-range counts
```

Set one of these values in `config.env`. Packed and auto modes also accept:

```bash
NIXL_PACKED_STAGING_MIB=64
NIXL_PACKED_STAGING_SLOTS=64
NIXL_PACKED_AUTO_RANGE_THRESHOLD=64
```

The conservative auto rule chooses packed only when `K >= 64` and the Decode
worker has at least one free local staging slot; block count no longer provides
an independent packed fallback. The range threshold comes from the standalone
wall-time cuts and remains configurable. Both Prefill and Decode receive the
same staging and selector settings from the runner.
The resolved mode and staging parameters are written to
`experiment_config.json`, while detailed traces record the selected path and
the exact `B` and `K` for every transfer.

When detailed PD tracing is enabled, the connector also estimates the remaining
pair-preserving reorder opportunity without changing the submitted descriptors:

- `reverse_canonicalized_range_count` is the exact forward-only range count
  after applying the existing paired-reverse transform in analysis only;
- `reordered_optimal_range_count` is the forward-only range count after stably
  sorting corresponding local/remote pairs by local block ID;
- `additional_reorderable_edge_count` is the extra range reduction available
  beyond paired-reverse canonicalization;
- `generalized_reordered_block_count` is the number of pair positions changed
  by that stable sort.

These fields are collected whether canonicalization is OFF or ON. Therefore a
trace-enabled `VARIANT_MODE=off` diagnostic run is sufficient to characterize
both the reverse-only and generalized-reorder opportunities without applying
either optimization to the submitted transfer. The diagnostic analysis itself
sorts the block-pair metadata and emits JSONL records, so use a separate
trace-disabled OFF run for headline latency, TTFT, and throughput numbers.

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
fractions. New traces also report the request coverage and total/p50/p90/p99
range reduction available from generalized pair reordering beyond the existing
reverse transform. Diagnostic profiles are restricted to the formal benchmark
time window, excluding health checks and Random warm-up requests.

## Mooncake long-input performance

Use the same isolated OFF/ON runner with a Mooncake FAST'25 trace. The native
loader preserves request order and timestamps and maps every prefix `hash_id`
to a deterministic 512-token block. Repeated hash IDs therefore produce
identical prompt tokens and exercise vLLM prefix caching. Download one of the
official JSONL traces locally:

```bash
for name in conversation toolagent synthetic; do
  wget -O "dataset/${name}_trace.jsonl" \
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/${name}_trace.jsonl"
done
```

Prepare each trace with the same context limit. For example:

```bash
python tests/pd_transfer/prepare_mooncake_trace.py \
  dataset/synthetic_trace.jsonl \
  dataset/mooncake_synthetic_filtered.jsonl \
  --max-total-tokens 40960
```

JSONL output preserves timestamps, lengths, order, and `hash_ids`; the
neighboring statistics file records the context-limit filtering. The adapter
rejects malformed rows, non-monotonic timestamps, and invalid hashes. Pass
`--force` to replace an existing conversion. CSV output remains available only
for historical length-only experiments and cannot reproduce prefix sharing.

The following one-based, post-filtering window is a 1000-request Synthetic
sample whose infinite-capacity token hit rate (59.22%) closely matches the
complete filtered trace (59.18%):

```bash
python tests/pd_transfer/prepare_mooncake_trace.py \
  dataset/mooncake_synthetic_filtered.jsonl \
  dataset/mooncake_synthetic_sample_1000.jsonl \
  --max-total-tokens 39000 \
  --start-request 2468 \
  --num-requests 1000
```

Use conversation as the primary real workload, toolagent as the high-reuse
real workload, and synthetic as the public long-context stress workload. Set
`DATASET_PATH`, `WORKLOAD_NAME`, and `WORKLOAD_SLUG` separately for each run.

Copy both the filtered JSONL and the repository changes to the server, then
prepare the separate configuration:

```bash
cp tests/pd_transfer/long_context_config.example.env \
  tests/pd_transfer/long_context_config.env

bash tests/pd_transfer/run_pd_transfer_generalization_bench.sh \
  tests/pd_transfer/long_context_config.env

python tests/pd_transfer/compare_pd_generalization_perf.py \
  results/pd_transfer_long_context/<run-id>
```

Both `parse_pd_trace.py` and `plot_pd_transfer.py` also accept a bare run ID,
such as `20260729-031022`. From the current directory they recursively locate
the one matching result directory containing `run_manifest.json`; pass the
full path instead if more than one match exists.

For the Mooncake loader, `REQUEST_RATES` are recorded arrival-rate multipliers:
`1.0` replays native timestamps, `0.5` runs at half the recorded rate, and
`2.0` doubles it. The example's `0.1 0.2 0.4` values are only calibration
starting points. Prefix caching is explicitly enabled on both servers, output
lengths are enforced with `--ignore-eos`, and the benchmark's duplicate
first-prompt readiness request is disabled because the runner already checks
P, D, and proxy health independently.

Set `VARIANT_MODE=paired` (the default) for the OFF/ON comparison: odd
repetitions run OFF then ON, while even repetitions reverse that order. Set
`VARIANT_MODE=off` or `VARIANT_MODE=on` to run only one canonicalization
variant, which is useful for smoke tests or workload characterization. A
single-variant run is not compatible with the OFF/ON comparison aggregator.
Set `TRANSFER_DELAY_MS_LIST` to a space-separated list such as
`"0 50 100 200"` to run a transfer-latency sensitivity sweep. The runner
restarts isolated P/D processes for every delay, injects the delay only on the
Decode side after NIXL completion, and records it in the case ID, result path,
trace events, and benchmark metadata. Multi-delay sweeps require
`VARIANT_MODE=off` or `VARIANT_MODE=on` so the OFF/ON aggregator cannot mix
different delays. Even repetitions reverse the delay order. Use
`DIAGNOSTIC_TRANSFER_DELAY_MS_LIST`, for example `"0 200"`, to trace only
selected validation points; it defaults to the first performance delay.
Set `NUM_PROMPTS=0` together with `RUN_DIAGNOSTIC_TRACE=1` to skip the main
trace-disabled performance cases and run only the trace-enabled diagnostic
cases. Diagnostic cases repeat `REPETITIONS` times by default. Set
`DIAGNOSTIC_REPETITIONS` to use an independent repeat count; every repetition
starts fresh P/D processes and writes to its own `rep-N` directory. Odd
diagnostic repetitions use the configured rate, variant, and delay order,
while even repetitions reverse all three orders to reduce ordering bias.
For a diagnostic arrival-rate sweep, set `DIAGNOSTIC_REQUEST_RATES` to a
space-separated list such as `"0.5 1.0 2.0"`. It takes precedence over the
backward-compatible single-value `DIAGNOSTIC_REQUEST_RATE` setting.

The runner accepts `DATASET_LOADER=mooncake` in addition to the existing
BurstGPT path. Existing configs using `BURSTGPT_DATASET_PATH` remain supported
as a compatibility fallback.

This v0.11 branch has NIXL pull support only; push-mode comparison is not part
of this harness.

## Standalone pack/transfer/scatter feasibility benchmark

`nixl_pack_scatter_bench.py` compares the current direct region-by-block NIXL
READ path against bounded GPU gather, contiguous READ, and GPU scatter without
starting vLLM. It includes fused Triton and PyTorch reference kernels, four
synthetic mapping patterns, exact `runs_per_region` mappings for controlled
bytes/range cost-model sweeps, fixed-size chunking, CUDA-event timing,
randomized A/B ordering, and independent correctness checks.

See [NIXL_PACK_SCATTER_BENCH.md](NIXL_PACK_SCATTER_BENCH.md) for the design,
GPU-memory calculation, commands, output fields, and interpretation guidance.

## Visualization

After running the trace parser, `plot_pd_transfer.py` writes PNG figures plus
`plot_summary.csv` and `generalization_delay_summary.csv` under
`<run-id>/plots` by default:

- `01_sleep_sweep`: fixed input and concurrency, varying injected sleep
- `01_generalization_delay_sweep`: mixed-length generalization results grouped
  by phase, workload, canonicalization variant, and configured request rate;
  repeated measurements use the median with min--max error bars, and diagnostic
  plots are explicitly labeled as trace-enabled
- `02_kv_transfer_breakdown`: one pie per benchmark case, showing the
  mean-per-request connector time split across handshake, descriptor build,
  NIXL preparation/submission, observed completion, and other connector work
- `03_kv_transfer_ttft_share`: one pie per benchmark case, showing the mean
  KV load-to-connector-finish time as a share of mean time to first token
  (TTFT), so output-generation length does not affect the ratio
- `04_request_e2e_scatter`: one raw scatter plot per benchmark case; every
  point is one request's end-to-end latency, ordered by proxy arrival time
- `05_kv_transfer_scatter`: one raw scatter plot per benchmark case; every
  point is one Decode-side KV load-to-connector-finish latency; cold handshake
  requests are excluded by default

Both sleep-sweep chart types show mean TTFT, p99 TTFT, and request throughput.
The generalization path reads `injected_transfer_delay_ms` directly from each
`result.json`, so it supports both performance and diagnostic delay sweeps. The
pie charts require `pd_request_timeline_ms.csv`, so run
`parse_pd_trace.py` first. Use `--output-dir <path>` to place artifacts elsewhere.
By default, pie charts exclude requests with a cold NIXL handshake so they
represent steady-state transfer latency. Pass `--include-cold-handshake` to
include startup cost instead.

```bash
python tests/pd_transfer/plot_pd_transfer.py results/pd_transfer/<run-id>
```
