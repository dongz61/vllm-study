# NIXL P2-D1 microbenchmark

This benchmark isolates the NIXL data path used by a single-host P2-D1 KV
handoff. Two producer processes register long-lived GPU buffers. One consumer
process imports their metadata and issues two sequential NIXL `READ` requests,
matching the shape of vLLM's decode-side pull path without running a model.

The primary sweep keeps bytes per producer fixed while varying descriptor size.
It runs two descriptor orders:

- `interleaved`: even chunks followed by odd chunks. This models the non-adjacent
  K/V-style access order and prevents most adjacent-descriptor merging.
- `contiguous`: monotonically increasing chunks. This is the merge-friendly
  control case.

Both modes transfer every byte, so their payload and validation are identical.
The benchmark reports descriptor-list setup separately from the timed request.
Each iteration records:

- `make_prepped_xfer` and synchronous `transfer()` time for each P rank;
- the first-to-last submit gap and completion wait;
- NIXL `postDuration`, `xferDuration`, derived data duration, descriptor count,
  bytes, and selected backend;
- total P2-D1 request time.

## Run

Run this in the same CUDA/NIXL container used for the vLLM experiments. All
three processes must see the same GPU ordinal mapping.

```bash
OUTPUT_DIR=/results/nixl-p2d1 \
bash /opt/pd-transfer/nixl_p2d1/run.sh
```

Useful overrides:

```bash
OUTPUT_DIR=/results/nixl-p2d1 \
BYTES_PER_PRODUCER=288MiB \
DESCRIPTOR_SIZES=16KiB,64KiB,256KiB,1MiB,all \
LAYOUTS="interleaved contiguous" \
WARMUP=5 ITERATIONS=20 \
bash benchmarks/pd_transfer/nixl_p2d1/run.sh
```

Defaults are P0/P1/D on CUDA devices 0/1/2 and the UCX backend with four
threads, matching the current vLLM NIXL configuration. The default 288 MiB per
producer yields 18,432 16-KiB descriptors, matching the representative request
in `20260806-102719`. Set
`SKIP_DESC_MERGE=1` only as an additional mechanism experiment; vLLM normally
allows NIXL to merge descriptors.

The result directory contains `iterations.jsonl`, `summary.json`, and logs for
all three processes. A successful case also performs full byte-wise validation
of both destination halves after its measured iterations.

## Interpretation

Compare cases at equal `bytes_per_producer`:

- If `postDuration`, `transfer_call_ms`, and total request time grow with the
  effective telemetry `desc_count`, descriptor processing/submission is the
  likely bottleneck.
- If `data_duration_us` (`xferDuration - postDuration`) grows while post time
  stays flat, the data movement itself is the likely bottleneck.
- If `contiguous` collapses to a small effective `desc_count` and is faster than
  `interleaved`, descriptor merging/layout is an important part of the issue.

This benchmark intentionally excludes scheduler delay, KV readiness, metadata
exchange, memory registration, and model execution from the timed region. It
therefore explains the transport mechanism, not the entire end-to-end gap.
