# NIXL versus CUDA IPC P2-D1 microbenchmark

This benchmark compares the two data paths used by a single-host P2-D1 KV
handoff without running a model. Two producer processes create long-lived GPU
buffers. One consumer first reads the same buffers with the integrated CUDA IPC
gather path, then issues two sequential NIXL `READ` requests.

The CUDA IPC case calls vLLM's real `CudaIpcGatherManager` and the same
`cuda_ipc_gather.cu` shared library used by the connector integration. It
interprets each producer buffer as 36 Qwen3 layers with 32-KiB remote blocks,
uses one fused gather launch for P0 and P1, and keeps persistent IPC mapping
outside the per-request timed region.

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
- CUDA IPC host launch, kernel, completion-observation, and total request time;
- the descriptor count required by the equivalent 16-KiB K/V geometry.

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
ENABLE_CUDA_IPC_GATHER=1 \
CUDA_IPC_REMOTE_BLOCK_BYTES=32KiB \
WARMUP=5 ITERATIONS=20 \
bash benchmarks/pd_transfer/nixl_p2d1/run.sh
```

Defaults are P0/P1/D on CUDA devices 0/1/2 and the UCX backend with four
threads, matching the current vLLM NIXL configuration. The default 288 MiB per
producer yields 18,432 16-KiB descriptors, matching the representative request
in `20260806-102719`. Set
`SKIP_DESC_MERGE=1` only as an additional mechanism experiment; vLLM normally
allows NIXL to merge descriptors. Set `ENABLE_CUDA_IPC_GATHER=0` for a NIXL-only
run. When CUDA IPC is enabled, `BYTES_PER_PRODUCER` must be a multiple of
`36 * CUDA_IPC_REMOTE_BLOCK_BYTES`.

The result directory contains NIXL `iterations.jsonl`, CUDA IPC
`cuda_ipc_iterations.jsonl`, the combined `summary.json`, and logs for all three
processes. The summary's `comparison` rows directly report equal-payload NIXL
and CUDA IPC request time and speedup. Successful cases perform full byte-wise
validation after their measured iterations.

## Interpretation

Compare cases at equal `bytes_per_producer`:

- If `postDuration`, `transfer_call_ms`, and total request time grow with the
  effective telemetry `desc_count`, descriptor processing/submission is the
  likely bottleneck.
- If `data_duration_us` (`xferDuration - postDuration`) grows while post time
  stays flat, the data movement itself is the likely bottleneck.
- If `contiguous` collapses to a small effective `desc_count` and is faster than
  `interleaved`, descriptor merging/layout is an important part of the issue.
- Compare the CUDA IPC `interleaved` case with NIXL's `interleaved`, 16-KiB case.
  They transfer the same bytes from the same producer allocations and represent
  the same 36-layer K/V geometry. CUDA IPC should avoid the two descriptor-post
  loops, so its `request_total_ms` should remain near its kernel time.
- The `persistent_setup_ms` value is reported but excluded from request time,
  matching the integration where IPC handles are opened during handshake and
  reused across requests.

This benchmark intentionally excludes scheduler delay, KV readiness, metadata
exchange, memory registration, and model execution from the timed region. It
therefore explains the transport mechanism, not the entire end-to-end gap.
