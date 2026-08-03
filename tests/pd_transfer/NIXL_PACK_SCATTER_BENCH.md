# NIXL pack/transfer/scatter microbenchmark

`nixl_pack_scatter_bench.py` compares two ways to pull the same paged KV data
between two GPUs:

- `direct`: submit one NIXL descriptor for every `(region, block)` pair;
- `packed`: gather source blocks into a bounded GPU staging buffer, pull one
  contiguous descriptor per chunk, then scatter into destination blocks.

This is a standalone feasibility experiment. It does not start vLLM, allocate
a model KV cache, modify the scheduler, or implement an adaptive selector.

## Design

The target process models the Prefill side and owns the source KV arena. The
initiator process models the Decode side and issues NIXL `READ` operations.
Each arena has `regions` independent logical regions. Every region contains
`physical_blocks` addressable blocks plus one unregistered gap block. The gap
prevents descriptors at adjacent region boundaries from becoming physically
contiguous in the synthetic allocation.

The direct descriptor list is region-major, matching the vLLM connector:

```text
region 0 block IDs, region 1 block IDs, ... region R-1 block IDs
```

The packed path preserves the same ordering in the staging buffer. For each
bounded chunk it performs:

```text
target GPU gather
  -> target acknowledges staging readiness
  -> initiator NIXL READ into local staging
  -> initiator GPU scatter
```

The target must acknowledge readiness because the current vLLM connector is a
Decode-initiated pull path. Reusing the target staging buffer is safe because
this first version processes chunks sequentially and waits for every READ to
finish before requesting the next pack.

The default 64 MiB staging size follows SGLang's Prefill-side staging default.
The Triton kernels use the same important structure as SGLang's implementation:
a device-resident region-pointer table and one fused launch spanning all
regions. They are adapted from token/head slicing to arbitrary block IDs.

References:

- [SGLang staging buffer and fused kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/common/staging_buffer.py)
- [SGLang staging design PR #19890](https://github.com/sgl-project/sglang/pull/19890)
- [SGLang NIXL staging integration PR #22536](https://github.com/sgl-project/sglang/pull/22536)

## Quick smoke test

Run this in the same CUDA/NIXL environment used for the descriptor-order
benchmark:

```bash
python tests/pd_transfer/nixl_pack_scatter_bench.py \
  --target-gpu 3 \
  --initiator-gpu 4 \
  --regions 72 \
  --physical-blocks 128 \
  --request-blocks 4,16 \
  --patterns forward,fragmented \
  --staging-mib 64 \
  --kernel triton \
  --warmup 1 \
  --repeats 3 \
  --nixl-label 0.6.0 \
  --output results/nixl_pack_scatter_smoke.jsonl
```

When a container exposes only the two selected GPUs, their visible indices are
normally `0` and `1`.

`--kernel auto` uses Triton when installed and otherwise falls back to
PyTorch. `--kernel torch` is useful for debugging correctness, but it should
not be used as the primary performance result because it is not the fused
production-style path.

## Controlled cost-model sweep

Use `--runs-per-region` instead of `--patterns` to construct mappings with an
exact paired forward-run count. Runs are ascending on both sides and separated
by one unused physical block, so address order is not an additional variable.
For a workload with `B` requested blocks and `K` runs per region:

```text
total_bytes = regions * B * block_bytes
direct_descriptor_count = regions * B
estimated_direct_backend_ranges = regions * K
```

The benchmark skips a requested `K` for cells where `K > B`. This produces a
triangular grid from one global list:

```bash
UCX_MODULE_DIR=/path/to/nixl/ucx \
UCX_MODULES=all \
UCX_TLS=tcp,cuda \
UCX_LOG_LEVEL=warn \
UCX_PROTO_INFO=n \
python tests/pd_transfer/nixl_pack_scatter_bench.py \
  --target-gpu 3 \
  --initiator-gpu 4 \
  --regions 72 \
  --physical-blocks 512 \
  --request-blocks 1,2,4,8,16 \
  --runs-per-region 1,2,4,8,16 \
  --block-bytes 32768 \
  --staging-mib 64 \
  --kernel triton \
  --warmup 5 \
  --repeats 30 \
  --nixl-label 0.6.0 \
  --output results/pack_scatter/cost-model-core-0.6.0.jsonl
```

This command creates 15 workloads and 900 measured samples across the direct
and packed paths. A logical block is 2.25 MiB with these settings, so the
largest 16-block request is 36 MiB and every packed workload uses exactly one
chunk. Holding `B` fixed isolates range count; holding `K` fixed varies request
bytes while preserving vLLM's real relationship between physical blocks and
submitted descriptors.

`--patterns` and `--runs-per-region` are mutually exclusive. Existing pattern
sweeps remain available for exploratory and external-validation experiments.

## Fragmentation sweep

After the smoke test passes, use more block counts and all mapping patterns:

```bash
python tests/pd_transfer/nixl_pack_scatter_bench.py \
  --target-gpu 3 \
  --initiator-gpu 4 \
  --regions 72 \
  --physical-blocks 512 \
  --request-blocks 1,2,4,8,16,32,64,128,256 \
  --patterns forward,reverse,mixed,fragmented \
  --mixed-run-length 4 \
  --block-bytes 32768 \
  --staging-mib 64 \
  --kernel triton \
  --warmup 5 \
  --repeats 30 \
  --nixl-label 0.6.0 \
  --output results/nixl_pack_scatter_sweep.jsonl
```

The source and destination arenas each require approximately:

```text
regions * (physical_blocks + 1) * block_bytes
```

For `72 * 513 * 32 KiB`, that is about 1.13 GiB per GPU, excluding the
64 MiB staging buffer and runtime allocations.

## Mapping patterns

- `forward`: both block sequences are ascending and jointly contiguous;
- `reverse`: both sequences are descending, stressing a forward-only merger;
- `mixed`: short ascending runs appear in shuffled order;
- `fragmented`: local and remote IDs are independent random permutations.

Every JSONL sample records the exact `forward_range_count` and the corresponding
`estimated_direct_backend_ranges = regions * forward_range_count`. Therefore,
the analysis should use the measured range count rather than assuming that a
pattern name always produces an exact number of fragments.

Controlled mappings additionally record `mapping = "controlled"` and the
numeric `runs_per_region`. Summaries group different run counts separately.

## Measurements

Each measured sample records:

- source and destination index construction;
- target-control wait time;
- GPU `pack_gpu_ns` and `scatter_gpu_ns`, measured with CUDA events;
- NIXL prepare, submit, polling, and transfer-completion time;
- `data_path_ns`, the serial sum of pack, transfer, and scatter;
- `wall_ns`, including the benchmark's control exchange and CPU work;
- effective bandwidth and chunk count.

The primary comparison for this sequential first version is:

```text
direct.transfer_total_ns
vs.
packed.data_path_ns
```

`wall_ns` is also reported, but the multiprocessing pipe is a benchmark control
channel rather than the final vLLM protocol. If a crossover appears only in
`data_path_ns` and disappears by a large margin in `wall_ns`, the next step is
to optimize or overlap the control path before integrating it.

Correctness is checked independently after timing for every workload and path.
The checker validates every byte of selected blocks and confirms that the first
byte of every unselected destination block retains its sentinel value.

## Current limitations

- Chunks are sequential; there is no ping-pong pipeline or ring allocator.
- Mapping inputs are synthetic; exact block-pair trace replay is not yet wired.
- The arena uses one allocation with a gap between logical regions rather than
  72 independent vLLM tensor allocations.
- The benchmark measures one request at a time and does not model staging
  contention between concurrent requests.
- It does not yet choose between direct and packed paths automatically.

These constraints are intentional: the benchmark first determines whether a
meaningful direct/packed crossover exists before production integration.
