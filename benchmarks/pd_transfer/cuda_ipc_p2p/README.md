# Two-source CUDA IPC P2P prototype

This is a standalone proof of concept for the P2-D1 KV transfer path:

1. Two producer processes allocate buffers on separate P GPUs.
2. Each producer exports its allocation with `cudaIpcGetMemHandle` and keeps
   the allocation alive.
3. One consumer process on the D GPU opens both handles with
   `cudaIpcOpenMemHandle`.
4. One CUDA kernel reads both remote pointers and concatenates them into a
   local D-GPU buffer.
5. The runner compares that fused kernel with two sequential single-source
   kernel launches.
6. A validation kernel checks the complete output, and CUDA events report the
   kernel time and aggregate remote-read bandwidth.

The prototype deliberately stays outside the vLLM/NIXL hot path. It answers
the first question before connector integration: can one D kernel correctly
read both P allocations through CUDA IPC, and how long does that operation
take on the target topology?

## Requirements

- Linux processes in the same host/PID-compatible CUDA environment.
- CUDA development toolkit with `nvcc`.
- Three GPUs with peer access from D to both P GPUs.
- All three processes must use the same CUDA device-ordinal mapping. For this
  standalone runner, expose all three GPUs in one `CUDA_VISIBLE_DEVICES` list.
- Producer allocations must remain alive until the consumer closes both IPC
  mappings. The runner enforces this lifetime.

Check the topology first:

```bash
nvidia-smi topo -m
```

The default architecture is `sm_80` for the current A100 server. Override
`CUDA_ARCH` for another GPU.

## Build and run

The repository may be mounted read-only in the experiment container, so the
default build and synchronization directories are under `/tmp`.

```bash
cd /workspace/vllm-study

BUILD_DIR=/tmp/vllm-cuda-ipc-p2p-build \
CUDA_ARCH=80 \
bash benchmarks/pd_transfer/cuda_ipc_p2p/build.sh

P0_DEVICE=0 \
P1_DEVICE=1 \
D_DEVICE=2 \
CUDA_VISIBLE_DEVICES=0,1,2 \
BYTES_PER_PRODUCER=268435456 \
WARMUP=5 \
ITERATIONS=20 \
bash benchmarks/pd_transfer/cuda_ipc_p2p/run.sh
```

The default transfers 256 MiB from each P GPU, or 512 MiB in total. For a
size close to the current trace's roughly 5.6 GB total request, try a size
that fits the available memory:

```bash
BYTES_PER_PRODUCER=2805202944 \
ITERATIONS=5 \
bash benchmarks/pd_transfer/cuda_ipc_p2p/run.sh
```

Successful output contains one JSON record similar to:

```json
{
  "consumer_device": 2,
  "producer0_device": 0,
  "producer1_device": 1,
  "peer_access0": true,
  "peer_access1": true,
  "bytes_total": 536870912,
  "ipc_open_ms": 1.234,
  "kernel_ms": 4.567,
  "sequential_kernel_ms": 7.891,
  "fused_speedup": 1.728,
  "aggregate_remote_read_gbps": 117.543,
  "validation_errors": 0
}
```

`aggregate_remote_read_gbps` counts source bytes read from both P GPUs. The
destination write traffic is not added to that value. `kernel_ms` is the
fused two-source kernel time; `sequential_kernel_ms` is the two-launch
baseline.

## What this prototype does not cover

- It copies two contiguous regions instead of gathering vLLM KV blocks.
- It uses standalone `cudaMalloc` allocations; exporting PyTorch/vLLM caching
  allocator storage will also require preserving its allocation-base offset.
- It does not synchronize against P-side KV production events.
- It does not manage per-request block lifetime or prefix-cache ownership.
- It does not replace NIXL notifications or failure handling.
- It does not yet support the real per-layer/per-KV-group layout.

The integration below is the follow-up to this standalone measurement: it
keeps IPC mappings open and passes request-specific block-index arrays to a
block-gather kernel.

## Experimental vLLM integration

The repository now includes that fixed-layout integration for this one target:
Qwen3-8B BF16, one host, P TP=2, D TP=1, FlashAttention HND KV cache. NIXL is
still used for handshakes, request leases, and completion notifications; the
two NIXL READ submissions are replaced by one CUDA IPC gather launch.

Rebuild `Dockerfile.pd-v24`, then use a benchmark config containing:

```bash
PREFILL_DEVICES="0,1"
PREFILL_TP_SIZE=2
DECODE_DEVICES="2"
DECODE_TP_SIZE=1
ENABLE_CUDA_IPC_GATHER=1
```

Run the normal benchmark command. Decode `transfer_rank_done_observed` records
have `backend="cuda_ipc_gather"`, `kernel_ms`, and
`submit_to_done_observed_ms`. The parsed CSV exposes
`max_cuda_ipc_kernel_ms`. Set the flag back to `0` for the NIXL data-path
baseline. This path intentionally rejects other model geometries, layouts,
host buffers, packed/cross-layer caches, and TP configurations.
