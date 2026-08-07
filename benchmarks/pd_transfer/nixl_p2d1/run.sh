#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

P0_DEVICE="${P0_DEVICE:-0}"
P1_DEVICE="${P1_DEVICE:-1}"
D_DEVICE="${D_DEVICE:-2}"
BYTES_PER_PRODUCER="${BYTES_PER_PRODUCER:-288MiB}"
DESCRIPTOR_SIZES="${DESCRIPTOR_SIZES:-16KiB,64KiB,256KiB,1MiB,all}"
LAYOUTS="${LAYOUTS:-interleaved contiguous}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-10}"
NIXL_NUM_THREADS="${NIXL_NUM_THREADS:-4}"
NIXL_BACKEND="${NIXL_BACKEND:-UCX}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SKIP_DESC_MERGE="${SKIP_DESC_MERGE:-0}"
ENABLE_CUDA_IPC_GATHER="${ENABLE_CUDA_IPC_GATHER:-1}"
CUDA_IPC_REMOTE_BLOCK_BYTES="${CUDA_IPC_REMOTE_BLOCK_BYTES:-32KiB}"

if [[ "${ENABLE_CUDA_IPC_GATHER}" != "0" \
      && "${ENABLE_CUDA_IPC_GATHER}" != "1" ]]; then
  echo "ENABLE_CUDA_IPC_GATHER must be 0 or 1" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/nixl-p2d1-${timestamp}}"
RUN_DIR="$(mktemp -d /tmp/vllm-nixl-p2d1.XXXXXX)"
producer0_pid=""
producer1_pid=""

cleanup() {
  touch -- "${RUN_DIR}/stop" 2>/dev/null || true
  if [[ -n "${producer0_pid}" ]]; then
    kill "${producer0_pid}" 2>/dev/null || true
  fi
  if [[ -n "${producer1_pid}" ]]; then
    kill "${producer1_pid}" 2>/dev/null || true
  fi
  case "${RUN_DIR}" in
    /tmp/vllm-nixl-p2d1.*)
      rm -rf -- "${RUN_DIR}"
      ;;
  esac
}
trap cleanup EXIT INT TERM

mkdir -p -- "${OUTPUT_DIR}"

cuda_ipc_args=()
if [[ "${ENABLE_CUDA_IPC_GATHER}" == "1" ]]; then
  cuda_ipc_args+=(--cuda-ipc-gather)
fi

"${PYTHON}" "${SCRIPT_DIR}/nixl_p2d1_microbench.py" producer \
  --rendezvous "${RUN_DIR}" \
  --rank 0 \
  --device "${P0_DEVICE}" \
  --bytes "${BYTES_PER_PRODUCER}" \
  --pattern 17 \
  --backend "${NIXL_BACKEND}" \
  --num-threads "${NIXL_NUM_THREADS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  "${cuda_ipc_args[@]}" \
  >"${OUTPUT_DIR}/producer0.log" 2>&1 &
producer0_pid=$!

"${PYTHON}" "${SCRIPT_DIR}/nixl_p2d1_microbench.py" producer \
  --rendezvous "${RUN_DIR}" \
  --rank 1 \
  --device "${P1_DEVICE}" \
  --bytes "${BYTES_PER_PRODUCER}" \
  --pattern 34 \
  --backend "${NIXL_BACKEND}" \
  --num-threads "${NIXL_NUM_THREADS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  "${cuda_ipc_args[@]}" \
  >"${OUTPUT_DIR}/producer1.log" 2>&1 &
producer1_pid=$!

read -r -a layout_args <<< "${LAYOUTS}"
consumer_args=(
  consumer
  --rendezvous "${RUN_DIR}"
  --device "${D_DEVICE}"
  --bytes "${BYTES_PER_PRODUCER}"
  --backend "${NIXL_BACKEND}"
  --num-threads "${NIXL_NUM_THREADS}"
  --timeout-seconds "${TIMEOUT_SECONDS}"
  --output-dir "${OUTPUT_DIR}"
  --descriptor-sizes "${DESCRIPTOR_SIZES}"
  --layouts "${layout_args[@]}"
  --warmup "${WARMUP}"
  --iterations "${ITERATIONS}"
  --cuda-ipc-remote-block-bytes "${CUDA_IPC_REMOTE_BLOCK_BYTES}"
  "${cuda_ipc_args[@]}"
)
if [[ "${SKIP_DESC_MERGE}" == "1" ]]; then
  consumer_args+=(--skip-desc-merge)
fi

"${PYTHON}" "${SCRIPT_DIR}/nixl_p2d1_microbench.py" "${consumer_args[@]}" \
  | tee "${OUTPUT_DIR}/consumer.log"

wait "${producer0_pid}"
producer0_pid=""
wait "${producer1_pid}"
producer1_pid=""

echo "NIXL P2-D1 benchmark results: ${OUTPUT_DIR}"
