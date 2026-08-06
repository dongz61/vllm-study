#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/vllm-cuda-ipc-p2p-build}"
BIN="${CUDA_IPC_P2P_BIN:-${BUILD_DIR}/cuda_ipc_two_source}"

P0_DEVICE="${P0_DEVICE:-0}"
P1_DEVICE="${P1_DEVICE:-1}"
D_DEVICE="${D_DEVICE:-2}"
BYTES_PER_PRODUCER="${BYTES_PER_PRODUCER:-268435456}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
KEEP_RUN_DIR="${KEEP_RUN_DIR:-0}"

if [[ ! -x "${BIN}" ]]; then
  BUILD_DIR="${BUILD_DIR}" bash "${SCRIPT_DIR}/build.sh"
fi

owns_run_dir=0
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$(mktemp -d /tmp/vllm-cuda-ipc-two-source.XXXXXX)"
  owns_run_dir=1
else
  mkdir -p -- "${RUN_DIR}"
fi

handle0="${RUN_DIR}/producer0.handle"
handle1="${RUN_DIR}/producer1.handle"
done0="${RUN_DIR}/producer0.done"
done1="${RUN_DIR}/producer1.done"
producer0_log="${RUN_DIR}/producer0.log"
producer1_log="${RUN_DIR}/producer1.log"
producer0_pid=""
producer1_pid=""

cleanup() {
  touch -- "${done0}" "${done1}" 2>/dev/null || true
  if [[ -n "${producer0_pid}" ]]; then
    kill "${producer0_pid}" 2>/dev/null || true
  fi
  if [[ -n "${producer1_pid}" ]]; then
    kill "${producer1_pid}" 2>/dev/null || true
  fi
  if [[ "${owns_run_dir}" == "1" && "${KEEP_RUN_DIR}" != "1" ]]; then
    case "${RUN_DIR}" in
      /tmp/vllm-cuda-ipc-two-source.*)
        rm -rf -- "${RUN_DIR}"
        ;;
    esac
  fi
}
trap cleanup EXIT INT TERM

"${BIN}" producer \
  --device "${P0_DEVICE}" \
  --bytes "${BYTES_PER_PRODUCER}" \
  --pattern 17 \
  --handle "${handle0}" \
  --done "${done0}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  >"${producer0_log}" 2>&1 &
producer0_pid=$!

"${BIN}" producer \
  --device "${P1_DEVICE}" \
  --bytes "${BYTES_PER_PRODUCER}" \
  --pattern 34 \
  --handle "${handle1}" \
  --done "${done1}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  >"${producer1_log}" 2>&1 &
producer1_pid=$!

"${BIN}" consumer \
  --device "${D_DEVICE}" \
  --handle0 "${handle0}" \
  --handle1 "${handle1}" \
  --done0 "${done0}" \
  --done1 "${done1}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}"

wait "${producer0_pid}"
producer0_pid=""
wait "${producer1_pid}"
producer1_pid=""

cat "${producer0_log}"
cat "${producer1_log}"

trap - EXIT INT TERM
if [[ "${owns_run_dir}" == "1" && "${KEEP_RUN_DIR}" != "1" ]]; then
  case "${RUN_DIR}" in
    /tmp/vllm-cuda-ipc-two-source.*)
      rm -rf -- "${RUN_DIR}"
      ;;
  esac
fi
