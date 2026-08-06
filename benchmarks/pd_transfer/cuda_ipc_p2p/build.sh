#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/vllm-cuda-ipc-p2p-build}"
CUDA_ARCH="${CUDA_ARCH:-80}"

mkdir -p -- "${BUILD_DIR}"

nvcc \
  -O3 \
  -std=c++17 \
  -lineinfo \
  "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
  "${SCRIPT_DIR}/cuda_ipc_two_source.cu" \
  -o "${BUILD_DIR}/cuda_ipc_two_source"

echo "Built ${BUILD_DIR}/cuda_ipc_two_source"
