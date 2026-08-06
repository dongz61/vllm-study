// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string last_error;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void gather_qwen3_p2d1_kernel(
    const uint64_t* source0_bases, const uint64_t* source1_bases,
    const uint64_t* destination_bases, const int64_t* remote0_block_ids,
    const int64_t* remote1_block_ids, const int64_t* local_block_ids,
    int num_layers, int num_request_blocks, uint64_t remote_block_bytes,
    uint64_t local_block_bytes) {
  const uint64_t task = blockIdx.x;
  const uint64_t tasks_per_layer =
      static_cast<uint64_t>(num_request_blocks) * 2;
  const int layer = task / tasks_per_layer;
  const uint64_t remainder = task % tasks_per_layer;
  const int request_block = remainder / 2;
  const int kv_index = remainder % 2;

  const uint64_t remote_half_bytes = remote_block_bytes / 2;
  const uint64_t local_half_bytes = local_block_bytes / 2;
  const uint64_t source0 =
      source0_bases[layer] + remote0_block_ids[request_block] *
                                   remote_block_bytes +
      kv_index * remote_half_bytes;
  const uint64_t source1 =
      source1_bases[layer] + remote1_block_ids[request_block] *
                                   remote_block_bytes +
      kv_index * remote_half_bytes;
  const uint64_t destination =
      destination_bases[layer] +
      local_block_ids[request_block] * local_block_bytes +
      kv_index * local_half_bytes;

  const uint4* source0_vectors =
      reinterpret_cast<const uint4*>(source0);
  const uint4* source1_vectors =
      reinterpret_cast<const uint4*>(source1);
  uint4* destination0_vectors = reinterpret_cast<uint4*>(destination);
  uint4* destination1_vectors =
      reinterpret_cast<uint4*>(destination + remote_half_bytes);
  const uint64_t vectors = remote_half_bytes / sizeof(uint4);
  for (uint64_t index = threadIdx.x; index < vectors;
       index += blockDim.x) {
    destination0_vectors[index] = source0_vectors[index];
    destination1_vectors[index] = source1_vectors[index];
  }
}

template <typename Function>
int protect(Function&& function) {
  try {
    function();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  }
}

}  // namespace

extern "C" int vllm_pd_ipc_open(const void* handle_bytes, size_t handle_size,
                                uint64_t* opened_base) {
  return protect([&] {
    if (handle_size != sizeof(cudaIpcMemHandle_t)) {
      throw std::invalid_argument("unexpected CUDA IPC handle size");
    }
    cudaIpcMemHandle_t handle{};
    std::memcpy(&handle, handle_bytes, sizeof(handle));
    void* base = nullptr;
    check_cuda(
        cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle");
    *opened_base = reinterpret_cast<uint64_t>(base);
  });
}

extern "C" int vllm_pd_ipc_close(uint64_t opened_base) {
  return protect([&] {
    check_cuda(cudaIpcCloseMemHandle(reinterpret_cast<void*>(opened_base)),
               "cudaIpcCloseMemHandle");
  });
}

extern "C" int vllm_pd_gather_launch(
    uint64_t source0_bases, uint64_t source1_bases,
    uint64_t destination_bases, uint64_t remote0_block_ids,
    uint64_t remote1_block_ids, uint64_t local_block_ids, int num_layers,
    int num_request_blocks, uint64_t remote_block_bytes,
    uint64_t local_block_bytes, uint64_t stream) {
  return protect([&] {
    if (num_layers != 36 || num_request_blocks <= 0 ||
        remote_block_bytes == 0 ||
        local_block_bytes != 2 * remote_block_bytes ||
        remote_block_bytes % (2 * sizeof(uint4)) != 0) {
      throw std::invalid_argument(
          "unsupported Qwen3 P2-D1 CUDA IPC gather geometry");
    }
    const uint64_t tasks =
        static_cast<uint64_t>(num_layers) * num_request_blocks * 2;
    gather_qwen3_p2d1_kernel<<<static_cast<unsigned int>(tasks), 256, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const uint64_t*>(source0_bases),
        reinterpret_cast<const uint64_t*>(source1_bases),
        reinterpret_cast<const uint64_t*>(destination_bases),
        reinterpret_cast<const int64_t*>(remote0_block_ids),
        reinterpret_cast<const int64_t*>(remote1_block_ids),
        reinterpret_cast<const int64_t*>(local_block_ids), num_layers,
        num_request_blocks, remote_block_bytes, local_block_bytes);
    check_cuda(cudaGetLastError(), "gather_qwen3_p2d1_kernel launch");
  });
}

extern "C" const char* vllm_pd_cuda_ipc_last_error() {
  return last_error.c_str();
}
