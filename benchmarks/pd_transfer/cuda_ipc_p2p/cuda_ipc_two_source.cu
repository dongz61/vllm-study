// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kRecordMagic = 0x56324c4c4d495043ULL;  // "V2LLMIPC"
constexpr uint32_t kRecordVersion = 1;

#define CUDA_CHECK(expr)                                                   \
  do {                                                                     \
    cudaError_t status = (expr);                                           \
    if (status != cudaSuccess) {                                           \
      throw std::runtime_error(std::string(#expr) + ": " +                \
                               cudaGetErrorString(status));                \
    }                                                                      \
  } while (false)

struct IpcRecord {
  uint64_t magic = kRecordMagic;
  uint32_t version = kRecordVersion;
  int32_t device = -1;
  uint64_t bytes = 0;
  uint8_t pattern = 0;
  uint8_t reserved[7]{};
  cudaIpcMemHandle_t handle{};
};

static_assert(std::is_trivially_copyable_v<IpcRecord>);

class CliArgs {
 public:
  CliArgs(int argc, char** argv, int start) {
    for (int i = start; i < argc; i += 2) {
      if (i + 1 >= argc || std::strncmp(argv[i], "--", 2) != 0) {
        throw std::invalid_argument("arguments must be --name value pairs");
      }
      values_.emplace(argv[i] + 2, argv[i + 1]);
    }
  }

  std::string get(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
      throw std::invalid_argument("missing --" + name);
    }
    return it->second;
  }

  std::string get(const std::string& name,
                  const std::string& default_value) const {
    auto it = values_.find(name);
    return it == values_.end() ? default_value : it->second;
  }

  int get_int(const std::string& name, int default_value) const {
    return std::stoi(get(name, std::to_string(default_value)));
  }

  uint64_t get_u64(const std::string& name, uint64_t default_value) const {
    return std::stoull(get(name, std::to_string(default_value)));
  }

 private:
  std::unordered_map<std::string, std::string> values_;
};

void write_record_atomically(const fs::path& path, const IpcRecord& record) {
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }
  fs::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot open " + temporary.string());
    }
    output.write(reinterpret_cast<const char*>(&record), sizeof(record));
    if (!output) {
      throw std::runtime_error("cannot write " + temporary.string());
    }
  }
  fs::rename(temporary, path);
}

void write_signal(const fs::path& path) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create signal " + path.string());
  }
  output << "done\n";
}

IpcRecord wait_for_record(const fs::path& path, int timeout_seconds) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    std::error_code error;
    if (fs::file_size(path, error) == sizeof(IpcRecord) && !error) {
      IpcRecord record;
      std::ifstream input(path, std::ios::binary);
      input.read(reinterpret_cast<char*>(&record), sizeof(record));
      if (input && record.magic == kRecordMagic &&
          record.version == kRecordVersion) {
        return record;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("timed out waiting for " + path.string());
}

void wait_for_signal(const fs::path& path, int timeout_seconds) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (fs::exists(path)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("timed out waiting for " + path.string());
}

__global__ void copy_two_sources_kernel(const uint4* source0,
                                        size_t source0_vectors,
                                        const uint4* source1,
                                        size_t source1_vectors,
                                        uint4* destination) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t count =
      source0_vectors > source1_vectors ? source0_vectors : source1_vectors;
  for (size_t index = start; index < count; index += stride) {
    if (index < source0_vectors) {
      destination[index] = source0[index];
    }
    if (index < source1_vectors) {
      destination[source0_vectors + index] = source1[index];
    }
  }
}

__global__ void copy_one_source_kernel(const uint4* source,
                                       size_t source_vectors,
                                       uint4* destination) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t index = start; index < source_vectors; index += stride) {
    destination[index] = source[index];
  }
}

__global__ void validate_two_sources_kernel(const uint4* destination,
                                            size_t source0_vectors,
                                            size_t source1_vectors,
                                            uint32_t expected0,
                                            uint32_t expected1,
                                            unsigned long long* errors) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t count =
      source0_vectors > source1_vectors ? source0_vectors : source1_vectors;
  for (size_t index = start; index < count; index += stride) {
    if (index < source0_vectors) {
      const uint4 value = destination[index];
      if (value.x != expected0 || value.y != expected0 ||
          value.z != expected0 || value.w != expected0) {
        atomicAdd(errors, 1ULL);
      }
    }
    if (index < source1_vectors) {
      const uint4 value = destination[source0_vectors + index];
      if (value.x != expected1 || value.y != expected1 ||
          value.z != expected1 || value.w != expected1) {
        atomicAdd(errors, 1ULL);
      }
    }
  }
}

void launch_copy_one(const void* source, uint64_t bytes, void* destination,
                     int blocks, cudaStream_t stream = nullptr) {
  constexpr int threads = 256;
  copy_one_source_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(source), bytes / sizeof(uint4),
      static_cast<uint4*>(destination));
  CUDA_CHECK(cudaGetLastError());
}

void launch_copy(const void* source0, uint64_t source0_bytes,
                 const void* source1, uint64_t source1_bytes, void* destination,
                 int blocks, cudaStream_t stream = nullptr) {
  constexpr int threads = 256;
  copy_two_sources_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(source0), source0_bytes / sizeof(uint4),
      static_cast<const uint4*>(source1), source1_bytes / sizeof(uint4),
      static_cast<uint4*>(destination));
  CUDA_CHECK(cudaGetLastError());
}

int run_producer(const CliArgs& args) {
  const int device = args.get_int("device", -1);
  const uint64_t bytes = args.get_u64("bytes", 256ULL << 20);
  const int pattern_value = args.get_int("pattern", 17);
  const int timeout_seconds = args.get_int("timeout-seconds", 300);
  const fs::path handle_path = args.get("handle");
  const fs::path done_path = args.get("done");

  if (device < 0 || pattern_value < 0 || pattern_value > 255 || bytes == 0 ||
      bytes % sizeof(uint4) != 0) {
    throw std::invalid_argument(
        "producer requires a valid device, byte pattern, and a positive "
        "16-byte-aligned allocation size");
  }

  CUDA_CHECK(cudaSetDevice(device));
  void* allocation = nullptr;
  CUDA_CHECK(cudaMalloc(&allocation, bytes));
  CUDA_CHECK(cudaMemset(allocation, pattern_value, bytes));
  CUDA_CHECK(cudaDeviceSynchronize());

  IpcRecord record;
  record.device = device;
  record.bytes = bytes;
  record.pattern = static_cast<uint8_t>(pattern_value);
  CUDA_CHECK(cudaIpcGetMemHandle(&record.handle, allocation));
  write_record_atomically(handle_path, record);

  std::cout << "producer ready: device=" << device << " bytes=" << bytes
            << " handle=" << handle_path << std::endl;
  wait_for_signal(done_path, timeout_seconds);
  CUDA_CHECK(cudaFree(allocation));
  std::cout << "producer released: device=" << device << std::endl;
  return 0;
}

int run_consumer(const CliArgs& args) {
  const int device = args.get_int("device", -1);
  const int warmup = args.get_int("warmup", 5);
  const int iterations = args.get_int("iterations", 20);
  const int timeout_seconds = args.get_int("timeout-seconds", 300);
  const fs::path handle0_path = args.get("handle0");
  const fs::path handle1_path = args.get("handle1");
  const fs::path done0_path = args.get("done0");
  const fs::path done1_path = args.get("done1");

  if (device < 0 || warmup < 0 || iterations <= 0) {
    throw std::invalid_argument("invalid consumer device or iteration count");
  }

  const IpcRecord record0 = wait_for_record(handle0_path, timeout_seconds);
  const IpcRecord record1 = wait_for_record(handle1_path, timeout_seconds);
  if (record0.bytes % sizeof(uint4) != 0 ||
      record1.bytes % sizeof(uint4) != 0) {
    throw std::invalid_argument("IPC allocation sizes must be 16-byte aligned");
  }

  CUDA_CHECK(cudaSetDevice(device));
  int can_access0 = 0;
  int can_access1 = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access0, device, record0.device));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access1, device, record1.device));
  if (!can_access0 || !can_access1) {
    throw std::runtime_error("consumer cannot peer-access both producer GPUs");
  }

  void* source0 = nullptr;
  void* source1 = nullptr;
  const auto open_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaIpcOpenMemHandle(
      &source0, record0.handle, cudaIpcMemLazyEnablePeerAccess));
  CUDA_CHECK(cudaIpcOpenMemHandle(
      &source1, record1.handle, cudaIpcMemLazyEnablePeerAccess));
  const auto open_end = std::chrono::steady_clock::now();

  const uint64_t total_bytes = record0.bytes + record1.bytes;
  void* destination = nullptr;
  CUDA_CHECK(cudaMalloc(&destination, total_bytes));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const int blocks = std::max(1, properties.multiProcessorCount * 8);

  for (int i = 0; i < warmup; ++i) {
    launch_copy_one(source0, record0.bytes, destination, blocks);
    launch_copy_one(source1, record1.bytes,
                    static_cast<uint4*>(destination) +
                        record0.bytes / sizeof(uint4),
                    blocks);
    launch_copy(source0, record0.bytes, source1, record1.bytes, destination,
                blocks);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_copy_one(source0, record0.bytes, destination, blocks);
    launch_copy_one(source1, record1.bytes,
                    static_cast<uint4*>(destination) +
                        record0.bytes / sizeof(uint4),
                    blocks);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float sequential_elapsed_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&sequential_elapsed_ms, start, stop));
  const double sequential_kernel_ms = sequential_elapsed_ms / iterations;

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_copy(source0, record0.bytes, source1, record1.bytes, destination,
                blocks);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  const double kernel_ms = elapsed_ms / iterations;

  unsigned long long* device_errors = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_errors),
                        sizeof(*device_errors)));
  CUDA_CHECK(cudaMemset(device_errors, 0, sizeof(*device_errors)));
  const uint32_t expected0 = 0x01010101U * record0.pattern;
  const uint32_t expected1 = 0x01010101U * record1.pattern;
  validate_two_sources_kernel<<<blocks, 256>>>(
      static_cast<const uint4*>(destination), record0.bytes / sizeof(uint4),
      record1.bytes / sizeof(uint4), expected0, expected1, device_errors);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long validation_errors = 0;
  CUDA_CHECK(cudaMemcpy(&validation_errors, device_errors,
                        sizeof(validation_errors), cudaMemcpyDeviceToHost));

  const double open_ms =
      std::chrono::duration<double, std::milli>(open_end - open_start).count();
  const double aggregate_remote_read_gbps =
      static_cast<double>(total_bytes) / kernel_ms / 1.0e6;
  const double fused_speedup = sequential_kernel_ms / kernel_ms;

  std::cout << std::fixed << std::setprecision(3)
            << "{\"consumer_device\":" << device
            << ",\"producer0_device\":" << record0.device
            << ",\"producer1_device\":" << record1.device
            << ",\"peer_access0\":" << (can_access0 ? "true" : "false")
            << ",\"peer_access1\":" << (can_access1 ? "true" : "false")
            << ",\"bytes_total\":" << total_bytes
            << ",\"ipc_open_ms\":" << open_ms
            << ",\"kernel_ms\":" << kernel_ms
            << ",\"sequential_kernel_ms\":" << sequential_kernel_ms
            << ",\"fused_speedup\":" << fused_speedup
            << ",\"aggregate_remote_read_gbps\":"
            << aggregate_remote_read_gbps
            << ",\"validation_errors\":" << validation_errors << "}"
            << std::endl;

  CUDA_CHECK(cudaFree(device_errors));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(destination));
  CUDA_CHECK(cudaIpcCloseMemHandle(source0));
  CUDA_CHECK(cudaIpcCloseMemHandle(source1));
  write_signal(done0_path);
  write_signal(done1_path);
  return validation_errors == 0 ? 0 : 2;
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " producer --device N --bytes N --pattern N --handle PATH --done PATH "
         "[--timeout-seconds N]\n"
      << "  " << program
      << " consumer --device N --handle0 PATH --handle1 PATH --done0 PATH "
         "--done1 PATH [--warmup N] [--iterations N] [--timeout-seconds N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }
  try {
    const std::string role = argv[1];
    const CliArgs args(argc, argv, 2);
    if (role == "producer") {
      return run_producer(args);
    }
    if (role == "consumer") {
      return run_consumer(args);
    }
    print_usage(argv[0]);
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}
