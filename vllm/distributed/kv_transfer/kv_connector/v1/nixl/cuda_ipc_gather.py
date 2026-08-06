# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental fixed-layout CUDA IPC gather path for P2-D1 Qwen3-8B."""

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    CudaIpcRegion,
)


def export_cuda_ipc_region(tensor: torch.Tensor) -> CudaIpcRegion:
    """Export a tensor's allocator allocation handle and data offset."""
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("CUDA IPC KV regions must be contiguous CUDA tensors")

    _, rebuild_args = reduce_tensor(tensor)
    if len(rebuild_args) < 10:
        raise RuntimeError("Unexpected torch CUDA IPC reduction tuple")

    tensor_offset_elements = int(rebuild_args[3])
    allocation_handle = bytes(rebuild_args[7])
    allocation_size_bytes = int(rebuild_args[8])
    storage_offset_bytes = int(rebuild_args[9])
    data_offset_bytes = (
        storage_offset_bytes + tensor_offset_elements * tensor.element_size()
    )
    region_size_bytes = tensor.numel() * tensor.element_size()
    if not allocation_handle:
        raise RuntimeError("PyTorch returned an empty CUDA IPC handle")
    if (
        data_offset_bytes < 0
        or data_offset_bytes + region_size_bytes > allocation_size_bytes
    ):
        raise RuntimeError("CUDA IPC tensor region exceeds its allocator allocation")

    return CudaIpcRegion(
        handle=allocation_handle,
        data_offset_bytes=data_offset_bytes,
        region_size_bytes=region_size_bytes,
        allocation_size_bytes=allocation_size_bytes,
    )


@dataclass
class CudaIpcGatherTransfer:
    start_event: torch.cuda.Event
    done_event: torch.cuda.Event
    block_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    remote_engine_id: str
    remote_request_id: str
    remote_ranks: tuple[int, int]
    submit_ns: int
    launch_host_ms: float
    num_blocks: int
    bytes_transferred: int

    def is_done(self) -> bool:
        return self.done_event.query()

    def kernel_ms(self) -> float:
        return float(self.start_event.elapsed_time(self.done_event))


class CudaIpcGatherManager:
    """Own persistent CUDA IPC mappings and launch the fixed gather kernel."""

    def __init__(self, device_id: int, local_bases: list[int]) -> None:
        library_path = os.getenv("VLLM_PD_CUDA_IPC_SO")
        if library_path is None:
            library_path = str(
                Path(__file__).with_name("libvllm_pd_cuda_ipc_gather.so")
            )
        self._library = ctypes.CDLL(library_path)
        self._configure_signatures()
        self._device_id = device_id
        self._stream = torch.cuda.Stream(device=device_id)
        self._opened_allocations: dict[bytes, int] = {}
        self._remote_region_ptrs: dict[tuple[str, int], list[int]] = {}
        self._remote_base_tensors: dict[tuple[str, int], torch.Tensor] = {}
        with torch.cuda.device(device_id), torch.cuda.stream(self._stream):
            self._local_base_tensor = torch.tensor(
                local_bases, dtype=torch.int64, device="cuda"
            )

    def _configure_signatures(self) -> None:
        self._library.vllm_pd_ipc_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._library.vllm_pd_ipc_open.restype = ctypes.c_int
        self._library.vllm_pd_ipc_close.argtypes = [ctypes.c_uint64]
        self._library.vllm_pd_ipc_close.restype = ctypes.c_int
        self._library.vllm_pd_gather_launch.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self._library.vllm_pd_gather_launch.restype = ctypes.c_int
        self._library.vllm_pd_cuda_ipc_last_error.argtypes = []
        self._library.vllm_pd_cuda_ipc_last_error.restype = ctypes.c_char_p

    def _check(self, status: int) -> None:
        if status == 0:
            return
        error = self._library.vllm_pd_cuda_ipc_last_error()
        message = error.decode("utf-8") if error else "unknown CUDA IPC error"
        raise RuntimeError(message)

    def register_remote_regions(
        self,
        engine_id: str,
        remote_rank: int,
        regions: list[CudaIpcRegion],
    ) -> None:
        key = (engine_id, remote_rank)
        if key in self._remote_base_tensors:
            return

        region_ptrs: list[int] = []
        for region in regions:
            allocation_base = self._opened_allocations.get(region.handle)
            if allocation_base is None:
                handle_buffer = (
                    ctypes.c_ubyte * len(region.handle)
                ).from_buffer_copy(region.handle)
                opened_base = ctypes.c_uint64()
                self._check(
                    self._library.vllm_pd_ipc_open(
                        ctypes.cast(handle_buffer, ctypes.c_void_p),
                        len(region.handle),
                        ctypes.byref(opened_base),
                    )
                )
                allocation_base = int(opened_base.value)
                self._opened_allocations[region.handle] = allocation_base
            region_ptrs.append(allocation_base + region.data_offset_bytes)

        with torch.cuda.device(self._device_id), torch.cuda.stream(self._stream):
            base_tensor = torch.tensor(
                region_ptrs, dtype=torch.int64, device="cuda"
            )
        self._remote_region_ptrs[key] = region_ptrs
        self._remote_base_tensors[key] = base_tensor

    def launch(
        self,
        engine_id: str,
        remote_request_id: str,
        remote_ranks: tuple[int, int],
        local_block_ids: list[int],
        remote0_block_ids: list[int],
        remote1_block_ids: list[int],
        remote_block_bytes: int,
        local_block_bytes: int,
    ) -> CudaIpcGatherTransfer:
        if not (
            len(local_block_ids)
            == len(remote0_block_ids)
            == len(remote1_block_ids)
        ):
            raise ValueError("local and remote CUDA IPC block lists must match")
        if not local_block_ids:
            raise ValueError("CUDA IPC gather cannot launch with no blocks")

        rank0, rank1 = remote_ranks
        source0_bases = self._remote_base_tensors[(engine_id, rank0)]
        source1_bases = self._remote_base_tensors[(engine_id, rank1)]
        if not (
            source0_bases.numel()
            == source1_bases.numel()
            == self._local_base_tensor.numel()
        ):
            raise ValueError("P0, P1 and D CUDA IPC region counts must match")

        launch_start_ns = time.perf_counter_ns()
        with torch.cuda.device(self._device_id), torch.cuda.stream(self._stream):
            local_blocks = torch.tensor(
                local_block_ids, dtype=torch.int64, device="cuda"
            )
            remote0_blocks = torch.tensor(
                remote0_block_ids, dtype=torch.int64, device="cuda"
            )
            remote1_blocks = torch.tensor(
                remote1_block_ids, dtype=torch.int64, device="cuda"
            )
            start_event = torch.cuda.Event(enable_timing=True)
            done_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self._stream)
            self._check(
                self._library.vllm_pd_gather_launch(
                    source0_bases.data_ptr(),
                    source1_bases.data_ptr(),
                    self._local_base_tensor.data_ptr(),
                    remote0_blocks.data_ptr(),
                    remote1_blocks.data_ptr(),
                    local_blocks.data_ptr(),
                    int(source0_bases.numel()),
                    len(local_block_ids),
                    remote_block_bytes,
                    local_block_bytes,
                    self._stream.cuda_stream,
                )
            )
            done_event.record(self._stream)
        launch_host_ms = (time.perf_counter_ns() - launch_start_ns) / 1_000_000
        submit_ns = time.perf_counter_ns()
        bytes_transferred = (
            len(local_block_ids) * int(source0_bases.numel()) * local_block_bytes
        )
        return CudaIpcGatherTransfer(
            start_event=start_event,
            done_event=done_event,
            block_tensors=(local_blocks, remote0_blocks, remote1_blocks),
            remote_engine_id=engine_id,
            remote_request_id=remote_request_id,
            remote_ranks=remote_ranks,
            submit_ns=submit_ns,
            launch_host_ms=launch_host_ms,
            num_blocks=len(local_block_ids),
            bytes_transferred=bytes_transferred,
        )

    def unregister_engine(self, engine_id: str) -> None:
        """Drop per-engine pointer tables; mappings stay valid until shutdown."""
        for key in [key for key in self._remote_base_tensors if key[0] == engine_id]:
            del self._remote_base_tensors[key]
            del self._remote_region_ptrs[key]

    def close(self) -> None:
        with torch.cuda.device(self._device_id):
            torch.cuda.synchronize(self._device_id)
        for allocation_base in self._opened_allocations.values():
            self._check(self._library.vllm_pd_ipc_close(allocation_base))
        self._opened_allocations.clear()
        self._remote_region_ptrs.clear()
        self._remote_base_tensors.clear()
