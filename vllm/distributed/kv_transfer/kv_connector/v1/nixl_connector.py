# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import msgspec
import numpy as np
import torch
import zmq

from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp, KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)
from vllm.distributed.kv_transfer.pd_trace import is_trace_enabled, trace_event
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.distributed.utils import divide
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform
from vllm.utils import make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str

GET_META_MSG = b"get_meta_msg"
_PACK_CONTROL_VERSION = 2
_PACK_COPY_TILE_BYTES = 4096
_NIXL_TRANSFER_MODES = frozenset(("direct", "packed", "auto"))

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None

try:
    from nixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": ("cuda", ),
    "tpu": ("cpu", ),
    "xpu": ("cpu", ),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


class NixlAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_lens: list[int]
    attn_backend_name: str
    kv_cache_layout: str
    packed_staging_base_addr: int = 0
    packed_staging_slot_bytes: int = 0
    packed_staging_slots: int = 0
    packed_blocks_per_slot: int = 0
    packed_block_bytes: int = 0
    packed_num_regions: int = 0


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int


@dataclass
class _PackedChunkState:
    key: str
    request_id: str
    chunk_id: int
    local_block_ids: list[int]
    remote_block_ids: list[int]
    local_slot: Optional[int] = None
    remote_slot: Optional[int] = None
    status: str = "queued"
    handle: Optional[int] = None
    pack_request_ns: Optional[int] = None
    scatter_start_event: Optional[torch.cuda.Event] = None
    scatter_event: Optional[torch.cuda.Event] = None
    retry_at: float = 0.0
    # Slot-contention diagnostics (see background 15.10 P1). created_ns marks
    # when the chunk was queued; local_slot_acquired_ns marks when it finally
    # obtained a D-side local slot. busy_count counts P-side "busy" replies
    # (no free source slot); retry_count counts reschedules from those.
    created_ns: Optional[int] = None
    local_slot_acquired_ns: Optional[int] = None
    busy_count: int = 0
    retry_count: int = 0


@dataclass
class _PackedSourceRequestState:
    total_chunks: int
    released_chunks: int = 0
    default_stream_ready: bool = False


@dataclass
class _PackedRequestState:
    request_id: str
    dst_engine_id: str
    remote_host: str
    remote_port: int
    remote_tp_size: int
    notif_id: bytes
    chunks: list[_PackedChunkState]
    read_chunks_done: int = 0
    scatter_chunks_done: int = 0
    notification_sent: bool = False


def _should_use_packed_path(mode: str, num_blocks: int, forward_ranges: int,
                            auto_range_threshold: int,
                            available_packed_slots: int) -> bool:
    """Select the experimental transfer path from exact request geometry."""
    if mode not in _NIXL_TRANSFER_MODES:
        raise ValueError(f"unsupported NIXL transfer mode: {mode}")
    if mode == "direct" or num_blocks <= 0:
        return False
    if mode == "packed":
        return True
    return (forward_ranges >= auto_range_threshold
            and available_packed_slots > 0)


_PACK_TRITON_KERNELS: Optional[tuple[Any, Any, Any]] = None


def _get_pack_triton_kernels() -> tuple[Any, Any, Any]:
    """Define pack/scatter kernels lazily for CPU-only imports and tests."""
    global _PACK_TRITON_KERNELS
    if _PACK_TRITON_KERNELS is not None:
        return _PACK_TRITON_KERNELS

    import triton
    import triton.language as tl

    @triton.jit
    def pack_regions_kernel(region_ptrs, block_ids, staging,
                            request_block_count, block_bytes,
                            COPY_TILE: tl.constexpr):
        row_id = tl.program_id(0)
        tile_id = tl.program_id(1)
        region_id = row_id // request_block_count
        request_block_id = row_id % request_block_count
        region_ptr = tl.load(region_ptrs + region_id).to(staging.dtype)
        physical_block_id = tl.load(block_ids + request_block_id)
        offsets = tile_id * COPY_TILE + tl.arange(0, COPY_TILE)
        mask = offsets < block_bytes
        source_offsets = (
            physical_block_id.to(tl.int64) * block_bytes.to(tl.int64) +
            offsets)
        destination_offsets = (row_id.to(tl.int64) * block_bytes.to(tl.int64) +
                               offsets)
        values = tl.load(region_ptr + source_offsets, mask=mask)
        tl.store(staging + destination_offsets, values, mask=mask)

    @triton.jit
    def scatter_regions_kernel(region_ptrs, block_ids, staging,
                               request_block_count, block_bytes,
                               COPY_TILE: tl.constexpr):
        row_id = tl.program_id(0)
        tile_id = tl.program_id(1)
        region_id = row_id // request_block_count
        request_block_id = row_id % request_block_count
        region_ptr = tl.load(region_ptrs + region_id).to(staging.dtype)
        offsets = tile_id * COPY_TILE + tl.arange(0, COPY_TILE)
        mask = offsets < block_bytes
        source_offsets = (row_id.to(tl.int64) * block_bytes.to(tl.int64) +
                          offsets)
        physical_block_id = tl.load(block_ids + request_block_id)
        destination_offsets = (
            physical_block_id.to(tl.int64) * block_bytes.to(tl.int64) +
            offsets)
        values = tl.load(staging + source_offsets, mask=mask)
        tl.store(region_ptr + destination_offsets, values, mask=mask)

    _PACK_TRITON_KERNELS = (triton, pack_regions_kernel,
                            scatter_regions_kernel)
    return _PACK_TRITON_KERNELS


def _launch_pack_regions(region_ptrs: torch.Tensor, block_ids: torch.Tensor,
                         staging: torch.Tensor, block_bytes: int) -> None:
    triton, pack_kernel, _ = _get_pack_triton_kernels()
    grid = (region_ptrs.numel() * block_ids.numel(),
            triton.cdiv(block_bytes, _PACK_COPY_TILE_BYTES))
    pack_kernel[grid](region_ptrs,
                      block_ids,
                      staging,
                      block_ids.numel(),
                      block_bytes,
                      COPY_TILE=_PACK_COPY_TILE_BYTES)


def _launch_scatter_regions(region_ptrs: torch.Tensor, block_ids: torch.Tensor,
                            staging: torch.Tensor, block_bytes: int) -> None:
    triton, _, scatter_kernel = _get_pack_triton_kernels()
    grid = (region_ptrs.numel() * block_ids.numel(),
            triton.cdiv(block_bytes, _PACK_COPY_TILE_BYTES))
    scatter_kernel[grid](region_ptrs,
                         block_ids,
                         staging,
                         block_ids.numel(),
                         block_bytes,
                         COPY_TILE=_PACK_COPY_TILE_BYTES)


@dataclass(frozen=True)
class _BlockPairStats:
    """Contiguity summary for corresponding local/remote KV blocks."""

    paired_forward_run_count: int
    paired_reverse_run_count: int
    paired_fragment_run_count: int
    forward_only_range_count: int
    reverse_only_range_count: int
    theoretical_merged_range_count: int
    reverse_canonicalized_range_count: int
    reordered_optimal_range_count: int
    additional_reorderable_edge_count: int
    generalized_reordered_block_count: int
    longest_paired_forward_run: int
    longest_paired_reverse_run: int
    local_first_block_id: Optional[int]
    local_last_block_id: Optional[int]
    local_min_block_id: Optional[int]
    local_max_block_id: Optional[int]
    remote_first_block_id: Optional[int]
    remote_last_block_id: Optional[int]
    remote_min_block_id: Optional[int]
    remote_max_block_id: Optional[int]


def _analyze_block_pairs(local_block_ids: list[int],
                         remote_block_ids: list[int]) -> _BlockPairStats:
    """Summarize maximal paired-contiguous runs without reordering blocks.

    A forward run advances both local and remote block IDs by ``+1``. A
    reverse run advances both by ``-1``. Every block that cannot be included
    in either kind of run is counted as a one-block fragment. The theoretical
    range count is the number of transfer ranges needed if both forward runs
    and reverse runs (after pair-preserving canonicalization) can be merged.
    """
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError("local and remote block ID counts must match")

    num_blocks = len(local_block_ids)
    if num_blocks == 0:
        return _BlockPairStats(
            paired_forward_run_count=0,
            paired_reverse_run_count=0,
            paired_fragment_run_count=0,
            forward_only_range_count=0,
            reverse_only_range_count=0,
            theoretical_merged_range_count=0,
            reverse_canonicalized_range_count=0,
            reordered_optimal_range_count=0,
            additional_reorderable_edge_count=0,
            generalized_reordered_block_count=0,
            longest_paired_forward_run=0,
            longest_paired_reverse_run=0,
            local_first_block_id=None,
            local_last_block_id=None,
            local_min_block_id=None,
            local_max_block_id=None,
            remote_first_block_id=None,
            remote_last_block_id=None,
            remote_min_block_id=None,
            remote_max_block_id=None,
        )

    # Each tuple is (direction, number of blocks), where direction is +1 for
    # paired-forward, -1 for paired-reverse, and 0 for a singleton fragment.
    runs: list[tuple[int, int]] = []
    run_start = 0
    run_direction: Optional[int] = None
    forward_only_range_count = 1
    reverse_only_range_count = 1

    for index in range(num_blocks - 1):
        local_delta = local_block_ids[index + 1] - local_block_ids[index]
        remote_delta = remote_block_ids[index + 1] - remote_block_ids[index]
        pair_direction = (local_delta if local_delta == remote_delta
                          and local_delta in (-1, 1) else 0)
        if pair_direction != 1:
            forward_only_range_count += 1
        if pair_direction != -1:
            reverse_only_range_count += 1

        if run_direction is None:
            if pair_direction == 0:
                runs.append((0, 1))
                run_start = index + 1
            else:
                run_direction = pair_direction
        elif pair_direction != run_direction:
            runs.append((run_direction, index - run_start + 1))
            run_start = index + 1
            run_direction = None

    runs.append((run_direction or 0, num_blocks - run_start))
    forward_lengths = [length for direction, length in runs if direction == 1]
    reverse_lengths = [length for direction, length in runs if direction == -1]

    # Measure the exact forward range count after applying the existing reverse
    # canonicalization. This can be smaller than ``len(runs)`` when reversing a
    # run also makes it contiguous with one or both neighboring runs.
    reverse_local, reverse_remote, _, _ = _canonicalize_paired_reverse_runs(
        local_block_ids, remote_block_ids)
    reverse_canonicalized_range_count = _count_forward_ranges(
        reverse_local, reverse_remote)

    # A pair-preserving reorder may move copy operations freely while keeping
    # every local destination mapped to the same remote source. With unique
    # local block IDs, sorting by local ID exposes every possible successor
    # pair ``(local + 1, remote + 1)``, so this is the minimum forward range
    # count attainable by reordering the submitted block pairs.
    reorder = sorted(range(num_blocks), key=local_block_ids.__getitem__)
    reordered_local = [local_block_ids[index] for index in reorder]
    reordered_remote = [remote_block_ids[index] for index in reorder]
    reordered_optimal_range_count = _count_forward_ranges(
        reordered_local, reordered_remote)
    generalized_reordered_block_count = sum(
        original_index != submitted_index
        for submitted_index, original_index in enumerate(reorder))
    additional_reorderable_edge_count = max(
        0, reverse_canonicalized_range_count - reordered_optimal_range_count)

    return _BlockPairStats(
        paired_forward_run_count=len(forward_lengths),
        paired_reverse_run_count=len(reverse_lengths),
        paired_fragment_run_count=sum(direction == 0 for direction, _ in runs),
        forward_only_range_count=forward_only_range_count,
        reverse_only_range_count=reverse_only_range_count,
        theoretical_merged_range_count=len(runs),
        reverse_canonicalized_range_count=(reverse_canonicalized_range_count),
        reordered_optimal_range_count=reordered_optimal_range_count,
        additional_reorderable_edge_count=(additional_reorderable_edge_count),
        generalized_reordered_block_count=(generalized_reordered_block_count),
        longest_paired_forward_run=max(forward_lengths, default=0),
        longest_paired_reverse_run=max(reverse_lengths, default=0),
        local_first_block_id=local_block_ids[0],
        local_last_block_id=local_block_ids[-1],
        local_min_block_id=min(local_block_ids),
        local_max_block_id=max(local_block_ids),
        remote_first_block_id=remote_block_ids[0],
        remote_last_block_id=remote_block_ids[-1],
        remote_min_block_id=min(remote_block_ids),
        remote_max_block_id=max(remote_block_ids),
    )


def _count_forward_ranges(local_block_ids: list[int],
                          remote_block_ids: list[int]) -> int:
    """Count ranges mergeable by a forward-only descriptor backend."""
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError("local and remote block ID counts must match")
    if not local_block_ids:
        return 0
    return 1 + sum(
        local_block_ids[index + 1] != local_block_ids[index] +
        1 or remote_block_ids[index + 1] != remote_block_ids[index] + 1
        for index in range(len(local_block_ids) - 1))


def _canonicalize_paired_reverse_runs(
    local_block_ids: list[int],
    remote_block_ids: list[int],
) -> tuple[list[int], list[int], int, int]:
    """Turn paired ``-1`` runs into paired ``+1`` runs.

    NIXL receives matching local and remote descriptor ID sequences. Reversing
    both sides of a paired-reverse run preserves every local-to-remote block
    mapping while presenting ascending contiguous descriptors to the transfer
    backend.

    Returns the submitted local and remote block lists, the number of reversed
    runs, and the number of blocks in those runs. The input lists are returned
    unchanged when no paired-reverse run exists.
    """
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError("local and remote block ID counts must match")

    reverse_runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    for index in range(len(local_block_ids) - 1):
        is_paired_reverse = (
            local_block_ids[index + 1] - local_block_ids[index] == -1
            and remote_block_ids[index + 1] - remote_block_ids[index] == -1)
        if is_paired_reverse:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            reverse_runs.append((run_start, index + 1))
            run_start = None

    if run_start is not None:
        reverse_runs.append((run_start, len(local_block_ids)))
    if not reverse_runs:
        return local_block_ids, remote_block_ids, 0, 0

    submitted_local_block_ids = list(local_block_ids)
    submitted_remote_block_ids = list(remote_block_ids)
    canonicalized_block_count = 0
    for start, end in reverse_runs:
        submitted_local_block_ids[start:end] = reversed(
            submitted_local_block_ids[start:end])
        submitted_remote_block_ids[start:end] = reversed(
            submitted_remote_block_ids[start:end])
        canonicalized_block_count += end - start

    return (submitted_local_block_ids, submitted_remote_block_ids,
            len(reverse_runs), canonicalized_block_count)


@dataclass
class _NixlTransferTraceState:
    """Low-overhead, request-local timestamps for PD transfer analysis."""

    remote_engine_id: str
    num_local_blocks: int
    num_remote_blocks: int
    kv_load_start_ns: int
    handshake_cached: Optional[bool] = None
    handshake_start_ns: Optional[int] = None
    handshake_end_ns: Optional[int] = None
    desc_build_start_ns: Optional[int] = None
    desc_build_end_ns: Optional[int] = None
    desc_build_total_ns: int = 0
    xfer_prepare_start_ns: Optional[int] = None
    xfer_prepare_end_ns: Optional[int] = None
    xfer_prepare_total_ns: int = 0
    xfer_submit_start_ns: Optional[int] = None
    xfer_submit_end_ns: Optional[int] = None
    xfer_submit_total_ns: int = 0
    first_poll_ns: Optional[int] = None
    done_observed_ns: Optional[int] = None
    connector_finished_ns: Optional[int] = None
    poll_rounds: int = 0
    proc_checks: int = 0
    num_handles: int = 0
    num_local_descs: int = 0
    num_remote_descs: int = 0
    total_bytes: int = 0
    block_pair_stats_total_ns: int = 0
    paired_forward_run_count: int = 0
    paired_reverse_run_count: int = 0
    paired_fragment_run_count: int = 0
    forward_only_range_count: int = 0
    reverse_only_range_count: int = 0
    theoretical_merged_range_count: int = 0
    reverse_canonicalized_range_count: int = 0
    reordered_optimal_range_count: int = 0
    additional_reorderable_edge_count: int = 0
    generalized_reordered_block_count: int = 0
    longest_paired_forward_run: int = 0
    longest_paired_reverse_run: int = 0
    local_first_block_id: Optional[int] = None
    local_last_block_id: Optional[int] = None
    local_min_block_id: Optional[int] = None
    local_max_block_id: Optional[int] = None
    remote_first_block_id: Optional[int] = None
    remote_last_block_id: Optional[int] = None
    remote_min_block_id: Optional[int] = None
    remote_max_block_id: Optional[int] = None
    reverse_block_pair_canonicalization_enabled: bool = False
    canonicalized_reverse_run_count: int = 0
    canonicalized_reverse_block_count: int = 0
    transfer_skipped: bool = False
    configured_transfer_mode: str = "direct"
    selected_transfer_path: str = "direct"
    selector_num_blocks: int = 0
    selector_forward_ranges: int = 0
    selector_available_packed_slots: int = 0
    auto_range_threshold: int = 64
    packed_chunk_count: int = 0
    packed_pack_control_total_ns: int = 0
    packed_pack_gpu_total_ns: int = 0
    packed_source_handler_total_ns: int = 0
    packed_source_handler_max_ns: int = 0
    packed_source_sync_wall_total_ns: int = 0
    packed_source_sync_wall_max_ns: int = 0
    packed_source_stream_wait_gpu_total_ns: int = 0
    packed_source_stream_wait_gpu_max_ns: int = 0
    packed_source_wait_count: int = 0
    packed_source_readiness_event_wait_count: int = 0
    packed_source_default_stream_fallback_count: int = 0
    packed_scatter_gpu_total_ns: int = 0
    # Slot-contention diagnostics (background 15.10 P1). Populated D-side so the
    # NIXL_PACKED_STAGING_SLOTS sweep can show slots -> contention -> wall.
    packed_local_slot_queue_wait_total_ns: int = 0
    packed_local_slot_queue_wait_max_ns: int = 0
    packed_local_slot_queue_wait_count: int = 0
    packed_source_busy_count: int = 0
    packed_chunk_retry_count: int = 0


class NixlConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        save_to_host: bool = False,
    ):
        # save and load are mutually exclusive
        assert load_remote_cache ^ save_to_host
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
        )
        if save_to_host:
            self.reqs_to_save[request_id] = _req
        if load_remote_cache:
            self.reqs_to_recv[request_id] = _req


class NixlConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[NixlConnectorScheduler] = \
                NixlConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker: Optional[NixlConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(
                vllm_config, self.engine_id)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once("Unable to detect current VLLM config. "
                                "Fallback to default kv cache layout.")
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once("NixlConnector setting KV cache "
                         "layout to HND for better xfer performance.")
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[Optional[int], bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        assert self.connector_worker is not None
        self.connector_worker.set_host_xfer_buffer_ops(copy_operation)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        assert self.connector_worker is not None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: Optional[dict[str,
                                Any]] = None) -> Optional[KVConnectorStats]:
        return NixlKVConnectorStats(data=data) if data is not None \
            else NixlKVConnectorStats()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """NixlConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        if self.connector_worker.use_host_buffer and \
           self.connector_worker.copy_blocks:
            self.connector_worker.save_kv_to_host(self._connector_metadata)

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)
        self.use_host_buffer = \
            vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            count = len(request.prompt_token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if not params:
            return

        if params.get("do_remote_decode"):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.

            # save all blocks
            block_ids = blocks.get_block_ids()[0]
            # TODO: skip the blocks that are already in the host xfer buffer.
            # Currently, the host xfer buffer block is 1-to-1 mapped to device
            # kv blocks, so host blocks won't be flushed as long as its device
            # block is not overwritten; and it will be safe to skip saving them
            # to host xfer buffer.
            if block_ids:
                self._reqs_need_save[request.request_id] = \
                    (request, block_ids)
        elif params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host",
                                             "remote_port")):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the D worker. We need to call
                    # send_notif in _read_blocks to free the memory on the P.
                    local_block_ids = (blocks.get_unhashed_block_ids()
                                       if num_external_tokens > 0 else [])
                    trace_event(
                        "pull_decode_kv_allocated",
                        request.request_id,
                        role="decode",
                        num_external_tokens=num_external_tokens,
                        num_local_blocks=len(local_block_ids),
                        num_remote_blocks=len(params["remote_block_ids"]),
                    )
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request, local_block_ids)

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=True,
                save_to_host=False,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=False,
                save_to_host=True,
            )

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_in_batch = set()
        self._reqs_need_send = {}

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            trace_event(
                "pull_prefill_finished",
                request.request_id,
                role="prefill",
                num_computed_tokens=request.num_computed_tokens,
                num_blocks=len(block_ids),
            )
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = time.perf_counter(
            ) + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size)


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._pd_transfer_sleep_ms = self._get_pd_transfer_sleep_ms()
        transfer_mode = vllm_config.kv_transfer_config.get_from_extra_config(
            "nixl_transfer_mode", "direct")
        if not isinstance(transfer_mode, str):
            raise ValueError("nixl_transfer_mode must be a string")
        transfer_mode = transfer_mode.lower()
        if transfer_mode not in _NIXL_TRANSFER_MODES:
            raise ValueError(
                "nixl_transfer_mode must be one of direct, packed, "
                "or auto")
        self._nixl_transfer_mode = transfer_mode

        def positive_int_config(name: str, default: int) -> int:
            value = vllm_config.kv_transfer_config.get_from_extra_config(
                name, default)
            if isinstance(value,
                          bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return value

        self._packed_staging_mib = positive_int_config(
            "nixl_packed_staging_mib", 64)
        self._packed_staging_slots = positive_int_config(
            "nixl_packed_staging_slots", 64)
        self._packed_auto_range_threshold = positive_int_config(
            "nixl_packed_auto_range_threshold", 64)
        logger.info(
            "NIXL transfer mode=%s, packed staging=%s MiB x %s slots, "
            "auto range threshold=%s",
            self._nixl_transfer_mode, self._packed_staging_mib,
            self._packed_staging_slots, self._packed_auto_range_threshold)
        canonicalize_reverse_block_pairs = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "canonicalize_reverse_block_pairs", False))
        if not isinstance(canonicalize_reverse_block_pairs, bool):
            raise ValueError(
                "canonicalize_reverse_block_pairs must be a boolean")
        self._canonicalize_reverse_block_pairs = (
            canonicalize_reverse_block_pairs)
        logger.info(
            "NIXL paired-reverse block canonicalization is %s", "enabled"
            if self._canonicalize_reverse_block_pairs else "disabled")

        self.nixl_backends = \
            vllm_config.kv_transfer_config.get_from_extra_config(
                "backends", ["UCX"])
        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        if nixl_agent_config is None:
            config = None
        else:
            config = nixl_agent_config(backends=self.nixl_backends) if len(
                non_ucx_backends) > 0 else nixl_agent_config(num_threads=8)

        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
        # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # NIXL handshake port.
        # NOTE(rob): Within a DP group, each DP rank gets its own
        # base port (which is sent in the KVTransferParams).
        # Each TP rank listens/queries on the base_port + tp_rank.
        self.side_channel_port: int = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = \
            vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[
                self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        self.use_host_buffer = self.kv_buffer_device == "cpu"
        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        self.nixl_memory_type = current_platform.get_nixl_memory_type()
        if self.nixl_memory_type is None:
            if self.kv_buffer_device == "cuda":
                self.nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                self.nixl_memory_type = "DRAM"
        if self.nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: Optional[CopyBlocksOp] = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # nixl_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # Experimental GPU pack/READ/scatter resources. They are initialized
        # after the KV cache layout is known in register_kv_caches().
        self._packed_available = False
        self._packed_staging: Optional[torch.Tensor] = None
        self._packed_region_ptrs: Optional[torch.Tensor] = None
        self._packed_block_bytes = 0
        self._packed_slot_bytes = 0
        self._packed_blocks_per_slot = 0
        self._packed_src_xfer_side_handle = 0
        self._packed_dst_xfer_side_handles: dict[EngineId, int] = {}
        self._packed_remote_blocks_per_slot: dict[EngineId, int] = {}
        self._packed_pack_stream: Optional[torch.cuda.Stream] = None
        self._packed_scatter_stream: Optional[torch.cuda.Stream] = None
        self._packed_source_free_slots: deque[int] = deque()
        self._packed_source_slots: dict[str, int] = {}
        self._packed_source_requests: dict[str, _PackedSourceRequestState] = {}
        # Request-specific KV readiness events. Recorded on the P-side default
        # (compute) stream the moment a request enters _reqs_to_send, i.e. right
        # after its Prefill KV writes were enqueued. The pack stream waits on the
        # matching event instead of blocking on the whole default stream, so it
        # no longer waits for unrelated later Prefill work (see background 15.10).
        # Written by the worker thread, read by the side-channel thread -> guard
        # with a lock.
        self._packed_source_ready_events: dict[str, torch.cuda.Event] = {}
        self._packed_source_ready_lock = threading.Lock()
        self._packed_source_device: Optional[torch.device] = None
        self._packed_local_free_slots: deque[int] = deque()
        self._packed_requests: dict[ReqId, _PackedRequestState] = {}
        self._packed_chunks_by_key: dict[str, _PackedChunkState] = {}
        self._packed_control_context: Optional[zmq.Context] = None
        self._packed_control_sockets: dict[EngineId, zmq.Socket] = {}
        self._packed_control_endpoints: dict[EngineId, tuple[str, int,
                                                             int]] = {}

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        # Transfers that are physically complete but intentionally not ready.
        # Map of request id -> (ready time, number of transfer handles).
        self._delayed_recving_transfers: dict[ReqId, tuple[float, int]] = {}
        # Detailed timing is collected only for explicit PD trace runs. Each
        # request emits one summary record after the connector reports it done.
        self._pd_trace_enabled = is_trace_enabled()
        self._transfer_trace_states: dict[ReqId, _NixlTransferTraceState] = {}
        self._transfer_trace_lock = threading.Lock()
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator")
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[Optional[int]] = []
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(self.model_config.get_head_size(),
                                   self.model_config.dtype,
                                   self.cache_config.cache_dtype,
                                   self.block_size,
                                   use_mla=self.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_flashinfer = attn_backend == _Backend.FLASHINFER
        self._use_pallas = attn_backend == _Backend.PALLAS
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.xfer_stats = NixlKVConnectorStats()

    @staticmethod
    def _nixl_handshake_listener(metadata: NixlAgentMetadata,
                                 ready_event: threading.Event,
                                 base_port: int,
                                 tp_rank: int,
                                 packed_control_handler: Optional[Any] = None):
        """Serve metadata handshakes and experimental pack control messages."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, base_port + tp_rank)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                frames = sock.recv_multipart()
                identity = frames[0]
                has_req_delimiter = len(frames) >= 3 and frames[-2] == b""
                msg = frames[-1]
                if msg == GET_META_MSG:
                    response = encoded_data
                else:
                    try:
                        request = decoder.decode(msg)
                        if (packed_control_handler is None
                                or not isinstance(request, dict)):
                            raise RuntimeError(
                                "packed transfer control is unavailable")
                        reply = packed_control_handler(request)
                        if reply is None:
                            continue
                        response = encoder.encode(reply)
                    except BaseException as exc:
                        logger.exception("NIXL packed control request failed")
                        response = encoder.encode({
                            "version": _PACK_CONTROL_VERSION,
                            "type": "error",
                            "error": str(exc),
                        })
                if has_req_delimiter:
                    sock.send_multipart((identity, b"", response))
                else:
                    sock.send_multipart((identity, response))

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        # Handshake only with the remote TP rank that current local rank will
        # pull from. With homogeneous TP it happens to be the same rank_i.
        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        p_remote_rank = self.tp_rank // tp_ratio
        path = make_zmq_path("tcp", host, port + p_remote_rank)
        logger.debug("Querying metadata on path: %s at remote rank %s", path,
                     p_remote_rank)

        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.debug("NIXL handshake: get metadata took: %s",
                         got_metadata_time - start_time)

            # Ensure engine id matches.
            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(f"Remote NIXL agent engine ID mismatch. "
                                   f"Expected {expected_engine_id},"
                                   f"received {metadata.engine_id}.")

            # Register Remote agent.
            remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,
                                                      remote_tp_size)
            setup_agent_time = time.perf_counter()
            logger.debug("NIXL handshake: add agent took: %s",
                         setup_agent_time - got_metadata_time)

        # Remote rank -> agent name.
        return {p_remote_rank: remote_agent_name}

    def initialize_host_xfer_buffer(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape,
                                                       dtype=kv_dtype,
                                                       device="cpu")
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _initialize_packed_staging(self) -> None:
        """Allocate and register a bounded GPU staging ring."""
        if self._nixl_transfer_mode == "direct":
            return

        unsupported_reason: Optional[str] = None
        if self.vllm_config.kv_transfer_config.kv_role == "kv_both":
            unsupported_reason = "the initial staging pool does not support kv_both"
        elif self.device_type != "cuda" or self.use_host_buffer:
            unsupported_reason = "only CUDA VRAM buffers are supported"
        elif self.use_mla or self._use_flashinfer or self._use_pallas:
            unsupported_reason = "MLA, FlashInfer, and Pallas are not supported"
        elif self.block_window_per_layer:
            unsupported_reason = "hybrid/local attention is not supported"
        elif not self.block_len_per_layer or len(set(
                self.block_len_per_layer)) != 1:
            unsupported_reason = "all registered regions must use one block size"
        elif len(self.kv_caches_base_addr[self.engine_id]) != self.num_regions:
            unsupported_reason = "registered region pointers do not match regions"

        if unsupported_reason is not None:
            logger.warning("NIXL packed path disabled: %s; using direct READ",
                           unsupported_reason)
            return

        # Import/define the kernels now so missing Triton dependencies are
        # reported at startup rather than on the first request.
        try:
            _get_pack_triton_kernels()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "NIXL packed path disabled because Triton is "
                "unavailable: %s", exc)
            return

        self._packed_block_bytes = self.block_len_per_layer[0]
        request_block_bytes = self.num_regions * self._packed_block_bytes
        configured_slot_bytes = self._packed_staging_mib * 1024 * 1024
        self._packed_blocks_per_slot = configured_slot_bytes // request_block_bytes
        if self._packed_blocks_per_slot == 0:
            logger.warning(
                "NIXL packed path disabled: %s MiB staging cannot hold one "
                "%s-byte logical request block", self._packed_staging_mib,
                request_block_bytes)
            return
        self._packed_slot_bytes = (self._packed_blocks_per_slot *
                                   request_block_bytes)

        first_cache_or_caches = next(iter(self.device_kv_caches.values()))
        first_cache = (first_cache_or_caches[0] if isinstance(
            first_cache_or_caches, (list, tuple)) else first_cache_or_caches)
        device = first_cache.device
        total_bytes = self._packed_staging_slots * self._packed_slot_bytes
        self._packed_staging = torch.empty(total_bytes,
                                           dtype=torch.uint8,
                                           device=device)
        self._packed_region_ptrs = torch.tensor(
            self.kv_caches_base_addr[self.engine_id],
            dtype=torch.int64,
            device=device)
        self._packed_pack_stream = torch.cuda.Stream(device=device)
        self._packed_scatter_stream = torch.cuda.Stream(device=device)
        self._packed_source_device = device

        registration = self.nixl_wrapper.get_reg_descs(
            [(self._packed_staging.data_ptr(), total_bytes, self.tp_rank, "")],
            self.nixl_memory_type)
        self.nixl_wrapper.register_memory(registration,
                                          backends=self.nixl_backends)
        self._registered_descs.append(registration)

        staging_data = []
        base_addr = self._packed_staging.data_ptr()
        for slot in range(self._packed_staging_slots):
            slot_addr = base_addr + slot * self._packed_slot_bytes
            for count in range(1, self._packed_blocks_per_slot + 1):
                staging_data.append(
                    (slot_addr, count * request_block_bytes, self.tp_rank))
        descs = self.nixl_wrapper.get_xfer_descs(staging_data,
                                                 self.nixl_memory_type)
        self._packed_src_xfer_side_handle = (self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs))
        self._packed_source_free_slots = deque(
            range(self._packed_staging_slots))
        self._packed_local_free_slots = deque(range(
            self._packed_staging_slots))
        self._packed_available = True
        logger.info(
            "NIXL packed staging ready: %s slots, %.2f MiB/slot, "
            "%s request blocks/slot, %s regions x %s bytes",
            self._packed_staging_slots,
            self._packed_slot_bytes / (1024 * 1024),
            self._packed_blocks_per_slot, self.num_regions,
            self._packed_block_bytes)

    def _record_packed_source_ready_event(self, req_id: str) -> None:
        """Record a KV-readiness event on the default stream for this request.

        Runs on the worker thread. Only meaningful when the packed path is
        available on this (source/Prefill) worker; otherwise it is a no-op.
        """
        if not self._packed_available:
            return
        device = self._packed_source_device
        if device is None:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.default_stream(device))
        with self._packed_source_ready_lock:
            # Replace any stale event for the same id (should not normally
            # happen, but keeps the map bounded and avoids leaks).
            self._packed_source_ready_events[req_id] = event

    def _pop_packed_source_ready_event(
            self, req_id: str) -> Optional[torch.cuda.Event]:
        """Fetch and remove the readiness event for a request, if present."""
        with self._packed_source_ready_lock:
            return self._packed_source_ready_events.pop(req_id, None)

    def _discard_packed_source_ready_event(self, req_id: str) -> None:
        """Drop a readiness event without using it (request done/expired)."""
        with self._packed_source_ready_lock:
            self._packed_source_ready_events.pop(req_id, None)

    def _handle_packed_control(
            self, request: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Run in the side-channel thread on the source/Prefill worker."""
        handler_start_ns = time.perf_counter_ns()
        if request.get("version") != _PACK_CONTROL_VERSION:
            raise ValueError("unsupported packed control protocol version")
        message_type = request.get("type")
        key = request.get("key")
        if not isinstance(key, str):
            raise ValueError("packed control message is missing its key")
        request_key = key.rsplit("/", 1)[0]

        if message_type == "release":
            slot = self._packed_source_slots.pop(key, None)
            if slot is not None:
                self._packed_source_free_slots.append(slot)
                source_request = self._packed_source_requests.get(request_key)
                if source_request is not None:
                    source_request.released_chunks += 1
                    if (source_request.released_chunks
                            >= source_request.total_chunks):
                        self._packed_source_requests.pop(request_key, None)
            return None
        if message_type != "pack":
            raise ValueError(f"unknown packed control message: {message_type}")
        if not self._packed_available:
            return {
                "version": _PACK_CONTROL_VERSION,
                "type": "unavailable",
                "key": key,
            }
        total_chunks = request.get("total_chunks")
        if not isinstance(total_chunks, int) or total_chunks <= 0:
            raise ValueError("packed control message has invalid total_chunks")
        source_request = self._packed_source_requests.get(request_key)
        if source_request is None:
            source_request = _PackedSourceRequestState(
                total_chunks=total_chunks)
            self._packed_source_requests[request_key] = source_request
        elif source_request.total_chunks != total_chunks:
            raise ValueError(
                "packed control total_chunks changed within request")
        if key in self._packed_source_slots:
            return {
                "version": _PACK_CONTROL_VERSION,
                "type": "ready",
                "key": key,
                "remote_slot": self._packed_source_slots[key],
                "block_count": len(request.get("block_ids", [])),
                "pack_gpu_ns": 0,
                "source_handler_ns": 0,
                "source_sync_wall_ns": 0,
                "source_stream_wait_gpu_ns": 0,
                "source_default_stream_wait_count": 0,
            }
        if not self._packed_source_free_slots:
            return {
                "version": _PACK_CONTROL_VERSION,
                "type": "busy",
                "key": key,
            }

        block_ids = request.get("block_ids")
        if (not isinstance(block_ids, list) or not block_ids
                or len(block_ids) > self._packed_blocks_per_slot
                or any(not isinstance(block_id, int) or block_id < 0
                       or block_id >= self.num_blocks
                       for block_id in block_ids)):
            raise ValueError("invalid packed source block IDs")

        slot = self._packed_source_free_slots.popleft()
        self._packed_source_slots[key] = slot
        wait_for_default_stream = not source_request.default_stream_ready
        # Only the first chunk of a request establishes the dependency on the
        # request's KV writes; prefer a request-specific readiness event so we
        # do not block on unrelated later Prefill work. Fall back to a full
        # default-stream wait if the event is missing (e.g. it was never
        # recorded, or a race dropped it) so correctness is never at risk.
        req_id = request.get("request_id")
        readiness_event = None
        used_readiness_event = 0
        if wait_for_default_stream and isinstance(req_id, str):
            readiness_event = self._pop_packed_source_ready_event(req_id)
        try:
            assert self._packed_staging is not None
            assert self._packed_region_ptrs is not None
            assert self._packed_pack_stream is not None
            staging = self._packed_staging.narrow(
                0, slot * self._packed_slot_bytes, self._packed_slot_bytes)
            device = self._packed_staging.device
            stream_wait_start_event = None
            stream_wait_end_event = None
            if self._pd_trace_enabled and wait_for_default_stream:
                stream_wait_start_event = torch.cuda.Event(enable_timing=True)
                stream_wait_end_event = torch.cuda.Event(enable_timing=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(device), torch.cuda.stream(
                    self._packed_pack_stream):
                if stream_wait_start_event is not None:
                    stream_wait_start_event.record(self._packed_pack_stream)
                if wait_for_default_stream:
                    if readiness_event is not None:
                        # Wait only for this request's KV writes to complete,
                        # not for every op currently queued on the default
                        # stream. Later chunks stay ordered on the pack stream.
                        self._packed_pack_stream.wait_event(readiness_event)
                        used_readiness_event = 1
                    else:
                        # Fallback: establish the dependency by waiting on the
                        # whole default stream. Correct but may block on
                        # unrelated Prefill work.
                        self._packed_pack_stream.wait_stream(
                            torch.cuda.default_stream(device))
                if stream_wait_end_event is not None:
                    stream_wait_end_event.record(self._packed_pack_stream)
                ids = torch.tensor(block_ids, dtype=torch.int64, device=device)
                start_event.record(self._packed_pack_stream)
                _launch_pack_regions(self._packed_region_ptrs, ids, staging,
                                     self._packed_block_bytes)
                end_event.record(self._packed_pack_stream)
            sync_start_ns = time.perf_counter_ns()
            end_event.synchronize()
            sync_end_ns = time.perf_counter_ns()
            pack_gpu_ns = int(start_event.elapsed_time(end_event) * 1_000_000)
            stream_wait_gpu_ns = 0
            if (stream_wait_start_event is not None
                    and stream_wait_end_event is not None):
                stream_wait_gpu_ns = int(
                    stream_wait_start_event.elapsed_time(stream_wait_end_event)
                    * 1_000_000)
            if wait_for_default_stream:
                source_request.default_stream_ready = True
        except BaseException:
            self._packed_source_slots.pop(key, None)
            self._packed_source_free_slots.appendleft(slot)
            raise

        handler_end_ns = time.perf_counter_ns()
        return {
            "version": _PACK_CONTROL_VERSION,
            "type": "ready",
            "key": key,
            "remote_slot": slot,
            "block_count": len(block_ids),
            "pack_gpu_ns": pack_gpu_ns,
            "source_handler_ns": handler_end_ns - handler_start_ns,
            "source_sync_wall_ns": sync_end_ns - sync_start_ns,
            "source_stream_wait_gpu_ns": stream_wait_gpu_ns,
            "source_default_stream_wait_count": int(wait_for_default_stream),
            "source_readiness_event_wait_count": used_readiness_event,
            "source_default_stream_fallback_count": int(
                wait_for_default_stream and used_readiness_event == 0),
        }

    def _get_packed_control_socket(self,
                                   state: _PackedRequestState) -> zmq.Socket:
        socket = self._packed_control_sockets.get(state.dst_engine_id)
        tp_ratio = self._tp_size[self.engine_id] // state.remote_tp_size
        remote_rank = self.tp_rank // tp_ratio
        endpoint = (state.remote_host, state.remote_port, remote_rank)
        if socket is not None:
            if self._packed_control_endpoints[state.dst_engine_id] != endpoint:
                raise RuntimeError(
                    "packed control endpoint changed for engine")
            return socket

        if self._packed_control_context is None:
            self._packed_control_context = zmq.Context(
            )  # type: ignore[attr-defined]
        path = make_zmq_path("tcp", state.remote_host,
                             state.remote_port + remote_rank)
        socket = self._packed_control_context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(path)
        self._packed_control_sockets[state.dst_engine_id] = socket
        self._packed_control_endpoints[state.dst_engine_id] = endpoint
        return socket

    def _send_packed_control(self, state: _PackedRequestState,
                             message: dict[str, Any]) -> None:
        message["version"] = _PACK_CONTROL_VERSION
        self._get_packed_control_socket(state).send(
            msgspec.msgpack.encode(message))

    def _background_nixl_handshake(self, req_id: str,
                                   remote_engine_id: EngineId, meta: ReqMeta):
        # Do NIXL handshake in background and add to _ready_requests when done.
        self._record_handshake_start(req_id)
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake, meta.remote_host, meta.remote_port,
                meta.tp_size, remote_engine_id)
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            fut.add_done_callback(done_callback)

        # TODO: handle failure state of future in the
        # callback, we want to fail the request in this case.
        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            self._record_handshake_end(req_id)
            self._ready_requests.put(entry)

        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}")
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}")

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s", self.use_mla, self.kv_buffer_device,
            self.use_host_buffer)

        caches_data = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are registered in the same region
        # to better exploit the memory layout (ie num_blocks is the first dim).
        split_k_and_v = not (self.use_mla or self._use_pallas
                             or self._use_flashinfer)
        tensor_size_bytes = None
        # Enable different block lengths for different layers when MLA is used.
        self.block_len_per_layer = list[int]()
        self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
        for layer_name, cache_or_caches in xfer_buffers.items():
            cache_list = cache_or_caches if split_k_and_v else [
                cache_or_caches
            ]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.numel() * cache.element_size()

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert cache.shape[0] == self.num_blocks, \
                    "All kv cache tensors must have the same number of blocks"

                self.block_len_per_layer.append(curr_tensor_size_bytes //
                                                self.num_blocks)
                self.slot_size_per_layer.append(self.block_len_per_layer[-1] //
                                                self.block_size)

                if not self.use_mla:
                    # Different kv cache shape is not supported by HeteroTP
                    assert tensor_size_bytes == curr_tensor_size_bytes, \
                        "All kv cache tensors must have the same size"
                caches_data.append(
                    (base_addr, curr_tensor_size_bytes, self.tp_rank, ""))

        logger.debug("Different block lengths collected: %s",
                     set(self.block_len_per_layer))
        assert len(self.block_len_per_layer) == len(seen_base_addresses)
        assert self.num_blocks != 0

        self.kv_caches_base_addr[self.engine_id] = seen_base_addresses
        self.num_regions = len(caches_data)
        self.num_layers = len(xfer_buffers.keys())

        descs = self.nixl_wrapper.get_reg_descs(caches_data,
                                                self.nixl_memory_type)
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        if self._use_flashinfer:
            for i in range(len(self.slot_size_per_layer)):
                assert self.slot_size_per_layer[i] % 2 == 0
                self.slot_size_per_layer[i] //= 2

            # NOTE (NickLucche) When FlashInfer is used, memory is registered
            # with joint KV for each block. This minimizes the overhead in
            # registerMem allowing faster descs queries. In order to be able to
            # split on kv_heads dim as required by heterogeneous TP, one must
            # be able to index K/V separately. Hence we double the number
            # of 'virtual' regions here and halve `block_len` below.
            self.num_regions *= 2

        # Register local/src descr for NIXL xfer.
        blocks_data = []
        for i, base_addr in enumerate(seen_base_addresses):
            kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            # NOTE With heter-TP, more blocks are prepared than what are
            # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
            # could create fewer, but then _get_block_descs_ids needs to
            # select agent_meta.num_blocks instead of self.num_blocks for
            # local descr, and that makes handling regular flow less clean.
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len_per_layer[i]
                addr = base_addr + block_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, self.tp_rank))

            if self._use_flashinfer:
                # Separate and interleave K/V regions to maintain the same
                # descs ordering. This is needed for selecting contiguous heads
                # when split across TP ranks.
                for block_id in range(self.num_blocks):
                    block_offset = block_id * self.block_len_per_layer[i]
                    addr = base_addr + block_offset
                    # Register addresses for V cache (K registered first).
                    v_addr = addr + kv_block_len
                    blocks_data.append((v_addr, kv_block_len, self.tp_rank))
        logger.debug("Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.tp_rank)

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs)

        # TODO(mgoin): Hybrid memory allocator is currently disabled for
        # models with local attention (Llama 4). Can remove this once enabled.
        if self.vllm_config.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig
            assert isinstance(self.vllm_config.model_config.hf_text_config,
                              Llama4TextConfig)
            llama4_config = self.vllm_config.model_config.hf_text_config
            no_rope_layers = llama4_config.no_rope_layers
            chunk_size = llama4_config.attention_chunk_size
            chunk_block_size = math.ceil(chunk_size / self.block_size)
            for layer_idx in range(self.num_layers):
                # no_rope_layers[layer_idx] == 0 means NoPE (global)
                # Any other value means RoPE (local chunked)
                is_local_attention = no_rope_layers[layer_idx] != 0
                block_window = chunk_block_size if is_local_attention else None
                self.block_window_per_layer.append(block_window)
            logger.debug("Llama 4 block window per layer mapping: %s",
                         self.block_window_per_layer)
            assert len(self.block_window_per_layer) == self.num_layers

        self._initialize_packed_staging()

        # After KV Caches registered, listen for new connections.
        metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            attn_backend_name=self.backend_name,
            kv_cache_layout=self.kv_cache_layout,
            packed_staging_base_addr=(
                self._packed_staging.data_ptr() if self._packed_available
                and self._packed_staging is not None else 0),
            packed_staging_slot_bytes=(self._packed_slot_bytes
                                       if self._packed_available else 0),
            packed_staging_slots=(self._packed_staging_slots
                                  if self._packed_available else 0),
            packed_blocks_per_slot=(self._packed_blocks_per_slot
                                    if self._packed_available else 0),
            packed_block_bytes=(self._packed_block_bytes
                                if self._packed_available else 0),
            packed_num_regions=(self.num_regions
                                if self._packed_available else 0))
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank,
                  self._handle_packed_control),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def add_remote_agent(self,
                         nixl_agent_meta: NixlAgentMetadata,
                         remote_tp_rank: int = 0,
                         remote_tp_size: int = 1) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i. 
        The latter, assuming D.world_size > P.world_size, requires that two or 
        more local TP worker share the xfer from a single TP worker.

        Here's an example (non-MLA case):

        rank_offset     p_remote_tp_rank
        (kv split no)    
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2                  
                                
        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]  
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio 
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.   
        
        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.
        """ # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size
        # TODO We may eventually want to skip enforcing the same attn backend.
        assert nixl_agent_meta.attn_backend_name == self.backend_name

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata)

        # Number of D TP workers reading from a single P TP worker. This is
        # 1 when P and D `--tensor-parallel-size` match.
        tp_ratio = divide(self._tp_size[self.engine_id],
                          self._tp_size[engine_id])
        assert tp_ratio > 0, "Decode TP cannot be smaller than prefill TP"
        assert not self._use_pallas or tp_ratio == 1, \
               "TPU (pallas_v1) DOES NOT support heterogeneous TP yet."

        # Handle tp_size>num_kv_heads: replicate KV cache.
        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1

        remote_block_len = nixl_agent_meta.block_lens[0]
        if self.use_mla or is_kv_replicated:
            # With replicated KV cache, only the number of blocks can differ.
            assert self.block_len_per_layer == nixl_agent_meta.block_lens, \
                "KV cache sizes must match between P and D when replicated"
            remote_block_size = remote_block_len // (
                self.slot_size_per_layer[0])
        else:
            # When MLA is not used, this is a list of the same block length
            for block_len in nixl_agent_meta.block_lens:
                assert block_len == remote_block_len, \
                    "All remote layers must have the same block size"
            remote_block_size = remote_block_len // (
                self.slot_size_per_layer[0] * tp_ratio)
            if self._use_flashinfer:
                # With flashinfer, KV are sent in the same message.
                remote_block_size //= 2
            if tp_ratio > 1:
                # Heterogeneous TP expects same kv_cache_layout.
                assert nixl_agent_meta.kv_cache_layout == self.kv_cache_layout
                if self.device_type == "xpu":
                    raise ValueError(
                        "Heterogeneous TP is not supported on XPU")

            assert remote_block_len == self.block_len_per_layer[0] * tp_ratio, (
                "Remote P worker KV layer cache must be of shape [2, N, "
                "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
            )

        assert self.block_size == remote_block_size, (
            "Remote P worker with different page/block size is not supported "
            f"{self.block_size=}, {remote_block_size=}")

        # Create dst descs and xfer side handles. TP workers have same #blocks.
        if engine_id in self.dst_num_blocks:
            assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
        else:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        blocks_data = []
        # With homogeneous TP, D pulls the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr

        assert len(nixl_agent_meta.kv_caches_base_addr) == len(
            self.block_len_per_layer)
        # Register all remote blocks, but only the corresponding kv heads.
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            rank_offset = self.tp_rank % tp_ratio * kv_block_len \
                if not (self.use_mla or is_kv_replicated) else 0
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_lens[i]
                # For each block, grab the heads chunk belonging to rank_i
                # of size remote_nheads // tp_ratio, which correspond to
                # self.block_len == remote_block_len//tp_ratio bytes.
                addr = base_addr + block_offset + rank_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, remote_tp_rank))

            if self._use_flashinfer:
                # With FlashInfer index V separately to allow head splitting.
                for block_id in range(nixl_agent_meta.num_blocks):
                    block_offset = block_id * nixl_agent_meta.block_lens[i]
                    addr = base_addr + block_offset + rank_offset
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    blocks_data.append((v_addr, kv_block_len, remote_tp_rank))

        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and "
            "local rank %s", len(blocks_data), engine_id, remote_tp_rank,
            self.tp_rank)

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        self.dst_xfer_side_handles[
            engine_id] = self.nixl_wrapper.prep_xfer_dlist(
                remote_agent_name, descs)

        # The packed path is intentionally enabled only when both peers expose
        # an identical homogeneous staging layout. Direct READ remains the
        # compatibility fallback for mixed versions or unsupported backends.
        if (self._packed_available and nixl_agent_meta.packed_staging_base_addr
                and nixl_agent_meta.packed_staging_slots > 0
                and nixl_agent_meta.packed_blocks_per_slot > 0
                and nixl_agent_meta.packed_num_regions == self.num_regions and
                nixl_agent_meta.packed_block_bytes == self._packed_block_bytes
                and tp_ratio == 1):
            request_block_bytes = self.num_regions * self._packed_block_bytes
            remote_staging_data = []
            for slot in range(nixl_agent_meta.packed_staging_slots):
                slot_addr = (nixl_agent_meta.packed_staging_base_addr +
                             slot * nixl_agent_meta.packed_staging_slot_bytes)
                for count in range(1,
                                   nixl_agent_meta.packed_blocks_per_slot + 1):
                    remote_staging_data.append(
                        (slot_addr, count * request_block_bytes,
                         remote_tp_rank))
            packed_descs = self.nixl_wrapper.get_xfer_descs(
                remote_staging_data, self.nixl_memory_type)
            self._packed_dst_xfer_side_handles[engine_id] = (
                self.nixl_wrapper.prep_xfer_dlist(remote_agent_name,
                                                  packed_descs))
            self._packed_remote_blocks_per_slot[engine_id] = (
                nixl_agent_meta.packed_blocks_per_slot)
        elif self._nixl_transfer_mode != "direct":
            logger.warning_once(
                "Remote NIXL engine %s does not expose a compatible packed "
                "staging pool; requests will use direct READ", engine_id)

        return remote_agent_name

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        local_block_ids = meta.local_block_ids
        self.copy_blocks(self.host_xfer_buffers, self.device_kv_caches,
                         local_block_ids, local_block_ids, "h2d")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ", req_id,
                ",".join(map(str, meta.local_block_ids)))

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        for req_id, meta in metadata.reqs_to_save.items():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "save_load_kv for request[%s] to host xfer buffer."
                    "local_block_ids: %s. ", req_id,
                    ",".join(map(str, meta.local_block_ids)))
            # blocking
            self.copy_blocks(self.device_kv_caches, self.host_xfer_buffers,
                             meta.local_block_ids, meta.local_block_ids, "d2h")

    def _start_transfer_trace(self, req_id: str, meta: ReqMeta) -> None:
        if not self._pd_trace_enabled:
            return
        now_ns = time.perf_counter_ns()
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is None:
                self._transfer_trace_states[req_id] = \
                    _NixlTransferTraceState(
                        remote_engine_id=meta.remote_engine_id,
                        num_local_blocks=0,
                        num_remote_blocks=0,
                        kv_load_start_ns=now_ns,
                        configured_transfer_mode=self._nixl_transfer_mode,
                        auto_range_threshold=(
                            self._packed_auto_range_threshold),
                        reverse_block_pair_canonicalization_enabled=(
                            self._canonicalize_reverse_block_pairs),
                    )

    def _record_handshake_start(self, req_id: str) -> None:
        if not self._pd_trace_enabled:
            return
        now_ns = time.perf_counter_ns()
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is not None:
                state.handshake_cached = False
                if state.handshake_start_ns is None:
                    state.handshake_start_ns = now_ns

    def _record_handshake_end(self, req_id: str) -> None:
        if not self._pd_trace_enabled:
            return
        now_ns = time.perf_counter_ns()
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is not None:
                state.handshake_end_ns = now_ns

    def _record_cached_handshake(self, req_id: str) -> None:
        if not self._pd_trace_enabled:
            return
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is not None:
                state.handshake_cached = True

    def _record_trace_phase(self, req_id: str, phase: str, start_ns: int,
                            end_ns: int) -> None:
        if not self._pd_trace_enabled:
            return
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is None:
                return
            start_field = f"{phase}_start_ns"
            end_field = f"{phase}_end_ns"
            total_field = f"{phase}_total_ns"
            if getattr(state, start_field) is None:
                setattr(state, start_field, start_ns)
            setattr(state, end_field, end_ns)
            setattr(state, total_field,
                    getattr(state, total_field) + end_ns - start_ns)

    def _record_transfer_shape(
            self,
            req_id: str,
            num_local_blocks: int,
            num_remote_blocks: int,
            num_local_descs: int,
            num_remote_descs: int,
            total_bytes: int,
            block_pair_stats: Optional[_BlockPairStats] = None,
            block_pair_stats_ns: int = 0,
            canonicalized_reverse_run_count: int = 0,
            canonicalized_reverse_block_count: int = 0,
            num_handles: int = 1) -> None:
        if not self._pd_trace_enabled:
            return
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is None:
                return
            state.num_local_blocks += num_local_blocks
            state.num_remote_blocks += num_remote_blocks
            state.num_local_descs += num_local_descs
            state.num_remote_descs += num_remote_descs
            state.total_bytes += total_bytes
            state.num_handles += num_handles
            state.canonicalized_reverse_run_count += (
                canonicalized_reverse_run_count)
            state.canonicalized_reverse_block_count += (
                canonicalized_reverse_block_count)
            if block_pair_stats is None:
                return
            state.block_pair_stats_total_ns += block_pair_stats_ns
            state.paired_forward_run_count += (
                block_pair_stats.paired_forward_run_count)
            state.paired_reverse_run_count += (
                block_pair_stats.paired_reverse_run_count)
            state.paired_fragment_run_count += (
                block_pair_stats.paired_fragment_run_count)
            state.forward_only_range_count += (
                block_pair_stats.forward_only_range_count)
            state.reverse_only_range_count += (
                block_pair_stats.reverse_only_range_count)
            state.theoretical_merged_range_count += (
                block_pair_stats.theoretical_merged_range_count)
            state.reverse_canonicalized_range_count += (
                block_pair_stats.reverse_canonicalized_range_count)
            state.reordered_optimal_range_count += (
                block_pair_stats.reordered_optimal_range_count)
            state.additional_reorderable_edge_count += (
                block_pair_stats.additional_reorderable_edge_count)
            state.generalized_reordered_block_count += (
                block_pair_stats.generalized_reordered_block_count)
            state.longest_paired_forward_run = max(
                state.longest_paired_forward_run,
                block_pair_stats.longest_paired_forward_run)
            state.longest_paired_reverse_run = max(
                state.longest_paired_reverse_run,
                block_pair_stats.longest_paired_reverse_run)

            if block_pair_stats.local_first_block_id is not None:
                if state.local_first_block_id is None:
                    state.local_first_block_id = (
                        block_pair_stats.local_first_block_id)
                    state.remote_first_block_id = (
                        block_pair_stats.remote_first_block_id)
                state.local_last_block_id = (
                    block_pair_stats.local_last_block_id)
                state.remote_last_block_id = (
                    block_pair_stats.remote_last_block_id)
                state.local_min_block_id = min(
                    value for value in (state.local_min_block_id,
                                        block_pair_stats.local_min_block_id)
                    if value is not None)
                state.local_max_block_id = max(
                    value for value in (state.local_max_block_id,
                                        block_pair_stats.local_max_block_id)
                    if value is not None)
                state.remote_min_block_id = min(
                    value for value in (state.remote_min_block_id,
                                        block_pair_stats.remote_min_block_id)
                    if value is not None)
                state.remote_max_block_id = max(
                    value for value in (state.remote_max_block_id,
                                        block_pair_stats.remote_max_block_id)
                    if value is not None)

    def _record_transfer_poll(self, req_id: str, proc_checks: int,
                              done: bool) -> None:
        if not self._pd_trace_enabled:
            return
        now_ns = time.perf_counter_ns()
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is None:
                return
            if state.first_poll_ns is None:
                state.first_poll_ns = now_ns
            state.poll_rounds += 1
            state.proc_checks += proc_checks
            if done:
                state.done_observed_ns = now_ns

    def _record_transfer_skipped(self, req_id: str) -> None:
        if not self._pd_trace_enabled:
            return
        now_ns = time.perf_counter_ns()
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.get(req_id)
            if state is not None:
                state.transfer_skipped = True
                state.done_observed_ns = now_ns

    def _get_transfer_nbytes(self, local_block_ids: list[int]) -> int:
        """Return bytes represented by the selected local descriptors."""
        if not self.block_window_per_layer:
            return len(local_block_ids) * sum(self.block_len_per_layer)

        total_bytes = 0
        for layer_idx, block_window in enumerate(self.block_window_per_layer):
            num_blocks = (len(local_block_ids) if block_window is None else
                          min(len(local_block_ids), block_window))
            if self._use_flashinfer:
                layer_bytes = self.block_len_per_layer[layer_idx]
            elif self.num_layers < self.num_regions:
                first_region = 2 * layer_idx
                layer_bytes = sum(
                    self.block_len_per_layer[first_region:first_region + 2])
            else:
                layer_bytes = self.block_len_per_layer[layer_idx]
            total_bytes += num_blocks * layer_bytes
        return total_bytes

    def _emit_transfer_trace(self, req_id: str) -> None:
        if not self._pd_trace_enabled:
            return
        now_ns = time.perf_counter_ns()
        with self._transfer_trace_lock:
            state = self._transfer_trace_states.pop(req_id, None)
            if state is None:
                return
            state.connector_finished_ns = now_ns

        def elapsed_ms(start_ns: Optional[int],
                       end_ns: Optional[int]) -> Optional[float]:
            if start_ns is None or end_ns is None:
                return None
            return round((end_ns - start_ns) / 1_000_000, 6)

        trace_event(
            "pull_transfer_profile",
            req_id,
            role="decode",
            tp_rank=self.tp_rank,
            remote_engine_id=state.remote_engine_id,
            configured_transfer_mode=state.configured_transfer_mode,
            selected_transfer_path=state.selected_transfer_path,
            selector_num_blocks=state.selector_num_blocks,
            selector_forward_ranges=state.selector_forward_ranges,
            selector_available_packed_slots=(
                state.selector_available_packed_slots),
            auto_range_threshold=state.auto_range_threshold,
            packed_chunk_count=state.packed_chunk_count,
            packed_pack_control_ms=round(
                state.packed_pack_control_total_ns / 1_000_000, 6),
            packed_pack_gpu_ms=round(
                state.packed_pack_gpu_total_ns / 1_000_000, 6),
            packed_source_handler_ms=round(
                state.packed_source_handler_total_ns / 1_000_000, 6),
            packed_source_handler_max_ms=round(
                state.packed_source_handler_max_ns / 1_000_000, 6),
            packed_source_sync_wall_ms=round(
                state.packed_source_sync_wall_total_ns / 1_000_000, 6),
            packed_source_sync_wall_max_ms=round(
                state.packed_source_sync_wall_max_ns / 1_000_000, 6),
            packed_source_stream_wait_gpu_ms=round(
                state.packed_source_stream_wait_gpu_total_ns / 1_000_000, 6),
            packed_source_stream_wait_gpu_max_ms=round(
                state.packed_source_stream_wait_gpu_max_ns / 1_000_000, 6),
            packed_source_default_stream_wait_count=(
                state.packed_source_wait_count),
            packed_source_readiness_event_wait_count=(
                state.packed_source_readiness_event_wait_count),
            packed_source_default_stream_fallback_count=(
                state.packed_source_default_stream_fallback_count),
            packed_pack_control_other_ms=round(
                max(
                    0, state.packed_pack_control_total_ns -
                    state.packed_source_handler_total_ns) / 1_000_000, 6),
            packed_scatter_gpu_ms=round(
                state.packed_scatter_gpu_total_ns / 1_000_000, 6),
            packed_local_slot_queue_wait_ms=round(
                state.packed_local_slot_queue_wait_total_ns / 1_000_000, 6),
            packed_local_slot_queue_wait_max_ms=round(
                state.packed_local_slot_queue_wait_max_ns / 1_000_000, 6),
            packed_local_slot_queue_wait_count=(
                state.packed_local_slot_queue_wait_count),
            packed_source_busy_count=state.packed_source_busy_count,
            packed_chunk_retry_count=state.packed_chunk_retry_count,
            handshake_cached=state.handshake_cached,
            num_local_blocks=state.num_local_blocks,
            num_remote_blocks=state.num_remote_blocks,
            num_local_descs=state.num_local_descs,
            num_remote_descs=state.num_remote_descs,
            num_handles=state.num_handles,
            total_bytes=state.total_bytes,
            block_pair_stats_ms=round(
                state.block_pair_stats_total_ns / 1_000_000, 6),
            paired_forward_run_count=state.paired_forward_run_count,
            paired_reverse_run_count=state.paired_reverse_run_count,
            paired_fragment_run_count=state.paired_fragment_run_count,
            forward_only_range_count=state.forward_only_range_count,
            reverse_only_range_count=state.reverse_only_range_count,
            theoretical_merged_range_count=(
                state.theoretical_merged_range_count),
            reverse_canonicalized_range_count=(
                state.reverse_canonicalized_range_count),
            reordered_optimal_range_count=(
                state.reordered_optimal_range_count),
            additional_reorderable_edge_count=(
                state.additional_reorderable_edge_count),
            generalized_reordered_block_count=(
                state.generalized_reordered_block_count),
            longest_paired_forward_run=state.longest_paired_forward_run,
            longest_paired_reverse_run=state.longest_paired_reverse_run,
            local_first_block_id=state.local_first_block_id,
            local_last_block_id=state.local_last_block_id,
            local_min_block_id=state.local_min_block_id,
            local_max_block_id=state.local_max_block_id,
            remote_first_block_id=state.remote_first_block_id,
            remote_last_block_id=state.remote_last_block_id,
            remote_min_block_id=state.remote_min_block_id,
            remote_max_block_id=state.remote_max_block_id,
            reverse_block_pair_canonicalization_enabled=(
                state.reverse_block_pair_canonicalization_enabled),
            canonicalized_reverse_run_count=(
                state.canonicalized_reverse_run_count),
            canonicalized_reverse_block_count=(
                state.canonicalized_reverse_block_count),
            transfer_skipped=state.transfer_skipped,
            poll_rounds=state.poll_rounds,
            proc_checks=state.proc_checks,
            injected_sleep_ms=self._pd_transfer_sleep_ms,
            kv_load_start_perf_ns=state.kv_load_start_ns,
            handshake_start_perf_ns=state.handshake_start_ns,
            handshake_end_perf_ns=state.handshake_end_ns,
            desc_build_start_perf_ns=state.desc_build_start_ns,
            desc_build_end_perf_ns=state.desc_build_end_ns,
            xfer_prepare_start_perf_ns=state.xfer_prepare_start_ns,
            xfer_prepare_end_perf_ns=state.xfer_prepare_end_ns,
            xfer_submit_start_perf_ns=state.xfer_submit_start_ns,
            xfer_submit_end_perf_ns=state.xfer_submit_end_ns,
            first_poll_perf_ns=state.first_poll_ns,
            xfer_done_observed_perf_ns=state.done_observed_ns,
            connector_finished_perf_ns=state.connector_finished_ns,
            handshake_wait_ms=elapsed_ms(state.handshake_start_ns,
                                         state.handshake_end_ns),
            desc_build_ms=round(state.desc_build_total_ns / 1_000_000, 6),
            xfer_prepare_ms=round(state.xfer_prepare_total_ns / 1_000_000, 6),
            xfer_submit_ms=round(state.xfer_submit_total_ns / 1_000_000, 6),
            kv_load_to_desc_build_ms=elapsed_ms(state.kv_load_start_ns,
                                                state.desc_build_start_ns),
            handshake_to_desc_build_ms=elapsed_ms(state.handshake_end_ns,
                                                  state.desc_build_start_ns),
            prepare_to_submit_gap_ms=elapsed_ms(state.xfer_prepare_end_ns,
                                                state.xfer_submit_start_ns),
            submit_to_done_observed_ms=elapsed_ms(state.xfer_submit_end_ns,
                                                  state.done_observed_ns),
            first_poll_delay_ms=elapsed_ms(state.xfer_submit_end_ns,
                                           state.first_poll_ns),
            done_to_connector_finished_ms=elapsed_ms(
                state.done_observed_ns, state.connector_finished_ns),
            kv_load_to_connector_finished_ms=elapsed_ms(
                state.kv_load_start_ns, state.connector_finished_ns),
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        done_recving.update(self._poll_packed_transfers())
        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving", self.tp_rank,
                len(done_sending), len(done_recving))

        if self.use_host_buffer:
            for req_id in done_recving:
                meta = self._recving_metadata.pop(req_id)
                assert meta, f"{req_id} not found in recving_metadata list"
                self.sync_recved_kv_to_device(req_id, meta)

        for req_id in done_recving:
            self._emit_transfer_trace(req_id)

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            # Sorted dict, oldest requests are put first so we can exit early.
            if now < expires:
                break
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d decode worker(s) within %d seconds.", req_id,
                count, envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT)
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)
            # Drop the readiness event for expired requests to avoid leaks.
            self._discard_packed_source_ready_event(req_id)

        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                req_id, tp_ratio = notif.decode("utf-8").rsplit(":", 1)
                if (req_id not in self._reqs_to_send
                        and req_id not in self._reqs_to_process):
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.", req_id)
                    continue

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if self.consumer_notification_counts_by_req[req_id] == int(
                        tp_ratio):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
                    # Drop any unused readiness event (e.g. direct path, or a
                    # request that finished before its first pack chunk).
                    self._discard_packed_source_ready_event(req_id)
        return notified_req_ids

    def _pop_done_transfers(
            self, transfers: dict[str, list[tuple[int, float]]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids = self._pop_delayed_recv_transfers()
        newly_done: list[tuple[str, int]] = []
        for req_id, handles in list(transfers.items()):
            in_progress = False
            proc_checks = 0
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    self.nixl_wrapper.release_xfer_handle(handle)
                    # TODO (NickLucche) Get from NIXL telemetry once integrated
                    self.xfer_stats.record_transfer()
                elif xfer_state == "PROC":
                    in_progress = True
                    proc_checks += 1
                    continue
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if not in_progress:
                newly_done.append((req_id, len(handles)))
                del transfers[req_id]
            if self._pd_trace_enabled:
                self._record_transfer_poll(req_id,
                                           proc_checks,
                                           done=not in_progress)

        if not newly_done:
            return done_req_ids

        if self._pd_transfer_sleep_ms <= 0:
            for req_id, num_handles in newly_done:
                self._trace_recv_transfer_done(req_id, num_handles)
                done_req_ids.add(req_id)
            return done_req_ids

        # All transfers observed in this poll get the same deadline. The worker
        # remains available to poll and process other requests in the meantime.
        ready_at = (time.perf_counter() + self._pd_transfer_sleep_ms / 1000.0)
        for req_id, num_handles in newly_done:
            trace_event(
                "pull_transfer_sleep_start",
                req_id,
                role="decode",
                sleep_ms=self._pd_transfer_sleep_ms,
                num_handles=num_handles,
            )
            previous = self._delayed_recving_transfers.get(req_id)
            if previous is None:
                self._delayed_recving_transfers[req_id] = (ready_at,
                                                           num_handles)
            else:
                self._delayed_recving_transfers[req_id] = (max(
                    previous[0], ready_at), previous[1] + num_handles)
        return done_req_ids

    def _pop_delayed_recv_transfers(self) -> set[str]:
        now = time.perf_counter()
        done_req_ids: set[str] = set()
        for req_id, (ready_at, num_handles) in list(
                self._delayed_recving_transfers.items()):
            if now < ready_at:
                continue
            trace_event(
                "pull_transfer_sleep_end",
                req_id,
                role="decode",
                sleep_ms=self._pd_transfer_sleep_ms,
                num_handles=num_handles,
            )
            self._trace_recv_transfer_done(req_id, num_handles)
            done_req_ids.add(req_id)
            del self._delayed_recving_transfers[req_id]
        return done_req_ids

    def _trace_recv_transfer_done(self, req_id: str, num_handles: int) -> None:
        trace_event(
            "pull_transfer_end",
            req_id,
            role="decode",
            num_handles=num_handles,
            injected_sleep_ms=self._pd_transfer_sleep_ms,
        )
        trace_event(
            "pull_kv_recv_done",
            req_id,
            role="decode",
            num_handles=num_handles,
            injected_sleep_ms=self._pd_transfer_sleep_ms,
        )

    @staticmethod
    def _get_pd_transfer_sleep_ms() -> float:
        raw_value = os.getenv("VLLM_PD_TRANSFER_SLEEP_MS", "0")
        try:
            return float(raw_value)
        except ValueError:
            return 0.0

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote_engine_id
            self._start_transfer_trace(req_id, meta)
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))
            if self.use_host_buffer:
                self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(
                            req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._record_cached_handshake(req_id)
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time
                # Record a request-specific KV readiness event on the default
                # (compute) stream. The request finished Prefill in an earlier
                # step, so its KV writes are already enqueued ahead of this
                # point; the pack stream can later wait on this event instead of
                # the whole default stream and skip unrelated later Prefill work.
                self._record_packed_source_ready_event(req_id)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id, req_id)
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_port=meta.remote_port,
            remote_tp_size=meta.tp_size,
        )

    def _start_packed_transfer(self, request_id: str, dst_engine_id: str,
                               local_block_ids: list[int],
                               remote_block_ids: list[int], remote_host: str,
                               remote_port: int, remote_tp_size: int,
                               notif_id: bytes,
                               block_pair_stats: Optional[_BlockPairStats],
                               block_pair_stats_ns: int,
                               canonicalized_reverse_run_count: int,
                               canonicalized_reverse_block_count: int) -> None:
        remote_blocks_per_slot = self._packed_remote_blocks_per_slot[
            dst_engine_id]
        blocks_per_chunk = min(self._packed_blocks_per_slot,
                               remote_blocks_per_slot)
        chunks: list[_PackedChunkState] = []
        nonce = uuid.uuid4().hex
        for chunk_id, start in enumerate(
                range(0, len(local_block_ids), blocks_per_chunk)):
            end = min(start + blocks_per_chunk, len(local_block_ids))
            key = f"{self.engine_id}/{request_id}/{nonce}/{chunk_id}"
            chunk = _PackedChunkState(
                key=key,
                request_id=request_id,
                chunk_id=chunk_id,
                local_block_ids=local_block_ids[start:end],
                remote_block_ids=remote_block_ids[start:end],
                created_ns=time.perf_counter_ns())
            chunks.append(chunk)
            self._packed_chunks_by_key[key] = chunk

        state = _PackedRequestState(request_id=request_id,
                                    dst_engine_id=dst_engine_id,
                                    remote_host=remote_host,
                                    remote_port=remote_port,
                                    remote_tp_size=remote_tp_size,
                                    notif_id=notif_id,
                                    chunks=chunks)
        self._packed_requests[request_id] = state
        self._record_transfer_shape(
            request_id,
            num_local_blocks=len(local_block_ids),
            num_remote_blocks=len(remote_block_ids),
            num_local_descs=len(chunks),
            num_remote_descs=len(chunks),
            total_bytes=self._get_transfer_nbytes(local_block_ids),
            block_pair_stats=block_pair_stats,
            block_pair_stats_ns=block_pair_stats_ns,
            canonicalized_reverse_run_count=(canonicalized_reverse_run_count),
            canonicalized_reverse_block_count=(
                canonicalized_reverse_block_count),
            num_handles=len(chunks))
        if self._pd_trace_enabled:
            with self._transfer_trace_lock:
                trace_state = self._transfer_trace_states.get(request_id)
                if trace_state is not None:
                    trace_state.packed_chunk_count = len(chunks)
        trace_event("pull_transfer_start",
                    request_id,
                    role="decode",
                    remote_engine_id=dst_engine_id,
                    transfer_path="packed",
                    num_local_blocks=len(local_block_ids),
                    num_remote_blocks=len(remote_block_ids),
                    num_chunks=len(chunks),
                    num_local_descs=len(chunks),
                    num_remote_descs=len(chunks))
        self._schedule_packed_chunks()

    def _schedule_packed_chunks(self) -> None:
        """Reserve local slots and asynchronously request source-side packs."""
        now = time.perf_counter()
        for state in self._packed_requests.values():
            for chunk in state.chunks:
                if chunk.status not in ("queued", "retry"):
                    continue
                if chunk.status == "retry" and now < chunk.retry_at:
                    continue
                if chunk.local_slot is None:
                    if not self._packed_local_free_slots:
                        return
                    chunk.local_slot = self._packed_local_free_slots.popleft()
                    chunk.local_slot_acquired_ns = time.perf_counter_ns()
                    if (self._pd_trace_enabled
                            and chunk.created_ns is not None):
                        wait_ns = (chunk.local_slot_acquired_ns
                                   - chunk.created_ns)
                        with self._transfer_trace_lock:
                            trace_state = self._transfer_trace_states.get(
                                chunk.request_id)
                            if trace_state is not None:
                                profile = trace_state
                                profile.\
                                    packed_local_slot_queue_wait_total_ns += (
                                        wait_ns)
                                profile.packed_local_slot_queue_wait_max_ns = (
                                    max(
                                        profile
                                        .packed_local_slot_queue_wait_max_ns,
                                        wait_ns))
                                profile.\
                                    packed_local_slot_queue_wait_count += 1
                self._send_packed_control(
                    state, {
                        "type": "pack",
                        "key": chunk.key,
                        "request_id": chunk.request_id,
                        "chunk_id": chunk.chunk_id,
                        "total_chunks": len(state.chunks),
                        "block_ids": chunk.remote_block_ids,
                    })
                chunk.pack_request_ns = time.perf_counter_ns()
                chunk.status = "pack_requested"

    def _start_packed_read(self, state: _PackedRequestState,
                           chunk: _PackedChunkState, remote_slot: int) -> None:
        assert chunk.local_slot is not None
        block_count = len(chunk.local_block_ids)
        local_index = (chunk.local_slot * self._packed_blocks_per_slot +
                       block_count - 1)
        remote_blocks_per_slot = self._packed_remote_blocks_per_slot[
            state.dst_engine_id]
        remote_index = remote_slot * remote_blocks_per_slot + block_count - 1
        local_indices = np.asarray([local_index], dtype=np.int32)
        remote_indices = np.asarray([remote_index], dtype=np.int32)
        prepare_start_ns = time.perf_counter_ns()
        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ", self._packed_src_xfer_side_handle, local_indices,
            self._packed_dst_xfer_side_handles[state.dst_engine_id],
            remote_indices)
        self._record_trace_phase(req_id=state.request_id,
                                 phase="xfer_prepare",
                                 start_ns=prepare_start_ns,
                                 end_ns=time.perf_counter_ns())
        submit_start_ns = time.perf_counter_ns()
        self.nixl_wrapper.transfer(handle)
        self._record_trace_phase(req_id=state.request_id,
                                 phase="xfer_submit",
                                 start_ns=submit_start_ns,
                                 end_ns=time.perf_counter_ns())
        chunk.remote_slot = remote_slot
        chunk.handle = handle
        chunk.status = "reading"

    def _poll_packed_control_responses(self) -> None:
        decoder = msgspec.msgpack.Decoder()
        for socket in self._packed_control_sockets.values():
            while socket.poll(timeout=0, flags=zmq.POLLIN):
                response = decoder.decode(socket.recv())
                if not isinstance(response, dict):
                    raise RuntimeError("invalid packed control response")
                response_type = response.get("type")
                key = response.get("key")
                if response_type == "error":
                    raise RuntimeError("remote packed control failed: " +
                                       str(response.get("error")))
                if not isinstance(
                        key, str) or key not in self._packed_chunks_by_key:
                    logger.warning("Ignoring stale packed response for key %s",
                                   key)
                    continue
                chunk = self._packed_chunks_by_key[key]
                state = self._packed_requests[chunk.request_id]
                if response_type == "busy":
                    chunk.status = "retry"
                    chunk.retry_at = time.perf_counter() + 0.001
                    chunk.busy_count += 1
                    chunk.retry_count += 1
                    if self._pd_trace_enabled:
                        with self._transfer_trace_lock:
                            trace_state = self._transfer_trace_states.get(
                                chunk.request_id)
                            if trace_state is not None:
                                trace_state.packed_source_busy_count += 1
                                trace_state.packed_chunk_retry_count += 1
                    continue
                if response_type == "unavailable":
                    raise RuntimeError(
                        "remote packed staging became unavailable")
                if response_type != "ready":
                    raise RuntimeError(
                        f"unexpected packed control response: {response_type}")
                if response.get("block_count") != len(chunk.local_block_ids):
                    raise RuntimeError("packed control block count mismatch")
                if self._pd_trace_enabled:
                    ready_ns = time.perf_counter_ns()
                    with self._transfer_trace_lock:
                        trace_state = self._transfer_trace_states.get(
                            chunk.request_id)
                        if trace_state is not None:
                            if chunk.pack_request_ns is not None:
                                trace_state.packed_pack_control_total_ns += (
                                    ready_ns - chunk.pack_request_ns)
                            pack_gpu_ns = response.get("pack_gpu_ns", 0)
                            if isinstance(pack_gpu_ns, int):
                                trace_state.packed_pack_gpu_total_ns += (
                                    pack_gpu_ns)
                            source_handler_ns = response.get(
                                "source_handler_ns", 0)
                            if isinstance(source_handler_ns, int):
                                trace_state.packed_source_handler_total_ns += (
                                    source_handler_ns)
                                trace_state.packed_source_handler_max_ns = max(
                                    trace_state.packed_source_handler_max_ns,
                                    source_handler_ns)
                            source_sync_wall_ns = response.get(
                                "source_sync_wall_ns", 0)
                            if isinstance(source_sync_wall_ns, int):
                                trace_state.packed_source_sync_wall_total_ns += (
                                    source_sync_wall_ns)
                                trace_state.packed_source_sync_wall_max_ns = max(
                                    trace_state.packed_source_sync_wall_max_ns,
                                    source_sync_wall_ns)
                            source_stream_wait_gpu_ns = response.get(
                                "source_stream_wait_gpu_ns", 0)
                            if isinstance(source_stream_wait_gpu_ns, int):
                                profile = trace_state
                                profile.packed_source_stream_wait_gpu_total_ns += (
                                    source_stream_wait_gpu_ns)
                                profile.packed_source_stream_wait_gpu_max_ns = max(
                                    profile
                                    .packed_source_stream_wait_gpu_max_ns,
                                    source_stream_wait_gpu_ns)
                            wait_count = response.get(
                                "source_default_stream_wait_count", 0)
                            if isinstance(wait_count, int):
                                trace_state.packed_source_wait_count += wait_count
                            event_wait = response.get(
                                "source_readiness_event_wait_count", 0)
                            if isinstance(event_wait, int):
                                trace_state.\
                                    packed_source_readiness_event_wait_count += (
                                        event_wait)
                            fallback = response.get(
                                "source_default_stream_fallback_count", 0)
                            if isinstance(fallback, int):
                                trace_state.\
                                    packed_source_default_stream_fallback_count \
                                    += fallback
                remote_slot = response.get("remote_slot")
                if not isinstance(remote_slot, int) or remote_slot < 0:
                    raise RuntimeError("invalid remote packed staging slot")
                self._start_packed_read(state, chunk, remote_slot)

    def _launch_packed_scatter(
            self, chunk: _PackedChunkState
    ) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        assert self._packed_staging is not None
        assert self._packed_region_ptrs is not None
        assert self._packed_scatter_stream is not None
        assert chunk.local_slot is not None
        staging = self._packed_staging.narrow(
            0, chunk.local_slot * self._packed_slot_bytes,
            self._packed_slot_bytes)
        device = self._packed_staging.device
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device), torch.cuda.stream(
                self._packed_scatter_stream):
            ids = torch.tensor(chunk.local_block_ids,
                               dtype=torch.int64,
                               device=device)
            start_event.record(self._packed_scatter_stream)
            _launch_scatter_regions(self._packed_region_ptrs, ids, staging,
                                    self._packed_block_bytes)
            end_event.record(self._packed_scatter_stream)
        return start_event, end_event

    def _packed_remote_agent_name(self, state: _PackedRequestState) -> str:
        tp_ratio = self._tp_size[self.engine_id] // state.remote_tp_size
        remote_rank = self.tp_rank // tp_ratio
        return self._remote_agents[state.dst_engine_id][remote_rank]

    def _poll_packed_transfers(self) -> set[str]:
        if not getattr(self, "_packed_requests", None):
            return set()
        self._poll_packed_control_responses()
        proc_checks_by_req: defaultdict[str, int] = defaultdict(int)

        for state in list(self._packed_requests.values()):
            for chunk in state.chunks:
                if chunk.status != "reading":
                    continue
                assert chunk.handle is not None
                xfer_state = self.nixl_wrapper.check_xfer_state(chunk.handle)
                if xfer_state == "PROC":
                    proc_checks_by_req[state.request_id] += 1
                    continue
                if xfer_state != "DONE":
                    raise RuntimeError(
                        f"Packed transfer failed with state {xfer_state}")
                self.nixl_wrapper.release_xfer_handle(chunk.handle)
                self.xfer_stats.record_transfer()
                self._send_packed_control(state, {
                    "type": "release",
                    "key": chunk.key,
                })
                (chunk.scatter_start_event,
                 chunk.scatter_event) = self._launch_packed_scatter(chunk)
                chunk.status = "scattering"
                state.read_chunks_done += 1

            if (state.read_chunks_done == len(state.chunks)
                    and not state.notification_sent):
                self.nixl_wrapper.send_notif(
                    self._packed_remote_agent_name(state),
                    notif_msg=state.notif_id)
                state.notification_sent = True

        done_req_ids: set[str] = set()
        for state in list(self._packed_requests.values()):
            for chunk in state.chunks:
                if (chunk.status != "scattering" or chunk.scatter_event is None
                        or not chunk.scatter_event.query()):
                    continue
                assert chunk.local_slot is not None
                if (self._pd_trace_enabled
                        and chunk.scatter_start_event is not None):
                    scatter_gpu_ns = int(
                        chunk.scatter_start_event.elapsed_time(
                            chunk.scatter_event) * 1_000_000)
                    with self._transfer_trace_lock:
                        trace_state = self._transfer_trace_states.get(
                            state.request_id)
                        if trace_state is not None:
                            trace_state.packed_scatter_gpu_total_ns += (
                                scatter_gpu_ns)
                self._packed_local_free_slots.append(chunk.local_slot)
                chunk.local_slot = None
                chunk.status = "done"
                state.scatter_chunks_done += 1
            done = state.scatter_chunks_done == len(state.chunks)
            self._record_transfer_poll(state.request_id,
                                       proc_checks_by_req[state.request_id],
                                       done=done)
            if done:
                num_handles = len(state.chunks)
                if self._pd_transfer_sleep_ms <= 0:
                    self._trace_recv_transfer_done(state.request_id,
                                                   num_handles)
                    done_req_ids.add(state.request_id)
                else:
                    ready_at = (time.perf_counter() +
                                self._pd_transfer_sleep_ms / 1000.0)
                    trace_event("pull_transfer_sleep_start",
                                state.request_id,
                                role="decode",
                                sleep_ms=self._pd_transfer_sleep_ms,
                                num_handles=num_handles)
                    self._delayed_recving_transfers[state.request_id] = (
                        ready_at, num_handles)
                for chunk in state.chunks:
                    self._packed_chunks_by_key.pop(chunk.key, None)
                del self._packed_requests[state.request_id]

        self._schedule_packed_chunks()
        return done_req_ids

    def _read_blocks(self,
                     local_block_ids: list[int],
                     remote_block_ids: list[int],
                     dst_engine_id: str,
                     request_id: str,
                     remote_host: Optional[str] = None,
                     remote_port: Optional[int] = None,
                     remote_tp_size: Optional[int] = None):
        # The experimental packed path stages on both peers: the remote/source
        # worker gathers before READ, and the local/reader worker scatters only
        # after NIXL reports DONE. Both phases are advanced asynchronously from
        # get_finished(), so start_load_kv() remains non-blocking.

        # Number of D TP workers that will read from dst P. Propagate tp_ratio
        # on notification so that dst worker can wait before freeing blocks.
        tp_ratio = self._tp_size[
            self.engine_id] // self._tp_size[dst_engine_id]
        notif_id = f"{request_id}:{tp_ratio}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            remote_rank = self.tp_rank // tp_ratio
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            self._record_transfer_skipped(request_id)
            self._emit_transfer_trace(request_id)
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        block_pair_stats: Optional[_BlockPairStats] = None
        block_pair_stats_ns = 0
        if self._pd_trace_enabled:
            block_pair_stats_start_ns = time.perf_counter_ns()
            block_pair_stats = _analyze_block_pairs(local_block_ids,
                                                    remote_block_ids)
            block_pair_stats_ns = (time.perf_counter_ns() -
                                   block_pair_stats_start_ns)

        canonicalized_reverse_run_count = 0
        canonicalized_reverse_block_count = 0

        submitted_local_block_ids = local_block_ids
        submitted_remote_block_ids = remote_block_ids
        if (not self.block_window_per_layer
                and self._canonicalize_reverse_block_pairs):
            (submitted_local_block_ids, submitted_remote_block_ids,
             canonicalized_reverse_run_count, canonicalized_reverse_block_count
             ) = _canonicalize_paired_reverse_runs(local_block_ids,
                                                   remote_block_ids)

        configured_mode = getattr(self, "_nixl_transfer_mode", "direct")
        forward_ranges = 0
        if configured_mode != "direct" or self._pd_trace_enabled:
            forward_ranges = _count_forward_ranges(submitted_local_block_ids,
                                                   submitted_remote_block_ids)
        available_packed_slots = len(
            getattr(self, "_packed_local_free_slots", ()))
        wants_packed = _should_use_packed_path(
            configured_mode, num_local_blocks, forward_ranges,
            getattr(self, "_packed_auto_range_threshold", 64),
            available_packed_slots)
        packed_supported = (getattr(self, "_packed_available", False)
                            and not self.block_window_per_layer and
                            dst_engine_id in self._packed_dst_xfer_side_handles
                            and remote_host is not None
                            and remote_port is not None
                            and remote_tp_size is not None
                            and request_id not in self._packed_requests)
        selected_path = "packed" if wants_packed and packed_supported else "direct"
        trace_event(
            "pull_transfer_path_selected",
            request_id,
            role="decode",
            configured_mode=configured_mode,
            selected_path=selected_path,
            num_blocks=num_local_blocks,
            forward_ranges=forward_ranges,
            available_packed_slots=available_packed_slots,
            auto_range_threshold=getattr(self, "_packed_auto_range_threshold",
                                         64),
            packed_supported=packed_supported)
        if wants_packed and not packed_supported:
            logger.warning_once(
                "NIXL %s mode selected packed for B=%s K=%s, but this peer or "
                "layout is unsupported; falling back to direct READ",
                configured_mode, num_local_blocks, forward_ranges)
        if self._pd_trace_enabled:
            with self._transfer_trace_lock:
                trace_state = self._transfer_trace_states.get(request_id)
                if trace_state is not None:
                    trace_state.configured_transfer_mode = configured_mode
                    trace_state.selected_transfer_path = selected_path
                    trace_state.selector_num_blocks = num_local_blocks
                    trace_state.selector_forward_ranges = forward_ranges
                    trace_state.selector_available_packed_slots = (
                        available_packed_slots)

        if selected_path == "packed":
            assert remote_host is not None
            assert remote_port is not None
            assert remote_tp_size is not None
            self._start_packed_transfer(
                request_id=request_id,
                dst_engine_id=dst_engine_id,
                local_block_ids=submitted_local_block_ids,
                remote_block_ids=submitted_remote_block_ids,
                remote_host=remote_host,
                remote_port=remote_port,
                remote_tp_size=remote_tp_size,
                notif_id=notif_id,
                block_pair_stats=block_pair_stats,
                block_pair_stats_ns=block_pair_stats_ns,
                canonicalized_reverse_run_count=(
                    canonicalized_reverse_run_count),
                canonicalized_reverse_block_count=(
                    canonicalized_reverse_block_count))
            return

        # Get side handles.
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        desc_build_start_ns = (time.perf_counter_ns()
                               if self._pd_trace_enabled else 0)
        local_block_descs_ids: np.ndarray
        remote_block_descs_ids: np.ndarray
        if not self.block_window_per_layer:
            # Default case: assume global attention
            remote_block_descs_ids = self._get_block_descs_ids(
                dst_engine_id, submitted_remote_block_ids)
            local_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, submitted_local_block_ids)
        else:
            # TODO(mgoin): remove this once we have hybrid memory allocator
            # Optimization for models with local attention (Llama 4)
            local_descs_list = []
            remote_descs_list = []
            for layer_idx, block_window in enumerate(
                    self.block_window_per_layer):
                # For each layer:
                if block_window is None:
                    # If not chunked, we just use the
                    # full block lists (global attention)
                    layer_local_block_ids = local_block_ids
                    layer_remote_block_ids = remote_block_ids
                else:
                    # If chunked, get the last block_window blocks
                    layer_local_block_ids = local_block_ids[-block_window:]
                    layer_remote_block_ids = remote_block_ids[-block_window:]

                if self._canonicalize_reverse_block_pairs:
                    (layer_local_block_ids, layer_remote_block_ids,
                     layer_reverse_run_count, layer_reverse_block_count
                     ) = _canonicalize_paired_reverse_runs(
                         layer_local_block_ids, layer_remote_block_ids)
                    canonicalized_reverse_run_count += layer_reverse_run_count
                    canonicalized_reverse_block_count += (
                        layer_reverse_block_count)

                # Get descs ids for the layer.
                layer_local_desc_ids = self._get_block_descs_ids(
                    self.engine_id, layer_local_block_ids, layer_idx)
                layer_remote_desc_ids = self._get_block_descs_ids(
                    dst_engine_id, layer_remote_block_ids, layer_idx)

                local_descs_list.append(layer_local_desc_ids)
                remote_descs_list.append(layer_remote_desc_ids)

            local_block_descs_ids = np.concatenate(local_descs_list)
            remote_block_descs_ids = np.concatenate(remote_descs_list)

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)
        if self._pd_trace_enabled:
            self._record_trace_phase(request_id,
                                     "desc_build", desc_build_start_ns,
                                     time.perf_counter_ns())

        # Prepare transfer with Nixl.
        xfer_prepare_start_ns = (time.perf_counter_ns()
                                 if self._pd_trace_enabled else 0)
        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg=notif_id,
        )
        if self._pd_trace_enabled:
            self._record_trace_phase(request_id, "xfer_prepare",
                                     xfer_prepare_start_ns,
                                     time.perf_counter_ns())
            self._record_transfer_shape(
                request_id,
                num_local_blocks=len(local_block_ids),
                num_remote_blocks=len(remote_block_ids),
                num_local_descs=len(local_block_descs_ids),
                num_remote_descs=len(remote_block_descs_ids),
                total_bytes=self._get_transfer_nbytes(local_block_ids),
                block_pair_stats=block_pair_stats,
                block_pair_stats_ns=block_pair_stats_ns,
                canonicalized_reverse_run_count=(
                    canonicalized_reverse_run_count),
                canonicalized_reverse_block_count=(
                    canonicalized_reverse_block_count),
            )

        # Begin async xfer.
        trace_event(
            "pull_transfer_start",
            request_id,
            role="decode",
            remote_engine_id=dst_engine_id,
            transfer_path="direct",
            num_local_blocks=len(local_block_ids),
            num_remote_blocks=len(remote_block_ids),
            num_local_descs=len(local_block_descs_ids),
            num_remote_descs=len(remote_block_descs_ids),
        )
        xfer_submit_start_ns = (time.perf_counter_ns()
                                if self._pd_trace_enabled else 0)
        self.nixl_wrapper.transfer(handle)
        if self._pd_trace_enabled:
            self._record_trace_phase(request_id, "xfer_submit",
                                     xfer_submit_start_ns,
                                     time.perf_counter_ns())

        # Use handle to check completion in future step().
        self._recving_transfers[request_id].append(
            (handle, time.perf_counter()))

    def _get_block_descs_ids(self,
                             engine_id: str,
                             block_ids: list[int],
                             layer_idx: Optional[int] = None) -> np.ndarray:
        """
        Get the descs ids for a set of block ids.
        If layer_idx is provided, we use the region_ids for the given layer.
        Otherwise, we use all regions.
        """
        if layer_idx is None:
            region_ids = np.arange(self.num_regions)
        else:
            assert layer_idx < self.num_layers
            if self.num_layers < self.num_regions:
                # If we have more regions than layers, we assume that
                # the regions are organized as [K0, V0, K1, V1, ...]
                # and we select K_i and V_i
                assert 2 * self.num_layers == self.num_regions
                region_ids = np.arange(2 * layer_idx, 2 * layer_idx + 2)
            else:
                # Otherwise, we assume we have MLA and select i-th layer
                assert self.num_layers == self.num_regions
                region_ids = np.arange(layer_idx, layer_idx + 1)

        num_blocks = self.dst_num_blocks[engine_id]

        # Compute the desc ids for each block.
        region_ids = region_ids[:, None]
        block_ids = np.array(block_ids)[None, :]
        descs_ids = region_ids * num_blocks + block_ids
        return descs_ids.flatten()

    def get_backend_aware_kv_block_len(self, layer_idx: int):
        """
        Get the block length for one K/V element (K and V have the same size).

        For FA and other backends, this is equal to the length of the whole 
        block, as K and V are in separate regions.
        For FlashInfer, this is half the length of the whole block, as K and V
        share the same region.
        """
        if self._use_flashinfer:
            # For indexing only half (either just the K or V part).
            block_len = self.block_len_per_layer[layer_idx] // 2
        else:
            block_len = self.block_len_per_layer[layer_idx]
        return block_len

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def shutdown(self):
        """Shutdown the connector worker."""
        self._handshake_initiation_executor.shutdown(wait=False)
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join(timeout=0)
            self._nixl_handshake_listener_t = None
        for handles in self._recving_transfers.values():
            for handle, _ in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        for state in self._packed_requests.values():
            for chunk in state.chunks:
                if chunk.status == "reading" and chunk.handle is not None:
                    self.nixl_wrapper.release_xfer_handle(chunk.handle)
        self._packed_requests.clear()
        self._packed_chunks_by_key.clear()
        self._packed_source_requests.clear()
        with self._packed_source_ready_lock:
            self._packed_source_ready_events.clear()
        self._delayed_recving_transfers.clear()
        self._transfer_trace_states.clear()
        if self.src_xfer_side_handle:
            self.nixl_wrapper.release_dlist_handle(self.src_xfer_side_handle)
            self.src_xfer_side_handle = 0
        for dst_xfer_side_handle in self.dst_xfer_side_handles.values():
            self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)
        self.dst_xfer_side_handles.clear()
        if self._packed_src_xfer_side_handle:
            self.nixl_wrapper.release_dlist_handle(
                self._packed_src_xfer_side_handle)
            self._packed_src_xfer_side_handle = 0
        for handle in self._packed_dst_xfer_side_handles.values():
            self.nixl_wrapper.release_dlist_handle(handle)
        self._packed_dst_xfer_side_handles.clear()
        for socket in self._packed_control_sockets.values():
            socket.close(linger=0)
        self._packed_control_sockets.clear()
        if self._packed_control_context is not None:
            self._packed_control_context.destroy(linger=0)
            self._packed_control_context = None
        for remote_agents in self._remote_agents.values():
            for agent_name in remote_agents.values():
                self.nixl_wrapper.remove_remote_agent(agent_name)
        self._remote_agents.clear()
        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if "num_successful_transfers" not in self.data:
            self.data["num_successful_transfers"] = 0

    def reset(self):
        self.data = {"num_successful_transfers": 0}

    def record_transfer(self):
        # TODO: record actual transfer stats when available
        self.data["num_successful_transfers"] += 1

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.data["num_successful_transfers"] == 0

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            self.data["num_successful_transfers"] += other.data[
                "num_successful_transfers"]
        return self

    def reduce(self) -> dict[str, Union[int, float]]:
        # TODO: reduce stats to a single value, calculate latency/throughput
        return {
            "num_successful_transfers": self.data["num_successful_transfers"]
        }
