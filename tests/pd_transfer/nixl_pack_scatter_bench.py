#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone NIXL direct versus pack/transfer/scatter benchmark.

The benchmark models vLLM's region-major NIXL descriptor layout without
starting a model server.  A target process owns a paged source KV arena and an
initiator process pulls data into a second GPU.  Every workload is executed by
two paths:

* ``direct`` submits one descriptor for every (region, block) pair;
* ``packed`` gathers those pairs into a bounded contiguous target buffer,
  pulls one descriptor per chunk, and scatters it into destination blocks.

The fused copy kernels follow the pointer-table and single-launch structure of
SGLang's staging buffer kernels, adapted from token/head slices to vLLM-style
region/block mappings.  The torch backend is an independent correctness and
portability fallback.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
import platform
import random
import socket
import statistics
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np


LOG = logging.getLogger("nixl-pack-scatter-bench")
PATTERN_MODULUS = 251
DESTINATION_SENTINEL = 255
SUPPORTED_PATTERNS = ("forward", "reverse", "mixed", "fragmented")
SUPPORTED_PATHS = ("direct", "packed")
COPY_TILE_BYTES = 1024

SGLANG_STAGING_URL = (
    "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/"
    "disaggregation/common/staging_buffer.py"
)
SGLANG_NIXL_PR_URL = "https://github.com/sgl-project/sglang/pull/22536"

_TRITON_KERNELS: tuple[Any, Any, Any] | None = None


@dataclass(frozen=True)
class Workload:
    pattern: str
    request_blocks: int
    local_block_ids: tuple[int, ...]
    remote_block_ids: tuple[int, ...]
    forward_range_count: int


def _load_nixl_api():
    """Import the API from both the NIXL 0.6 and current package layouts."""
    try:
        from nixl import nixl_agent, nixl_agent_config
    except ImportError:
        from nixl._api import nixl_agent, nixl_agent_config
    return nixl_agent, nixl_agent_config


def _make_agent(name: str):
    nixl_agent, nixl_agent_config = _load_nixl_api()
    config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=False,
        backends=["UCX"],
    )
    return nixl_agent(name, config)


def _agent_name_text(agent_name: str | bytes) -> str:
    if isinstance(agent_name, bytes):
        return agent_name.decode("utf-8")
    return agent_name


def _get_triton_kernels() -> tuple[Any, Any, Any]:
    """Define the optional kernels lazily so CPU-only unit tests can import."""
    global _TRITON_KERNELS
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS

    import triton
    import triton.language as tl

    @triton.jit
    def pack_regions_kernel(
        region_ptrs,
        block_ids,
        staging,
        request_block_count,
        block_bytes,
        COPY_TILE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        tile_id = tl.program_id(1)
        region_id = row_id // request_block_count
        request_block_id = row_id % request_block_count
        region_ptr = tl.load(region_ptrs + region_id).to(staging.dtype)
        physical_block_id = tl.load(block_ids + request_block_id)
        offsets = tile_id * COPY_TILE + tl.arange(0, COPY_TILE)
        mask = offsets < block_bytes
        source_offsets = (
            physical_block_id.to(tl.int64) * block_bytes.to(tl.int64)
            + offsets
        )
        destination_offsets = (
            row_id.to(tl.int64) * block_bytes.to(tl.int64) + offsets
        )
        values = tl.load(region_ptr + source_offsets, mask=mask)
        tl.store(staging + destination_offsets, values, mask=mask)

    @triton.jit
    def scatter_regions_kernel(
        region_ptrs,
        block_ids,
        staging,
        request_block_count,
        block_bytes,
        COPY_TILE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        tile_id = tl.program_id(1)
        region_id = row_id // request_block_count
        request_block_id = row_id % request_block_count
        region_ptr = tl.load(region_ptrs + region_id).to(staging.dtype)
        physical_block_id = tl.load(block_ids + request_block_id)
        offsets = tile_id * COPY_TILE + tl.arange(0, COPY_TILE)
        mask = offsets < block_bytes
        source_offsets = (
            row_id.to(tl.int64) * block_bytes.to(tl.int64) + offsets
        )
        destination_offsets = (
            physical_block_id.to(tl.int64) * block_bytes.to(tl.int64)
            + offsets
        )
        values = tl.load(staging + source_offsets, mask=mask)
        tl.store(region_ptr + destination_offsets, values, mask=mask)

    _TRITON_KERNELS = (triton, pack_regions_kernel, scatter_regions_kernel)
    return _TRITON_KERNELS


def _resolve_kernel(requested: str) -> str:
    if requested == "torch":
        return requested
    try:
        _get_triton_kernels()
    except (ImportError, ModuleNotFoundError):
        if requested == "triton":
            raise
        LOG.warning(
            "Triton is unavailable; falling back to torch gather/scatter"
        )
        return "torch"
    return "triton"


def _region_ptrs(torch, arena):
    return torch.tensor(
        [arena[region].data_ptr() for region in range(arena.shape[0])],
        dtype=torch.int64,
        device=arena.device,
    )


def _launch_pack(
    torch,
    kernel: str,
    source_arena,
    source_region_ptrs,
    block_ids,
    staging,
    block_bytes: int,
) -> None:
    region_count = source_arena.shape[0]
    request_block_count = block_ids.numel()
    used_bytes = region_count * request_block_count * block_bytes
    if kernel == "torch":
        output = staging[:used_bytes].view(
            region_count, request_block_count, block_bytes
        )
        torch.index_select(
            source_arena[:, :-1, :], 1, block_ids, out=output
        )
        return

    triton, pack_kernel, _ = _get_triton_kernels()
    grid = (
        region_count * request_block_count,
        triton.cdiv(block_bytes, COPY_TILE_BYTES),
    )
    pack_kernel[grid](
        source_region_ptrs,
        block_ids,
        staging,
        request_block_count,
        block_bytes,
        COPY_TILE=COPY_TILE_BYTES,
    )


def _launch_scatter(
    torch,
    kernel: str,
    destination_arena,
    destination_region_ptrs,
    block_ids,
    staging,
    block_bytes: int,
) -> None:
    region_count = destination_arena.shape[0]
    request_block_count = block_ids.numel()
    used_bytes = region_count * request_block_count * block_bytes
    if kernel == "torch":
        source = staging[:used_bytes].view(
            region_count, request_block_count, block_bytes
        )
        destination_arena[:, :-1, :].index_copy_(1, block_ids, source)
        return

    triton, _, scatter_kernel = _get_triton_kernels()
    grid = (
        region_count * request_block_count,
        triton.cdiv(block_bytes, COPY_TILE_BYTES),
    )
    scatter_kernel[grid](
        destination_region_ptrs,
        block_ids,
        staging,
        request_block_count,
        block_bytes,
        COPY_TILE=COPY_TILE_BYTES,
    )


def _count_forward_ranges(
    local_block_ids: tuple[int, ...] | list[int],
    remote_block_ids: tuple[int, ...] | list[int],
) -> int:
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError("local and remote block ID counts must match")
    if not local_block_ids:
        return 0
    return 1 + sum(
        local_block_ids[index + 1] != local_block_ids[index] + 1
        or remote_block_ids[index + 1] != remote_block_ids[index] + 1
        for index in range(len(local_block_ids) - 1)
    )


def _shuffled_runs(
    start: int,
    count: int,
    run_length: int,
    random_generator: random.Random,
) -> list[int]:
    values = list(range(start, start + count))
    runs = [values[index : index + run_length]
            for index in range(0, count, run_length)]
    random_generator.shuffle(runs)
    return [value for run in runs for value in run]


def _build_mapping(
    pattern: str,
    request_blocks: int,
    physical_blocks: int,
    seed: int,
    mixed_run_length: int,
) -> Workload:
    if pattern not in SUPPORTED_PATTERNS:
        raise ValueError(f"unsupported mapping pattern: {pattern}")
    if request_blocks <= 0 or request_blocks > physical_blocks:
        raise ValueError("request_blocks must be in [1, physical_blocks]")
    if mixed_run_length <= 0:
        raise ValueError("mixed_run_length must be positive")

    remote_start = physical_blocks - request_blocks
    if pattern == "forward":
        local = list(range(request_blocks))
        remote = list(range(remote_start, physical_blocks))
    elif pattern == "reverse":
        local = list(range(request_blocks - 1, -1, -1))
        remote = list(range(physical_blocks - 1, remote_start - 1, -1))
    elif pattern == "mixed":
        local_rng = random.Random(seed ^ 0x51A7)
        remote_rng = random.Random(seed ^ 0xA715)
        local = _shuffled_runs(
            0, request_blocks, mixed_run_length, local_rng
        )
        remote = _shuffled_runs(
            remote_start, request_blocks, mixed_run_length, remote_rng
        )
    else:
        local_rng = random.Random(seed ^ 0xF12A)
        remote_rng = random.Random(seed ^ 0x2AF1)
        local = local_rng.sample(range(physical_blocks), request_blocks)
        remote = remote_rng.sample(range(physical_blocks), request_blocks)

    local_tuple = tuple(local)
    remote_tuple = tuple(remote)
    return Workload(
        pattern=pattern,
        request_blocks=request_blocks,
        local_block_ids=local_tuple,
        remote_block_ids=remote_tuple,
        forward_range_count=_count_forward_ranges(local_tuple, remote_tuple),
    )


def _descriptor_indices(
    block_ids: tuple[int, ...] | list[int],
    region_count: int,
    physical_blocks: int,
) -> np.ndarray:
    region_offsets = (
        np.arange(region_count, dtype=np.int64) * physical_blocks
    )[:, None]
    blocks = np.asarray(block_ids, dtype=np.int64)[None, :]
    indices = region_offsets + blocks
    if indices.size and indices.max() > np.iinfo(np.int32).max:
        raise ValueError("descriptor index exceeds NIXL int32 index range")
    return indices.ravel().astype(np.int32)


def _chunk_ranges(
    request_blocks: int, blocks_per_chunk: int
) -> list[tuple[int, int]]:
    if request_blocks <= 0 or blocks_per_chunk <= 0:
        raise ValueError("request_blocks and blocks_per_chunk must be positive")
    return [
        (start, min(start + blocks_per_chunk, request_blocks))
        for start in range(0, request_blocks, blocks_per_chunk)
    ]


def _make_kv_views(
    arena, region_count: int, physical_blocks: int, block_bytes: int
):
    views = []
    for region in range(region_count):
        flat_region = arena[region].view(-1)
        views.extend(
            flat_region.narrow(0, block * block_bytes, block_bytes)
            for block in range(physical_blocks)
        )
    return views


def _make_staging_views(
    staging, region_count: int, max_blocks: int, block_bytes: int
):
    return [
        staging.narrow(0, 0, region_count * count * block_bytes)
        for count in range(1, max_blocks + 1)
    ]


def _fill_source_pattern(torch, arena, physical_blocks: int) -> None:
    region_values = torch.arange(
        arena.shape[0], dtype=torch.int64, device=arena.device
    )[:, None]
    block_values = torch.arange(
        physical_blocks, dtype=torch.int64, device=arena.device
    )[None, :]
    values = ((region_values * 131 + block_values * 17) % PATTERN_MODULUS)
    valid = arena[:, :physical_blocks, :]
    valid.copy_(values.to(torch.uint8)[..., None])
    # Encode the full region/block identity in the first four bytes.  The
    # remaining bytes retain the inexpensive constant fill, so verification
    # catches both wrong row mappings and partial copies.
    valid[:, :, 0].copy_((block_values & 0xFF).to(torch.uint8))
    valid[:, :, 1].copy_(((block_values >> 8) & 0xFF).to(torch.uint8))
    valid[:, :, 2].copy_((region_values & 0xFF).to(torch.uint8))
    valid[:, :, 3].copy_(((region_values >> 8) & 0xFF).to(torch.uint8))
    arena[:, physical_blocks, :].zero_()


def _time_gpu_copy(torch, stream, operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.wait_stream(torch.cuda.default_stream(stream.device))
    with torch.cuda.stream(stream):
        start.record(stream)
        operation()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end) * 1_000_000.0)


def _target_worker(connection, config: dict[str, Any]) -> None:
    agent = None
    remote_name = None
    registrations = []
    try:
        import torch

        gpu = config["target_gpu"]
        region_count = config["regions"]
        physical_blocks = config["physical_blocks"]
        block_bytes = config["block_bytes"]
        staging_bytes = config["staging_bytes"]
        blocks_per_chunk = config["blocks_per_chunk"]
        kernel = config["kernel"]

        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        source_arena = torch.empty(
            (region_count, physical_blocks + 1, block_bytes),
            dtype=torch.uint8,
            device=device,
        )
        send_staging = torch.empty(
            staging_bytes, dtype=torch.uint8, device=device
        )
        _fill_source_pattern(torch, source_arena, physical_blocks)
        source_region_ptrs = _region_ptrs(torch, source_arena)
        pack_stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)

        agent = _make_agent("target")
        registrations.append(
            agent.register_memory(source_arena, backends=["UCX"])
        )
        registrations.append(
            agent.register_memory(send_staging, backends=["UCX"])
        )
        source_xfer_descs = agent.get_xfer_descs(
            _make_kv_views(
                source_arena, region_count, physical_blocks, block_bytes
            )
        )
        staging_xfer_descs = agent.get_xfer_descs(
            _make_staging_views(
                send_staging, region_count, blocks_per_chunk, block_bytes
            )
        )
        connection.send(
            {
                "type": "ready",
                "metadata": agent.get_agent_metadata(),
                "source_descs": agent.get_serialized_descs(source_xfer_descs),
                "staging_descs": agent.get_serialized_descs(
                    staging_xfer_descs
                ),
            }
        )

        while True:
            if not connection.poll(0.01):
                agent.get_new_notifs()
                continue
            message = connection.recv()
            if message == "stop":
                break
            if message.get("type") == "add_remote":
                remote_name = agent.add_remote_agent(message["metadata"])
                if _agent_name_text(remote_name) != "initiator":
                    raise RuntimeError(
                        "target loaded unexpected remote agent name: "
                        f"{remote_name!r}"
                    )
                connection.send(
                    {
                        "type": "remote_added",
                        "remote_name": _agent_name_text(remote_name),
                    }
                )
                continue
            if message.get("type") != "pack":
                raise RuntimeError(f"unexpected target message: {message!r}")

            index_start_ns = time.perf_counter_ns()
            block_ids = torch.tensor(
                message["remote_block_ids"],
                dtype=torch.int64,
                device=device,
            )
            index_end_ns = time.perf_counter_ns()
            used_bytes = region_count * block_ids.numel() * block_bytes
            if used_bytes > staging_bytes:
                raise RuntimeError(
                    f"packed chunk needs {used_bytes} bytes, staging has "
                    f"{staging_bytes}"
                )
            pack_gpu_ns = _time_gpu_copy(
                torch,
                pack_stream,
                lambda: _launch_pack(
                    torch,
                    kernel,
                    source_arena,
                    source_region_ptrs,
                    block_ids,
                    send_staging,
                    block_bytes,
                ),
            )
            connection.send(
                {
                    "type": "packed",
                    "chunk_id": message["chunk_id"],
                    "used_bytes": used_bytes,
                    "source_index_build_ns": index_end_ns - index_start_ns,
                    "pack_gpu_ns": pack_gpu_ns,
                }
            )
    except BaseException:
        try:
            connection.send(
                {"type": "error", "traceback": traceback.format_exc()}
            )
        except BaseException:
            pass
    finally:
        if agent is not None and remote_name is not None:
            try:
                agent.remove_remote_agent(remote_name)
            except BaseException:
                pass
        if agent is not None:
            for registration in registrations:
                try:
                    agent.deregister_memory(registration, backends=["UCX"])
                except BaseException:
                    pass
        connection.close()


def _run_nixl_transfer(
    *,
    agent,
    local_prepped,
    local_indices: np.ndarray,
    remote_prepped,
    remote_indices: np.ndarray,
    timeout_seconds: float,
) -> dict[str, int | str]:
    handle = None
    try:
        make_start_ns = time.perf_counter_ns()
        handle = agent.make_prepped_xfer(
            "READ",
            local_prepped,
            local_indices,
            remote_prepped,
            remote_indices,
            backends=["UCX"],
        )
        make_end_ns = time.perf_counter_ns()
        if handle is None:
            raise RuntimeError("make_prepped_xfer returned no handle")

        post_start_ns = make_end_ns
        initial_state = agent.transfer(handle)
        post_end_ns = time.perf_counter_ns()
        if initial_state == "ERR":
            raise RuntimeError("NIXL transfer() returned ERR")

        state = initial_state
        poll_count = 0
        deadline = time.perf_counter() + timeout_seconds
        while state != "DONE":
            state = agent.check_xfer_state(handle)
            poll_count += 1
            if state == "ERR":
                raise RuntimeError("NIXL check_xfer_state() returned ERR")
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "NIXL transfer did not finish within "
                    f"{timeout_seconds}s (initial={initial_state}, "
                    f"last={state}, polls={poll_count})"
                )
        done_ns = time.perf_counter_ns()
        return {
            "make_xfer_ns": make_end_ns - make_start_ns,
            "post_xfer_ns": post_end_ns - post_start_ns,
            "poll_xfer_ns": done_ns - post_end_ns,
            "transfer_total_ns": done_ns - post_start_ns,
            "initial_state": initial_state,
            "poll_count": poll_count,
        }
    finally:
        if handle is not None:
            agent.release_xfer_handle(handle)


def _base_sample(
    workload: Workload,
    path: str,
    region_count: int,
    block_bytes: int,
) -> dict[str, Any]:
    total_bytes = region_count * workload.request_blocks * block_bytes
    return {
        "path": path,
        "mapping": workload.pattern,
        "request_blocks": workload.request_blocks,
        "regions": region_count,
        "block_bytes": block_bytes,
        "total_bytes": total_bytes,
        "forward_range_count": workload.forward_range_count,
        "direct_descriptor_count": region_count * workload.request_blocks,
        "estimated_direct_backend_ranges": (
            region_count * workload.forward_range_count
        ),
    }


def _run_direct(
    *,
    agent,
    local_prepped,
    remote_prepped,
    workload: Workload,
    region_count: int,
    physical_blocks: int,
    block_bytes: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    index_start_ns = start_ns
    local_indices = _descriptor_indices(
        workload.local_block_ids, region_count, physical_blocks
    )
    remote_indices = _descriptor_indices(
        workload.remote_block_ids, region_count, physical_blocks
    )
    index_end_ns = time.perf_counter_ns()
    transfer = _run_nixl_transfer(
        agent=agent,
        local_prepped=local_prepped,
        local_indices=local_indices,
        remote_prepped=remote_prepped,
        remote_indices=remote_indices,
        timeout_seconds=timeout_seconds,
    )
    done_ns = time.perf_counter_ns()
    sample = _base_sample(workload, "direct", region_count, block_bytes)
    sample.update(transfer)
    sample.update(
        {
            "chunks": 1,
            "packed_descriptor_count": 0,
            "source_index_build_ns": 0,
            "destination_index_build_ns": index_end_ns - index_start_ns,
            "pack_control_wait_ns": 0,
            "pack_gpu_ns": 0.0,
            "scatter_gpu_ns": 0.0,
            "data_path_ns": transfer["transfer_total_ns"],
            "wall_ns": done_ns - start_ns,
            "effective_gbps": (
                sample["total_bytes"] / transfer["transfer_total_ns"]
                if transfer["transfer_total_ns"]
                else None
            ),
        }
    )
    return sample


def _receive_target_message(connection, timeout_seconds: float) -> dict:
    if not connection.poll(timeout_seconds):
        raise TimeoutError("target did not respond within timeout")
    message = connection.recv()
    if message.get("type") == "error":
        raise RuntimeError(
            "target process failed:\n"
            + message.get("traceback", "unknown target error")
        )
    return message


def _run_packed(
    *,
    torch,
    connection,
    agent,
    local_prepped,
    remote_prepped,
    destination_arena,
    destination_region_ptrs,
    receive_staging,
    scatter_stream,
    kernel: str,
    workload: Workload,
    region_count: int,
    block_bytes: int,
    blocks_per_chunk: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    totals: defaultdict[str, float] = defaultdict(float)
    initial_states = []
    poll_count = 0
    chunks = _chunk_ranges(workload.request_blocks, blocks_per_chunk)

    for chunk_id, (chunk_start, chunk_end) in enumerate(chunks):
        remote_ids = workload.remote_block_ids[chunk_start:chunk_end]
        local_ids = workload.local_block_ids[chunk_start:chunk_end]
        chunk_blocks = chunk_end - chunk_start

        pack_wait_start_ns = time.perf_counter_ns()
        connection.send(
            {
                "type": "pack",
                "chunk_id": chunk_id,
                "remote_block_ids": remote_ids,
            }
        )
        packed = _receive_target_message(connection, timeout_seconds)
        pack_wait_end_ns = time.perf_counter_ns()
        if packed.get("type") != "packed" or packed.get("chunk_id") != chunk_id:
            raise RuntimeError(f"unexpected packed response: {packed!r}")
        expected_bytes = region_count * chunk_blocks * block_bytes
        if packed["used_bytes"] != expected_bytes:
            raise RuntimeError(
                f"target packed {packed['used_bytes']} bytes; expected "
                f"{expected_bytes}"
            )

        packed_index = np.asarray([chunk_blocks - 1], dtype=np.int32)
        transfer = _run_nixl_transfer(
            agent=agent,
            local_prepped=local_prepped,
            local_indices=packed_index,
            remote_prepped=remote_prepped,
            remote_indices=packed_index,
            timeout_seconds=timeout_seconds,
        )

        destination_index_start_ns = time.perf_counter_ns()
        local_ids_tensor = torch.tensor(
            local_ids, dtype=torch.int64, device=destination_arena.device
        )
        destination_index_end_ns = time.perf_counter_ns()
        scatter_gpu_ns = _time_gpu_copy(
            torch,
            scatter_stream,
            lambda: _launch_scatter(
                torch,
                kernel,
                destination_arena,
                destination_region_ptrs,
                local_ids_tensor,
                receive_staging,
                block_bytes,
            ),
        )

        totals["source_index_build_ns"] += packed["source_index_build_ns"]
        totals["destination_index_build_ns"] += (
            destination_index_end_ns - destination_index_start_ns
        )
        totals["pack_control_wait_ns"] += (
            pack_wait_end_ns - pack_wait_start_ns
        )
        totals["pack_gpu_ns"] += packed["pack_gpu_ns"]
        totals["scatter_gpu_ns"] += scatter_gpu_ns
        for field in (
            "make_xfer_ns",
            "post_xfer_ns",
            "poll_xfer_ns",
            "transfer_total_ns",
        ):
            totals[field] += transfer[field]
        initial_states.append(transfer["initial_state"])
        poll_count += int(transfer["poll_count"])

    done_ns = time.perf_counter_ns()
    data_path_ns = (
        totals["pack_gpu_ns"]
        + totals["transfer_total_ns"]
        + totals["scatter_gpu_ns"]
    )
    sample = _base_sample(workload, "packed", region_count, block_bytes)
    sample.update(
        {
            "chunks": len(chunks),
            "packed_descriptor_count": len(chunks),
            "source_index_build_ns": totals["source_index_build_ns"],
            "destination_index_build_ns": totals[
                "destination_index_build_ns"
            ],
            "pack_control_wait_ns": totals["pack_control_wait_ns"],
            "pack_gpu_ns": totals["pack_gpu_ns"],
            "scatter_gpu_ns": totals["scatter_gpu_ns"],
            "make_xfer_ns": totals["make_xfer_ns"],
            "post_xfer_ns": totals["post_xfer_ns"],
            "poll_xfer_ns": totals["poll_xfer_ns"],
            "transfer_total_ns": totals["transfer_total_ns"],
            "data_path_ns": data_path_ns,
            "wall_ns": done_ns - start_ns,
            "effective_gbps": (
                sample["total_bytes"] / data_path_ns
                if data_path_ns
                else None
            ),
            "initial_state": ",".join(initial_states),
            "poll_count": poll_count,
        }
    )
    return sample


def _verify_destination(torch, destination_arena, workload: Workload) -> bool:
    physical_blocks = destination_arena.shape[1] - 1
    local_ids = torch.tensor(
        workload.local_block_ids,
        dtype=torch.int64,
        device=destination_arena.device,
    )
    selected = torch.index_select(
        destination_arena[:, :physical_blocks, :], 1, local_ids
    )
    regions = torch.arange(
        destination_arena.shape[0],
        dtype=torch.int64,
        device=destination_arena.device,
    )[:, None]
    remote_ids = torch.tensor(
        workload.remote_block_ids,
        dtype=torch.int64,
        device=destination_arena.device,
    )[None, :]
    expected = ((regions * 131 + remote_ids * 17) % PATTERN_MODULUS)
    selected_ok = bool(
        selected[:, :, 4:].eq(expected.to(torch.uint8)[..., None]).all()
    )
    selected_ok = selected_ok and bool(
        selected[:, :, 0].eq((remote_ids & 0xFF).to(torch.uint8)).all()
    )
    selected_ok = selected_ok and bool(
        selected[:, :, 1]
        .eq(((remote_ids >> 8) & 0xFF).to(torch.uint8))
        .all()
    )
    selected_ok = selected_ok and bool(
        selected[:, :, 2].eq((regions & 0xFF).to(torch.uint8)).all()
    )
    selected_ok = selected_ok and bool(
        selected[:, :, 3]
        .eq(((regions >> 8) & 0xFF).to(torch.uint8))
        .all()
    )

    untouched_mask = torch.ones(
        physical_blocks, dtype=torch.bool, device=destination_arena.device
    )
    untouched_mask[local_ids] = False
    untouched_first_bytes = destination_arena[
        :, :physical_blocks, 0
    ][:, untouched_mask]
    untouched_ok = bool(untouched_first_bytes.eq(DESTINATION_SENTINEL).all())
    return selected_ok and untouched_ok


def _percentile(values: list[int | float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summaries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (sample["mapping"], sample["request_blocks"], sample["path"])
        groups[key].append(sample)

    timing_fields = (
        "source_index_build_ns",
        "destination_index_build_ns",
        "pack_control_wait_ns",
        "pack_gpu_ns",
        "make_xfer_ns",
        "post_xfer_ns",
        "poll_xfer_ns",
        "transfer_total_ns",
        "scatter_gpu_ns",
        "data_path_ns",
        "wall_ns",
    )
    output = []
    for key in sorted(groups):
        group = groups[key]
        summary: dict[str, Any] = {
            "record_type": "summary",
            "mapping": key[0],
            "request_blocks": key[1],
            "path": key[2],
            "sample_count": len(group),
            "regions": group[0]["regions"],
            "block_bytes": group[0]["block_bytes"],
            "total_bytes": group[0]["total_bytes"],
            "forward_range_count": group[0]["forward_range_count"],
            "estimated_direct_backend_ranges": group[0][
                "estimated_direct_backend_ranges"
            ],
            "chunks": group[0]["chunks"],
        }
        for field in timing_fields:
            values = [sample[field] for sample in group]
            summary[f"{field}_median"] = statistics.median(values)
            summary[f"{field}_p95"] = _percentile(values, 0.95)
        bandwidths = [sample["effective_gbps"] for sample in group]
        summary["effective_gbps_median"] = statistics.median(bandwidths)
        summary["effective_gbps_p95"] = _percentile(bandwidths, 0.95)
        output.append(summary)
    return output


def _distribution_version() -> str:
    versions = []
    for distribution in ("nixl", "nixl-cu12", "nixl-cu13"):
        try:
            versions.append(
                f"{distribution}={importlib.metadata.version(distribution)}"
            )
        except importlib.metadata.PackageNotFoundError:
            pass
    return ",".join(versions) if versions else "unknown"


@contextmanager
def _output_stream(path: str, overwrite: bool) -> Iterator[TextIO]:
    if path == "-":
        yield sys.stdout
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as stream:
        yield stream


def _write_record(stream: TextIO, record: dict[str, Any]) -> None:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()


def _runtime_record(args, torch, kernel: str, blocks_per_chunk: int):
    initiator_properties = torch.cuda.get_device_properties(args.initiator_gpu)
    target_properties = torch.cuda.get_device_properties(args.target_gpu)
    return {
        "record_type": "metadata",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "nixl_distribution": _distribution_version(),
        "nixl_label": args.nixl_label,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "initiator_gpu": args.initiator_gpu,
        "initiator_gpu_name": initiator_properties.name,
        "target_gpu": args.target_gpu,
        "target_gpu_name": target_properties.name,
        "regions": args.regions,
        "physical_blocks": args.physical_blocks,
        "request_blocks": args.request_blocks,
        "patterns": args.patterns,
        "block_bytes": args.block_bytes,
        "staging_bytes": args.staging_mib * 1024 * 1024,
        "blocks_per_chunk": blocks_per_chunk,
        "requested_kernel": args.kernel,
        "resolved_kernel": kernel,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "mixed_run_length": args.mixed_run_length,
        "operation": "READ",
        "backend": "UCX",
        "pipeline": "sequential",
        "source_reference": SGLANG_STAGING_URL,
        "nixl_reference": SGLANG_NIXL_PR_URL,
    }


def _parse_csv_ints(parser, text: str, option: str) -> list[int]:
    try:
        values = [
            int(value.strip())
            for value in text.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error(f"{option} must be a comma-separated integer list")
    if not values or any(value <= 0 for value in values):
        parser.error(f"{option} must contain positive integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct NIXL READ with bounded GPU "
            "pack/READ/scatter for paged KV mappings."
        )
    )
    parser.add_argument("--initiator-gpu", type=int, default=4)
    parser.add_argument("--target-gpu", type=int, default=3)
    parser.add_argument("--regions", type=int, default=72)
    parser.add_argument("--physical-blocks", type=int, default=256)
    parser.add_argument(
        "--request-blocks",
        default="4,16,64",
        help="Comma-separated logical block counts.",
    )
    parser.add_argument(
        "--patterns",
        default="forward,mixed,fragmented",
        help="Comma-separated values from: " + ",".join(SUPPORTED_PATTERNS),
    )
    parser.add_argument("--mixed-run-length", type=int, default=4)
    parser.add_argument("--block-bytes", type=int, default=32 * 1024)
    parser.add_argument("--staging-mib", type=int, default=64)
    parser.add_argument(
        "--kernel", choices=("auto", "triton", "torch"), default="auto"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--nixl-label", default="auto")
    parser.add_argument(
        "--output", default="nixl_pack_scatter_results.jsonl"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.initiator_gpu == args.target_gpu:
        parser.error("initiator and target GPUs must be different")
    for name in ("regions", "physical_blocks", "block_bytes", "staging_mib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.block_bytes < 4:
        parser.error("--block-bytes must be at least 4 for data verification")
    if args.mixed_run_length <= 0:
        parser.error("--mixed-run-length must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    args.request_block_counts = _parse_csv_ints(
        parser, args.request_blocks, "--request-blocks"
    )
    if max(args.request_block_counts) > args.physical_blocks:
        parser.error("request block count cannot exceed --physical-blocks")
    args.pattern_names = [
        value.strip() for value in args.patterns.split(",") if value.strip()
    ]
    invalid_patterns = sorted(set(args.pattern_names) - set(SUPPORTED_PATTERNS))
    if not args.pattern_names or invalid_patterns:
        parser.error(f"invalid --patterns values: {invalid_patterns}")

    staging_bytes = args.staging_mib * 1024 * 1024
    logical_block_bytes = args.regions * args.block_bytes
    if staging_bytes < logical_block_bytes:
        parser.error(
            "staging buffer must fit at least one logical block across all "
            f"regions ({logical_block_bytes} bytes)"
        )
    return args


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import torch
    except ImportError:
        LOG.error("PyTorch is required")
        return 2
    if not torch.cuda.is_available():
        LOG.error("CUDA is not available")
        return 2
    if max(args.initiator_gpu, args.target_gpu) >= torch.cuda.device_count():
        LOG.error(
            "requested GPU index is unavailable; visible device count is %d",
            torch.cuda.device_count(),
        )
        return 2

    try:
        kernel = _resolve_kernel(args.kernel)
    except (ImportError, ModuleNotFoundError):
        LOG.error("--kernel triton requested, but Triton is unavailable")
        return 2

    staging_bytes = args.staging_mib * 1024 * 1024
    logical_block_bytes = args.regions * args.block_bytes
    blocks_per_chunk = staging_bytes // logical_block_bytes
    workloads = [
        _build_mapping(
            pattern,
            request_blocks,
            args.physical_blocks,
            args.seed ^ request_blocks,
            args.mixed_run_length,
        )
        for request_blocks in args.request_block_counts
        for pattern in args.pattern_names
    ]

    config = {
        "target_gpu": args.target_gpu,
        "regions": args.regions,
        "physical_blocks": args.physical_blocks,
        "block_bytes": args.block_bytes,
        "staging_bytes": staging_bytes,
        "blocks_per_chunk": blocks_per_chunk,
        "kernel": kernel,
    }
    context = mp.get_context("spawn")
    parent_connection, target_connection = context.Pipe()
    target_process = context.Process(
        target=_target_worker,
        args=(target_connection, config),
        name="nixl-pack-target",
    )
    target_process.start()
    target_connection.close()

    agent = None
    remote_name = None
    registrations = []
    prepped_handles = []
    samples: list[dict[str, Any]] = []
    try:
        target_message = _receive_target_message(
            parent_connection, args.timeout_seconds
        )
        if target_message.get("type") != "ready":
            raise RuntimeError(
                f"unexpected target ready message: {target_message!r}"
            )

        torch.cuda.set_device(args.initiator_gpu)
        device = torch.device("cuda", args.initiator_gpu)
        destination_arena = torch.full(
            (args.regions, args.physical_blocks + 1, args.block_bytes),
            DESTINATION_SENTINEL,
            dtype=torch.uint8,
            device=device,
        )
        receive_staging = torch.empty(
            staging_bytes, dtype=torch.uint8, device=device
        )
        destination_region_ptrs = _region_ptrs(torch, destination_arena)
        scatter_stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)

        agent = _make_agent("initiator")
        registrations.append(
            agent.register_memory(destination_arena, backends=["UCX"])
        )
        registrations.append(
            agent.register_memory(receive_staging, backends=["UCX"])
        )
        remote_name = agent.add_remote_agent(target_message["metadata"])
        if _agent_name_text(remote_name) != "target":
            raise RuntimeError(
                f"loaded unexpected remote agent name: {remote_name!r}"
            )
        parent_connection.send(
            {"type": "add_remote", "metadata": agent.get_agent_metadata()}
        )
        remote_ack = _receive_target_message(
            parent_connection, args.timeout_seconds
        )
        if (
            remote_ack.get("type") != "remote_added"
            or remote_ack.get("remote_name") != "initiator"
        ):
            raise RuntimeError(
                f"unexpected remote acknowledgement: {remote_ack!r}"
            )

        local_direct_descs = agent.get_xfer_descs(
            _make_kv_views(
                destination_arena,
                args.regions,
                args.physical_blocks,
                args.block_bytes,
            )
        )
        local_staging_descs = agent.get_xfer_descs(
            _make_staging_views(
                receive_staging,
                args.regions,
                blocks_per_chunk,
                args.block_bytes,
            )
        )
        remote_direct_descs = agent.deserialize_descs(
            target_message["source_descs"]
        )
        remote_staging_descs = agent.deserialize_descs(
            target_message["staging_descs"]
        )
        local_direct_prepped = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_direct_descs, backends=["UCX"]
        )
        remote_direct_prepped = agent.prep_xfer_dlist(
            remote_name, remote_direct_descs, backends=["UCX"]
        )
        local_staging_prepped = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_staging_descs, backends=["UCX"]
        )
        remote_staging_prepped = agent.prep_xfer_dlist(
            remote_name, remote_staging_descs, backends=["UCX"]
        )
        prepped_handles.extend(
            [
                local_direct_prepped,
                remote_direct_prepped,
                local_staging_prepped,
                remote_staging_prepped,
            ]
        )

        def run_cell(workload: Workload, path: str) -> dict[str, Any]:
            if path == "direct":
                return _run_direct(
                    agent=agent,
                    local_prepped=local_direct_prepped,
                    remote_prepped=remote_direct_prepped,
                    workload=workload,
                    region_count=args.regions,
                    physical_blocks=args.physical_blocks,
                    block_bytes=args.block_bytes,
                    timeout_seconds=args.timeout_seconds,
                )
            return _run_packed(
                torch=torch,
                connection=parent_connection,
                agent=agent,
                local_prepped=local_staging_prepped,
                remote_prepped=remote_staging_prepped,
                destination_arena=destination_arena,
                destination_region_ptrs=destination_region_ptrs,
                receive_staging=receive_staging,
                scatter_stream=scatter_stream,
                kernel=kernel,
                workload=workload,
                region_count=args.regions,
                block_bytes=args.block_bytes,
                blocks_per_chunk=blocks_per_chunk,
                timeout_seconds=args.timeout_seconds,
            )

        cells = [(workload, path) for workload in workloads
                 for path in SUPPORTED_PATHS]
        random_generator = random.Random(args.seed)
        with _output_stream(args.output, args.overwrite) as output:
            _write_record(
                output,
                _runtime_record(args, torch, kernel, blocks_per_chunk),
            )

            warmup_schedule = cells * args.warmup
            random_generator.shuffle(warmup_schedule)
            LOG.info("running %d warmup cells", len(warmup_schedule))
            for workload, path in warmup_schedule:
                run_cell(workload, path)

            LOG.info("running %d measured cells", args.repeats * len(cells))
            for repeat in range(args.repeats):
                schedule = list(cells)
                random_generator.shuffle(schedule)
                for workload, path in schedule:
                    sample = run_cell(workload, path)
                    sample.update(
                        {"record_type": "sample", "repeat": repeat}
                    )
                    samples.append(sample)
                    _write_record(output, sample)

            for workload, path in cells:
                destination_arena.fill_(DESTINATION_SENTINEL)
                torch.cuda.synchronize(device)
                run_cell(workload, path)
                torch.cuda.synchronize(device)
                passed = _verify_destination(
                    torch, destination_arena, workload
                )
                correctness = {
                    "record_type": "correctness",
                    "mapping": workload.pattern,
                    "request_blocks": workload.request_blocks,
                    "path": path,
                    "passed": passed,
                }
                _write_record(output, correctness)
                if not passed:
                    raise RuntimeError(
                        "data verification failed for "
                        f"{workload.pattern}/{workload.request_blocks}/{path}"
                    )

            for summary in _summaries(samples):
                summary["nixl_label"] = args.nixl_label
                summary["kernel"] = kernel
                _write_record(output, summary)

        LOG.info("benchmark complete; results written to %s", args.output)
        return 0
    except BaseException:
        LOG.error("benchmark failed:\n%s", traceback.format_exc())
        return 1
    finally:
        if agent is not None:
            for prepped in prepped_handles:
                try:
                    agent.release_dlist_handle(prepped)
                except BaseException:
                    pass
            if remote_name is not None:
                try:
                    agent.remove_remote_agent(remote_name)
                except BaseException:
                    pass
            for registration in registrations:
                try:
                    agent.deregister_memory(registration, backends=["UCX"])
                except BaseException:
                    pass

        if target_process.is_alive():
            try:
                parent_connection.send("stop")
            except BaseException:
                pass
        parent_connection.close()
        target_process.join(timeout=10)
        if target_process.is_alive():
            LOG.warning("target did not stop cleanly; terminating it")
            target_process.terminate()
            target_process.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
