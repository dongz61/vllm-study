#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare NIXL and CUDA IPC data paths for a single-host P2-D1 handoff."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any


_CUDA_IPC_NUM_LAYERS = 36


def _parse_size(value: str) -> int:
    suffixes = {
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
        "kb": 1_000,
        "mb": 1_000_000,
        "gb": 1_000_000_000,
        "b": 1,
    }
    normalized = value.strip().lower()
    for suffix, multiplier in suffixes.items():
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            return int(float(number) * multiplier)
    return int(normalized)


def _parse_sizes(value: str, total_bytes: int) -> list[int]:
    sizes = []
    for item in value.split(","):
        item = item.strip()
        sizes.append(total_bytes if item == "all" else _parse_size(item))
    if not sizes or any(size <= 0 or size > total_bytes for size in sizes):
        raise ValueError("descriptor sizes must be in (0, bytes_per_producer]")
    return list(dict.fromkeys(sizes))


def _wait_for(paths: Sequence[Path], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not all(path.exists() for path in paths):
        if time.monotonic() >= deadline:
            missing = ", ".join(str(path) for path in paths if not path.exists())
            raise TimeoutError(f"timed out waiting for {missing}")
        time.sleep(0.01)


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(temporary, path)


def _agent_config(backend: str, num_threads: int, telemetry: bool):
    from nixl._api import nixl_agent_config

    if backend == "UCX":
        return nixl_agent_config(
            num_threads=num_threads,
            capture_telemetry=telemetry,
        )
    return nixl_agent_config(
        backends=[backend],
        capture_telemetry=telemetry,
    )


def _export_cuda_ipc_allocation(tensor) -> dict[str, Any]:
    from torch.multiprocessing.reductions import reduce_tensor

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.cuda_ipc_gather import (
        _unwrap_torch_cuda_ipc_handle,
    )

    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("CUDA IPC benchmark buffers must be contiguous CUDA tensors")

    _, rebuild_args = reduce_tensor(tensor)
    if len(rebuild_args) < 10:
        raise RuntimeError("Unexpected torch CUDA IPC reduction tuple")

    tensor_offset_elements = int(rebuild_args[3])
    allocation_handle = _unwrap_torch_cuda_ipc_handle(bytes(rebuild_args[7]))
    allocation_size_bytes = int(rebuild_args[8])
    storage_offset_bytes = int(rebuild_args[9])
    data_offset_bytes = (
        storage_offset_bytes + tensor_offset_elements * tensor.element_size()
    )
    region_size_bytes = tensor.numel() * tensor.element_size()
    if data_offset_bytes + region_size_bytes > allocation_size_bytes:
        raise RuntimeError("CUDA IPC benchmark tensor exceeds its allocation")
    return {
        "handle": allocation_handle,
        "data_offset_bytes": data_offset_bytes,
        "region_size_bytes": region_size_bytes,
        "allocation_size_bytes": allocation_size_bytes,
    }


def _cuda_ipc_geometry(
    bytes_per_producer: int,
    remote_block_bytes: int,
) -> tuple[int, int]:
    bytes_per_layer, remainder = divmod(
        bytes_per_producer, _CUDA_IPC_NUM_LAYERS
    )
    num_request_blocks, block_remainder = divmod(
        bytes_per_layer, remote_block_bytes
    )
    if remainder or block_remainder or num_request_blocks <= 0:
        raise ValueError(
            "CUDA IPC gather requires bytes_per_producer to be a positive "
            f"multiple of {_CUDA_IPC_NUM_LAYERS} layers * "
            "remote_block_bytes"
        )
    return bytes_per_layer, num_request_blocks


def _descriptor_array(
    base_address: int,
    total_bytes: int,
    descriptor_bytes: int,
    device: int,
):
    import numpy as np

    count = (total_bytes + descriptor_bytes - 1) // descriptor_bytes
    offsets = np.arange(count, dtype=np.uint64) * descriptor_bytes
    lengths = np.minimum(descriptor_bytes, total_bytes - offsets)
    devices = np.full(count, device, dtype=np.uint64)
    return np.column_stack((base_address + offsets, lengths, devices))


def _descriptor_indices(count: int, layout: str):
    import numpy as np

    if layout == "contiguous":
        return np.arange(count, dtype=np.int32)
    if layout == "interleaved":
        evens = np.arange(0, count, 2, dtype=np.int32)
        odds = np.arange(1, count, 2, dtype=np.int32)
        return np.concatenate((evens, odds))
    raise ValueError(f"unknown layout: {layout}")


def _telemetry_dict(agent, handle) -> dict[str, Any]:
    telemetry = agent.get_xfer_telemetry(handle)
    post_duration_us = int(telemetry.postDuration)
    xfer_duration_us = int(telemetry.xferDuration)
    return {
        "start_time_us": int(telemetry.startTime),
        "post_duration_us": post_duration_us,
        "xfer_duration_us": xfer_duration_us,
        "data_duration_us": max(0, xfer_duration_us - post_duration_us),
        "total_bytes": int(telemetry.totalBytes),
        "desc_count": int(telemetry.descCount),
    }


def _producer(args: argparse.Namespace) -> None:
    import torch
    from nixl._api import nixl_agent

    torch.cuda.set_device(args.device)
    tensor = torch.empty(args.bytes, dtype=torch.uint8, device=f"cuda:{args.device}")
    tensor.fill_(args.pattern)
    torch.cuda.synchronize(args.device)

    agent = nixl_agent(
        f"nixl-p2d1-p{args.rank}-{uuid.uuid4().hex}",
        _agent_config(args.backend, args.num_threads, telemetry=True),
    )
    registration = agent.register_memory(tensor, backends=[args.backend])
    metadata = {
        "agent_metadata": agent.get_agent_metadata(),
        "base_address": tensor.data_ptr(),
        "bytes": args.bytes,
        "device": args.device,
        "pattern": args.pattern,
        "rank": args.rank,
    }
    if args.cuda_ipc_gather:
        metadata["cuda_ipc_allocation"] = _export_cuda_ipc_allocation(tensor)
    metadata_path = args.rendezvous / f"producer{args.rank}.pkl"
    stop_path = args.rendezvous / "stop"
    _atomic_pickle(metadata_path, metadata)
    print(
        json.dumps(
            {
                "event": "producer_ready",
                "rank": args.rank,
                "device": args.device,
                "bytes": args.bytes,
            }
        ),
        flush=True,
    )

    try:
        _wait_for([stop_path], args.timeout_seconds)
    finally:
        agent.deregister_memory(registration, backends=[args.backend])


def _wait_for_transfers(
    agent,
    handles: list[Any],
    initial_states: list[str],
    submit_done_ns: list[int],
    timeout_s: float,
) -> list[int]:
    completion_ns = [0] * len(handles)
    pending = set(range(len(handles)))
    for index, state in enumerate(initial_states):
        if state == "ERR":
            raise RuntimeError(f"transfer {index} failed during post")
        if state == "DONE":
            completion_ns[index] = submit_done_ns[index]
            pending.remove(index)

    deadline = time.monotonic() + timeout_s
    while pending:
        for index in tuple(pending):
            state = agent.check_xfer_state(handles[index])
            if state == "ERR":
                raise RuntimeError(f"transfer {index} entered ERR state")
            if state == "DONE":
                completion_ns[index] = time.perf_counter_ns()
                pending.remove(index)
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for NIXL transfers")
    return completion_ns


def _run_iteration(
    agent,
    local_sides: list[Any],
    remote_sides: list[Any],
    indices,
    backend: str,
    timeout_s: float,
    skip_desc_merge: bool,
) -> dict[str, Any]:
    handles = []
    ranks = []
    request_start_ns = time.perf_counter_ns()
    try:
        for rank, (local_side, remote_side) in enumerate(
            zip(local_sides, remote_sides, strict=True)
        ):
            make_start_ns = time.perf_counter_ns()
            handle = agent.make_prepped_xfer(
                "READ",
                local_side,
                indices,
                remote_side,
                indices,
                backends=[backend],
                skip_desc_merge=skip_desc_merge,
            )
            make_done_ns = time.perf_counter_ns()
            state = agent.transfer(handle)
            submit_done_ns = time.perf_counter_ns()
            handles.append(handle)
            ranks.append(
                {
                    "rank": rank,
                    "make_prepped_ms": (make_done_ns - make_start_ns) / 1e6,
                    "transfer_call_ms": (submit_done_ns - make_done_ns) / 1e6,
                    "submit_done_ns": submit_done_ns,
                    "initial_state": state,
                }
            )

        completion_ns = _wait_for_transfers(
            agent,
            handles,
            [rank["initial_state"] for rank in ranks],
            [rank["submit_done_ns"] for rank in ranks],
            timeout_s,
        )
        all_done_ns = max(completion_ns)
        first_submit_ns = min(rank["submit_done_ns"] for rank in ranks)
        last_submit_ns = max(rank["submit_done_ns"] for rank in ranks)
        for rank, handle, done_ns in zip(
            ranks, handles, completion_ns, strict=True
        ):
            rank["completion_ns"] = done_ns
            rank["submit_to_done_ms"] = (
                done_ns - rank["submit_done_ns"]
            ) / 1e6
            rank["backend"] = agent.query_xfer_backend(handle)
            rank["telemetry"] = _telemetry_dict(agent, handle)
        return {
            "request_total_ms": (all_done_ns - request_start_ns) / 1e6,
            "first_to_last_submit_ms": (last_submit_ns - first_submit_ns) / 1e6,
            "first_submit_to_all_done_ms": (all_done_ns - first_submit_ns) / 1e6,
            "last_submit_to_all_done_ms": (all_done_ns - last_submit_ns) / 1e6,
            "ranks": ranks,
        }
    finally:
        for handle in handles:
            agent.release_xfer_handle(handle)


def _validate(destination, bytes_per_producer: int, patterns: list[int]) -> None:
    import torch

    for rank, pattern in enumerate(patterns):
        view = destination[
            rank * bytes_per_producer : (rank + 1) * bytes_per_producer
        ]
        if not bool(torch.all(view == pattern).item()):
            mismatch = int(torch.count_nonzero(view != pattern).item())
            raise RuntimeError(
                f"rank {rank} validation failed: {mismatch} bytes differ"
            )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "request_total_ms",
        "first_to_last_submit_ms",
        "first_submit_to_all_done_ms",
        "last_submit_to_all_done_ms",
    )
    summary: dict[str, Any] = {"iterations": len(records)}
    for field in fields:
        values = [float(record[field]) for record in records]
        summary[field] = {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    for rank in range(2):
        rank_summary = {}
        for field in ("make_prepped_ms", "transfer_call_ms", "submit_to_done_ms"):
            values = [float(record["ranks"][rank][field]) for record in records]
            rank_summary[field] = {
                "mean": statistics.fmean(values),
                "p50": statistics.median(values),
                "p95": _percentile(values, 0.95),
            }
        for field in (
            "post_duration_us",
            "data_duration_us",
            "xfer_duration_us",
            "desc_count",
        ):
            values = [
                float(record["ranks"][rank]["telemetry"][field])
                for record in records
            ]
            rank_summary[field] = {
                "mean": statistics.fmean(values),
                "p50": statistics.median(values),
                "p95": _percentile(values, 0.95),
            }
        summary[f"rank{rank}"] = rank_summary
    return summary


def _summarize_cuda_ipc(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"iterations": len(records)}
    for field in (
        "request_total_ms",
        "launch_host_ms",
        "submit_to_done_observed_ms",
        "kernel_ms",
        "aggregate_remote_read_gbps",
    ):
        values = [float(record[field]) for record in records]
        summary[field] = {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _build_path_comparison(
    nixl_cases: list[dict[str, Any]],
    cuda_ipc_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    comparisons = []
    for cuda_case in cuda_ipc_summary["cases"]:
        nixl_case = next(
            (
                case
                for case in nixl_cases
                if case["layout"] == cuda_case["layout"]
                and case["descriptor_bytes"]
                == cuda_case["equivalent_descriptor_bytes"]
            ),
            None,
        )
        if nixl_case is None:
            continue
        nixl_metrics = nixl_case["metrics"]
        cuda_metrics = cuda_case["metrics"]
        nixl_request_ms = nixl_metrics["request_total_ms"]["mean"]
        cuda_request_ms = cuda_metrics["request_total_ms"]["mean"]
        comparisons.append(
            {
                "layout": cuda_case["layout"],
                "bytes_total": cuda_case["bytes_total"],
                "equivalent_descriptor_bytes": cuda_case[
                    "equivalent_descriptor_bytes"
                ],
                "equivalent_descriptors_per_rank": cuda_case[
                    "equivalent_descriptors_per_rank"
                ],
                "nixl_effective_descriptors_per_rank": nixl_metrics[
                    "rank0"
                ]["desc_count"]["mean"],
                "nixl_request_total_ms": nixl_request_ms,
                "nixl_transfer_call_sum_ms": (
                    nixl_metrics["rank0"]["transfer_call_ms"]["mean"]
                    + nixl_metrics["rank1"]["transfer_call_ms"]["mean"]
                ),
                "cuda_ipc_request_total_ms": cuda_request_ms,
                "cuda_ipc_launch_host_ms": cuda_metrics["launch_host_ms"][
                    "mean"
                ],
                "cuda_ipc_kernel_ms": cuda_metrics["kernel_ms"]["mean"],
                "cuda_ipc_speedup": nixl_request_ms / cuda_request_ms,
            }
        )
    return comparisons


def _validate_cuda_ipc_destination(
    destination,
    num_request_blocks: int,
    remote_block_bytes: int,
    patterns: list[int],
) -> None:
    import torch

    remote_half_bytes = remote_block_bytes // 2
    view = destination.view(
        _CUDA_IPC_NUM_LAYERS,
        num_request_blocks,
        2,
        2,
        remote_half_bytes,
    )
    for rank, pattern in enumerate(patterns):
        rank_view = view[:, :, :, rank, :]
        if not bool(torch.all(rank_view == pattern).item()):
            mismatch = int(torch.count_nonzero(rank_view != pattern).item())
            raise RuntimeError(
                f"CUDA IPC rank {rank} validation failed: "
                f"{mismatch} bytes differ"
            )


def _run_cuda_ipc_benchmark(
    args: argparse.Namespace,
    producers: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.cuda_ipc_gather import (
        CudaIpcGatherManager,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        CudaIpcRegion,
    )

    remote_block_bytes = args.cuda_ipc_remote_block_bytes
    if remote_block_bytes <= 0 or remote_block_bytes % 32:
        raise ValueError(
            "CUDA IPC remote block bytes must be positive and 32-byte aligned"
        )
    bytes_per_layer, num_request_blocks = _cuda_ipc_geometry(
        args.bytes, remote_block_bytes
    )
    local_block_bytes = remote_block_bytes * 2
    destination = torch.zeros(
        args.bytes * 2,
        dtype=torch.uint8,
        device=f"cuda:{args.device}",
    )
    local_bases = [
        destination.data_ptr() + layer * bytes_per_layer * 2
        for layer in range(_CUDA_IPC_NUM_LAYERS)
    ]
    setup_start_ns = time.perf_counter_ns()
    manager = CudaIpcGatherManager(args.device, local_bases)
    engine_id = "nixl-p2d1-microbench"
    try:
        for rank, producer in enumerate(producers):
            allocation = producer.get("cuda_ipc_allocation")
            if allocation is None:
                raise RuntimeError(
                    f"producer {rank} did not export a CUDA IPC allocation"
                )
            regions = [
                CudaIpcRegion(
                    handle=allocation["handle"],
                    data_offset_bytes=(
                        allocation["data_offset_bytes"]
                        + layer * bytes_per_layer
                    ),
                    region_size_bytes=bytes_per_layer,
                    allocation_size_bytes=allocation["allocation_size_bytes"],
                )
                for layer in range(_CUDA_IPC_NUM_LAYERS)
            ]
            manager.register_remote_regions(engine_id, rank, regions)
        torch.cuda.synchronize(args.device)
        setup_ms = (time.perf_counter_ns() - setup_start_ns) / 1e6

        output_path = args.output_dir / "cuda_ipc_iterations.jsonl"
        summaries = []
        with output_path.open("w", encoding="utf-8") as output:
            for layout in args.layouts:
                remote_block_ids = _descriptor_indices(
                    num_request_blocks, layout
                ).tolist()
                local_block_ids = list(range(num_request_blocks))
                records = []
                for iteration in range(args.warmup + args.iterations):
                    if iteration == args.warmup + args.iterations - 1:
                        destination.zero_()
                        torch.cuda.synchronize(args.device)
                    request_start_ns = time.perf_counter_ns()
                    transfer = manager.launch(
                        engine_id=engine_id,
                        remote_request_id=f"{layout}-{iteration}",
                        remote_ranks=(0, 1),
                        local_block_ids=local_block_ids,
                        remote0_block_ids=remote_block_ids,
                        remote1_block_ids=remote_block_ids,
                        remote_block_bytes=remote_block_bytes,
                        local_block_bytes=local_block_bytes,
                    )
                    deadline = time.monotonic() + args.timeout_seconds
                    while not transfer.is_done():
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "timed out waiting for CUDA IPC gather"
                            )
                    done_observed_ns = time.perf_counter_ns()
                    kernel_ms = transfer.kernel_ms()
                    record = {
                        "path": "cuda_ipc_gather",
                        "layout": layout,
                        "iteration": iteration - args.warmup,
                        "warmup": iteration < args.warmup,
                        "request_total_ms": (
                            done_observed_ns - request_start_ns
                        )
                        / 1e6,
                        "launch_host_ms": transfer.launch_host_ms,
                        "submit_to_done_observed_ms": (
                            done_observed_ns - transfer.submit_ns
                        )
                        / 1e6,
                        "kernel_ms": kernel_ms,
                        "aggregate_remote_read_gbps": (
                            transfer.bytes_transferred / kernel_ms / 1e6
                        ),
                    }
                    if iteration >= args.warmup:
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                        records.append(record)

                _validate_cuda_ipc_destination(
                    destination,
                    num_request_blocks,
                    remote_block_bytes,
                    [item["pattern"] for item in producers],
                )
                case = {
                    "path": "cuda_ipc_gather",
                    "layout": layout,
                    "num_layers": _CUDA_IPC_NUM_LAYERS,
                    "num_request_blocks": num_request_blocks,
                    "remote_block_bytes": remote_block_bytes,
                    "local_block_bytes": local_block_bytes,
                    "bytes_per_producer": args.bytes,
                    "bytes_total": args.bytes * 2,
                    "equivalent_descriptor_bytes": remote_block_bytes // 2,
                    "equivalent_descriptors_per_rank": (
                        _CUDA_IPC_NUM_LAYERS * num_request_blocks * 2
                    ),
                    "metrics": _summarize_cuda_ipc(records),
                    "validation_errors": 0,
                }
                summaries.append(case)
                print(json.dumps(case), flush=True)
        return {
            "persistent_setup_ms": setup_ms,
            "cases": summaries,
        }
    finally:
        manager.close()


def _consumer(args: argparse.Namespace) -> None:
    import torch
    from nixl._api import nixl_agent

    metadata_paths = [
        args.rendezvous / "producer0.pkl",
        args.rendezvous / "producer1.pkl",
    ]
    _wait_for(metadata_paths, args.timeout_seconds)
    producers = [pickle.loads(path.read_bytes()) for path in metadata_paths]
    if any(item["bytes"] != args.bytes for item in producers):
        raise ValueError("consumer and producer buffer sizes do not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    cuda_ipc_summary = None
    if args.cuda_ipc_gather:
        cuda_ipc_summary = _run_cuda_ipc_benchmark(args, producers)

    destination = torch.zeros(
        args.bytes * 2,
        dtype=torch.uint8,
        device=f"cuda:{args.device}",
    )
    agent = nixl_agent(
        f"nixl-p2d1-d-{uuid.uuid4().hex}",
        _agent_config(args.backend, args.num_threads, telemetry=True),
    )
    registration = agent.register_memory(destination, backends=[args.backend])
    remote_names = [
        agent.add_remote_agent(item["agent_metadata"]) for item in producers
    ]
    descriptor_sizes = _parse_sizes(args.descriptor_sizes, args.bytes)
    output_path = args.output_dir / "iterations.jsonl"
    summaries = []

    try:
        with output_path.open("w", encoding="utf-8") as output:
            for layout in args.layouts:
                for descriptor_bytes in descriptor_sizes:
                    setup_start_ns = time.perf_counter_ns()
                    local_sides = []
                    remote_sides = []
                    descriptor_count = (
                        args.bytes + descriptor_bytes - 1
                    ) // descriptor_bytes
                    indices = _descriptor_indices(descriptor_count, layout)
                    try:
                        for rank, producer in enumerate(producers):
                            local_descs = _descriptor_array(
                                destination.data_ptr() + rank * args.bytes,
                                args.bytes,
                                descriptor_bytes,
                                args.device,
                            )
                            remote_descs = _descriptor_array(
                                producer["base_address"],
                                args.bytes,
                                descriptor_bytes,
                                producer["device"],
                            )
                            local_sides.append(
                                agent.prep_xfer_dlist(
                                    "NIXL_INIT_AGENT",
                                    local_descs,
                                    mem_type="VRAM",
                                    backends=[args.backend],
                                )
                            )
                            remote_sides.append(
                                agent.prep_xfer_dlist(
                                    remote_names[rank],
                                    remote_descs,
                                    mem_type="VRAM",
                                    backends=[args.backend],
                                )
                            )
                        setup_ms = (time.perf_counter_ns() - setup_start_ns) / 1e6
                        records = []
                        for iteration in range(args.warmup + args.iterations):
                            if iteration == args.warmup + args.iterations - 1:
                                destination.zero_()
                                torch.cuda.synchronize(args.device)
                            record = _run_iteration(
                                agent,
                                local_sides,
                                remote_sides,
                                indices,
                                args.backend,
                                args.timeout_seconds,
                                args.skip_desc_merge,
                            )
                            record.update(
                                {
                                    "layout": layout,
                                    "descriptor_bytes": descriptor_bytes,
                                    "input_descriptor_count": descriptor_count,
                                    "iteration": iteration - args.warmup,
                                    "warmup": iteration < args.warmup,
                                }
                            )
                            if iteration >= args.warmup:
                                output.write(json.dumps(record) + "\n")
                                output.flush()
                                records.append(record)
                        _validate(
                            destination,
                            args.bytes,
                            [item["pattern"] for item in producers],
                        )
                        case = {
                            "path": "nixl",
                            "layout": layout,
                            "descriptor_bytes": descriptor_bytes,
                            "input_descriptor_count": descriptor_count,
                            "dlist_setup_ms": setup_ms,
                            "metrics": _summarize(records),
                        }
                        summaries.append(case)
                        print(json.dumps(case), flush=True)
                    finally:
                        for side in local_sides + remote_sides:
                            agent.release_dlist_handle(side)

        summary = {
            "config": {
                "backend": args.backend,
                "bytes_per_producer": args.bytes,
                "consumer_device": args.device,
                "producer_devices": [item["device"] for item in producers],
                "layouts": args.layouts,
                "descriptor_sizes": descriptor_sizes,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "num_threads": args.num_threads,
                "skip_desc_merge": args.skip_desc_merge,
            },
            "cases": summaries,
        }
        if cuda_ipc_summary is not None:
            summary["cuda_ipc_gather"] = cuda_ipc_summary
            summary["comparison"] = _build_path_comparison(
                summaries, cuda_ipc_summary
            )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"results: {args.output_dir}", flush=True)
    finally:
        try:
            for remote_name in remote_names:
                agent.remove_remote_agent(remote_name)
        finally:
            try:
                agent.deregister_memory(registration, backends=[args.backend])
            finally:
                (args.rendezvous / "stop").touch()


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rendezvous", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bytes", type=_parse_size, required=True)
    parser.add_argument("--backend", default="UCX")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--cuda-ipc-gather", action="store_true")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    producer = subparsers.add_parser("producer")
    _common_parser(producer)
    producer.add_argument("--rank", type=int, choices=(0, 1), required=True)
    producer.add_argument("--pattern", type=int, choices=range(256), required=True)

    consumer = subparsers.add_parser("consumer")
    _common_parser(consumer)
    consumer.add_argument("--output-dir", type=Path, required=True)
    consumer.add_argument(
        "--descriptor-sizes",
        default="16KiB,64KiB,256KiB,1MiB,all",
    )
    consumer.add_argument(
        "--layouts",
        nargs="+",
        choices=("interleaved", "contiguous"),
        default=("interleaved", "contiguous"),
    )
    consumer.add_argument("--warmup", type=int, default=3)
    consumer.add_argument("--iterations", type=int, default=10)
    consumer.add_argument("--skip-desc-merge", action="store_true")
    consumer.add_argument(
        "--cuda-ipc-remote-block-bytes",
        type=_parse_size,
        default=32 << 10,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bytes <= 0 or args.num_threads < 0 or args.timeout_seconds <= 0:
        raise ValueError(
            "bytes and timeout must be positive; threads cannot be negative"
        )
    if args.role == "consumer" and (args.warmup < 0 or args.iterations <= 0):
        raise ValueError("warmup cannot be negative and iterations must be positive")
    args.rendezvous.mkdir(parents=True, exist_ok=True)
    if args.role == "producer":
        _producer(args)
    else:
        _consumer(args)


if __name__ == "__main__":
    main()
