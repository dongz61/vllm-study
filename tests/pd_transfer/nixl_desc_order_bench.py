#!/usr/bin/env python3
"""Standalone NIXL descriptor-order microbenchmark.

This benchmark isolates the order-sensitive path used by vLLM's NIXL pull
connector. It prepares one contiguous GPU buffer as individually addressable
descriptors, then compares four cells:

    ascending / mitigation OFF
    descending / mitigation OFF
    ascending / mitigation ON
    descending / mitigation ON

The mitigation turns paired reverse (-1) runs into paired forward (+1) runs
before descriptor indices are submitted to NIXL. Run this same file in
separate NIXL 0.6.0 and NIXL 1.3.2 environments to obtain the full 2 x 2 x 2
experiment.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


LOG = logging.getLogger("nixl-desc-order-bench")
PATTERN_MODULUS = 251
CELLS = (
    ("ascending", False),
    ("descending", False),
    ("ascending", True),
    ("descending", True),
)


def _load_nixl_api():
    """Import the API from both the 0.6 and current package layouts."""
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


def _canonicalize_paired_reverse_runs(
    local_block_ids: list[int],
    remote_block_ids: list[int],
) -> tuple[list[int], list[int], int, int]:
    """Turn paired -1 runs into paired +1 runs.

    This intentionally mirrors the helper in vLLM's NIXL connector. Reversing
    both sides of a paired-reverse run preserves every local-to-remote block
    mapping while presenting ascending contiguous descriptors to NIXL.
    """
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError("local and remote block ID counts must match")

    reverse_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(len(local_block_ids) - 1):
        is_paired_reverse = (
            local_block_ids[index + 1] - local_block_ids[index] == -1
            and remote_block_ids[index + 1] - remote_block_ids[index] == -1
        )
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
            submitted_local_block_ids[start:end]
        )
        submitted_remote_block_ids[start:end] = reversed(
            submitted_remote_block_ids[start:end]
        )
        canonicalized_block_count += end - start

    return (
        submitted_local_block_ids,
        submitted_remote_block_ids,
        len(reverse_runs),
        canonicalized_block_count,
    )


def _sequence_direction(values: list[int]) -> str:
    if len(values) < 2:
        return "singleton"
    deltas = (right - left for left, right in zip(values, values[1:]))
    directions = set(deltas)
    if directions == {1}:
        return "ascending"
    if directions == {-1}:
        return "descending"
    return "mixed"


def _agent_name_text(agent_name: str | bytes) -> str:
    """Normalize binding return values without changing API-facing values."""
    if isinstance(agent_name, bytes):
        return agent_name.decode("utf-8")
    return agent_name


def _make_views(buffer, descriptor_count: int, descriptor_bytes: int):
    return [
        buffer.narrow(0, index * descriptor_bytes, descriptor_bytes)
        for index in range(descriptor_count)
    ]


def _fill_target_pattern(torch, buffer, descriptor_count: int, descriptor_bytes: int):
    values = (
        torch.arange(descriptor_count, device=buffer.device, dtype=torch.int64)
        % PATTERN_MODULUS
    ).to(torch.uint8)
    buffer.view(descriptor_count, descriptor_bytes).copy_(values[:, None])


def _verify_target_pattern(
    torch,
    buffer,
    descriptor_count: int,
    descriptor_bytes: int,
) -> bool:
    values = (
        torch.arange(descriptor_count, device=buffer.device, dtype=torch.int64)
        % PATTERN_MODULUS
    ).to(torch.uint8)
    return bool(
        buffer.view(descriptor_count, descriptor_bytes)
        .eq(values[:, None])
        .all()
        .item()
    )


def _target_worker(
    connection,
    gpu: int,
    descriptor_count: int,
    descriptor_bytes: int,
) -> None:
    """Own the remote buffer and keep the target NIXL agent alive."""
    agent = None
    reg_descs = None
    remote_name = None
    try:
        import torch

        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        total_bytes = descriptor_count * descriptor_bytes
        target_buffer = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        _fill_target_pattern(
            torch, target_buffer, descriptor_count, descriptor_bytes
        )
        torch.cuda.synchronize(device)

        agent = _make_agent("target")
        reg_descs = agent.register_memory(target_buffer, backends=["UCX"])
        target_views = _make_views(
            target_buffer, descriptor_count, descriptor_bytes
        )
        target_xfer_descs = agent.get_xfer_descs(target_views)

        connection.send(
            {
                "type": "ready",
                "metadata": agent.get_agent_metadata(),
                "serialized_descs": agent.get_serialized_descs(
                    target_xfer_descs
                ),
            }
        )

        while True:
            if connection.poll(0.01):
                message = connection.recv()
                if message == "stop":
                    break
                if (
                    isinstance(message, dict)
                    and message.get("type") == "add_remote"
                ):
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
            else:
                # This also gives configurations without an effective UCX
                # progress thread a chance to advance control traffic.
                agent.get_new_notifs()
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
        if agent is not None and reg_descs is not None:
            try:
                agent.deregister_memory(reg_descs, backends=["UCX"])
            except BaseException:
                pass
        connection.close()


def _input_blocks(order: str, descriptor_count: int) -> tuple[list[int], list[int]]:
    if order == "ascending":
        blocks = list(range(descriptor_count))
    elif order == "descending":
        blocks = list(range(descriptor_count - 1, -1, -1))
    else:
        raise ValueError(f"unsupported order: {order}")
    return blocks, list(blocks)


def _run_one(
    *,
    agent,
    local_prepped,
    remote_prepped,
    order: str,
    mitigation: bool,
    descriptor_count: int,
    descriptor_bytes: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    import numpy as np

    local_blocks, remote_blocks = _input_blocks(order, descriptor_count)

    start_ns = time.perf_counter_ns()
    canonicalize_start_ns = start_ns
    if mitigation:
        (
            submitted_local_blocks,
            submitted_remote_blocks,
            canonicalized_run_count,
            canonicalized_block_count,
        ) = _canonicalize_paired_reverse_runs(local_blocks, remote_blocks)
    else:
        submitted_local_blocks = local_blocks
        submitted_remote_blocks = remote_blocks
        canonicalized_run_count = 0
        canonicalized_block_count = 0
    canonicalize_end_ns = time.perf_counter_ns()

    index_build_start_ns = canonicalize_end_ns
    local_indices = np.asarray(submitted_local_blocks, dtype=np.int32)
    remote_indices = np.asarray(submitted_remote_blocks, dtype=np.int32)
    index_build_end_ns = time.perf_counter_ns()

    handle = None
    try:
        make_start_ns = index_build_end_ns
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
                    f"{order}/{('on' if mitigation else 'off')} transfer "
                    f"did not finish within {timeout_seconds}s "
                    f"(initial_state={initial_state}, last_state={state}, "
                    f"poll_count={poll_count})"
                )
        done_ns = time.perf_counter_ns()
    finally:
        if handle is not None:
            agent.release_xfer_handle(handle)

    total_bytes = descriptor_count * descriptor_bytes
    transfer_total_ns = done_ns - post_start_ns
    return {
        "input_order": order,
        "mitigation": "on" if mitigation else "off",
        "submitted_order": _sequence_direction(submitted_local_blocks),
        "descriptor_count": descriptor_count,
        "descriptor_bytes": descriptor_bytes,
        "total_bytes": total_bytes,
        "canonicalized_run_count": canonicalized_run_count,
        "canonicalized_block_count": canonicalized_block_count,
        "canonicalize_ns": canonicalize_end_ns - canonicalize_start_ns,
        "index_build_ns": index_build_end_ns - index_build_start_ns,
        "make_xfer_ns": make_end_ns - make_start_ns,
        "post_xfer_ns": post_end_ns - post_start_ns,
        "poll_xfer_ns": done_ns - post_end_ns,
        "transfer_total_ns": transfer_total_ns,
        "end_to_end_ns": done_ns - start_ns,
        "effective_gbps": (
            total_bytes / transfer_total_ns if transfer_total_ns else None
        ),
        "initial_state": initial_state,
        "poll_count": poll_count,
    }


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
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(sample["input_order"], sample["mitigation"])].append(sample)

    timing_fields = (
        "canonicalize_ns",
        "index_build_ns",
        "make_xfer_ns",
        "post_xfer_ns",
        "poll_xfer_ns",
        "transfer_total_ns",
        "end_to_end_ns",
    )
    output = []
    for order, mitigation in (
        ("ascending", "off"),
        ("descending", "off"),
        ("ascending", "on"),
        ("descending", "on"),
    ):
        group = groups[(order, mitigation)]
        summary: dict[str, Any] = {
            "record_type": "summary",
            "input_order": order,
            "mitigation": mitigation,
            "submitted_order": group[0]["submitted_order"],
            "sample_count": len(group),
            "descriptor_count": group[0]["descriptor_count"],
            "descriptor_bytes": group[0]["descriptor_bytes"],
            "total_bytes": group[0]["total_bytes"],
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


def _runtime_record(args, torch) -> dict[str, Any]:
    initiator_properties = torch.cuda.get_device_properties(
        args.initiator_gpu
    )
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
        "descriptor_count": args.descriptors,
        "descriptor_bytes": args.descriptor_bytes,
        "total_bytes": args.descriptors * args.descriptor_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "operation": "READ",
        "backend": "UCX",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ascending and descending NIXL descriptor submission with "
            "paired-reverse canonicalization OFF and ON."
        )
    )
    parser.add_argument("--initiator-gpu", type=int, default=4)
    parser.add_argument("--target-gpu", type=int, default=3)
    parser.add_argument("--descriptors", type=int, default=1024)
    parser.add_argument("--descriptor-bytes", type=int, default=32 * 1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--nixl-label",
        default="auto",
        help="Human-readable environment label, for example 0.6.0 or 1.3.2.",
    )
    parser.add_argument(
        "--output",
        default="nixl_desc_order_results.jsonl",
        help="JSONL output path, or '-' for stdout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args()

    if args.initiator_gpu == args.target_gpu:
        parser.error("initiator and target GPUs must be different")
    if args.descriptors <= 0:
        parser.error("--descriptors must be positive")
    if args.descriptor_bytes <= 0:
        parser.error("--descriptor-bytes must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
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

    context = mp.get_context("spawn")
    parent_connection, target_connection = context.Pipe()
    target_process = context.Process(
        target=_target_worker,
        args=(
            target_connection,
            args.target_gpu,
            args.descriptors,
            args.descriptor_bytes,
        ),
        name="nixl-target",
    )
    target_process.start()
    target_connection.close()

    agent = None
    reg_descs = None
    local_prepped = None
    remote_prepped = None
    remote_name = None
    samples: list[dict[str, Any]] = []
    try:
        if not parent_connection.poll(args.timeout_seconds):
            raise TimeoutError("target did not become ready")
        target_message = parent_connection.recv()
        if target_message.get("type") == "error":
            raise RuntimeError(
                "target initialization failed:\n"
                + target_message.get("traceback", "unknown error")
            )
        if target_message.get("type") != "ready":
            raise RuntimeError(f"unexpected target message: {target_message!r}")

        torch.cuda.set_device(args.initiator_gpu)
        device = torch.device("cuda", args.initiator_gpu)
        total_bytes = args.descriptors * args.descriptor_bytes
        local_buffer = torch.zeros(total_bytes, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)

        agent = _make_agent("initiator")
        reg_descs = agent.register_memory(local_buffer, backends=["UCX"])
        remote_name = agent.add_remote_agent(target_message["metadata"])
        if _agent_name_text(remote_name) != "target":
            raise RuntimeError(
                f"loaded unexpected remote agent name: {remote_name!r}"
            )
        # Current NIXL two-peer examples establish metadata in both directions.
        # NIXL 0.6.0 allowed this one-sided READ benchmark to work with only
        # target metadata loaded at the initiator, but using a symmetric
        # exchange avoids relying on that version-specific behavior.
        parent_connection.send(
            {"type": "add_remote", "metadata": agent.get_agent_metadata()}
        )
        if not parent_connection.poll(args.timeout_seconds):
            raise TimeoutError("target did not acknowledge initiator metadata")
        remote_ack = parent_connection.recv()
        if remote_ack.get("type") == "error":
            raise RuntimeError(
                "target metadata import failed:\n"
                + remote_ack.get("traceback", "unknown error")
            )
        if (
            remote_ack.get("type") != "remote_added"
            or remote_ack.get("remote_name") != "initiator"
        ):
            raise RuntimeError(
                f"unexpected target metadata acknowledgement: {remote_ack!r}"
            )

        local_views = _make_views(
            local_buffer, args.descriptors, args.descriptor_bytes
        )
        local_xfer_descs = agent.get_xfer_descs(local_views)
        remote_xfer_descs = agent.deserialize_descs(
            target_message["serialized_descs"]
        )
        local_prepped = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_xfer_descs, backends=["UCX"]
        )
        remote_prepped = agent.prep_xfer_dlist(
            remote_name, remote_xfer_descs, backends=["UCX"]
        )

        with _output_stream(args.output, args.overwrite) as output:
            _write_record(output, _runtime_record(args, torch))

            random_generator = random.Random(args.seed)
            warmup_schedule = list(CELLS) * args.warmup
            random_generator.shuffle(warmup_schedule)
            LOG.info("running %d warmup transfers", len(warmup_schedule))
            for order, mitigation in warmup_schedule:
                _run_one(
                    agent=agent,
                    local_prepped=local_prepped,
                    remote_prepped=remote_prepped,
                    order=order,
                    mitigation=mitigation,
                    descriptor_count=args.descriptors,
                    descriptor_bytes=args.descriptor_bytes,
                    timeout_seconds=args.timeout_seconds,
                )

            LOG.info("running %d measured transfers", args.repeats * len(CELLS))
            for repeat in range(args.repeats):
                schedule = list(CELLS)
                random_generator.shuffle(schedule)
                for order, mitigation in schedule:
                    sample = _run_one(
                        agent=agent,
                        local_prepped=local_prepped,
                        remote_prepped=remote_prepped,
                        order=order,
                        mitigation=mitigation,
                        descriptor_count=args.descriptors,
                        descriptor_bytes=args.descriptor_bytes,
                        timeout_seconds=args.timeout_seconds,
                    )
                    sample.update(
                        {"record_type": "sample", "repeat": repeat}
                    )
                    samples.append(sample)
                    _write_record(output, sample)

            # Verify each cell independently after timing. Clearing the local
            # buffer prevents an earlier successful cell from masking a later
            # incorrect descriptor mapping.
            for order, mitigation in CELLS:
                local_buffer.zero_()
                torch.cuda.synchronize(device)
                _run_one(
                    agent=agent,
                    local_prepped=local_prepped,
                    remote_prepped=remote_prepped,
                    order=order,
                    mitigation=mitigation,
                    descriptor_count=args.descriptors,
                    descriptor_bytes=args.descriptor_bytes,
                    timeout_seconds=args.timeout_seconds,
                )
                torch.cuda.synchronize(device)
                passed = _verify_target_pattern(
                    torch,
                    local_buffer,
                    args.descriptors,
                    args.descriptor_bytes,
                )
                correctness = {
                    "record_type": "correctness",
                    "input_order": order,
                    "mitigation": "on" if mitigation else "off",
                    "passed": passed,
                }
                _write_record(output, correctness)
                if not passed:
                    raise RuntimeError(
                        f"data verification failed for {order}, "
                        f"mitigation={'on' if mitigation else 'off'}"
                    )

            for summary in _summaries(samples):
                summary["nixl_label"] = args.nixl_label
                _write_record(output, summary)

        LOG.info("benchmark complete; results written to %s", args.output)
        return 0
    except BaseException:
        LOG.error("benchmark failed:\n%s", traceback.format_exc())
        return 1
    finally:
        if agent is not None:
            if local_prepped is not None:
                try:
                    agent.release_dlist_handle(local_prepped)
                except BaseException:
                    pass
            if remote_prepped is not None:
                try:
                    agent.release_dlist_handle(remote_prepped)
                except BaseException:
                    pass
            if remote_name is not None:
                try:
                    agent.remove_remote_agent(remote_name)
                except BaseException:
                    pass
            if reg_descs is not None:
                try:
                    agent.deregister_memory(reg_descs, backends=["UCX"])
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
