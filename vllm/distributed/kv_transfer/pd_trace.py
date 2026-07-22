# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight JSONL tracing for PD transfer experiments.

Tracing is disabled unless ``VLLM_PD_TRACE_PATH`` is set. This module is
best-effort: tracing must not affect serving correctness.
"""

import atexit
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, TextIO

_TRACE_PATH_ENV = "VLLM_PD_TRACE_PATH"
_TRACE_ROLE_ENV = "VLLM_PD_TRACE_ROLE"

_lock = threading.Lock()
_hostname = socket.gethostname()
_trace_files: dict[tuple[int, Path], TextIO] = {}


def _close_trace_files() -> None:
    with _lock:
        for trace_file in _trace_files.values():
            try:
                trace_file.close()
            except Exception:
                pass
        _trace_files.clear()


atexit.register(_close_trace_files)


def is_trace_enabled() -> bool:
    """Return whether PD experiment tracing is enabled for this process."""
    return bool(os.getenv(_TRACE_PATH_ENV))


def _resolve_trace_path() -> Path | None:
    trace_path = os.getenv(_TRACE_PATH_ENV)
    if not trace_path:
        return None

    path = Path(trace_path)
    if trace_path.endswith(os.sep) or path.is_dir():
        role = os.getenv(_TRACE_ROLE_ENV, "unknown")
        path = path / f"{role}-{os.getpid()}.jsonl"
    return path


def trace_event(
    event: str,
    request_id: str | None = None,
    role: str | None = None,
    **fields: Any,
) -> None:
    path = _resolve_trace_path()
    if path is None:
        return

    record: dict[str, Any] = {
        "ts_ns": time.time_ns(),
        "perf_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "host": _hostname,
        "role": role or os.getenv(_TRACE_ROLE_ENV, "unknown"),
        "event": event,
    }
    if request_id is not None:
        record["request_id"] = request_id
    record.update(fields)

    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            key = (os.getpid(), path)
            trace_file = _trace_files.get(key)
            if trace_file is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                trace_file = path.open("a", encoding="utf-8", buffering=1)
                _trace_files[key] = trace_file
            trace_file.write(line + "\n")
    except Exception:
        return
