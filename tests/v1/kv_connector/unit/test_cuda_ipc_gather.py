# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.cuda_ipc_gather import (
    _unwrap_torch_cuda_ipc_handle,
)


def test_unwrap_legacy_cuda_ipc_handle() -> None:
    raw_handle = bytes(range(64))

    assert _unwrap_torch_cuda_ipc_handle(raw_handle) == raw_handle


def test_unwrap_torch_v2_cuda_malloc_handle() -> None:
    raw_handle = bytes(range(64))

    assert _unwrap_torch_cuda_ipc_handle(b"\x02c" + raw_handle) == raw_handle


@pytest.mark.parametrize(
    ("handle", "error"),
    [
        (b"\x02e" + bytes(range(64)), "expandable-segment"),
        (b"\x03c" + bytes(range(64)), "version 3"),
        (b"\x02x" + bytes(range(64)), "handle type"),
        (b"\x02cshort", "expected 64 payload bytes"),
    ],
)
def test_reject_unsupported_torch_cuda_ipc_handle(
    handle: bytes, error: str
) -> None:
    with pytest.raises(RuntimeError, match=error):
        _unwrap_torch_cuda_ipc_handle(handle)
