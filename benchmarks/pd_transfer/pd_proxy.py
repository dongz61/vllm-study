# SPDX-License-Identifier: Apache-2.0
"""Minimal pull-mode PD proxy for transfer-latency experiments."""

import argparse
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from vllm.distributed.kv_transfer.pd_trace import trace_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BENCHMARK_REQUEST_SUFFIX = "-pdreq"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefiller-host", default="127.0.0.1")
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-host", default="127.0.0.1")
    parser.add_argument("--decoder-port", type=int, default=8200)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_client = httpx.AsyncClient(
        timeout=None,
        base_url=(
            f"http://{global_args.prefiller_host}:"
            f"{global_args.prefiller_port}/v1"
        ),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    app.state.decode_client = httpx.AsyncClient(
        timeout=None,
        base_url=(
            f"http://{global_args.decoder_host}:{global_args.decoder_port}/v1"
        ),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    yield
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


app = FastAPI(lifespan=lifespan)


def _headers(request_id: str) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _prefill_payload(request_data: dict[str, Any]) -> dict[str, Any]:
    payload = request_data.copy()
    payload["stream"] = False
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return payload


async def _send_prefill(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
) -> httpx.Response:
    trace_event("proxy_prefill_start", request_id, role="proxy")
    response = await client.post(endpoint, json=payload, headers=_headers(request_id))
    response.raise_for_status()
    await response.aread()
    trace_event("proxy_prefill_end", request_id, role="proxy")
    return response


async def _stream_decode(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
):
    trace_event("proxy_decode_start", request_id, role="proxy")
    first_chunk = True
    async with client.stream(
        "POST", endpoint, json=payload, headers=_headers(request_id)
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first_chunk:
                first_chunk = False
                trace_event("proxy_first_response_chunk", request_id, role="proxy")
            yield chunk
    trace_event("proxy_decode_end", request_id, role="proxy")


async def _handle(endpoint: str, request: Request):
    request_data = await request.json()
    client_request_id = request.headers.get("x-request-id")
    request_id = (
        f"{client_request_id}{_BENCHMARK_REQUEST_SUFFIX}"
        if client_request_id
        else str(uuid.uuid4())
    )
    prompt = request_data.get("prompt")
    input_len = len(prompt) if isinstance(prompt, list) else None
    output_len = request_data.get("max_tokens")
    trace_event(
        "proxy_request_received",
        request_id,
        role="proxy",
        mode="pull",
        input_len=input_len,
        output_len=output_len,
    )

    response = await _send_prefill(
        request.app.state.prefill_client,
        endpoint,
        _prefill_payload(request_data),
        request_id,
    )
    decode_payload = request_data.copy()
    kv_transfer_params = response.json().get("kv_transfer_params", {})
    if not kv_transfer_params:
        raise RuntimeError("Prefill response did not contain kv_transfer_params")
    decode_payload["kv_transfer_params"] = kv_transfer_params

    return StreamingResponse(
        _stream_decode(
            request.app.state.decode_client,
            endpoint,
            decode_payload,
            request_id,
        ),
        media_type="text/event-stream",
    )


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "mode": "pull"}


if __name__ == "__main__":
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
