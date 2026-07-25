# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small pull-mode PD proxy for NIXL transfer latency experiments."""

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from vllm.distributed.kv_transfer.pd_trace import trace_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = [
        {
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=f"http://{host}:{port}/v1",
                limits=httpx.Limits(max_connections=None,
                                    max_keepalive_connections=None),
            ),
            "host": host,
            "port": port,
        }
        for host, port in global_args.prefiller_instances
    ]
    app.state.decode_clients = [
        {
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=f"http://{host}:{port}/v1",
                limits=httpx.Limits(max_connections=None,
                                    max_keepalive_connections=None),
            ),
            "host": host,
            "port": port,
        }
        for host, port in global_args.decoder_instances
    ]
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    yield
    for info in app.state.prefill_clients + app.state.decode_clients:
        await info["client"].aclose()


app = FastAPI(lifespan=lifespan)

_BENCHMARK_REQUEST_SUFFIX = "-pdreq"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefiller-hosts", nargs="+", default=["127.0.0.1"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8100])
    parser.add_argument("--decoder-hosts", nargs="+", default=["127.0.0.1"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8200])
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("prefiller host/port counts differ")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("decoder host/port counts differ")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def _next_client(app: FastAPI, service_type: str) -> dict[str, Any]:
    if service_type == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iterator)]
    return app.state.decode_clients[next(app.state.decode_iterator)]


def _headers(request_id: str) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _prefill_payload(req_data: dict[str, Any]) -> dict[str, Any]:
    payload = req_data.copy()
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


async def _send_prefill(client_info: dict[str, Any], endpoint: str,
                        payload: dict[str, Any], request_id: str):
    trace_event("proxy_prefill_request_start", request_id, role="proxy")
    response = await client_info["client"].post(
        endpoint, json=payload, headers=_headers(request_id))
    response.raise_for_status()
    await response.aread()
    trace_event("proxy_prefill_request_end", request_id, role="proxy")
    return response


async def _stream_decode(client_info: dict[str, Any], endpoint: str,
                         payload: dict[str, Any], request_id: str):
    trace_event("proxy_decode_request_start", request_id, role="proxy")
    first_chunk = True
    async with client_info["client"].stream(
            "POST", endpoint, json=payload, headers=_headers(request_id)) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first_chunk:
                first_chunk = False
                trace_event("proxy_first_response_chunk", request_id, role="proxy")
            yield chunk
    trace_event("proxy_decode_request_end", request_id, role="proxy")


async def _handle(endpoint: str, request: Request):
    req_data = await request.json()
    client_request_id = request.headers.get("x-request-id")
    request_id = (
        f"{client_request_id}{_BENCHMARK_REQUEST_SUFFIX}"
        if client_request_id else str(uuid.uuid4())
    )
    trace_event("proxy_request_received", request_id, role="proxy", mode="pull")

    prefill_client = _next_client(request.app, "prefill")
    decode_client = _next_client(request.app, "decode")
    response = await _send_prefill(prefill_client, endpoint,
                                   _prefill_payload(req_data), request_id)

    decode_payload = req_data.copy()
    kv_transfer_params = response.json().get("kv_transfer_params", {})
    if kv_transfer_params:
        decode_payload["kv_transfer_params"] = kv_transfer_params

    async def generate():
        async for chunk in _stream_decode(decode_client, endpoint, decode_payload,
                                          request_id):
            yield chunk

    return StreamingResponse(generate(), media_type="application/json")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "mode": "pull"}


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
