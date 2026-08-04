# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
import os
import tempfile
import textwrap
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import ray
import torch

from vllm import LLM
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorStats)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    KVConnectorRole, NixlAgentMetadata, NixlConnector, NixlConnectorMetadata,
    NixlConnectorWorker, NixlKVConnectorStats, _analyze_block_pairs,
    _canonicalize_paired_reverse_runs, _NixlTransferTraceState,
    _PackedSourceRequestState, _should_use_packed_path)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_shutdown, has_kv_transfer_group)
from vllm.forward_context import ForwardContext
from vllm.platforms.interface import Platform
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from .utils import create_request, create_scheduler, create_vllm_config


@pytest.fixture(scope="module", autouse=True)
def clear_kv_transfer():
    """
    The test cases in this file use `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
    causing the global variable `_KV_CONNECTOR_AGENT` 
    to be assigned but never deleted.

    Since the current pytest process does not terminate and instead 
    continues running tests from other files,
    this global variable remains in memory and interferes 
    with test cases in other modules.
    
    So we use this fixture to ensure that the global variable
    `_KV_CONNECTOR_AGENT` is properly cleaned up after each test.
    """
    yield
    if has_kv_transfer_group():
        ensure_kv_transfer_shutdown()


class FakeNixlWrapper:
    """Mock implementation of NixlWrapper for testing.

    We don't inherit from nixl._api.nixl_agent because nixl may not be
    installed.
    
    Note: The complete source of this class is also used in the
    `_make_fake_nixl_pkg` function to create a fake nixl package
    for Ray workers.
    """

    AGENT_METADATA = b"fake_agent_metadata"
    REMOTE_AGENT_NAME = "remote_agent"

    def __init__(self, agent_name: str, *args, **kwargs):
        self._cycles_before_xfer_done = 0
        self._check_xfer_state_cycles: defaultdict[int, int] = defaultdict(
            lambda: 0)

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in caches_data]

    def register_memory(self, descs, backends) -> None:
        pass

    def deregister_memory(self, descs) -> None:
        pass

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in blocks_data]

    def prep_xfer_dlist(self, agent_name: str, descs: list) -> int:
        return uuid.uuid4().int

    def get_agent_metadata(self) -> bytes:
        return self.AGENT_METADATA

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return self.REMOTE_AGENT_NAME

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        # Used to collect done_sending, which we don't test yet.
        return {}

    def check_xfer_state(self, handle: int) -> str:
        if self._check_xfer_state_cycles[
                handle] >= self._cycles_before_xfer_done:
            return "DONE"
        self._check_xfer_state_cycles[handle] += 1
        return "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def release_dlist_handle(self, handle: int) -> None:
        pass

    def remove_remote_agent(self, agent: str) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def make_prepped_xfer(self,
                          xfer_type: str,
                          local_xfer_side_handle: int,
                          local_block_descs_ids: list[int],
                          remote_xfer_side_handle: int,
                          remote_block_descs_ids: list[int],
                          notif_msg: Optional[bytes] = None) -> int:
        return uuid.uuid4().int

    def transfer(self, handle: int) -> str:
        return "PROC"

    ############################################################
    # Follow are for changing the behavior during testing.
    ############################################################

    def set_cycles_before_xfer_done(self, cycles: int):
        """Set the number of cycles before a transfer is considered done."""


@contextlib.contextmanager
def _make_fake_nixl_pkg():
    """Context manager that creates a temporary package making
       `from nixl._api import nixl_agent` resolve to our FakeNixlWrapper.
       
    Automatically cleans up the temporary directory when done.
    """
    with tempfile.TemporaryDirectory() as td:
        pkg_root = os.path.join(td, "nixl", "_api")
        os.makedirs(pkg_root, exist_ok=True)

        # Get the source code of FakeNixlWrapper class and dedent it
        fake_nixl_source = inspect.getsource(FakeNixlWrapper)
        fake_nixl_source = textwrap.dedent(fake_nixl_source)

        stub = f"""\
# Copy of FakeNixlWrapper implementation for Ray workers
import uuid
from collections import defaultdict
from typing import Optional

{fake_nixl_source}

# Export as nixl_agent
nixl_agent = FakeNixlWrapper
"""
        with open(os.path.join(pkg_root, "__init__.py"), "w") as f:
            f.write(stub)

        # touch parent package
        open(os.path.join(td, "nixl", "__init__.py"), "w").close()
        yield td


def test_basic_interface():
    """Unit test for basic NixlConnector interface functionality."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(request_id=1,
                             block_size=BLOCK_SIZE,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv
    req_meta = kv_connector_metadata.reqs_to_recv[request_id]

    for block_id, block in zip(
            req_meta.local_block_ids, scheduler.kv_cache_manager.coordinator.
            single_type_managers[0].req_to_blocks[request_id]):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """
    Test that we can handle case where prompt is < block.

    In this case, the P worker will still send remote_block_ids of the
    partial block. The D worker should schedule an async read
    in this case.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(request_id=1,
                             block_size=BLOCK_SIZE,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True,
                             num_remote_blocks=1)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0


class FakeNixlConnectorWorker(NixlConnectorWorker):

    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(self, *args, hand_shake_latency: float = 1.8, **kwargs):
        super().__init__(*args, **kwargs)
        self._hand_shake_latency = hand_shake_latency

    def _nixl_handshake(self, host: str, port: int, remote_tp_size: int,
                        expected_engine_id: str) -> dict[int, str]:
        # Mimic slow _nixl_handshake, as well as bypass zmq communication.
        time.sleep(self._hand_shake_latency)
        # These should've been done in register_kv_caches(), called by
        # gpu_model_runner. Here we just hardcode some dummy values.
        slot_size_bytes = 4096
        self.slot_size_per_layer = [slot_size_bytes]
        self.block_len_per_layer = [slot_size_bytes * self.block_size]
        self.num_blocks = 1
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        assert expected_engine_id == self.REMOTE_ENGINE_ID

        remote_agent_name = self.add_remote_agent(
            NixlAgentMetadata(
                engine_id=self.REMOTE_ENGINE_ID,
                agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                num_blocks=1,
                block_lens=self.block_len_per_layer,
                attn_backend_name=self.backend_name,
                # `self.kv_cache_layout` is only forced to HND when vllm engine
                # is started. We mock HND here.
                kv_cache_layout="HND",
            ),
            remote_tp_size=remote_tp_size)
        return {0: remote_agent_name}


class TestNixlHandshake:

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    def test_multi_xfer_one_engine(
        self,
        # dist_init is a fixture that initializes the distributed environment.
        dist_init):
        """Test case where multiple xfers are initiated to the same engine.
        
        This test triggers the connector to load remote KV for the same
        `request_id`. The transfer is not done immediately due to
        `set_cycles_before_xfer_done`, so there is a state where there are
        multiple transfer states for the same `request_id`, and `get_finished`
        should handle it correctly (wait for all transfers to be done).
        """
        vllm_config = create_vllm_config()

        request_id = "req_id"

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id, hand_shake_latency=0)
        assert isinstance(connector.connector_worker.nixl_wrapper,
                          FakeNixlWrapper)
        connector.connector_worker.nixl_wrapper.set_cycles_before_xfer_done(3)
        num_xfers = 4
        while True:
            # For the same request_id, initiate multiple xfers across different
            # round of `execute_model` calls.
            metadata = NixlConnectorMetadata()
            if num_xfers > 0:
                num_xfers -= 1
                metadata.add_new_req(
                    request_id=request_id,
                    local_block_ids=[
                        num_xfers + 1, num_xfers + 2, num_xfers + 3
                    ],
                    kv_transfer_params={
                        "remote_block_ids":
                        [num_xfers + 4, num_xfers + 5, num_xfers + 6],
                        "remote_engine_id":
                        FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                        "remote_host":
                        "localhost",
                        "remote_port":
                        1234,
                        "remote_tp_size":
                        1,
                    })
            connector.bind_connector_metadata(metadata)

            # Mimic maybe_setup_kv_connector in gpu_model_runner.
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, "start_load_kv took " \
                f"{_after_load - _before_load} seconds"

            # Mimic get_finished_kv_transfers in gpu_model_runner.
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                assert request_id in done_recving
                break

            connector.clear_connector_metadata()

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    @pytest.mark.parametrize("decode_tp_size, prefill_tp_size", [
        (1, 1),
        (2, 1),
        (4, 2),
        (4, 4),
    ])
    def test_async_load_kv(
            self,
            # Fixture that initializes the distributed environment.
            dist_init,
            # Simulate consumer-producer TP sizes.
            decode_tp_size,
            prefill_tp_size):
        """Test that NixlConnector's start_load_kv should be non-blocking."""

        vllm_config = create_vllm_config()
        vllm_config.parallel_config.tensor_parallel_size = decode_tp_size

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id)
        metadata = NixlConnectorMetadata()
        metadata.add_new_req(request_id="id",
                             local_block_ids=[1, 2, 3],
                             kv_transfer_params={
                                 "remote_block_ids": [4, 5, 6],
                                 "remote_engine_id":
                                 FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                                 "remote_host": "localhost",
                                 "remote_port": 1234,
                                 "remote_tp_size": prefill_tp_size,
                             })
        connector.bind_connector_metadata(metadata)

        timeout = 2.5
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, "start_load_kv took " \
                f"{_after_load - _before_load} seconds"
            time.sleep(0.5)  # backoff for the async handshake to complete.
            connector.bind_connector_metadata(NixlConnectorMetadata())
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                return
        raise TimeoutError("Took too long to complete async handshake.")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    def test_concurrent_load_kv(
        self,
        # dist_init is a fixture that initializes the distributed environment.
        dist_init):
        """Test that multiple start_load_kv calls should occur concurrently."""

        vllm_config = create_vllm_config()

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id)
        metadata = NixlConnectorMetadata()
        total_reqs = 5
        for i in range(total_reqs):
            metadata.add_new_req(request_id=f"id_{i}",
                                 local_block_ids=[1, 2, 3],
                                 kv_transfer_params={
                                     "remote_block_ids": [4, 5, 6],
                                     "remote_engine_id":
                                     FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                                     "remote_host": "localhost",
                                     "remote_port": 1234,
                                     "remote_tp_size": 1,
                                 })
        connector.bind_connector_metadata(metadata)

        timeout = 2.5 * total_reqs
        cnt_finished_reqs = 0
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, "start_load_kv took " \
                f"{_after_load - _before_load} seconds"
            time.sleep(0.5)  # backoff for the async handshake to complete.
            connector.bind_connector_metadata(NixlConnectorMetadata())
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                cnt_finished_reqs += len(done_recving)
                if cnt_finished_reqs == total_reqs:
                    return
        raise TimeoutError("Took too long to complete async handshake.")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    def test_handshake_fails_on_kv_cache_layout_mismatch(self, dist_init):
        """
        Verify that adding a remote agent fails if kv_cache_layout differs.
        This test is only relevant for heterogeneous TP.
        """
        vllm_config = create_vllm_config()

        # Mock TP world size to 2 to force heterogeneous TP when
        # remote_tp_size=1
        with patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.get_tensor_model_parallel_world_size",  # noqa: E501
                return_value=2):
            # Initialize connector and worker (with fake NIXL wrapper)
            connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
            connector.connector_worker = FakeNixlConnectorWorker(
                vllm_config, connector.engine_id, hand_shake_latency=0)
            worker = connector.connector_worker

            # Minimal local registration params used by add_remote_agent
            worker.slot_size_per_layer = [4096]
            worker.block_len_per_layer = [4096 * worker.block_size]
            worker.num_blocks = 1
            worker.dst_num_blocks[worker.engine_id] = worker.num_blocks

            # Metadata with different kv_cache_layout than local worker
            mismatched_layout = "HND" if worker.kv_cache_layout != "HND" \
                else "NHD"
            meta = NixlAgentMetadata(
                engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                num_blocks=1,
                block_lens=worker.block_len_per_layer,
                attn_backend_name=worker.backend_name,
                kv_cache_layout=mismatched_layout,
            )

            # We don't check layout for homogeneous TP and MLA for now, as the
            # whole block is moved.
            worker.add_remote_agent(meta, remote_tp_size=2)
            with pytest.raises(AssertionError):
                worker.add_remote_agent(meta, remote_tp_size=1)


# NOTE: resource cleanup in mp backend is a bit finicky, so the order in which
# we put here is important. First run ray, it will clean up the resources, then
# the rest of the tests.
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper)
def test_kv_connector_stats(dist_init):
    """Test that KV transfer stats are properly recorded and retrieved."""
    vllm_config = create_vllm_config()

    # Test worker role in decode server.
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(vllm_config,
                                                         connector.engine_id,
                                                         hand_shake_latency=0)

    # Verify that xfer_stats starts empty
    initial_stats = connector.get_kv_connector_stats()
    assert initial_stats is None

    # Create transfer metadata
    request_id = "test_req_for_stats"
    metadata = NixlConnectorMetadata()
    metadata.add_new_req(request_id=request_id,
                         local_block_ids=[1, 2, 3],
                         kv_transfer_params={
                             "remote_block_ids": [4, 5, 6],
                             "remote_engine_id":
                             FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                             "remote_host": "localhost",
                             "remote_port": 1234,
                             "remote_tp_size": 1,
                         })
    connector.bind_connector_metadata(metadata)

    # Start the transfer
    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
    )
    connector.start_load_kv(dummy_ctx)

    # Verify stats are recorded after transfer is complete
    max_iterations = 2
    # Clear metadata before start_load_kv to prevent reprocessing same request
    connector.bind_connector_metadata(NixlConnectorMetadata())
    for _ in range(max_iterations):
        # Need to call start_load_kv to process completed handshakes
        connector.start_load_kv(dummy_ctx)
        _, done_recving = connector.get_finished(finished_req_ids=set())
        if len(done_recving) > 0 and request_id in done_recving:
            break
        time.sleep(
            0.1)  # Small delay to allow background handshake to complete
    else:
        assert "Transfer did not complete within expected iterations"

    # Now check that stats were recorded
    stats_after_transfer = connector.get_kv_connector_stats()
    assert isinstance(stats_after_transfer, NixlKVConnectorStats)

    # Verify stats values are recorded
    assert not stats_after_transfer.is_empty()
    assert stats_after_transfer.data["num_successful_transfers"] == 1

    # Verify stats are reset after retrieval
    stats_after_reset = connector.get_kv_connector_stats()
    assert stats_after_reset is None


def test_kv_connector_stats_aggregation():
    """
    Test KV transfer stats aggregation across TP ranks using 
    KVOutputAggregator (used by MultiprocExecutor).
    """

    # Create KVOutputAggregator for 3 workers (simulating TP=3), same thing
    # done in MultiprocExecutor.execute_model
    aggregator = KVOutputAggregator(world_size=3)

    # Create stats for multiple workers with different transfer patterns
    worker1_stats = NixlKVConnectorStats()
    worker2_stats = NixlKVConnectorStats()
    worker3_stats = NixlKVConnectorStats()

    # Record different transfers on each worker
    # Worker 1: 2 transfers
    worker1_stats.record_transfer()
    worker1_stats.record_transfer()

    # Worker 2: 1 transfer
    worker2_stats.record_transfer()

    # Worker 3: 3 transfers
    worker3_stats.record_transfer()
    worker3_stats.record_transfer()
    worker3_stats.record_transfer()

    # Create ModelRunnerOutput instances for each worker
    worker_outputs = []
    for i, worker_stats in enumerate(
        [worker1_stats, worker2_stats, worker3_stats]):
        output = ModelRunnerOutput(
            req_ids=[f"req_{i}"],
            req_id_to_index={f"req_{i}": 0},
            sampled_token_ids=[[123]],  # dummy token
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(
                finished_sending=set([f"req_{i}_send"])
                if i < 2 else None,  # Workers 0,1 finished sending
                finished_recving=set([f"req_{i}_recv"])
                if i > 0 else None,  # Workers 1,2 finished receiving
                kv_connector_stats=worker_stats,
            ))
        worker_outputs.append(output)

    # Use the real aggregation mechanism (like MultiprocExecutor.execute_model)
    aggregated_output = aggregator.aggregate(worker_outputs, output_rank=0)
    kv_connector_stats = \
        aggregated_output.kv_connector_output.kv_connector_stats
    assert isinstance(kv_connector_stats, NixlKVConnectorStats)
    # Number of total transfers across all workers.
    assert kv_connector_stats.data["num_successful_transfers"] == 6


def test_multi_kv_connector_stats_aggregation():
    """
    Test MultiKVConnectorStats aggregation across TP ranks using
    KVOutputAggregator (used by MultiprocExecutor).
    """

    aggregator = KVOutputAggregator(world_size=3)

    from dataclasses import dataclass

    @dataclass
    class FooKVConnectorStats(KVConnectorStats):

        def reset(self):
            self.data = {"num_foo_transfers": 0}

        def record_transfer(self):
            if "num_foo_transfers" not in self.data:
                self.data["num_foo_transfers"] = 0
            self.data["num_foo_transfers"] += 1

        def is_empty(self) -> bool:
            return self.data["num_foo_transfers"] == 0

        def aggregate(self,
                      other: "FooKVConnectorStats") -> "FooKVConnectorStats":
            if not other.is_empty():
                self.data["num_foo_transfers"] += other.data[
                    "num_foo_transfers"]
            return self

    def make_multi_stats(nixl_count: int,
                         foo_count: int) -> MultiKVConnectorStats:
        data: dict[str, KVConnectorStats] = {}
        if nixl_count > 0:
            nixl_stats = NixlKVConnectorStats()
            for _ in range(nixl_count):
                nixl_stats.record_transfer()
            data["NixlConnector"] = nixl_stats
        if foo_count > 0:
            foo_stats = FooKVConnectorStats()
            for _ in range(foo_count):
                foo_stats.record_transfer()
            data["FooConnector"] = foo_stats
        return MultiKVConnectorStats(data=data)

    # Create heterogeneous stats across 3 workers
    worker_patterns = [(2, 1), (3, 0), (0, 5)]  # (Nixl, Foo)

    worker_outputs: list[ModelRunnerOutput] = []
    for i, (nixl, foo) in enumerate(worker_patterns):
        stats = make_multi_stats(nixl, foo)
        output = ModelRunnerOutput(
            req_ids=[f"req_{i}"],
            req_id_to_index={f"req_{i}": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(
                finished_sending=set([f"req_{i}_send"]) if i < 2 else None,
                finished_recving=set([f"req_{i}_recv"]) if i > 0 else None,
                kv_connector_stats=stats,
            ),
        )
        worker_outputs.append(output)

    aggregated_output = aggregator.aggregate(worker_outputs, output_rank=0)
    kv_connector_stats = \
        aggregated_output.kv_connector_output.kv_connector_stats
    assert isinstance(kv_connector_stats, MultiKVConnectorStats)

    # Validate per-connector totals across workers
    assert kv_connector_stats["NixlConnector"].data[
        "num_successful_transfers"] == 5
    assert kv_connector_stats["FooConnector"].data["num_foo_transfers"] == 6


@pytest.mark.parametrize("distributed_executor_backend", ["ray", None])
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper)
def test_abort_timeout_on_prefiller(monkeypatch, distributed_executor_backend):
    """
    Test lifecycle of an aborted Remote Prefill request hitting the timeout.
    -----> P 
            |  {process request}
     <-/--- |  {result is NOT delivered, eg proxy is down}
            |
            |
            |  {eventually free blocks}
    """
    model_name = "Qwen/Qwen3-0.6B"
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.5,
        "kv_transfer_config": kv_transfer_config,
        "distributed_executor_backend": distributed_executor_backend,
    }

    timeout = 6
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", str(timeout))

    # Build runtime_env only if we're using Ray
    if distributed_executor_backend == "ray":
        with _make_fake_nixl_pkg() as working_dir:
            runtime_env = {
                "working_dir": working_dir,  # ship fake nixl package
                "env_vars": {
                    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": str(timeout),
                },
            }
            ray.init(runtime_env=runtime_env)

            _run_abort_timeout_test(llm_kwargs, timeout)
    else:
        _run_abort_timeout_test(llm_kwargs, timeout)


def _run_abort_timeout_test(llm_kwargs: dict, timeout: int):
    """Helper function to run the abort timeout test logic."""
    llm = LLM(**llm_kwargs)
    remote_prefill_opts = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    # Simulate sidecar request
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"kv_transfer_params": remote_prefill_opts})
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0].req_to_blocks

    padding = "Just making this request a little longer so that we're sure "
    "we're not hitting the small-request lower bound beneath which we don't "
    "actually trigger the whole kv transfer, but rather just recompute the "
    "blocks on D."
    _ = llm.generate([f"What is the capital of Japan? {padding}"],
                     sampling_params)

    # Request finished but not freed
    assert '0' in scheduler.finished_req_ids and '0' in req_to_blocks
    # Some other request, 0 still not freed
    _ = llm.generate([f"What is the capital of Italy? {padding}"],
                     sampling_params)
    assert '0' in req_to_blocks
    assert '1' in scheduler.finished_req_ids and '1' in req_to_blocks

    # Wait for timeout and trigger another scheduler loop
    time.sleep(timeout)
    _ = llm.generate([f"What is the capital of France? {padding}"],
                     sampling_params)
    # Request-0 times out and is cleared!
    assert '0' not in req_to_blocks


def test_register_kv_caches(dist_init):
    """
    Test that register_kv_caches() properly calls nixl_wrapper methods with
    correct data.
    
    This test verifies:
    1. nixl_wrapper.get_reg_descs() is called with caches_data containing
       tensor metadata
    2. nixl_wrapper.get_xfer_descs() is called with blocks_data containing
       block layout info
    """

    vllm_config = create_vllm_config()

    # Create test kv cache tensors using proper backend shape
    kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(num_blocks=2,
                                                              block_size=16,
                                                              num_kv_heads=4,
                                                              head_size=64)
    shared_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    unique_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    kv_caches = {
        "layer0": shared_tensor,
        "layer1": unique_tensor,
        "layer2": shared_tensor,
    }

    # Store tensor info for validation
    expected_tensor_size = shared_tensor[0].element_size(
    ) * shared_tensor[0].numel()
    expected_base_addrs = [
        shared_tensor[0].data_ptr(), shared_tensor[1].data_ptr(),
        unique_tensor[0].data_ptr(), unique_tensor[1].data_ptr()
    ]

    with patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper") as mock_nixl_wrapper, \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.threading.Event"), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.threading.Thread"):  # noqa: E501

        # Create connector
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id, hand_shake_latency=0)

        # Get the mock instance
        mock_wrapper_instance = mock_nixl_wrapper.return_value
        connector.connector_worker.nixl_wrapper = mock_wrapper_instance

        # Execute register_kv_caches
        connector.register_kv_caches(kv_caches)

        # Verify get_reg_descs was called with caches_data
        assert mock_wrapper_instance.get_reg_descs.called
        caches_data, _ = mock_wrapper_instance.get_reg_descs.call_args[0]
        assert len(caches_data) == 4

        for i, cache_entry in enumerate(caches_data):
            base_addr, size, _tp_rank, _ = cache_entry
            assert size == expected_tensor_size, \
                f"Entry {i}: Expected tensor size {expected_tensor_size}, " \
                f"got {size}"
            assert base_addr == expected_base_addrs[i], \
                f"Entry {i}: Expected base address {expected_base_addrs[i]}, " \
                f"got {base_addr}"

        # Verify get_xfer_descs was called with blocks_data
        assert mock_wrapper_instance.get_xfer_descs.called
        blocks_data, _ = mock_wrapper_instance.get_xfer_descs.call_args[0]

        # Validate blocks_data structure and size
        expected_blocks_count = 8
        assert len(blocks_data) == expected_blocks_count, \
            f"Expected {expected_blocks_count} blocks, " \
            f"got {len(blocks_data)}"

        expected_block_len = expected_tensor_size // 2
        for i, block_entry in enumerate(blocks_data):
            block_start_addr, block_len, tp_rank = block_entry
            assert block_len == expected_block_len, \
                f"Block entry {i}: Expected block len {expected_block_len}, " \
                f"got {block_len}"


class FakePlatform(Platform):
    device_type: str = "oot"

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported 
        kv_buffer_device for nixl.
        """
        return {'oot': ('oot', )}

    @classmethod
    def get_nixl_memory_type(cls) -> Optional[str]:
        """
        Returns the nixl memory type for the current platform.
        """
        return 'VRAM'


@pytest.mark.parametrize("kv_buffer_device, nixl_memory_type", [
    ("oot", "VRAM"),
])
def test_kv_buffer_to_nixl_memory_types(dist_init, kv_buffer_device,
                                        nixl_memory_type):
    """
    Test that register_kv_caches() passes the correct memory types from the
    config to the nixl_wrapper.
    """
    vllm_config = create_vllm_config()
    # Override the default memory types in the config
    vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        _NIXL_SUPPORTED_DEVICE)
    _NIXL_SUPPORTED_DEVICE.update(FakePlatform.get_nixl_supported_devices())

    with patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper"), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.threading.Event"), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.threading.Thread"), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.current_platform", FakePlatform), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector._NIXL_SUPPORTED_DEVICE", _NIXL_SUPPORTED_DEVICE):  # noqa: E501

        # Create connector and replace its worker with a fake one for isolation
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)

        # Verify get_reg_descs was called with the correct memory_type
        assert connector.connector_worker.kv_buffer_device == kv_buffer_device
        assert connector.connector_worker.nixl_memory_type == nixl_memory_type


def test_pd_transfer_delay_is_non_blocking():
    worker = object.__new__(NixlConnectorWorker)
    worker._pd_transfer_sleep_ms = 80
    worker._pd_trace_enabled = False
    worker._delayed_recving_transfers = {}
    worker.nixl_wrapper = FakeNixlWrapper("test")
    worker.xfer_stats = NixlKVConnectorStats()
    transfers = {
        "req1": [(1, 0.0)],
        "req2": [(2, 0.0)],
    }
    clock_values = [10.0, 10.0, 10.079, 10.081]
    module = "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"

    with patch(f"{module}.time.perf_counter",
               side_effect=clock_values), \
            patch(f"{module}.time.sleep") as sleep_mock, \
            patch(f"{module}.trace_event"):
        assert worker._pop_done_transfers(transfers) == set()
        assert transfers == {}
        deadlines = {
            ready_at
            for ready_at, _ in worker._delayed_recving_transfers.values()
        }
        assert len(deadlines) == 1

        assert worker._pop_done_transfers(transfers) == set()
        assert worker._pop_done_transfers(transfers) == {"req1", "req2"}

    sleep_mock.assert_not_called()
    assert worker._delayed_recving_transfers == {}


@pytest.mark.parametrize(
    "local_blocks,remote_blocks,expected",
    [
        ([], [], (0, 0, 0, 0, 0, 0, 0, 0)),
        ([1], [4], (0, 0, 1, 1, 1, 1, 0, 0)),
        ([1, 2, 3], [4, 5, 6], (1, 0, 0, 1, 3, 1, 3, 0)),
        ([3, 2, 1], [6, 5, 4], (0, 1, 0, 3, 1, 1, 0, 3)),
        ([1, 2, 3], [6, 5, 4], (0, 0, 3, 3, 3, 3, 0, 0)),
        ([1, 2, 3, 10, 9, 8, 20], [4, 5, 6, 30, 29, 28, 40],
         (1, 1, 1, 5, 5, 3, 3, 3)),
    ],
)
def test_analyze_block_pairs(local_blocks, remote_blocks, expected):
    stats = _analyze_block_pairs(local_blocks, remote_blocks)

    assert (
        stats.paired_forward_run_count,
        stats.paired_reverse_run_count,
        stats.paired_fragment_run_count,
        stats.forward_only_range_count,
        stats.reverse_only_range_count,
        stats.theoretical_merged_range_count,
        stats.longest_paired_forward_run,
        stats.longest_paired_reverse_run,
    ) == expected
    if local_blocks:
        assert stats.local_first_block_id == local_blocks[0]
        assert stats.local_last_block_id == local_blocks[-1]
        assert stats.local_min_block_id == min(local_blocks)
        assert stats.local_max_block_id == max(local_blocks)
        assert stats.remote_first_block_id == remote_blocks[0]
        assert stats.remote_last_block_id == remote_blocks[-1]
        assert stats.remote_min_block_id == min(remote_blocks)
        assert stats.remote_max_block_id == max(remote_blocks)


def test_analyze_block_pairs_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="counts must match"):
        _analyze_block_pairs([1, 2], [3])


@pytest.mark.parametrize(
    "mode,num_blocks,forward_ranges,available_slots,expected",
    [
        ("direct", 1, 1, 64, False),
        ("direct", 512, 512, 64, False),
        ("packed", 1, 1, 0, True),
        ("packed", 512, 1, 0, True),
        ("auto", 32, 1, 64, False),
        ("auto", 512, 63, 64, False),
        ("auto", 512, 64, 1, True),
        ("auto", 512, 64, 0, False),
        ("auto", 2400, 64, 64, True),
        ("auto", 0, 0, 64, False),
    ],
)
def test_packed_path_selector(mode, num_blocks, forward_ranges,
                              available_slots, expected):
    assert _should_use_packed_path(mode,
                                   num_blocks,
                                   forward_ranges,
                                   auto_range_threshold=64,
                                   available_packed_slots=(
                                       available_slots)) is expected


def test_packed_path_selector_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported"):
        _should_use_packed_path("unknown", 1, 1, 64, 1)


@pytest.mark.parametrize(
    "local_blocks,remote_blocks,expected",
    [
        ([], [], (0, 0, 0, 0)),
        ([1, 2, 3], [11, 12, 13], (1, 1, 0, 0)),
        ([3, 2, 1], [13, 12, 11], (1, 1, 0, 2)),
        # The existing reverse transform fixes this permutation completely by
        # reversing its middle [3, 2] run.
        ([1, 3, 2, 4], [11, 13, 12, 14], (1, 1, 0, 2)),
        # No paired -1 run exists, but sorting the preserved pairs by local ID
        # exposes one fully contiguous range.
        ([1, 3, 5, 2, 4], [11, 13, 15, 12, 14], (5, 1, 4, 4)),
        # Independent local/remote layouts remain fragmented after sorting.
        ([1, 3, 5, 2, 4], [21, 11, 40, 18, 30], (5, 5, 0, 4)),
    ],
)
def test_analyze_block_pairs_reports_generalized_reorder_opportunity(
        local_blocks, remote_blocks, expected):
    stats = _analyze_block_pairs(local_blocks, remote_blocks)

    assert (
        stats.reverse_canonicalized_range_count,
        stats.reordered_optimal_range_count,
        stats.additional_reorderable_edge_count,
        stats.generalized_reordered_block_count,
    ) == expected


@pytest.mark.parametrize(
    "local_blocks,remote_blocks,expected_local,expected_remote,"
    "expected_runs,expected_blocks",
    [
        ([], [], [], [], 0, 0),
        ([1], [11], [1], [11], 0, 0),
        ([1, 2, 3], [11, 12, 13], [1, 2, 3], [11, 12, 13], 0, 0),
        ([3, 2, 1], [13, 12, 11], [1, 2, 3], [11, 12, 13], 1, 3),
        ([1, 2, 3], [13, 12, 11], [1, 2, 3], [13, 12, 11], 0, 0),
        (
            [1, 2, 3, 10, 9, 8, 20, 19],
            [11, 12, 13, 30, 29, 28, 40, 41],
            [1, 2, 3, 8, 9, 10, 20, 19],
            [11, 12, 13, 28, 29, 30, 40, 41],
            1,
            3,
        ),
    ],
)
def test_canonicalize_paired_reverse_runs(local_blocks, remote_blocks,
                                          expected_local, expected_remote,
                                          expected_runs, expected_blocks):
    original_pairs = list(zip(local_blocks, remote_blocks))

    (submitted_local, submitted_remote, run_count,
     block_count) = _canonicalize_paired_reverse_runs(local_blocks,
                                                      remote_blocks)

    assert submitted_local == expected_local
    assert submitted_remote == expected_remote
    assert run_count == expected_runs
    assert block_count == expected_blocks
    assert sorted(zip(submitted_local,
                      submitted_remote)) == sorted(original_pairs)
    assert local_blocks == [pair[0] for pair in original_pairs]
    assert remote_blocks == [pair[1] for pair in original_pairs]


def test_canonicalize_paired_reverse_runs_makes_runs_forward():
    local_blocks = [1, 2, 3, 10, 9, 8, 20]
    remote_blocks = [11, 12, 13, 30, 29, 28, 40]

    submitted_local, submitted_remote, _, _ = (
        _canonicalize_paired_reverse_runs(local_blocks, remote_blocks))
    submitted_stats = _analyze_block_pairs(submitted_local, submitted_remote)

    assert submitted_stats.paired_reverse_run_count == 0
    assert submitted_stats.forward_only_range_count == 3
    assert submitted_stats.theoretical_merged_range_count == 3


def test_canonicalize_paired_reverse_runs_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="counts must match"):
        _canonicalize_paired_reverse_runs([1, 2], [3])


@pytest.mark.parametrize(
    "enabled,expected_local_desc_ids,expected_remote_desc_ids",
    [
        (False, [3, 2, 1], [13, 12, 11]),
        (True, [1, 2, 3], [11, 12, 13]),
    ],
)
def test_read_blocks_canonicalizes_submitted_descriptor_order(
        enabled, expected_local_desc_ids, expected_remote_desc_ids):
    worker = object.__new__(NixlConnectorWorker)
    worker.engine_id = "decode"
    worker.tp_rank = 0
    worker._tp_size = {"decode": 1, "prefill": 1}
    worker._pd_trace_enabled = False
    worker._canonicalize_reverse_block_pairs = enabled
    worker.src_xfer_side_handle = 101
    worker.dst_xfer_side_handles = {"prefill": 202}
    worker.block_window_per_layer = []
    worker.num_regions = 1
    worker.dst_num_blocks = {"decode": 100, "prefill": 100}
    worker.block_len_per_layer = [16]
    worker._recving_transfers = defaultdict(list)
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.make_prepped_xfer.return_value = 303

    worker._read_blocks(
        local_block_ids=[3, 2, 1],
        remote_block_ids=[13, 12, 11],
        dst_engine_id="prefill",
        request_id="req",
    )

    args = worker.nixl_wrapper.make_prepped_xfer.call_args.args
    assert args[2].tolist() == expected_local_desc_ids
    assert args[4].tolist() == expected_remote_desc_ids
    worker.nixl_wrapper.transfer.assert_called_once_with(303)


def test_record_transfer_shape_accumulates_block_pair_stats():
    worker = object.__new__(NixlConnectorWorker)
    worker._pd_trace_enabled = True
    worker._transfer_trace_lock = threading.Lock()
    worker._transfer_trace_states = {
        "req":
        _NixlTransferTraceState(
            remote_engine_id="prefill",
            num_local_blocks=0,
            num_remote_blocks=0,
            kv_load_start_ns=1,
        )
    }

    worker._record_transfer_shape(
        "req",
        num_local_blocks=3,
        num_remote_blocks=3,
        num_local_descs=6,
        num_remote_descs=6,
        total_bytes=1024,
        block_pair_stats=_analyze_block_pairs([1, 2, 3], [11, 12, 13]),
        block_pair_stats_ns=100_000,
        canonicalized_reverse_run_count=1,
        canonicalized_reverse_block_count=3,
    )
    worker._record_transfer_shape(
        "req",
        num_local_blocks=2,
        num_remote_blocks=2,
        num_local_descs=4,
        num_remote_descs=4,
        total_bytes=512,
        block_pair_stats=_analyze_block_pairs([9, 8], [19, 18]),
        block_pair_stats_ns=200_000,
        canonicalized_reverse_run_count=1,
        canonicalized_reverse_block_count=2,
    )

    state = worker._transfer_trace_states["req"]
    assert state.num_handles == 2
    assert state.num_local_blocks == 5
    assert state.num_remote_blocks == 5
    assert state.num_local_descs == 10
    assert state.num_remote_descs == 10
    assert state.total_bytes == 1536
    assert state.block_pair_stats_total_ns == 300_000
    assert state.paired_forward_run_count == 1
    assert state.paired_reverse_run_count == 1
    assert state.paired_fragment_run_count == 0
    assert state.forward_only_range_count == 3
    assert state.reverse_only_range_count == 4
    assert state.theoretical_merged_range_count == 2
    assert state.reverse_canonicalized_range_count == 2
    assert state.reordered_optimal_range_count == 2
    assert state.additional_reorderable_edge_count == 0
    assert state.generalized_reordered_block_count == 2
    assert state.longest_paired_forward_run == 3
    assert state.longest_paired_reverse_run == 2
    assert state.local_first_block_id == 1
    assert state.local_last_block_id == 8
    assert state.local_min_block_id == 1
    assert state.local_max_block_id == 9
    assert state.remote_first_block_id == 11
    assert state.remote_last_block_id == 18
    assert state.remote_min_block_id == 11
    assert state.remote_max_block_id == 19
    assert state.canonicalized_reverse_run_count == 2
    assert state.canonicalized_reverse_block_count == 5


def test_pd_transfer_profile_is_emitted_once():
    worker = object.__new__(NixlConnectorWorker)
    worker._pd_trace_enabled = True
    worker._pd_transfer_sleep_ms = 0
    worker._transfer_trace_lock = threading.Lock()
    worker.tp_rank = 0
    worker._transfer_trace_states = {
        "req":
        _NixlTransferTraceState(
            remote_engine_id="prefill",
            num_local_blocks=2,
            num_remote_blocks=2,
            kv_load_start_ns=1_000_000,
            handshake_cached=True,
            desc_build_start_ns=2_000_000,
            desc_build_end_ns=3_000_000,
            desc_build_total_ns=1_000_000,
            xfer_prepare_start_ns=3_000_000,
            xfer_prepare_end_ns=4_000_000,
            xfer_prepare_total_ns=1_000_000,
            xfer_submit_start_ns=4_000_000,
            xfer_submit_end_ns=5_000_000,
            xfer_submit_total_ns=1_000_000,
            first_poll_ns=5_500_000,
            done_observed_ns=7_000_000,
            poll_rounds=3,
            proc_checks=2,
            num_handles=1,
            num_local_descs=4,
            num_remote_descs=4,
            total_bytes=8192,
            block_pair_stats_total_ns=250_000,
            paired_forward_run_count=1,
            paired_reverse_run_count=2,
            paired_fragment_run_count=3,
            forward_only_range_count=4,
            reverse_only_range_count=5,
            theoretical_merged_range_count=6,
            reverse_canonicalized_range_count=4,
            reordered_optimal_range_count=3,
            additional_reorderable_edge_count=1,
            generalized_reordered_block_count=9,
            longest_paired_forward_run=7,
            longest_paired_reverse_run=8,
            local_first_block_id=10,
            local_last_block_id=11,
            local_min_block_id=9,
            local_max_block_id=12,
            remote_first_block_id=20,
            remote_last_block_id=21,
            remote_min_block_id=19,
            remote_max_block_id=22,
            reverse_block_pair_canonicalization_enabled=True,
            canonicalized_reverse_run_count=2,
            canonicalized_reverse_block_count=16,
            packed_chunk_count=2,
            packed_pack_control_total_ns=10_000_000,
            packed_source_handler_total_ns=6_000_000,
            packed_source_handler_max_ns=4_000_000,
            packed_source_sync_wall_total_ns=5_000_000,
            packed_source_sync_wall_max_ns=3_000_000,
            packed_source_stream_wait_gpu_total_ns=4_000_000,
            packed_source_stream_wait_gpu_max_ns=2_500_000,
            packed_source_wait_count=1,
        )
    }
    module = "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"

    with patch(f"{module}.time.perf_counter_ns", return_value=8_000_000), \
            patch(f"{module}.trace_event") as trace_mock:
        worker._emit_transfer_trace("req")
        worker._emit_transfer_trace("req")

    trace_mock.assert_called_once()
    args, fields = trace_mock.call_args
    assert args == ("pull_transfer_profile", "req")
    assert fields["handshake_cached"] is True
    assert fields["desc_build_ms"] == 1.0
    assert fields["submit_to_done_observed_ms"] == 2.0
    assert fields["done_to_connector_finished_ms"] == 1.0
    assert fields["kv_load_to_connector_finished_ms"] == 7.0
    assert fields["total_bytes"] == 8192
    assert fields["block_pair_stats_ms"] == 0.25
    assert fields["paired_forward_run_count"] == 1
    assert fields["paired_reverse_run_count"] == 2
    assert fields["paired_fragment_run_count"] == 3
    assert fields["forward_only_range_count"] == 4
    assert fields["reverse_only_range_count"] == 5
    assert fields["theoretical_merged_range_count"] == 6
    assert fields["reverse_canonicalized_range_count"] == 4
    assert fields["reordered_optimal_range_count"] == 3
    assert fields["additional_reorderable_edge_count"] == 1
    assert fields["generalized_reordered_block_count"] == 9
    assert fields["longest_paired_forward_run"] == 7
    assert fields["longest_paired_reverse_run"] == 8
    assert fields["local_first_block_id"] == 10
    assert fields["remote_last_block_id"] == 21
    assert fields["reverse_block_pair_canonicalization_enabled"] is True
    assert fields["canonicalized_reverse_run_count"] == 2
    assert fields["canonicalized_reverse_block_count"] == 16
    assert fields["packed_source_handler_ms"] == 6.0
    assert fields["packed_source_handler_max_ms"] == 4.0
    assert fields["packed_source_sync_wall_ms"] == 5.0
    assert fields["packed_source_sync_wall_max_ms"] == 3.0
    assert fields["packed_source_stream_wait_gpu_ms"] == 4.0
    assert fields["packed_source_stream_wait_gpu_max_ms"] == 2.5
    assert fields["packed_source_default_stream_wait_count"] == 1
    assert fields["packed_pack_control_other_ms"] == 4.0
    assert worker._transfer_trace_states == {}


def test_packed_source_request_state_is_released_after_every_chunk():
    worker = object.__new__(NixlConnectorWorker)
    request_key = "decode/req/nonce"
    first_key = f"{request_key}/0"
    second_key = f"{request_key}/1"
    worker._packed_source_slots = {first_key: 3, second_key: 4}
    worker._packed_source_free_slots = deque()
    worker._packed_source_requests = {
        request_key:
        _PackedSourceRequestState(total_chunks=2, default_stream_ready=True)
    }

    for key in (first_key, second_key):
        assert worker._handle_packed_control({
            "version": 2,
            "type": "release",
            "key": key,
        }) is None

    assert list(worker._packed_source_free_slots) == [3, 4]
    assert worker._packed_source_requests == {}


def test_packed_source_waits_for_default_stream_once_per_request():
    worker = object.__new__(NixlConnectorWorker)
    worker._packed_available = True
    worker._packed_source_slots = {}
    worker._packed_source_free_slots = deque((3, 4))
    worker._packed_source_requests = {}
    worker._packed_blocks_per_slot = 1
    worker.num_blocks = 2
    worker._packed_staging = MagicMock()
    worker._packed_staging.device = "cuda"
    worker._packed_region_ptrs = MagicMock()
    worker._packed_pack_stream = MagicMock()
    worker._pd_trace_enabled = False
    event = MagicMock()
    event.elapsed_time.return_value = 0.0
    module = "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"

    with patch(f"{module}.torch.cuda.Event", return_value=event), \
            patch(f"{module}.torch.cuda.device",
                  return_value=contextlib.nullcontext()), \
            patch(f"{module}.torch.cuda.stream",
                  return_value=contextlib.nullcontext()), \
            patch(f"{module}.torch.cuda.default_stream"), \
            patch(f"{module}.torch.tensor"), \
            patch(f"{module}._launch_pack_regions"):
        responses = [
            worker._handle_packed_control({
                "version": 2,
                "type": "pack",
                "key": f"decode/req/nonce/{chunk_id}",
                "total_chunks": 2,
                "block_ids": [chunk_id],
            }) for chunk_id in range(2)
        ]

    assert [
        response["source_default_stream_wait_count"] for response in responses
    ] == [1, 0]
    worker._packed_pack_stream.wait_stream.assert_called_once()
    assert worker._packed_source_requests[
        "decode/req/nonce"].default_stream_ready is True


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper)
def test_shutdown_cleans_up_resources(dist_init):
    """Test that shutdown() properly cleans up all resources."""
    vllm_config = create_vllm_config()

    worker = NixlConnectorWorker(vllm_config,
                                 vllm_config.kv_transfer_config.engine_id)
    nixl_wrapper = worker.nixl_wrapper

    with patch.object(worker, '_handshake_initiation_executor') as mock_exec, \
         patch.object(worker, '_nixl_handshake_listener_t') as mock_listener, \
         patch.object(nixl_wrapper, 'release_xfer_handle') as mock_rel_xfer, \
         patch.object(nixl_wrapper, 'release_dlist_handle') as mock_rel_dlist, \
         patch.object(nixl_wrapper, 'remove_remote_agent') as mock_rem_agent, \
         patch.object(nixl_wrapper, 'deregister_memory') as mock_dereg:

        worker._recving_transfers = {"req1": [(123, time.perf_counter())]}
        worker.src_xfer_side_handle = 456
        worker.dst_xfer_side_handles = {"engine1": 789}
        worker._remote_agents = {"engine1": {0: "agent1"}}
        worker._registered_descs = ["desc1", "desc2"]

        worker.shutdown()

        # Test idempotency
        worker.shutdown()
        worker.shutdown()

        mock_exec.shutdown.assert_called_with(wait=False)
        mock_listener.join.assert_called_once_with(timeout=0)

        mock_rel_xfer.assert_called_once_with(123)
        assert mock_rel_dlist.call_count == 2
        mock_rel_dlist.assert_any_call(456)  # src handle
        mock_rel_dlist.assert_any_call(789)  # dst handle
        mock_rem_agent.assert_called_once_with("agent1")
        assert mock_dereg.call_count == 2
        mock_dereg.assert_any_call("desc1")
        mock_dereg.assert_any_call("desc2")
