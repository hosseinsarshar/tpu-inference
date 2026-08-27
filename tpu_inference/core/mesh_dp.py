# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mesh-based (MPMD) data parallelism.

One process hosts ``dp_size`` complete, independent vLLM engines. Each engine
is pinned to its own ``jax.sharding.Mesh`` built over a disjoint slice of
``jax.devices()``, so a DP rank is *defined* by its mesh rather than by hiding
physical chips from the runtime.

How this differs from the two existing DP paths:

``TPU_MULTIPROCESS_DP=1`` (multi-process MPMD)
    Also gives independent ranks, but isolates them by setting libtpu
    ``TPU_VISIBLE_CHIPS`` / ``TPU_CHIPS_PER_PROCESS_BOUNDS`` env vars in each
    spawned engine process, so every rank's JAX runtime believes it owns a
    whole (smaller) TPU. That requires the per-rank chip set to form a valid
    physical topology box, needs a private libtpu port per rank, and gives up
    everything that a shared address space buys (shared weights cache, cheap
    load-balancer queries, one profiler).

SPMD DP (the default)
    Keeps one mesh with a ``data`` axis of size ``dp_size`` and one fused XLA
    program. Because a single program has a single set of static shapes, every
    rank must be padded up to the longest rank's token count and every rank
    must step in lock-step. Correct, but the padding and the per-step barrier
    are exactly what make it slow on ragged batches.

Mesh-based DP keeps the independence of the first and the JAX-nativeness of the
second: ``dp_size`` separate meshes, ``dp_size`` separate schedulers, and
``dp_size`` separate ``jit`` dispatches that overlap on disjoint devices.

Independence and dispatch, concretely:

* Ranks share no JAX arrays, no collectives and no barrier. A rank's mesh spans
  only its own devices, so nothing it compiles can even name another rank's
  device.
* Each rank is driven by its own Python thread running the stock
  ``EngineCore.step()``. The threads never wait on each other.
* New requests reach a rank through a queue that the rank thread drains at the
  top of its loop, so routing never touches scheduler state concurrently.
* Dispatch does not serialise: JAX enqueues asynchronously and the blocking
  device->host fetch releases the GIL, so while rank *i* waits for its step to
  land, ranks *j != i* are free to run Python and enqueue theirs.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import Request
from vllm.v1.worker.worker_base import WorkerWrapperBase

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# How long a rank thread parks on its input queue when it has nothing to do.
_IDLE_POLL_S = 0.02
# How long the outer step() waits for the first output before reporting back to
# vLLM empty-handed. Short enough to stay responsive, long enough not to spin.
_STEP_WAIT_S = 0.05

_DEVICE_GROUPS_ATTR = "mesh_dp_device_groups"
_NEXT_RANK_ATTR = "mesh_dp_next_rank"

# One rank per thread means `dp_size` threads competing for a single GIL. A
# step alternates between Python work and GIL-releasing device waits, and at
# CPython's 5ms default every release sends the thread to the back of the
# queue -- with 8 ranks that convoy stretched single steps past two seconds.
# A coarser interval lets a rank finish more of its step per acquisition.
_SWITCH_INTERVAL_S = float(os.getenv("TPU_MESH_DP_SWITCH_INTERVAL", "0.05"))

# Rank being constructed, so engines can be built concurrently: the executor
# runs on the builder's own thread and reads its rank from here.
_building = threading.local()

# `init_device` reaches torch.distributed's *global* default process group, so
# concurrent builders race ("initialize the default process group twice").
# Serialise just that call -- weight loading and XLA compilation, which are
# the parts actually worth overlapping, stay outside the lock.
_DEVICE_INIT_LOCK = threading.Lock()


def is_mesh_dp_enabled(vllm_config: VllmConfig) -> bool:
    """True when this config should run on mesh-based DP."""
    if not envs.TPU_MESH_BASED_DP:
        return False
    sharding_config = getattr(vllm_config, "sharding_config", None)
    return sharding_config is not None and sharding_config.mesh_dp_size > 1


def assign_device_groups(vllm_config: VllmConfig) -> List[List[Any]]:
    """Split ``jax.devices()`` into one contiguous group per DP rank.

    Each group becomes one rank's mesh. Unlike the multi-process path the
    grouping is purely logical, so any split that divides the device list is
    valid -- there is no physical-topology box to satisfy.
    """
    sharding_config = vllm_config.sharding_config
    dp_size = sharding_config.mesh_dp_size
    per_rank = sharding_config.total_devices

    devices = jax.devices()
    needed = dp_size * per_rank
    if len(devices) < needed:
        raise ValueError(
            f"Mesh-based DP needs {needed} devices "
            f"(dp_size={dp_size} x {per_rank} per rank) but only "
            f"{len(devices)} are visible. Lower --data-parallel-size or "
            f"--tensor-parallel-size.")

    groups = [
        devices[r * per_rank:(r + 1) * per_rank] for r in range(dp_size)
    ]
    setattr(vllm_config.device_config, _DEVICE_GROUPS_ATTR, groups)
    setattr(vllm_config.device_config, _NEXT_RANK_ATTR, 0)
    logger.info(
        "Mesh-based DP | dp_size=%d | %d device(s) per rank | groups=%s",
        dp_size, per_rank,
        [[d.id for d in g] for g in groups])
    return groups


class MeshDPExecutor(UniProcExecutor):
    """UniProc executor whose worker is pinned to one rank's device group.

    Handing ``TPUWorker`` an explicit ``devices`` list is all it takes: the
    worker keeps that list instead of taking all of ``jax.local_devices()``,
    and ``TPUModelRunner`` then builds its mesh from exactly those devices.
    That mesh *is* the DP rank.

    Only ``_init_executor`` differs from the upstream UniProc path, so the
    execute/collective_rpc machinery (including async output futures) is
    inherited unchanged.
    """

    def _init_executor(self) -> None:
        device_config = self.vllm_config.device_config
        groups = getattr(device_config, _DEVICE_GROUPS_ATTR, None)
        if not groups:
            raise RuntimeError(
                "MeshDPExecutor used without device groups; "
                "assign_device_groups() must run first.")
        # MeshDPEngineCore builds the engines concurrently, so the rank comes
        # from the thread doing the building rather than from a shared cursor.
        rank = getattr(_building, "rank", None)
        if rank is None:
            rank = getattr(device_config, _NEXT_RANK_ATTR, 0)
            setattr(device_config, _NEXT_RANK_ATTR, rank + 1)
        if rank >= len(groups):
            raise RuntimeError(
                f"MeshDPExecutor asked for rank {rank} but only "
                f"{len(groups)} device groups were assigned.")

        self.mesh_dp_rank = rank
        self.devices = groups[rank]
        logger.info("MeshDPExecutor rank=%d devices=%s", rank,
                    [d.id for d in self.devices])

        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, worker_rank, local_rank = (
            self._distributed_args())
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=worker_rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
            devices=self.devices,
        )
        with _DEVICE_INIT_LOCK:
            self.driver_worker.init_worker(all_kwargs=[kwargs])
            self.driver_worker.init_device()
        self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)


class _SchedulerProxy:
    """Aggregate view over the per-rank schedulers.

    vLLM occasionally reaches for ``engine_core.scheduler`` for coarse
    questions ("is there anything in flight?"). Only aggregate, read-mostly
    operations are exposed; anything per-request goes through the owning rank.
    """

    def __init__(self, engines: List[vLLMEngineCore]):
        self._engines = engines

    def has_requests(self) -> bool:
        return any(e.scheduler.has_requests() for e in self._engines)

    def has_unfinished_requests(self) -> bool:
        return any(e.scheduler.has_unfinished_requests()
                   for e in self._engines)

    def has_finished_requests(self) -> bool:
        return any(e.scheduler.has_finished_requests() for e in self._engines)

    def get_num_unfinished_requests(self) -> int:
        return sum(e.scheduler.get_num_unfinished_requests()
                   for e in self._engines)

    def get_request_counts(self) -> Tuple[int, int]:
        running = waiting = 0
        for e in self._engines:
            r, w = e.scheduler.get_request_counts()
            running += r
            waiting += w
        return running, waiting

    def get_kv_connector(self):
        return self._engines[0].scheduler.get_kv_connector()

    def reset_prefix_cache(self, *args, **kwargs) -> bool:
        return all(
            e.scheduler.reset_prefix_cache(*args, **kwargs)
            for e in self._engines)

    def reset_encoder_cache(self) -> None:
        for e in self._engines:
            e.scheduler.reset_encoder_cache()

    def shutdown(self) -> None:
        for e in self._engines:
            e.scheduler.shutdown()


class _RankRouter:
    """Least-loaded routing over the DP ranks.

    Load is tracked here rather than read off the rank schedulers: a scheduler
    is owned by its rank thread, and walking its queues from the routing thread
    would be a data race. The two counters are maintained from the two events
    the router can observe locally -- a request being routed, and a request's
    first token / final token coming back.

    ``prefill_owed`` (prompt tokens not yet prefilled) is the primary key
    because prefill is what actually saturates a rank; in-flight request count
    breaks ties so decode-heavy ranks do not keep collecting new work.
    """

    def __init__(self, dp_size: int):
        self._lock = threading.Lock()
        self._prefill_owed = [0] * dp_size
        self._inflight = [0] * dp_size
        self._dp_size = dp_size

    def pick(self, num_tokens: int) -> int:
        with self._lock:
            rank = min(
                range(self._dp_size),
                key=lambda r: (self._prefill_owed[r], self._inflight[r], r))
            self._prefill_owed[rank] += num_tokens
            self._inflight[rank] += 1
            return rank

    def on_prefilled(self, rank: int, num_tokens: int) -> None:
        with self._lock:
            self._prefill_owed[rank] = max(
                0, self._prefill_owed[rank] - num_tokens)

    def on_finished(self, rank: int) -> None:
        with self._lock:
            self._inflight[rank] = max(0, self._inflight[rank] - 1)

    def snapshot(self) -> Tuple[List[int], List[int]]:
        with self._lock:
            return list(self._prefill_owed), list(self._inflight)


class MeshDPEngineCore(vLLMEngineCore):
    """An ``EngineCore`` that fans out over one independent engine per mesh.

    Stands in for ``vllm.v1.engine.core.EngineCore`` (see :func:`install`). The
    base ``__init__`` is deliberately not called: this object owns no executor,
    scheduler or KV cache of its own, it only owns the sub-engines and the
    threads that drive them.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
        include_finished_set: bool = False,
    ):
        self.vllm_config = vllm_config
        self.log_stats = log_stats
        self.dp_size = vllm_config.sharding_config.mesh_dp_size

        assign_device_groups(vllm_config)

        # Build the engines concurrently. Most of the ~2min per-engine cost is
        # XLA compilation and weight loading, which release the GIL, so this
        # turns dp_size sequential compiles into roughly one.
        t0 = time.perf_counter()
        built: List[Optional[vLLMEngineCore]] = [None] * self.dp_size
        errors: List[BaseException] = []

        def build(rank: int) -> None:
            _building.rank = rank
            try:
                built[rank] = vLLMEngineCore(
                    vllm_config,
                    MeshDPExecutor,
                    log_stats,
                    executor_fail_callback,
                    include_finished_set,
                )
            except BaseException as e:  # noqa: BLE001 - re-raised below
                logger.exception("Mesh-based DP | rank %d failed to build",
                                 rank)
                errors.append(e)
            finally:
                _building.rank = None

        builders = [
            threading.Thread(target=build, args=(r, ), name=f"mesh-dp-build-{r}")
            for r in range(self.dp_size)
        ]
        # `set_current_vllm_config` saves and restores a module-level global, so
        # concurrent builders clobber each other: one engine's `load_model`
        # context exits and restores `None` while another is still in KV-cache
        # init, which fails with "Current vLLM config is not set". Holding the
        # config for the whole build makes every nested save/restore land on
        # this same object, so the global is never unset while a builder runs.
        with set_current_vllm_config(vllm_config):
            for t in builders:
                t.start()
            for t in builders:
                t.join()
        if errors:
            raise errors[0]

        self.engines: List[vLLMEngineCore] = [e for e in built if e is not None]
        if len(self.engines) != self.dp_size:
            raise RuntimeError(
                f"Mesh-based DP built {len(self.engines)} of {self.dp_size} "
                f"engines")
        logger.info("Mesh-based DP | %d engines ready in %.1fs", self.dp_size,
                    time.perf_counter() - t0)

        prev = sys.getswitchinterval()
        sys.setswitchinterval(_SWITCH_INTERVAL_S)
        logger.info("Mesh-based DP | GIL switch interval %.4fs -> %.4fs", prev,
                    _SWITCH_INTERVAL_S)

        # --- vLLM-facing attributes normally set by EngineCore.__init__ ---
        first = self.engines[0]
        self.model_executor = first.model_executor
        self.structured_output_manager = first.structured_output_manager
        self.request_block_hasher = first.request_block_hasher
        self.mm_receiver_cache = MULTIMODAL_REGISTRY.engine_receiver_cache_from_config(
            vllm_config)
        self.scheduler = _SchedulerProxy(self.engines)
        self.batch_queue = None
        self.batch_queue_size = 1
        # Set by EngineCore.__init__ and read by inherited helpers; the rank
        # engines own the real queues, these just keep the base class happy.
        self.aborts_queue = queue.Queue()
        self.is_ec_consumer = first.is_ec_consumer
        self._pooler_config_logged = True
        self.use_spec_decode = first.use_spec_decode
        self.check_for_draft_tokens = first.check_for_draft_tokens
        self.is_pooling_model = first.is_pooling_model
        self.async_scheduling = first.async_scheduling
        self.available_gpu_memory_for_kv_cache = (
            first.available_gpu_memory_for_kv_cache)
        self._weight_version = "default"
        self.step_fn = self.step

        # --- orchestration state ---
        self._router = _RankRouter(self.dp_size)
        self._req_rank: Dict[str, int] = {}
        self._req_rank_lock = threading.Lock()
        # Prompt-token count per request, needed to undo the router's
        # prefill-owed charge once the request's prefill actually lands.
        self._req_prompt_tokens: Dict[str, int] = {}
        self._prefilled: set[str] = set()

        # Lightweight throughput accounting, logged periodically from step().
        self._stats = [{
            "steps": 0,
            "busy_s": 0.0,
            "idle": 0,
            "toks": 0
        } for _ in range(self.dp_size)]
        self._outer = {"calls": 0, "empty": 0, "wait_s": 0.0}
        self._last_report = time.perf_counter()

        self._in_q: List[queue.Queue] = [
            queue.Queue() for _ in range(self.dp_size)
        ]
        self._out_q: queue.Queue = queue.Queue()
        self._live = True
        self._threads = [
            threading.Thread(target=self._rank_loop,
                             args=(rank, ),
                             name=f"mesh-dp-rank-{rank}",
                             daemon=True) for rank in range(self.dp_size)
        ]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------
    # Rank threads
    # ------------------------------------------------------------------

    def _rank_loop(self, rank: int) -> None:
        """Drive one rank: drain its inbox, then step it, forever.

        This is the stock single-engine loop. Nothing in it refers to another
        rank, which is what keeps the ranks independent -- no rank can stall
        waiting for a peer, and no step is padded to a peer's shape.
        """
        engine = self.engines[rank]
        inbox = self._in_q[rank]
        st = self._stats[rank]
        while self._live:
            try:
                self._drain_inbox(engine, inbox, rank)
            except Exception:
                logger.exception("Mesh-based DP rank %d failed draining inbox",
                                 rank)

            if not engine.scheduler.has_requests():
                # Park on the inbox rather than spinning, so an idle rank
                # costs no CPU and steals no GIL time from busy ranks.
                st["idle"] += 1
                try:
                    op = inbox.get(timeout=_IDLE_POLL_S)
                except queue.Empty:
                    continue
                self._apply_op(engine, op, rank)
                continue

            t_step = time.perf_counter()
            try:
                outputs, model_executed = engine.step_fn()
                engine.post_step(model_executed=model_executed)
            except Exception:
                logger.exception("Mesh-based DP rank %d step failed", rank)
                continue
            st["steps"] += 1
            st["busy_s"] += time.perf_counter() - t_step

            if outputs:
                st["toks"] += sum(
                    len(o.new_token_ids) for eco in outputs.values()
                    for o in eco.outputs)
                for client_index, eco in outputs.items():
                    self._account(rank, eco)
                    self._out_q.put_nowait((client_index, eco))

    def _drain_inbox(self, engine: vLLMEngineCore, inbox: queue.Queue,
                     rank: int) -> None:
        while True:
            try:
                op = inbox.get_nowait()
            except queue.Empty:
                return
            self._apply_op(engine, op, rank)

    def _apply_op(self, engine: vLLMEngineCore, op: Tuple, rank: int) -> None:
        kind = op[0]
        if kind == "add":
            _, request, request_wave = op
            engine.add_request(request, request_wave)
        elif kind == "abort":
            _, request_ids = op
            engine.abort_requests(request_ids)
            for rid in request_ids:
                self._forget_request(rid, rank)
        elif kind == "call":
            _, fn, args, kwargs, result_box = op
            try:
                result_box.append(("ok", fn(engine, *args, **kwargs)))
            except Exception as e:  # surfaced to the caller
                result_box.append(("err", e))
        elif kind == "stop":
            pass

    def _account(self, rank: int, eco: EngineCoreOutputs) -> None:
        """Update router load counters from a rank's own outputs."""
        for out in eco.outputs:
            rid = out.request_id
            if out.new_token_ids and rid not in self._prefilled:
                # First sampled token => this request's prefill is done, so it
                # no longer contributes to the rank's prefill backlog.
                self._prefilled.add(rid)
                self._router.on_prefilled(
                    rank, self._req_prompt_tokens.get(rid, 0))
            if out.finish_reason is not None:
                self._router.on_finished(rank)
                self._forget_request(rid, rank)

    def _forget_request(self, rid: str, rank: int) -> None:
        self._prefilled.discard(rid)
        self._req_prompt_tokens.pop(rid, None)
        with self._req_rank_lock:
            self._req_rank.pop(rid, None)

    # ------------------------------------------------------------------
    # EngineCore API
    # ------------------------------------------------------------------

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        num_tokens = request.num_tokens
        rank = self._router.pick(num_tokens)
        with self._req_rank_lock:
            self._req_rank[request.request_id] = rank
        self._req_prompt_tokens[request.request_id] = num_tokens
        self._in_q[rank].put_nowait(("add", request, request_wave))

    def abort_requests(self, request_ids: List[str]) -> None:
        by_rank: Dict[int, List[str]] = {}
        with self._req_rank_lock:
            for rid in request_ids:
                rank = self._req_rank.get(rid)
                if rank is not None:
                    by_rank.setdefault(rank, []).append(rid)
        for rank, rids in by_rank.items():
            self._in_q[rank].put_nowait(("abort", rids))

    def step(self) -> Tuple[Dict[int, EngineCoreOutputs], bool]:
        """Collect whatever the rank threads have finished since last call.

        Ranks run ahead on their own; this only harvests. Outputs for the same
        client are merged so a client never loses a rank's tokens for a step.
        """
        self._outer["calls"] += 1
        self._maybe_report()
        t_wait = time.perf_counter()
        try:
            first = self._out_q.get(timeout=_STEP_WAIT_S)
        except queue.Empty:
            self._outer["empty"] += 1
            self._outer["wait_s"] += time.perf_counter() - t_wait
            return {}, self.scheduler.has_requests()
        self._outer["wait_s"] += time.perf_counter() - t_wait

        merged: Dict[int, EngineCoreOutputs] = {}
        self._merge_into(merged, first)
        while True:
            try:
                item = self._out_q.get_nowait()
            except queue.Empty:
                break
            self._merge_into(merged, item)
        return merged, True

    def _maybe_report(self) -> None:
        now = time.perf_counter()
        if now - self._last_report < 2.0:
            return
        dt = now - self._last_report
        self._last_report = now
        per_rank = " | ".join(
            f"r{r}: {s['steps']}st {s['busy_s']:.2f}s busy "
            f"{s['toks']}tok {s['idle']}idle"
            for r, s in enumerate(self._stats))
        logger.info(
            "Mesh-DP %.1fs window | outer: %d calls %d empty %.2fs waiting | "
            "%s", dt, self._outer["calls"], self._outer["empty"],
            self._outer["wait_s"], per_rank)
        for s in self._stats:
            s.update(steps=0, busy_s=0.0, idle=0, toks=0)
        self._outer.update(calls=0, empty=0, wait_s=0.0)

    @staticmethod
    def _merge_into(merged: Dict[int, EngineCoreOutputs],
                    item: Tuple[int, EngineCoreOutputs]) -> None:
        client_index, eco = item
        existing = merged.get(client_index)
        if existing is None:
            merged[client_index] = eco
            return
        existing.outputs.extend(eco.outputs)
        if eco.finished_requests:
            existing.finished_requests = (existing.finished_requests
                                          or set()) | eco.finished_requests
        if eco.scheduler_stats is not None:
            existing.scheduler_stats = eco.scheduler_stats

    def post_step(self, model_executed: bool) -> None:
        # Each rank already ran post_step for its own engine inside its thread.
        return

    def get_supported_tasks(self):
        return self.engines[0].get_supported_tasks()

    # ------------------------------------------------------------------
    # Fan-out helpers for the remaining control-plane calls
    # ------------------------------------------------------------------

    def _broadcast(self, fn: Callable[[vLLMEngineCore], Any]) -> List[Any]:
        """Run ``fn`` on every rank engine, on that rank's own thread.

        Control-plane calls (profile, reset caches, LoRA) must not touch a rank
        engine from the caller's thread while the rank thread is mid-step.
        """
        boxes = []
        for rank in range(self.dp_size):
            box: List[Tuple[str, Any]] = []
            boxes.append(box)
            self._in_q[rank].put_nowait(("call", fn, (), {}, box))

        results = []
        deadline = time.monotonic() + 300.0
        for rank, box in enumerate(boxes):
            while not box:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Mesh-based DP rank {rank} did not answer a "
                        f"control-plane call within 300s")
                time.sleep(0.001)
            status, value = box[0]
            if status == "err":
                raise value
            results.append(value)
        return results

    def profile(self, is_start: bool = True, profile_prefix=None) -> None:
        self._broadcast(lambda e: e.profile(is_start, profile_prefix))

    def reset_mm_cache(self) -> None:
        self._broadcast(lambda e: e.reset_mm_cache())

    def reset_prefix_cache(self,
                           reset_running_requests: bool = False,
                           reset_connector: bool = False) -> bool:
        return all(
            self._broadcast(lambda e: e.reset_prefix_cache(
                reset_running_requests, reset_connector)))

    def reset_encoder_cache(self) -> None:
        self._broadcast(lambda e: e.reset_encoder_cache())

    def execute_dummy_batch(self) -> None:
        self._broadcast(lambda e: e.execute_dummy_batch())

    def add_lora(self, lora_request) -> bool:
        return all(self._broadcast(lambda e: e.add_lora(lora_request)))

    def remove_lora(self, lora_id: int) -> bool:
        return all(self._broadcast(lambda e: e.remove_lora(lora_id)))

    def list_loras(self):
        result: set = set()
        for loras in self._broadcast(lambda e: e.list_loras()):
            result |= loras
        return result

    def pin_lora(self, lora_id: int) -> bool:
        return all(self._broadcast(lambda e: e.pin_lora(lora_id)))

    def save_sharded_state(self, path, pattern=None, max_size=None) -> None:
        raise NotImplementedError(
            "save_sharded_state is not supported under mesh-based DP")

    def collective_rpc(self,
                       method,
                       timeout=None,
                       args=(),
                       kwargs=None) -> List[Any]:
        kwargs = kwargs or {}
        out: List[Any] = []
        for results in self._broadcast(
                lambda e: e.collective_rpc(method, timeout, args, kwargs)):
            out.extend(results)
        return out

    def set_weight_version(self, weight_version: str) -> None:
        self._weight_version = weight_version

    def get_weight_version(self) -> str:
        return self._weight_version

    def sleep(self, level: int = 1, mode: str = "abort"):
        raise NotImplementedError(
            "sleep/wake_up is not supported under mesh-based DP")

    def wake_up(self, tags=None):
        raise NotImplementedError(
            "sleep/wake_up is not supported under mesh-based DP")

    def is_sleeping(self) -> bool:
        return False

    def shutdown(self) -> None:
        if not self._live:
            return
        self._live = False
        for q in self._in_q:
            q.put_nowait(("stop", ))
        for t in self._threads:
            t.join(timeout=10.0)
        for e in self.engines:
            try:
                e.shutdown()
            except Exception:
                logger.exception("Error shutting down a mesh-DP rank engine")


_INSTALLED = False


def install() -> None:
    """Route ``EngineCore`` construction through :class:`MeshDPEngineCore`.

    ``vllm.v1.engine.core_client`` binds ``EngineCore`` at import time, so both
    that name and the definition site have to be replaced.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import vllm.v1.engine.core as vllm_core

    vllm_core.EngineCore = MeshDPEngineCore
    try:
        import vllm.v1.engine.core_client as vllm_core_client
        vllm_core_client.EngineCore = MeshDPEngineCore
    except ImportError:  # pragma: no cover - core_client always exists
        pass

    _INSTALLED = True
    logger.info("Mesh-based DP installed: EngineCore -> MeshDPEngineCore")
