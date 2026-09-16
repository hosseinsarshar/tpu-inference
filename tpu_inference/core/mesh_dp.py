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
"""Mesh-based data parallelism: vLLM's own DP ranks, as threads.

One process hosts ``data_parallel_size`` complete, independent vLLM engines.
Each engine is pinned to its own ``jax.sharding.Mesh`` built over a disjoint
slice of ``jax.devices()``, so a DP rank is *defined* by its mesh rather than
by hiding physical chips from the runtime.

How this differs from the two existing DP paths:

``TPU_MULTIPROCESS_DP=1`` (multi-process MPMD)
    Also gives independent ranks, but isolates them by setting libtpu
    ``TPU_VISIBLE_CHIPS`` / ``TPU_CHIPS_PER_PROCESS_BOUNDS`` env vars in each
    spawned engine process, so every rank's JAX runtime believes it owns a
    whole (smaller) TPU. That requires the per-rank chip set to form a valid
    physical topology box, needs a private libtpu port per rank, and gives up
    everything a shared address space buys (shared weights cache, one
    profiler, no IPC hop for multimodal tensors).

SPMD DP (the default)
    Keeps one mesh with a ``data`` axis of size ``dp_size`` and one fused XLA
    program. A single program has a single set of static shapes, so every rank
    is padded up to the longest rank's token count and every rank steps in
    lock-step. Correct, but the padding and the per-step barrier are exactly
    what make it slow on ragged batches.

What this module actually is
----------------------------

Mesh DP is the multi-process path with *threads* instead of processes, so it
is built as a **launcher**, not as a parallel class hierarchy. The engines it
starts are stock ``vllm.v1.engine.core.EngineCoreProc`` objects running the
stock busy loop; request routing, load balancing, wave/idle coordination and
the whole ZMQ control plane are vLLM's own, unmodified. Two things are
mesh-specific and nothing else is:

1. ``CoreEngineThreadManager`` spawns **one** engine-core child process and
   starts all ``local_engine_count`` ranks inside it as threads, where
   ``CoreEngineProcManager`` would have spawned one process per rank. It
   subclasses that class and overrides only ``__init__``, so ``shutdown()``,
   ``monitor_engine_liveness()``, ``sentinels()`` and ``finished_procs()``
   are inherited verbatim and keep working over the single child.
2. Each rank's ``VllmConfig`` copy carries its own
   ``sharding_config.device_indexes``. That is an existing, first-class field
   ``TPUWorker.init_device`` already honours (``tpu_worker.py:379``), and it
   is what pins the rank's mesh to its own chips.

The child process is not an implementation detail. Threads are what make the
ranks share one JAX client and one weights cache, but they also make every
rank share a GIL with whatever else lives in that process -- and the process
that calls ``launch_core_engines`` under ``vllm serve`` is the API server,
running uvicorn, tokenization and detokenization. Hosting the ranks there
would put the frontend on the engines' GIL. Spawning one child keeps the
split the process path has: frontend in the parent, engines in a child.
It also keeps ``jax.devices()`` out of the parent, which must stay free of an
initialised TPU backend.

``install()`` therefore rebinds exactly one name. An earlier version of this
module substituted the engine core itself -- ``MeshDPEngineCore``,
``MeshDPEngineCoreProc``, a ``_SchedulerProxy`` and a hand-written rank router,
about 900 lines that re-implemented request routing, output merging and the
engine-core control plane. All of it duplicated behaviour vLLM already has for
multi-process DP, and every vLLM bump risked the two drifting apart. Deleting
it is the point of this file.

What this costs, measured
-------------------------

The old engine core hid all ``dp_size`` ranks behind one ``EngineCore``, so
vLLM saw a single engine with a single queue of ``dp_size * max_num_seqs``
slots and no DP coordinator ran at all. This module gives vLLM the eight real
engines it thinks it has, each with its own scheduler and its own
``max_num_seqs`` slots -- the same shape multi-process DP has. Eight queues of
32 are not one queue of 256, and the difference shows up in the tail, not the
mean. On v6e-8, dp8/tp1, 1024x1024, sync scheduling:

==========  ==========================  ==========================
metric      256 global slots            1024 global slots
==========  ==========================  ==========================
throughput  14,019 vs 14,160  (-1.0%)   24,755 vs 26,398  (-6.2%)
P99 TTFT    14,647 vs 2,571  (+470%)    5,823 vs 5,686   (+2.4%)
median e2e  15,588 vs 17,781 (-12.3%)   36,090 vs 36,365  (-0.8%)
==========  ==========================  ==========================

(new vs old, means of 3-6 runs each; the throughput deltas are inside the
run-to-run spread, which at 1024 slots ran 23.7k-29.6k on the old code.)

At 1024 slots nothing moves: each rank holds 128 slots, no rank ever fills,
and the queue split is invisible. At 256 slots each rank holds exactly 32 and
the benchmark keeps exactly 256 in flight, so every rank sits at capacity and
an unlucky request waits out a whole generation instead of taking the next
slot to free anywhere. The old code bought its 2.6s tail with the single
queue, not with clever routing. Capping the stock load balancer at
``max_num_seqs`` was tried and changed nothing (P99 15.3s over three runs),
which is the evidence that the queue split -- not the routing policy -- is
what moved.

That is the trade this module makes on purpose: vLLM's DP path, with vLLM's
tail, in exchange for ~900 lines that had to be kept in step with it by hand.

Independence and dispatch, concretely:

* Ranks share no JAX arrays, no collectives and no barrier. A rank's mesh spans
  only its own devices, so nothing it compiles can even name another rank's
  device.
* Each rank is driven by its own thread running the stock
  ``EngineCoreProc.run_busy_loop()``. The threads never wait on each other.
* Dispatch does not serialise: JAX enqueues asynchronously and the blocking
  device->host fetch releases the GIL, so while rank *i* waits for its step to
  land, ranks *j != i* are free to run Python and enqueue theirs.
"""

from __future__ import annotations

import copy
import os
import signal
import sys
import threading
import time
import weakref
from typing import Any, Dict, List, Optional

import jax
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.utils import CoreEngineProcManager, SignalCallback
from vllm.v1.utils import shutdown as shutdown_processes

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Guards `install()`, which is reached from `check_and_update_config` and so
# runs once per config built in a process.
_INSTALLED = False

# One rank per thread means `dp_size` threads competing for a single GIL. A
# step alternates between Python work and GIL-releasing device waits, and at
# CPython's 5ms default every release sends the thread to the back of the
# queue -- with 8 ranks that convoy stretched single steps past two seconds.
# A coarser interval lets a rank finish more of its step per acquisition.
_SWITCH_INTERVAL_S = float(os.getenv("TPU_MESH_DP_SWITCH_INTERVAL", "0.05"))


def is_mesh_dp_enabled(vllm_config: VllmConfig) -> bool:
    """True when this config should run its DP ranks as threads.

    Deliberately the same shape of test the multi-process path uses: the flag,
    plus more than one DP rank to spread. Mesh DP no longer collapses
    ``data_parallel_size``, so this stays true for the whole life of the
    config instead of having to be recovered from an env var.
    """
    if not envs.TPU_MESH_BASED_DP:
        return False
    return vllm_config.parallel_config.data_parallel_size > 1


def assign_device_indexes(vllm_config: VllmConfig) -> List[List[int]]:
    """Split the visible JAX device ids into one contiguous block per rank.

    Unlike the multi-process path the grouping is purely logical, so any split
    that divides the device list is valid -- there is no physical-topology box
    to satisfy. Each block is ``sharding_config.total_devices`` long, which is
    the per-rank mesh size once the ``data`` axis has been collapsed.

    A user-supplied ``device_indexes`` is honoured and sliced within, so mesh
    DP composes with an explicit device list instead of overriding it.

    Called once from the launching thread, never from a rank thread:
    ``jax.devices()`` initialises the backend on first call, and eight threads
    racing to do that is not worth finding out about the hard way.
    """
    sharding_config = vllm_config.sharding_config
    dp_size = vllm_config.parallel_config.data_parallel_size
    per_rank = sharding_config.total_devices

    base = sharding_config.device_indexes
    if not base:
        base = [d.id for d in jax.devices()]

    needed = dp_size * per_rank
    if len(base) < needed:
        raise ValueError(
            f"Mesh-based DP needs {needed} devices "
            f"(dp_size={dp_size} x {per_rank} per rank) but only "
            f"{len(base)} are visible. Lower --data-parallel-size or "
            f"--tensor-parallel-size.")
    groups = [base[r * per_rank:(r + 1) * per_rank] for r in range(dp_size)]
    logger.info("Mesh-based DP | dp_size=%d | %d device(s) per rank | %s",
                dp_size, per_rank, groups)
    return groups


class _RankThread:
    """One rank's thread, with a ``Process``-shaped ``exitcode``.

    The only reason this is not a bare ``threading.Thread`` is that a thread
    that raises leaves no trace a caller can test. Recording an exit code lets
    the group entrypoint decide the child process's own exit status the same
    way ``CoreEngineProcManager`` reads one off a spawned rank.
    """

    def __init__(self, name: str, target, args: tuple):
        self.name = name
        self.exitcode: Optional[int] = None
        self._thread = threading.Thread(target=self._run,
                                        args=(target, args),
                                        name=name,
                                        daemon=True)

    def _run(self, target, args) -> None:
        try:
            target(*args)
            self.exitcode = 0
        except BaseException:  # noqa: BLE001 - reported via exitcode
            logger.exception("Mesh-based DP | %s died", self.name)
            self.exitcode = 1

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _run_rank(vllm_config: VllmConfig, rank: int, device_indexes: List[int],
              engine_kwargs: Dict[str, Any], built: threading.Semaphore,
              engines: List[Any]) -> None:
    """Build one stock ``EngineCoreProc`` and run its stock busy loop.

    This is the body of ``EngineCoreProc.run_engine_core`` with the
    process-global parts lifted out into :func:`_run_rank_group`: process
    title, log decoration and signal handlers belong to the child's main
    thread and are installed once for all ranks, not once per rank. What is
    left -- the per-rank config, the engine build and the busy loop -- is
    unchanged.
    """
    from vllm.v1.engine.core import EngineCoreProc

    cfg = copy.deepcopy(vllm_config)
    cfg.sharding_config.device_indexes = device_indexes

    pc = cfg.parallel_config
    pc.data_parallel_rank_local = rank
    # Read by the phased profiler and by anything else that wants to know
    # which rank it is; `reconfigure_for_independent_dp_rank` deliberately
    # leaves it alone, which is why the multi-process path sets it too.
    pc.data_parallel_index = rank
    # Same call `run_engine_core` makes for a non-MoE rank: from here down the
    # engine believes it is a plain DP=1 engine, which is precisely what makes
    # these ranks independent.
    pc.reconfigure_for_independent_dp_rank()

    logger.info("Mesh-based DP | rank %d on devices %s", rank,
                cfg.sharding_config.device_indexes)

    engine = None
    try:
        engine = EngineCoreProc(vllm_config=cfg,
                                engine_index=rank,
                                **engine_kwargs)
        # Publish before releasing, so that once the last rank is built the
        # signal handler is guaranteed to see every engine.
        engines.append(engine)
    finally:
        # Release the config holder whether or not the build worked, or a
        # failing rank would strand the other seven behind it.
        built.release()

    try:
        engine.run_busy_loop()
    except SystemExit:
        logger.info("Mesh-based DP | rank %d busy loop exited", rank)
    finally:
        engine.shutdown()


def _run_rank_group(vllm_config: VllmConfig, ranks: List[int],
                    engine_kwargs: Dict[str, Any]) -> None:
    """Child-process entrypoint: host ``ranks`` as threads and wait for them.

    Stands where ``EngineCoreProc.run_engine_core`` stands on the process
    path, and does the same process-level setup it does -- register the
    config serializer, set the process title, decorate logs, install SIGTERM
    and SIGINT handlers -- once, for the group. Signals can only be caught on
    the main thread, which is exactly why this function exists and the work is
    not folded into :func:`_run_rank`.
    """
    from vllm.transformers_utils.config import \
        maybe_register_config_serialize_by_value
    from vllm.utils.system_utils import decorate_logs, set_process_title
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineShutdownState

    maybe_register_config_serialize_by_value()
    set_process_title(f"EngineCore_DP{ranks[0]}-{ranks[-1]}")
    decorate_logs()

    # One rank per thread means `len(ranks)` threads competing for a single
    # GIL. See `_SWITCH_INTERVAL_S`. Set here rather than in the parent: the
    # parent is the API server and has no reason to run coarse.
    prev = sys.getswitchinterval()
    sys.setswitchinterval(_SWITCH_INTERVAL_S)
    logger.info("Mesh-based DP | GIL switch interval %.4fs -> %.4fs", prev,
                _SWITCH_INTERVAL_S)

    groups = assign_device_indexes(vllm_config)

    # `_current_vllm_config` is a module-level global, not a thread-local, so
    # concurrent builders clobber each other: one rank's `load_model` context
    # exits and restores `None` while another is still in KV-cache init, which
    # fails with "Current vLLM config is not set". Holding the shared parent
    # config across the whole build makes every nested save/restore land on
    # this object instead of on `None`. The per-rank copies differ only in
    # device indexes and DP identity, neither of which is read through the
    # global, so the ranks stay correct.
    built = threading.Semaphore(0)
    engines: List[Any] = []

    threads = [
        _RankThread(
            name=f"EngineCore_DP{rank}",
            target=_run_rank,
            args=(vllm_config, rank, groups[rank], engine_kwargs, built,
                  engines),
        ) for rank in ranks
    ]

    def wakeup_engines() -> None:
        # Not safe from a signal handler: it takes each input queue's
        # non-reentrant mutex, which the interrupted thread may already hold.
        for engine in list(engines):
            engine.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    signal_callback = SignalCallback(wakeup_engines)

    def signal_handler(signum, frame):
        logger.info("[shutdown] Mesh-based DP: received signal=%s",
                    signal.Signals(signum).name)
        for engine in list(engines):
            engine.shutdown_state = EngineShutdownState.REQUESTED
        signal_callback.trigger()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        started = time.monotonic()
        with set_current_vllm_config(vllm_config):
            for t in threads:
                t.start()
            for _ in threads:
                built.acquire()
        # The one line that proves mesh DP engaged, rather than the config
        # having quietly fallen back to a single engine. Benchmark harnesses
        # gate on it, so it has to stay stable.
        logger.info("Mesh-based DP | %d engines ready in %.1fs", len(threads),
                    time.monotonic() - started)
        # Outside the config context: the build window is over, and holding it
        # for the life of the busy loops would leak the parent config into
        # anything that consults the global at request time.
        for t in threads:
            t.join()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal_callback.stop()

    failed = [t.name for t in threads if t.exitcode != 0]
    if failed:
        # Non-zero exit is how the parent's `monitor_engine_liveness` learns
        # this was a crash and not a clean shutdown.
        raise RuntimeError(f"Mesh-based DP rank(s) failed: {', '.join(failed)}")


class CoreEngineThreadManager(CoreEngineProcManager):
    """``CoreEngineProcManager`` that spawns one child for all local ranks.

    Only ``__init__`` differs. Everything the rest of vLLM asks of this object
    -- ``shutdown()``, ``monitor_engine_liveness()``, ``sentinels()``,
    ``finished_procs()`` -- is inherited, because after construction
    ``self.processes`` is an ordinary list of ``multiprocessing`` processes;
    it just happens to have one entry instead of ``local_engine_count``.

    Nothing upstream counts that list. ``wait_for_engine_startup`` waits for
    one handshake per entry in ``core_engines``, and each rank thread performs
    its own handshake, so the frontend still sees N engines come up.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type,
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        tensor_queue: Optional[Any] = None,
    ):
        if vllm_config.model_config.is_moe:
            # The MoE branch of `run_engine_core` builds a `DPEngineCoreProc`,
            # which joins a torch.distributed DP group -- one rank per
            # process, not per thread.
            raise NotImplementedError(
                "Mesh-based DP does not support MoE models; they need vLLM's "
                "DPEngineCoreProc, which requires one process per rank. Use "
                "TPU_MULTIPROCESS_DP=1 instead.")

        engine_kwargs: Dict[str, Any] = {
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }
        if client_handshake_address:
            engine_kwargs["client_handshake_address"] = client_handshake_address

        ranks = [start_index + i for i in range(local_engine_count)]

        self._request_shutdown_timeout = vllm_config.shutdown_timeout
        self.manager_stopped = threading.Event()
        self.failed_proc_name: Optional[str] = None

        # One child for all of them. `assign_device_indexes` runs inside it,
        # so the TPU backend is initialised there and only there.
        self.processes = [
            get_mp_context().Process(
                target=_run_rank_group,
                name=f"EngineCore_DP{ranks[0]}-{ranks[-1]}",
                kwargs={
                    "vllm_config": vllm_config,
                    "ranks": ranks,
                    "engine_kwargs": engine_kwargs,
                },
            )
        ]

        self._finalizer = weakref.finalize(self, shutdown_processes,
                                           self.processes)
        try:
            self.processes[0].start()
        finally:
            if self.finished_procs():
                self.shutdown()


def install() -> None:
    """Point vLLM's local-engine launcher at the thread manager.

    ``launch_core_engines`` resolves ``CoreEngineProcManager`` from
    ``vllm.v1.engine.utils``'s module globals at call time, and the two
    ``isinstance(launch.engine_manager, CoreEngineProcManager)`` guards in
    ``wait_for_engine_startup`` read the same global, so rebinding that one
    name both constructs the thread manager and keeps it recognised as the
    local-engine manager.

    ``vllm.v1.engine.core_client`` also imports the name, but only as a type
    annotation, so it does not need patching.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.engine import utils as engine_utils

    engine_utils.CoreEngineProcManager = CoreEngineThreadManager

    _INSTALLED = True
    logger.info("Mesh-based DP installed: DP ranks run as threads "
                "(CoreEngineProcManager -> CoreEngineThreadManager)")
