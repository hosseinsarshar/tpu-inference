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

1. ``CoreEngineThreadManager`` starts each rank on a ``threading.Thread``
   rather than a ``multiprocessing.Process``. It matches
   ``CoreEngineProcManager``'s constructor and its ``sentinels()`` /
   ``finished_procs()`` / ``shutdown()`` surface, so ``launch_core_engines``
   and ``wait_for_engine_startup`` drive it without knowing the difference.
2. Each rank's ``VllmConfig`` copy carries its own
   ``sharding_config.device_indexes``. That is an existing, first-class field
   ``TPUWorker.init_device`` already honours (``tpu_worker.py:379``), and it
   is what pins the rank's mesh to its own chips.

``install()`` therefore rebinds exactly one name. An earlier version of this
module substituted the engine core itself -- ``MeshDPEngineCore``,
``MeshDPEngineCoreProc``, a ``_SchedulerProxy`` and a hand-written rank router,
about 900 lines that re-implemented request routing, output merging and the
engine-core control plane. All of it duplicated behaviour vLLM already has for
multi-process DP, and every vLLM bump risked the two drifting apart. Deleting
it is the point of this file.

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
import sys
import threading
import weakref
from multiprocessing import connection
from typing import Any, Dict, List, Optional

import jax
from vllm.config import VllmConfig, set_current_vllm_config

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

# How long `shutdown()` waits for a rank thread to leave its busy loop. Threads
# cannot be killed, so this is a report-and-continue deadline, not a guarantee.
_JOIN_TIMEOUT_S = 30.0


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
    """A rank thread wearing just enough of a ``Process``'s face.

    ``wait_for_engine_startup`` registers ``proc.sentinel`` with a
    ``zmq.Poller`` and ``monitor_engine_liveness`` passes it to
    ``multiprocessing.connection.wait``, both of which want a file descriptor
    that becomes readable when the rank dies. A thread has no such fd, so it
    gets a pipe whose write end is closed on the way out -- same signal, same
    poll, no special-casing in the caller.
    """

    def __init__(self, name: str, target, args: tuple):
        self.name = name
        self.exitcode: Optional[int] = None
        self._r_fd, self._w_fd = os.pipe()
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
        finally:
            # Closing the write end is what wakes every poller watching this
            # rank. Do it last, and do it exactly once.
            try:
                os.close(self._w_fd)
            except OSError:
                pass

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def sentinel(self) -> int:
        return self._r_fd

    def close_sentinel(self) -> None:
        try:
            os.close(self._r_fd)
        except OSError:
            pass


def _run_rank(vllm_config: VllmConfig, rank: int, device_indexes: List[int],
              engine_kwargs: Dict[str, Any],
              built: threading.Semaphore) -> None:
    """Build one stock ``EngineCoreProc`` and run its stock busy loop.

    This is ``EngineCoreProc.run_engine_core`` with the three things a thread
    cannot do removed -- ``signal.signal`` (main thread only),
    ``set_process_title`` and ``decorate_logs`` (both process-global) -- and
    nothing added. Shutdown arrives over ZMQ from the frontend exactly as it
    does for a spawned rank, so the signal handling is not merely skipped, it
    is unnecessary.
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


class CoreEngineThreadManager:
    """``CoreEngineProcManager``, with threads.

    Constructor signature and public surface are copied from the upstream
    class on purpose: ``launch_core_engines`` builds it by keyword and
    ``wait_for_engine_startup`` reaches for ``sentinels()`` and
    ``finished_procs()``, so matching it exactly is what lets the rest of the
    DP machinery stay untouched.
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

        prev = sys.getswitchinterval()
        sys.setswitchinterval(_SWITCH_INTERVAL_S)
        logger.info("Mesh-based DP | GIL switch interval %.4fs -> %.4fs", prev,
                    _SWITCH_INTERVAL_S)

        self.manager_stopped = threading.Event()
        self.failed_proc_name: Optional[str] = None

        # `_current_vllm_config` is a module-level global, not a thread-local,
        # so concurrent builders clobber each other: one rank's `load_model`
        # context exits and restores `None` while another is still in KV-cache
        # init, which fails with "Current vLLM config is not set". Holding the
        # shared parent config across the whole build makes every nested
        # save/restore land on this object instead of on `None`. The per-rank
        # copies differ only in device indexes and DP identity, neither of
        # which is read through the global, so the ranks stay correct.
        #
        # This has to be held from a side thread rather than from __init__:
        # each rank's `EngineCoreProc.__init__` blocks in its ZMQ handshake
        # until the frontend answers, and the frontend does not answer until
        # after this constructor returns.
        built = threading.Semaphore(0)

        groups = assign_device_indexes(vllm_config)
        self.processes: List[_RankThread] = [
            _RankThread(
                name=f"EngineCore_DP{start_index + i}",
                target=_run_rank,
                args=(vllm_config, start_index + i,
                      groups[start_index + i], engine_kwargs, built),
            ) for i in range(local_engine_count)
        ]

        def hold_config() -> None:
            with set_current_vllm_config(vllm_config):
                for _ in self.processes:
                    built.acquire()

        self._config_holder = threading.Thread(target=hold_config,
                                               name="mesh-dp-config-holder",
                                               daemon=True)
        self._config_holder.start()

        self._finalizer = weakref.finalize(self, _shutdown_threads,
                                           self.processes)
        for t in self.processes:
            t.start()

    def shutdown(self, timeout: Optional[float] = None) -> None:
        self.manager_stopped.set()
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer()

    def monitor_engine_liveness(self) -> None:
        """Mirror of the upstream method; see :class:`_RankThread`."""
        sentinel_to_proc = {t.sentinel: t for t in self.processes}
        sentinels = set(sentinel_to_proc)

        while sentinels and not self.manager_stopped.is_set():
            died = connection.wait(list(sentinels), timeout=1)
            for sentinel in died:
                sentinels.discard(sentinel)
                proc = sentinel_to_proc.pop(sentinel, None)
                if (proc is not None and proc.exitcode != 0
                        and not self.manager_stopped.is_set()):
                    self.failed_proc_name = proc.name
            if died:
                break

        self.shutdown()

    def sentinels(self) -> list:
        return [t.sentinel for t in self.processes]

    def finished_procs(self) -> Dict[str, int]:
        return {
            t.name: t.exitcode
            for t in self.processes if t.exitcode is not None
        }


def _shutdown_threads(threads: List[_RankThread]) -> None:
    """Wait for the rank threads to leave their busy loops.

    A thread cannot be terminated, so unlike the process manager there is no
    escalation to SIGKILL. The frontend has already sent the shutdown message
    over ZMQ by the time this runs; all that is left is to notice whether the
    ranks acted on it, and to say so if they did not.
    """
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    stuck = [t.name for t in threads if t.is_alive()]
    if stuck:
        logger.warning(
            "Mesh-based DP | %d rank thread(s) still running after %.0fs: %s",
            len(stuck), _JOIN_TIMEOUT_S, ", ".join(stuck))
    for t in threads:
        t.close_sentinel()


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
