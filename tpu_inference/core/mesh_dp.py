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
stock busy loop; wave/idle coordination and the whole ZMQ control plane are
vLLM's own, unmodified. Four things are mesh-specific and nothing else is:

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
3. ``SlotAwareDPLBClient`` replaces one method of vLLM's DP load balancer so
   that routing is driven by the client's exact in-flight counts rather than
   by a 100 ms-old coordinator snapshot.
4. ``default_to_one_api_server`` stops ``vllm serve`` from starting one API
   server per DP rank, because 3 only works when one process owns the whole
   routing decision.

The last two are the tail fix and are documented where they are defined; the
numbers are below.

The child process is not an implementation detail. Threads are what make the
ranks share one JAX client and one weights cache, but they also make every
rank share a GIL with whatever else lives in that process -- and the process
that calls ``launch_core_engines`` under ``vllm serve`` is the API server,
running uvicorn, tokenization and detokenization. Hosting the ranks there
would put the frontend on the engines' GIL. Spawning one child keeps the
split the process path has: frontend in the parent, engines in a child.
It also keeps ``jax.devices()`` out of the parent, which must stay free of an
initialised TPU backend.

``install()`` therefore rebinds two names, both subclasses that override a
single method each. An earlier version of this
module substituted the engine core itself -- ``MeshDPEngineCore``,
``MeshDPEngineCoreProc``, a ``_SchedulerProxy`` and a hand-written rank router,
about 900 lines that re-implemented request routing, output merging and the
engine-core control plane. All of it duplicated behaviour vLLM already has for
multi-process DP, and every vLLM bump risked the two drifting apart. Deleting
it is the point of this file.

Where this lands, measured
--------------------------

The old engine core hid all ``dp_size`` ranks behind one ``EngineCore``, so
vLLM saw a single engine with a single queue of ``dp_size * max_num_seqs``
slots and no DP coordinator ran at all. This module gives vLLM the eight real
engines it thinks it has, each with its own scheduler and its own
``max_num_seqs`` slots -- the same shape multi-process DP has. Eight queues of
32 are not one queue of 256, so which queue a request lands in now matters,
and that is what items 3 and 4 above are for. With them in place, on v6e-8,
dp8/tp1, 1024x1024, sync scheduling:

==========  ================  ================  ================
metric      256 slots         512 slots         1024 slots
==========  ================  ================  ================
throughput  15,121 vs 14,160  22,823 vs 22,419  29,779 vs 26,869
P99 TTFT     2,510 vs  2,571   3,935 vs  3,828   6,030 vs  5,699
median e2e  16,477 vs 17,781  22,167 vs 22,308  32,168 vs 35,949
==========  ================  ================  ================

(new vs old, means of 3-7 runs per cell, all dp8/tp1 with no other flags set.
Throughput is +6.8% / +1.8% / +10.8% and P99 TTFT -2.4% / +2.8% / +5.8%
across the three sizes. Run-to-run spread is wide -- throughput at 1024 slots
ran 23.7k-29.7k on the old code alone -- so read throughput as "no worse"
rather than as a speedup, and read the tail as parity.)

Without items 3 and 4 the tail at 256 slots was 14,647 ms -- 5.7x the old
code -- while throughput and median were unchanged. That case is the hard one:
each rank holds exactly 32 slots, the benchmark keeps exactly 256 requests in
flight, so the system has zero slack and a single misrouted request waits out
a whole generation. Two things caused the misroutes. ``vllm serve`` defaults
``api_server_count`` to ``data_parallel_size``, so eight independent routers
each saw a different slice of the load, and the only shared signal between
them was a coordinator snapshot up to 100 ms stale. Fixing either one alone
did not help (P99 stayed at 15-17 s); fixing both together did. At 1024 slots
no rank ever fills, so routing barely matters and the numbers move little
either way.

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
from collections import Counter
from typing import Any, Dict, List, Optional

import jax
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.core_client import DPLBAsyncMPClient
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


class SlotAwareDPLBClient(DPLBAsyncMPClient):
    """vLLM's DP load balancer, routing on what it knows instead of what it saw.

    Stock ``get_core_engine_for_request`` scores an engine as
    ``max(client_count * engine_inflight, waiting + running)``, plus a KV
    pressure penalty when ``waiting`` is non-zero. ``engine_inflight`` is exact:
    it goes up when this client sends a request and down the moment the engine
    reports it finished. ``waiting``/``running`` come from a ``DPCoordinator``
    snapshot that is refreshed at most every 100 ms
    (``coordinator.py:261``), and nothing corrects it downward in between.

    The staleness is harmless while engines have spare slots and fatal when they
    do not. A closed-loop client submits the replacement request microseconds
    after the completion, so the snapshot is *always* stale for the one engine
    that just freed a slot: it still reads as full, the ``max()`` lets that
    larger stale value win over the exact count, some genuinely full engine
    ties with it, and the request is committed to a queue it will sit in for a
    whole generation. There is no work stealing between engines, so that
    commitment is final. Measured at dp8/tp1, ``max_num_seqs=32``, 256
    concurrent -- every rank exactly full -- the engines run like this:

        running = 32 31 32 32 32 32 32 25   waiting = 8   free slots = 8

    Eight requests queued, eight slots idle, and it persists because the client
    cannot start anything new until one of the queued requests finishes.

    Patching the snapshot on completion does not fix it: ``lb_engines`` is
    rebound wholesale by the next broadcast, so the correction is erased within
    100 ms and the tail does not move (measured: P99 14.9 s, unchanged). The
    stale term has to stop describing load this client already knows about.

    So this subclass splits the estimate by who owns the information::

        score = engine_inflight[e] + max(0, (waiting + running)[e] - mine[e])

    where ``mine[e]`` is this client's ``engine_inflight[e]`` sampled by the
    ``lb_engines`` setter below, at the instant the snapshot arrived. The
    subtraction removes this client's own
    contribution from the snapshot, leaving only what the *other* front ends
    have on that engine; adding back the live count makes our own half exact
    and instantaneous. Nothing is stale except other clients' churn during one
    100 ms window, which at these rates is about one request.

    Simulated at dp8, 32 slots/rank, 256 concurrent, 20k completions, counting
    decisions that queued a request while a slot was free elsewhere:

    ==============  ===========  ===========
    policy          1 front end  8 front ends
    ==============  ===========  ===========
    stock           6877         5612
    inflight only   0            7222
    this one        0            4
    ==============  ===========  ===========

    Scoring on ``engine_inflight`` alone is exact for one front end and useless
    for eight: each client balances its own share perfectly, but the shares are
    uneven and their remainders stack on the same engines. The snapshot is the
    only thing that can see across clients, which is why it is kept -- just not
    inside a ``max()`` that lets it overrule what we know for certain.

    Two upstream details are dropped. The local ``current_counts[i][0] +=
    client_count`` bump is gone because ``engine_inflight`` already records our
    own sends, exactly and immediately -- and leaving it in would inflate the
    snapshot we now subtract from. The KV-pressure penalty on ``waiting`` is
    gone because it only discriminates when prefix caching makes queues drain
    at different rates per engine; here it can only re-admit stale-snapshot
    noise into the decision.

    Not mesh-specific -- vLLM's multi-process DP path has the same tail for the
    same reason -- but it is installed here because this is the path that has to
    be well behaved at exactly 100% slot occupancy.
    """

    _inflight_at_snapshot: Counter = Counter()

    @property
    def lb_engines(self) -> Any:
        return self._lb_engines

    @lb_engines.setter
    def lb_engines(self, counts: Any) -> None:
        """Sample our own in-flight counts the instant a snapshot lands.

        The sample has to be taken here rather than lazily at the next routing
        decision. Our own completions between the two would be missing from
        ``mine`` while still being counted in the snapshot, so ``others`` would
        absorb them, the two ``inflight`` terms in the score would cancel, and
        the policy would silently decay back into scoring on the stale
        snapshot alone.

        ``DPLBAsyncMPClient.__init__`` assigns ``lb_engines`` before
        ``engine_inflight`` exists, hence the guard.
        """
        self._lb_engines = counts
        inflight = getattr(self, "engine_inflight", None)
        if inflight is not None:
            self._inflight_at_snapshot = inflight.copy()

    def get_core_engine_for_request(self, request: Any) -> Any:
        from vllm.v1.pool.late_interaction import (
            get_late_interaction_engine_index)

        # Both short circuits pin the request to a rank for correctness, not
        # for balance, so they bypass scoring entirely -- same as upstream.
        if (eng_index := request.data_parallel_rank) is None and (
                eng_index := get_late_interaction_engine_index(
                    request.pooling_params, len(self.core_engines))) is None:
            counts = self.lb_engines
            engines = self.core_engines
            inflight = self.engine_inflight
            num_engines = len(counts)
            mine = self._inflight_at_snapshot

            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Scan from a rotating origin so that ties -- which is every
                # decision while the engines are empty -- go round-robin
                # instead of always landing on rank 0.
                idx = (self.eng_start_index + i) % num_engines
                engine = engines[idx]
                waiting, running, _kv_usage = counts[idx]
                others = (waiting + running) - mine[engine]
                score = inflight[engine] + (others if others > 0 else 0)
                if score < min_score:
                    min_score = score
                    eng_index = idx
            self.eng_start_index = (self.eng_start_index + 1) % num_engines

        chosen_engine = self.core_engines[eng_index]
        # Recorded so that an abort can be forwarded to the right engine, and
        # so the completion decrements the counter we just incremented.
        self.reqs_in_flight[request.request_id] = chosen_engine
        self.engine_inflight[chosen_engine] += 1
        return chosen_engine


def default_to_one_api_server(parser: Any) -> None:
    """Stop ``vllm serve`` from defaulting to one API server per DP rank.

    ``ServeSubcommand.cmd`` sets ``api_server_count = data_parallel_size``
    whenever the flag was not given (``cli/serve.py:119``), so dp8 gets eight
    front ends behind ``SO_REUSEPORT``. Each front end runs its own
    ``DPLBAsyncMPClient`` with its own ``engine_inflight``, and they never
    compare notes except through the 100 ms coordinator snapshot.

    That is what breaks routing. ``SO_REUSEPORT`` splits connections badly --
    measured at 256 concurrent, one front end held 98 of them and another held
    5 -- so no front end can infer the global picture from its own share, and
    the only cross-client signal is a snapshot that is always stale for the
    engine that just freed a slot. Free slots and queued requests then coexist
    for a whole generation:

        waiting = 0 1 0 0 4 0 3 0   running = 31 32 31 29 32 30 32 30

    unchanged across 14 seconds of a run where a generation takes 15.

    With a single front end the router's ``engine_inflight`` is not a sample of
    the load, it *is* the load, so the replacement for a completed request goes
    back to the engine that just freed the slot, every time. Measured on v6e-8,
    dp8/tp1, 1024x1024, ``max_num_seqs=32``, 256 concurrent:

    ================  ===========  ============  ===========
    front ends        router       throughput    P99 TTFT
    ================  ===========  ============  ===========
    8 (vLLM default)  stock        14,019        14,647 ms
    8                 in-flight    13,915        16,538 ms
    1                 stock        13,423        15,025 ms
    1                 in-flight    17,363         2,334 ms
    ================  ===========  ============  ===========

    Both changes are needed and neither works alone. The last row also beats
    the pre-refactor engine core, which managed 14,160 and 2,571 ms.

    This sets the *default*, so ``--api-server-count`` still wins if given.
    Raise it if one front end cannot keep up with tokenization and HTTP, and
    accept the tail that comes back with it.

    Called from ``TpuPlatform.pre_register_and_update``, which vLLM invokes at
    the end of ``AsyncEngineArgs.add_cli_args`` -- after ``make_arg_parser``
    has registered ``--api-server-count``, so ``set_defaults`` finds the action
    and replaces its default rather than being overwritten by it.
    """
    if parser is None or not envs.TPU_MESH_BASED_DP:
        return
    parser.set_defaults(api_server_count=1)


def install() -> None:
    """Point vLLM's local-engine launcher and DP router at our subclasses.

    ``launch_core_engines`` resolves ``CoreEngineProcManager`` from
    ``vllm.v1.engine.utils``'s module globals at call time, and the two
    ``isinstance(launch.engine_manager, CoreEngineProcManager)`` guards in
    ``wait_for_engine_startup`` read the same global, so rebinding that one
    name both constructs the thread manager and keeps it recognised as the
    local-engine manager.

    ``make_async_mp_client`` resolves ``DPLBAsyncMPClient`` from
    ``vllm.v1.engine.core_client``'s globals the same way
    (``core_client.py:138``), so the router is rebound identically. This runs in
    every API server process, not just the one that launches the engines, so
    each front end gets the slot-aware router.

    ``vllm.v1.engine.core_client`` also imports ``CoreEngineProcManager``, but
    only as a type annotation, so it does not need patching.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.engine import core_client
    from vllm.v1.engine import utils as engine_utils

    engine_utils.CoreEngineProcManager = CoreEngineThreadManager
    core_client.DPLBAsyncMPClient = SlotAwareDPLBClient

    _INSTALLED = True
    logger.info("Mesh-based DP installed: DP ranks run as threads "
                "(CoreEngineProcManager -> CoreEngineThreadManager), "
                "DP router scores on exact in-flight counts "
                "(DPLBAsyncMPClient -> SlotAwareDPLBClient)")
