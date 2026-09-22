# Mesh-based data parallelism

`TPU_MESH_BASED_DP=1` runs `dp_size` engine cores inside one process, each with
its own `jax.sharding.Mesh` and its own rank thread, behind a single vLLM
engine core. The alternative, SPMD DP (`TPU_MESH_BASED_DP=0`), fuses DP into
one sharded XLA program driven by one runner over one global `InputBatch`.

The two are bound by different things, which is the whole point: mesh DP is
host-dispatch bound, SPMD DP is device bound. Neither wins everywhere.

## Best measured configuration

v7x-8 (8 chips, 2 cores/chip), Qwen3-0.6B, `vllm serve` + InferenceX
`benchmark_serving.py` @ `89ce6098e`, 1024 requests of 16384 in / 8192 out,
`--max-concurrency=218`, `--ignore-eos`, client in-pod against localhost.

```bash
cd /                    # a cwd inside a source tree shadows the installed package

env TPU_MESH_BASED_DP=1 TPU_MULTIPROCESS_DP=0 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    FUSE_H2D_METADATA=0 CONTINUE_DECODE_AFTER_PREFILL=0 \
    HOST_PHASE_STATS=0 CONTINUE_DECODE_GATE_STATS=0 \
    USE_BATCHED_RPA_KERNEL=1 SHARDED_SAMPLING=0 \
  python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
    --port 8000 \
    --max-model-len=24576 \
    --max-num-seqs=28 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization=0.9 \
    --no-async-scheduling \
    --block-size=256 \
    --additional-config '{"enable_continue_decode": true, "max_decode_steps": 64}'
```

| arm | dp/tp | out tok/s | total tok/s | med TTFT | P99 TTFT | med TPOT | P99 ITL |
|---|---|---|---|---|---|---|---|
| **mesh** | 8/1 | **6624** | 19872 | 1535 | 67671 | 32.7 | 53 |
| mesh | 4/2 | 6208 | 18623 | 664 | 18174 | 34.3 | 53 |
| SPMD | 4/2 | 5779 | 17337 | 885 | 16874 | 22.2 | 1588 |
| SPMD | 8/1 | 5707 | 17121 | 644 | 10958 | 23.7 | 1738 |

Mesh dp8/tp1 wins throughput by 14.6% over the best SPMD arm and is the worst
arm on P99 TTFT. If you are serving to a tail-latency SLO rather than
maximising tokens per second, take mesh dp4/tp2: it gives up 6% of throughput
and cuts P99 TTFT by 3.7x.

Reproduced from this branch at `e21e6949e`: 6619.79 out tok/s, 0.06% off.

## Things that are easy to get wrong

- **`--max-num-seqs` is per rank under mesh DP** and global under SPMD. Keep
  `max_num_seqs * dp_size` at or above the client's concurrency, or requests
  queue at admission and the comparison is not like-for-like.
- **`cd -` out of any source tree before launching.** `sys.path[0] == ''` beats
  every meta-path finder, so a run started inside a checkout imports that
  checkout regardless of which interpreter or venv was selected.
- **Never trust the arm label; gate on evidence.** A mesh run that silently
  fell back to one core still serves happily. Require
  `Mesh-based DP | N engines ready` in the log before believing a mesh number,
  and require `Mesh-based DP installed:` to be *absent* before believing an
  SPMD one. The KV cache size is a second check: the two arms should report
  the same aggregate (here 787,712/rank x 8 = 6.30M, vs 6,302,976 for SPMD).
- **Startup is slow** — 190-300 s to build 8 engines. Keep
  `VLLM_ENGINE_READY_TIMEOUT_S` generous.

## Regime

Mesh DP wins where the host cannot keep the device fed and loses where the
device is the bottleneck. At ~1k input the workload is host-dispatch bound and
mesh dp8/tp1 beats mesh dp4/tp2 by 55%; at 16k input it is device bound and
that margin collapses to 6%. Check the input length before generalising any
number here.
