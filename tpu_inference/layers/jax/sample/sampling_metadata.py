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

import functools
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.runner.input_batch import InputBatch
from tpu_inference.utils import device_array

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    top_k=0,
    top_p=1.0,
)


@functools.lru_cache(maxsize=32)
def _cached_collision_dummy(mesh: Mesh, size: int) -> jax.Array:
    """The compile-cache discriminator array, built once per (mesh, size)."""
    return device_array(
        mesh,
        np.zeros((size, ), dtype=np.int32),
        sharding=jax.sharding.NamedSharding(mesh,
                                            jax.sharding.PartitionSpec()))


_SAMPLING_META_CACHE: dict = {}


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "temperature",
        "top_k",
        "top_p",
        "_cache_collision_dummy",
    ],
    meta_fields=["do_sampling", "logprobs"],
)
@dataclass
class TPUSupportedSamplingMetadata:
    temperature: Optional[jnp.ndarray] = None
    top_k: Optional[jnp.ndarray] = None
    top_p: Optional[jnp.ndarray] = None
    _cache_collision_dummy: Optional[jnp.ndarray] = None
    do_sampling: bool = False
    logprobs: bool = False

    @classmethod
    def from_input_batch(
        cls,
        mesh: Mesh,
        input_batch: InputBatch,
        padded_num_reqs: int,
        req_indices_dp: dict,
        sharding: Optional[jax.sharding.Sharding] = None,
    ) -> "TPUSupportedSamplingMetadata":
        needs_logprobs = input_batch.max_num_logprobs > 0 if input_batch.max_num_logprobs else False

        # Use a dummy tensor with a unique shape for each logprobs config.
        # This avoids persistent cache collisions.
        cache_collision_dummy = _cached_collision_dummy(
            mesh, 1 if needs_logprobs else 2)

        if input_batch.all_greedy:
            return cls(do_sampling=False,
                       logprobs=needs_logprobs,
                       _cache_collision_dummy=cache_collision_dummy)

        dp_size = len(req_indices_dp)
        n = input_batch.num_reqs
        temp_cur = input_batch.temperature_cpu[:n]
        top_k_cur = input_batch.top_k_cpu[:n]
        top_p_cur = input_batch.top_p_cpu[:n]
        is_uniform = (n > 0 and np.all(temp_cur == temp_cur[0])
                      and np.all(top_k_cur == top_k_cur[0])
                      and np.all(top_p_cur == top_p_cur[0]))
        if is_uniform:
            rank_counts = tuple(
                len(req_indices_dp.get(r, ())) for r in range(dp_size))
            u_key = (
                padded_num_reqs,
                needs_logprobs,
                sharding,
                float(temp_cur[0]),
                int(top_k_cur[0]),
                float(top_p_cur[0]),
                rank_counts,
            )
            cached_u = _SAMPLING_META_CACHE.get((mesh, "uniform"))
            if cached_u is not None and cached_u[0] == u_key:
                return cached_u[1]

        def fill_slice(cpu_tensor_np: np.ndarray,
                       fill_val: float) -> np.ndarray:
            out_tensor = np.full((padded_num_reqs, ),
                                 fill_val,
                                 dtype=cpu_tensor_np.dtype)

            assert padded_num_reqs % dp_size == 0, f"padded_num_reqs ({padded_num_reqs}) must be divisible by dp_size ({dp_size})"
            padded_num_reqs_per_dp_rank = padded_num_reqs // dp_size
            for dp_rank in range(dp_size):
                req_indices = req_indices_dp.get(dp_rank, [])
                if req_indices:
                    start_idx = dp_rank * padded_num_reqs_per_dp_rank
                    out_tensor[start_idx:start_idx +
                               len(req_indices)] = cpu_tensor_np[req_indices]

            return out_tensor

        temp_tensor = fill_slice(input_batch.temperature_cpu,
                                 DEFAULT_SAMPLING_PARAMS["temperature"])
        top_k_tensor = fill_slice(input_batch.top_k_cpu,
                                  DEFAULT_SAMPLING_PARAMS["top_k"])
        top_p_tensor = fill_slice(input_batch.top_p_cpu,
                                  DEFAULT_SAMPLING_PARAMS["top_p"])

        # Slice persistent device tensors to a fixed pre-compiled padded shape
        # in a single batched device_array call.
        temperature_dev, top_p_dev, top_k_dev = device_array(
            mesh,
            (temp_tensor[:padded_num_reqs], top_p_tensor[:padded_num_reqs],
             top_k_tensor[:padded_num_reqs]),
            sharding=sharding,
        )
        result = cls(
            temperature=temperature_dev,
            top_p=top_p_dev,
            top_k=top_k_dev,
            _cache_collision_dummy=cache_collision_dummy,
            do_sampling=not input_batch.all_greedy,
            logprobs=needs_logprobs,
        )
        if is_uniform:
            _SAMPLING_META_CACHE[(mesh, "uniform")] = (u_key, result)
        return result
