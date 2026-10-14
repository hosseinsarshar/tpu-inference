# Copyright 2026 Google LLC
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
"""CPU tests for the `decode_only` flag of the v3 ragged paged attention.

The real kernel needs a TPU (Pallas interpret mode does not support its
reshaped DMA refs), so these tests replace `pl.pallas_call` with a fake that
records which kernels the wrapper launches.
"""

from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from tpu_inference.kernels.ragged_paged_attention.v3 import kernel as rpa

PAGE_SIZE = 16
HEAD_DIM = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
MAX_NUM_SEQS = 8
MAX_NUM_TOKENS = 8
PAGES_PER_SEQ = 3
# Three decode sequences (q_len 1 each), padded to MAX_NUM_SEQS.
KV_LENS = (5, 17, 33)
NUM_SEQS = len(KV_LENS)

# Explicit sizes, so the wrapper does not query the TPU.
KERNEL_KWARGS = dict(
    d_block_sizes=(1, 32, 1, 16),
    p_block_sizes=(4, 32, 4, 16),
    m_block_sizes=(8, 32, 8, 16),
    vmem_limit_bytes=64 * 1024 * 1024,
)


def make_decode_batch(dtype=jnp.bfloat16):
    """Returns the kernel's positional inputs for a decode-only batch."""
    num_pages = MAX_NUM_SEQS * PAGES_PER_SEQ
    q = jnp.ones((MAX_NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), dtype)
    k = jnp.ones((MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), dtype)
    v = jnp.ones((MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), dtype)
    kv_cache = jnp.zeros(
        rpa.get_kv_cache_shape(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                               dtype), dtype)
    kv_lens = jnp.array(KV_LENS + (0, ) * (MAX_NUM_SEQS - NUM_SEQS), jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array(
        list(range(NUM_SEQS + 1)) + [NUM_SEQS] * (MAX_NUM_SEQS - NUM_SEQS),
        jnp.int32)
    # (decode_end, prefill_end, mixed_end): every sequence is a decode.
    distribution = jnp.array([NUM_SEQS] * 3, jnp.int32)
    return q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution


class DecodeOnlyLaunchTest(parameterized.TestCase):
    """Checks which RPA kernels the wrapper launches, without a TPU."""

    def setUp(self):
        super().setUp()
        # Take the v6 path, which has no v7-only HBM memory-space constraints.
        self.enter_context(
            mock.patch.object(rpa, "get_tpu_version", return_value=6))
        # A cached trace would skip the fake pallas_call below, and a trace
        # built with the fake must not leak into other tests.
        jax.clear_caches()
        self.addCleanup(jax.clear_caches)

    def _launched_kernels(self, **kwargs):
        names = []

        def fake_pallas_call(kernel, *, out_shape, name, **unused_kwargs):
            del kernel, unused_kwargs
            names.append(name)

            def run(*args):
                # 7 scalar prefetches, then q, kv and kv_cache.
                q, _, kv_cache = args[-3:]
                self.assertEqual([q.shape, kv_cache.shape],
                                 [s.shape for s in out_shape])
                return q, kv_cache

            return run

        # This goes through jax.jit, so it also checks that decode_only is
        # a static argument.
        with mock.patch.object(rpa.pl, "pallas_call", fake_pallas_call):
            out, _ = rpa.ragged_paged_attention(*make_decode_batch(),
                                                **KERNEL_KWARGS, **kwargs)
        self.assertEqual(out.shape, (MAX_NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM))
        # Names look like "RPAd-p_16-bq_1_1-bkv_32_16"; keep the case part.
        return [name.split("-")[0] for name in names]

    @parameterized.named_parameters(
        ("default", False, None, ["RPAd", "RPAm"]),
        ("chunk_prefill", False, 4, ["RPAd", "RPAp", "RPAm"]),
        ("decode_only", True, None, ["RPAd"]),
        ("decode_only_chunk_prefill", True, 4, ["RPAd"]),
    )
    def test_launched_kernels(self, decode_only, chunk_prefill_size, expected):
        launched = self._launched_kernels(
            decode_only=decode_only, chunk_prefill_size=chunk_prefill_size)
        self.assertEqual(launched, expected)

    def test_skipped_cases_have_no_sequences(self):
        # On a decode-only batch the prefill and mixed kernels get an empty
        # sequence range, so skipping their launches drops no work.
        distribution = make_decode_batch()[-1]
        ranges = {
            case: tuple(int(i) for i in case.get_range(distribution))
            for case in rpa.RpaCase
        }
        self.assertEqual(
            ranges, {
                rpa.RpaCase.DECODE: (0, NUM_SEQS),
                rpa.RpaCase.PREFILL: (NUM_SEQS, NUM_SEQS),
                rpa.RpaCase.MIXED: (NUM_SEQS, NUM_SEQS),
            })


if __name__ == "__main__":
    absltest.main()
