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

import inspect
import random
from unittest import mock

import pytest

from tpu_inference.core.sched.utils import \
    patch_vllm_scheduler_for_continue_decode


@pytest.fixture(autouse=True)
def restore_scheduler_patch():
    """The continue_decode patch is a process-global, irreversible monkeypatch
    (including Scheduler.__init__, which forces num_lookahead_tokens >= 9).
    CI runs all of tests/ in one pytest process, so restore the originals after
    each test to avoid leaking the patch into unrelated tests."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    originals = (Scheduler._update_request_with_output, Scheduler.__init__,
                 AsyncScheduler._update_request_with_output)
    yield
    (Scheduler._update_request_with_output, Scheduler.__init__,
     AsyncScheduler._update_request_with_output) = originals
    for cls in (Scheduler, AsyncScheduler):
        if "_continue_decode_patched" in cls.__dict__:
            del cls._continue_decode_patched


def _make_request(num_computed_tokens=10, num_output_placeholders=1):
    request = mock.MagicMock()
    request.num_computed_tokens = num_computed_tokens
    request.num_output_placeholders = num_output_placeholders
    return request


def _make_scheduler(cls):
    # spec= so getattr(scheduler, "_cd_stale_in_flight", False) is False (a
    # bare MagicMock would auto-create a truthy attribute) and so the zero-arg
    # super() in the wrapped async original resolves (isinstance check).
    # Instance attributes read by the base implementation are set explicitly.
    scheduler = mock.MagicMock(spec=cls)
    scheduler.max_model_len = 1024
    return scheduler


class TestContinueDecodeSchedulerPatch:

    def test_patched_signatures_accept_is_stale(self):
        """vLLM passes is_stale=... as a kwarg from update_from_output; the
        patched wrappers must accept it or the EngineCore dies with a
        TypeError."""
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        for cls in (Scheduler, AsyncScheduler):
            params = inspect.signature(
                cls._update_request_with_output).parameters
            assert "is_stale" in params, (
                f"{cls.__name__}._update_request_with_output must accept "
                f"is_stale (got {list(params)})")
            assert any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in params.values()), (
                           f"{cls.__name__}._update_request_with_output must "
                           f"pass through future kwargs (**kwargs)")

    def test_sync_patch_advances_num_computed_tokens_unless_stale(self):
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler = _make_scheduler(Scheduler)
        with mock.patch("vllm.v1.core.sched.scheduler.check_stop",
                        return_value=False):
            # Fresh output: 3 tokens returned for 1 scheduled step -> advance
            # num_computed_tokens by the extra 2 on-device tokens.
            request = _make_request(num_computed_tokens=10)
            Scheduler._update_request_with_output(scheduler,
                                                  request, [1, 2, 3],
                                                  is_stale=False)
            assert request.num_computed_tokens == 12

            # Stale output predates the preemption rollback -> no advance.
            request = _make_request(num_computed_tokens=10)
            Scheduler._update_request_with_output(scheduler,
                                                  request, [1, 2, 3],
                                                  is_stale=True)
            assert request.num_computed_tokens == 10

    def test_async_patch_stale_handling(self):
        """The async original calls super() WITHOUT forwarding is_stale, so
        staleness reaches the patched base via the _cd_stale_in_flight flag.
        Assert both counters end-to-end through the full async chain."""
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        scheduler = _make_scheduler(AsyncScheduler)
        with mock.patch("vllm.v1.core.sched.scheduler.check_stop",
                        return_value=False):
            # Fresh output: placeholders pre-compensated by (N - 1) so the
            # original's -= N lands back at 0; num_computed_tokens advances.
            request = _make_request(num_computed_tokens=10,
                                    num_output_placeholders=1)
            AsyncScheduler._update_request_with_output(scheduler,
                                                       request, [1, 2, 3],
                                                       is_stale=False)
            assert request.num_output_placeholders == 0
            assert request.num_computed_tokens == 12

            # Stale output: placeholders were zeroed at preemption and
            # num_computed_tokens was rolled back; neither may move.
            request = _make_request(num_computed_tokens=10,
                                    num_output_placeholders=1)
            AsyncScheduler._update_request_with_output(scheduler,
                                                       request, [1, 2, 3],
                                                       is_stale=True)
            assert request.num_output_placeholders == 1
            assert request.num_computed_tokens == 10
            # The flag must not leak past the call.
            assert scheduler._cd_stale_in_flight is False


_EOS = 151645
_EOS2 = 151643


def _real_request(prompt_len,
                  max_tokens,
                  stop_token_ids,
                  ignore_eos,
                  min_tokens=0,
                  block_hasher=None):
    """A vLLM Request whose sampling params went through the same EOS setup
    as a served request (primary EOS plus the generation-config EOS ids)."""
    from vllm import SamplingParams
    from vllm.v1.request import Request

    params = SamplingParams(max_tokens=max_tokens,
                            min_tokens=min(min_tokens, max_tokens),
                            stop_token_ids=stop_token_ids,
                            ignore_eos=ignore_eos)
    params.update_from_generation_config({"eos_token_id": [_EOS, _EOS2]},
                                         eos_token_id=_EOS)
    return Request(request_id="r",
                   prompt_token_ids=list(range(prompt_len)),
                   sampling_params=params,
                   pooling_params=None,
                   block_hasher=block_hasher)


def _random_tokens(rng, n):
    # Mostly ordinary tokens, with EOS, the second EOS id and the stop id 7
    # at random positions.
    special = [3, 7, _EOS, _EOS2]
    return [
        rng.choice(special) if rng.random() < 0.02 else rng.randint(10, 1000)
        for _ in range(n)
    ]


def _request_state(request):
    return (list(request.output_token_ids), list(request.all_token_ids),
            list(request.block_hashes), request.status, request.stop_reason,
            request.num_computed_tokens, request.num_output_placeholders)


def _cached_num_tokens(scheduler):
    """The token counts AsyncScheduler passed to cache_blocks(), in order."""
    calls = scheduler.kv_cache_manager.cache_blocks.call_args_list
    return [c.args[1] for c in calls]


class TestContinueDecodeBulkAppend:

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_matches_per_token_update(self, use_async, seed):
        """Appending the leading tokens in one call must leave the request
        and the scheduler exactly as the per-token update (check_stop() per
        token) does."""
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                                 init_none_hash)
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import RequestStatus

        from tpu_inference.core.sched import utils as sched_utils

        patch_vllm_scheduler_for_continue_decode()
        cls = AsyncScheduler if use_async else Scheduler
        update = cls._update_request_with_output

        def per_token_update(*args):
            with mock.patch.object(sched_utils,
                                   "_num_tokens_before_possible_stop",
                                   return_value=0) as helper:
                out = update(*args)
            assert helper.called  # really took the per-token path
            return out

        init_none_hash(sha256)
        block_hasher = get_request_block_hasher(16, sha256)
        stop_token_ids = [None, [], [7], [7, _EOS]]
        rng = random.Random(seed)
        for _ in range(200):
            want_sched, got_sched = _make_scheduler(cls), _make_scheduler(cls)
            want_sched.max_model_len = rng.choice([48, 300, 100000])
            got_sched.max_model_len = want_sched.max_model_len
            want_sched.kv_cache_manager = mock.MagicMock()
            got_sched.kv_cache_manager = mock.MagicMock()
            kwargs = dict(prompt_len=rng.randint(1, 40),
                          max_tokens=rng.choice([1, 2, 37, 300, 5000]),
                          stop_token_ids=rng.choice(stop_token_ids),
                          ignore_eos=rng.random() < 0.3,
                          min_tokens=rng.choice([0, 0, 5, 100]),
                          block_hasher=block_hasher)
            want, got = _real_request(**kwargs), _real_request(**kwargs)
            want.status = got.status = RequestStatus.RUNNING
            # A few consecutive continue-decode steps on the same request.
            for _ in range(3):
                if want.is_finished():
                    break
                # schedule() adds one in-flight placeholder per step (async).
                want.num_output_placeholders = 1
                got.num_output_placeholders = 1
                tokens = _random_tokens(rng, rng.choice([1, 2, 64, 256]))
                want_ids, got_ids = list(tokens), list(tokens)
                want_out = per_token_update(want_sched, want, want_ids)
                got_out = update(got_sched, got, got_ids)
                assert got_out == want_out
                # Both trim the caller's list in place and return it.
                assert got_out[0] is got_ids and got_ids == want_ids
                assert _request_state(got) == _request_state(want)
            want_cached = _cached_num_tokens(want_sched)
            assert _cached_num_tokens(got_sched) == want_cached

    def test_check_stop_runs_only_from_first_possible_stop(self):
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched import scheduler as scheduler_module
        Scheduler = scheduler_module.Scheduler
        scheduler = _make_scheduler(Scheduler)
        scheduler.max_model_len = 100000
        real_check_stop = scheduler_module.check_stop

        def run(tokens, **kwargs):
            request = _real_request(prompt_len=8,
                                    max_tokens=5000,
                                    stop_token_ids=None,
                                    ignore_eos=False)
            for key, value in kwargs.items():
                setattr(request.sampling_params, key, value)
            with mock.patch.object(scheduler_module,
                                   "check_stop",
                                   side_effect=real_check_stop) as check:
                ids, stopped = Scheduler._update_request_with_output(
                    scheduler, request, list(tokens))
            return ids, stopped, check.call_count

        tokens = list(range(10, 1034))
        # No stop token and far from the limits: no per-token checks at all.
        assert run(tokens) == (tokens, False, 0)
        # EOS at index 500: one check, on the EOS token, which stops.
        tokens[500] = _EOS
        assert run(tokens) == (tokens[:501], True, 1)
        # Repetition detection keeps the per-token path.
        with mock.patch("vllm.v1.core.sched.utils.check_sequence_repetition",
                        return_value=False):
            assert run(tokens, repetition_detection=mock.sentinel.rd)[2] == 501
