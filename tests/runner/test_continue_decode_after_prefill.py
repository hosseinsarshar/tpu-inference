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
"""Tests for the continue-decode window that runs right after prefill."""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

from tpu_inference.runner.tpu_runner import (TPUModelRunner,
                                             _append_window_logprobs)

REQ_IDS = ["r0", "r1", "r2"]


def _logprobs(token_ids, ranks, cu):
    token_ids = np.asarray(token_ids, dtype=np.int32)
    # Two logprob columns per row; the values follow the token ids so that
    # rows can be told apart after merging.
    return LogprobsLists(
        logprob_token_ids=np.stack([token_ids, token_ids + 1000], axis=1),
        logprobs=np.stack([-token_ids / 100.0, -token_ids / 10.0], axis=1),
        sampled_token_ranks=np.asarray(ranks, dtype=np.int32),
        cu_num_generated_tokens=cu,
    )


def _runner_output(sampled_token_ids, logprobs=None):
    return ModelRunnerOutput(
        req_ids=list(REQ_IDS),
        req_id_to_index={
            r: i
            for i, r in enumerate(REQ_IDS)
        },
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _chain_runner():
    """A runner whose batch just finished a mixed prefill/decode step."""
    runner = MagicMock()
    runner.enable_continue_decode_after_prefill = True
    runner.scheduler_config.async_scheduling = False
    runner.speculative_config = None
    runner.is_multimodal_model = False
    runner.vllm_config.sharding_config.prefill_cp_size = 1
    runner.aux_output_worker = None
    runner._continue_decode_output = None
    batch = runner.input_batch
    batch.num_reqs = len(REQ_IDS)
    batch.num_prompt_logprobs = {}
    batch.req_ids = REQ_IDS + [None]
    # r0 is a new 7-token prompt, r1 is decoding, r2 ends a chunked prefill.
    batch.num_computed_tokens_cpu = np.array([0, 5, 9, 0], dtype=np.int32)
    batch.request_distribution = [1, 1, 3]
    return runner


def _prefill_step_output():
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={
            "r0": 7,
            "r1": 1,
            "r2": 4
        },
        total_num_scheduled_tokens=12,
        has_structured_output_requests=False,
    )
    output = _runner_output([[11], [12], [13]],
                            _logprobs([11, 12, 13, 0], [1, 1, 1, 0],
                                      [0, 1, 2, 3]))
    return scheduler_output, output


class TestAppendWindowLogprobs:

    def test_puts_each_window_after_its_step_row(self):
        # The step has a padding row at the end; the window gives r1 nothing.
        step = _logprobs([11, 12, 13, 0], [1, 1, 1, 0], [0, 1, 2, 3])
        window = _logprobs([21, 22, 41, 42, 43], [2, 3, 5, 6, 7], [0, 2, 2, 5])

        merged = _append_window_logprobs(step, window, num_reqs=3)

        assert merged.cu_num_generated_tokens == [0, 3, 4, 8]
        expected_ids = [11, 21, 22, 12, 13, 41, 42, 43]
        assert merged.logprob_token_ids[:, 0].tolist() == expected_ids
        np.testing.assert_array_equal(merged.logprob_token_ids[:, 1],
                                      np.array(expected_ids) + 1000)
        np.testing.assert_allclose(merged.logprobs[:, 0],
                                   -np.array(expected_ids) / 100.0)
        assert merged.sampled_token_ranks.tolist() == [1, 2, 3, 1, 1, 5, 6, 7]

    def test_step_rows_without_offsets(self):
        step = _logprobs([11, 12], [1, 1], None)
        window = _logprobs([21, 31, 32], [2, 3, 4], [0, 1, 3])

        merged = _append_window_logprobs(step, window, num_reqs=2)

        assert merged.cu_num_generated_tokens == [0, 2, 5]
        assert merged.logprob_token_ids[:, 0].tolist() == [11, 21, 12, 31, 32]


class TestCanChainContinueDecode:

    def _can_chain(self, runner, output, grammar_output=None, kv=False):
        scheduler_output, _ = _prefill_step_output()
        with patch("tpu_inference.runner.tpu_runner.has_kv_transfer_group",
                   return_value=kv):
            return TPUModelRunner._can_chain_continue_decode(
                runner, scheduler_output, grammar_output, output)

    def test_every_request_in_decode(self):
        _, output = _prefill_step_output()
        assert self._can_chain(_chain_runner(), output)

    def test_request_still_in_prefill(self):
        # A partial prefill chunk discards its sampled token.
        output = _runner_output([[11], [], [13]])
        assert not self._can_chain(_chain_runner(), output)

    @pytest.mark.parametrize("setting", [
        "flag_off", "async_scheduling", "spec_decode", "multimodal",
        "prompt_logprobs", "prefill_cp", "routed_experts"
    ])
    def test_unsupported_setting(self, setting):
        runner = _chain_runner()
        if setting == "flag_off":
            runner.enable_continue_decode_after_prefill = False
        elif setting == "async_scheduling":
            runner.scheduler_config.async_scheduling = True
        elif setting == "spec_decode":
            runner.speculative_config = MagicMock()
        elif setting == "multimodal":
            runner.is_multimodal_model = True
        elif setting == "prompt_logprobs":
            runner.input_batch.num_prompt_logprobs = {"r0": 1}
        elif setting == "prefill_cp":
            runner.vllm_config.sharding_config.prefill_cp_size = 2
        elif setting == "routed_experts":
            runner.aux_output_worker = MagicMock()
        _, output = _prefill_step_output()
        assert not self._can_chain(runner, output)

    def test_structured_output(self):
        _, output = _prefill_step_output()
        assert not self._can_chain(
            _chain_runner(), output, grammar_output=MagicMock())

    def test_kv_transfer(self):
        _, output = _prefill_step_output()
        assert not self._can_chain(_chain_runner(), output, kv=True)


class TestContinueDecodeAfterPrefill:

    def test_runs_the_next_window_and_appends_it(self):
        runner = _chain_runner()
        scheduler_output, output = _prefill_step_output()
        seen = {}

        def run_window(decode_sched):
            seen["sched"] = decode_sched
            seen["num_scheduled"] = dict(decode_sched.num_scheduled_tokens)
            seen["computed"] = (
                runner.input_batch.num_computed_tokens_cpu.copy())
            seen["distribution"] = list(
                runner.input_batch.request_distribution)
            runner._continue_decode_output = _runner_output(
                [[21, 22], [31], [41, 42, 43]],
                _logprobs([21, 22, 31, 41, 42, 43], [2, 3, 4, 5, 6, 7],
                          [0, 2, 3, 6]))

        runner._execute_continue_decode.side_effect = run_window

        result = TPUModelRunner._continue_decode_after_prefill(
            runner, scheduler_output, output)

        # The window saw the batch as the next step would: one decode token
        # per request, after this step's tokens.
        assert seen["sched"] is not scheduler_output
        assert seen["num_scheduled"] == {"r0": 1, "r1": 1, "r2": 1}
        assert seen["sched"].total_num_scheduled_tokens == 3
        assert seen["computed"].tolist() == [7, 6, 13, 0]
        assert seen["distribution"] == [3, 3, 3]
        # The batch state and this step's scheduler output are unchanged.
        batch = runner.input_batch
        assert batch.num_computed_tokens_cpu.tolist() == [0, 5, 9, 0]
        assert batch.request_distribution == [1, 1, 3]
        assert scheduler_output.num_scheduled_tokens == {
            "r0": 7,
            "r1": 1,
            "r2": 4
        }
        assert scheduler_output.total_num_scheduled_tokens == 12
        # The window output is consumed, not returned by sample_tokens later.
        assert runner._continue_decode_output is None

        assert result is output
        assert result.sampled_token_ids == [[11, 21, 22], [12, 31],
                                            [13, 41, 42, 43]]
        assert result.logprobs.cu_num_generated_tokens == [0, 3, 5, 9]
        assert result.logprobs.logprob_token_ids[:, 0].tolist() == [
            11, 21, 22, 12, 31, 13, 41, 42, 43
        ]

    def test_restores_batch_state_on_error(self):
        runner = _chain_runner()
        scheduler_output, output = _prefill_step_output()
        runner._execute_continue_decode.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            TPUModelRunner._continue_decode_after_prefill(
                runner, scheduler_output, output)

        batch = runner.input_batch
        assert batch.num_computed_tokens_cpu.tolist() == [0, 5, 9, 0]
        assert batch.request_distribution == [1, 1, 3]

    @patch("tpu_inference.runner.tpu_runner.continue_decode")
    @patch("jax.device_get")
    def test_window_matches_the_next_step_window(self, mock_device_get,
                                                 mock_continue_decode):
        """Runs the real `_execute_continue_decode` as the window.

        The window must be the full window that the next step would run:
        min(max_decode_steps, room left) steps from the current RNG state, on
        the tokens this step sampled.
        """
        runner = _chain_runner()
        runner._execute_continue_decode = functools.partial(
            TPUModelRunner._execute_continue_decode, runner)
        runner.max_num_reqs = 4
        runner.max_model_len = 64
        runner.dp_size = 1
        # Read by set_forward_context.
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.static_max_decode_steps = 4
        runner._get_min_remaining_slots.return_value = 10
        runner.eos_token_id = (99, )
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {}
        runner.continue_decode_eos_check_interval = 0
        batch = runner.input_batch
        # Tokens after this step: r0 has 7 prompt tokens + 1 sampled token,
        # r1 and r2 likewise end with the token sampled in this step.
        batch.num_tokens = np.array([8, 7, 14, 0])
        batch.num_tokens_no_spec = batch.num_tokens.copy()
        batch.token_ids_cpu = np.zeros((4, 64), dtype=np.int32)
        runner.requests = {
            r: MagicMock(output_token_ids=[t])
            for r, t in zip(REQ_IDS, [11, 12, 13])
        }
        seen = {}

        def prepare_inputs(decode_sched):
            seen["computed"] = batch.num_computed_tokens_cpu.copy()
            seen["num_scheduled"] = dict(decode_sched.num_scheduled_tokens)
            input_ids = np.array([11, 12, 13, 0], dtype=np.int32)
            attn_metadata = MagicMock(spec=[])
            logits_indices = np.array([0, 1, 2, -1], dtype=np.int32)
            return (input_ids, None, attn_metadata, None,
                    logits_indices) + (None, ) * 7

        runner._prepare_inputs.side_effect = prepare_inputs
        generated = MagicMock()
        mock_continue_decode.return_value = (generated,
                                             MagicMock(), MagicMock(),
                                             MagicMock(), None, None)
        tokens = np.zeros((4, 4), dtype=np.int32)
        tokens[:, 0] = [21, 22, 23, 24]
        tokens[:, 1] = [31, 99, 0, 0]
        tokens[:, 2] = [41, 42, 43, 44]
        mock_device_get.side_effect = lambda arg: (
            (tokens, np.int32(4))
            if isinstance(arg, tuple) and arg[0] is generated else arg)
        rng_after_step = runner.rng_params_for_sampling
        scheduler_output, _ = _prefill_step_output()
        output = _runner_output([[11], [12], [13]])

        result = TPUModelRunner._continue_decode_after_prefill(
            runner, scheduler_output, output)

        assert seen["computed"].tolist() == [7, 6, 13, 0]
        assert seen["num_scheduled"] == {"r0": 1, "r1": 1, "r2": 1}
        kwargs = mock_continue_decode.call_args.kwargs
        # A full window, as in the next step (not one step shorter).
        assert kwargs["max_decode_steps"] == 4
        assert kwargs["rng"] is rng_after_step
        np.testing.assert_array_equal(kwargs["init_state"].current_tokens,
                                      [11, 12, 13, 0])
        np.testing.assert_array_equal(kwargs["init_state"].active_mask,
                                      [True, True, True, False])
        assert result.sampled_token_ids == [[11, 21, 22, 23, 24], [12, 31, 99],
                                            [13, 41, 42, 43, 44]]
        # The host batch holds the window tokens, as after a normal window.
        assert batch.token_ids_cpu[0, 8:12].tolist() == [21, 22, 23, 24]
        assert batch.token_ids_cpu[1, 7:9].tolist() == [31, 99]
        assert batch.num_tokens_no_spec[:3].tolist() == [12, 9, 18]
        assert runner.requests["r1"].output_token_ids == [12, 31, 99]
        assert batch.num_computed_tokens_cpu.tolist() == [0, 5, 9, 0]


class TestSampleTokensChain:

    @pytest.mark.parametrize("can_chain", [True, False])
    def test_chains_only_when_allowed(self, can_chain):
        runner = MagicMock()
        runner._continue_decode_output = None
        step_output = MagicMock()
        chained_output = MagicMock()
        runner._sample_from_logits.return_value = step_output
        runner._can_chain_continue_decode.return_value = can_chain
        runner._continue_decode_after_prefill.return_value = chained_output
        state = runner.execute_model_state

        with patch("jax.set_mesh"):
            result = TPUModelRunner.sample_tokens(runner, None)

        runner._can_chain_continue_decode.assert_called_once_with(
            state.scheduler_output, None, step_output)
        if can_chain:
            runner._continue_decode_after_prefill.assert_called_once_with(
                state.scheduler_output, step_output)
            assert result is chained_output
        else:
            runner._continue_decode_after_prefill.assert_not_called()
            assert result is step_output
