"""Tests for pause / resume + agentic loop (Phase 4).

Pure unit tests first (no model / no GPU) for SamplingParams pause fields,
the Sequence pause surface and its batch ``append_tokens``. Then Scheduler
unit tests that build a real ``Config`` from the GPT-2 124M path (this reads
``config.json`` via ``AutoConfig.from_pretrained`` but loads no weights and
needs no GPU). Finally end-to-end tests that load GPT-2 124M and need a CUDA
GPU, exercising ``generate_stream`` pause events, ``resume()``, the agentic
``generate_agentic()`` tool loop, and the ``generate()`` rejection of pause
parameters.

Run directly (``python test_phase4.py``) or via pytest.
"""
import pytest

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus


MODEL_PATH = "/root/models/gpt2-nanovllm"


# --------------------------------------------------------------------------- #
# SamplingParams pause fields (pure, no model / no GPU)
# --------------------------------------------------------------------------- #
def test_sampling_params_pause_fields():
    sp = SamplingParams(pause=["STOP"], pause_token_ids=[99])
    assert sp.pause == ["STOP"]
    assert sp.pause_token_ids == [99]
    assert sp.has_pause is True


def test_sampling_params_pause_normalization():
    # A bare string is normalized to a single-element list.
    sp_str = SamplingParams(pause="STOP")
    assert sp_str.pause == ["STOP"]

    # A bare int is normalized to a single-element list.
    sp_int = SamplingParams(pause_token_ids=99)
    assert sp_int.pause_token_ids == [99]

    # Defaults stay None and has_pause is False.
    sp_default = SamplingParams()
    assert sp_default.pause is None
    assert sp_default.pause_token_ids is None
    assert sp_default.has_pause is False

    # has_pause flips on either field even when the other is None.
    assert SamplingParams(pause=["x"]).has_pause is True
    assert SamplingParams(pause_token_ids=[1]).has_pause is True


# --------------------------------------------------------------------------- #
# Sequence pause surface (pure, no model / no GPU)
# --------------------------------------------------------------------------- #
def test_sequence_pause_fields():
    seq = Sequence([1, 2, 3], SamplingParams(pause=["X"], pause_token_ids=[1]))
    assert seq.pause == ["X"]
    assert seq.pause_token_ids == [1]
    assert seq.pause_index == -1
    assert seq.pause_reason is None
    assert seq.status == SequenceStatus.WAITING
    assert seq.is_paused is False
    assert seq.is_finished is False


def test_sequence_append_tokens():
    seq = Sequence([1, 2, 3])
    assert seq.num_tokens == 3
    assert seq.token_ids == [1, 2, 3]
    assert seq.last_token == 3

    seq.append_tokens([4, 5])
    assert seq.num_tokens == 5
    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.last_token == 5

    # Appending an empty list is a no-op (last_token / num_tokens unchanged).
    seq.append_tokens([])
    assert seq.num_tokens == 5
    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.last_token == 5


def test_sequence_paused_status():
    seq = Sequence([1])
    seq.status = SequenceStatus.PAUSED
    assert seq.is_paused is True
    assert seq.is_finished is False


# --------------------------------------------------------------------------- #
# Scheduler pause / resume unit tests (Config from real model path, no GPU)
# --------------------------------------------------------------------------- #
def _scheduler():
    """Build a Scheduler backed by a real Config (reads config.json, no weights).

    Mirrors how LLMEngine wires things up: Sequence.block_size comes from the
    Config, and the tokenizer is attached for stop / pause-string decoding.
    """
    from transformers import AutoTokenizer
    from nanovllm.config import Config
    from nanovllm.engine.scheduler import Scheduler

    config = Config(model=MODEL_PATH, num_kvcache_blocks=100)
    Sequence.block_size = config.kvcache_block_size
    scheduler = Scheduler(config)
    scheduler.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    return scheduler


def test_scheduler_pause_retains_blocks():
    scheduler = _scheduler()

    seq = Sequence([1, 2, 3], SamplingParams(pause=["X"], pause_token_ids=[1]))
    # Simulate a sequence mid-decode: RUNNING with allocated KV blocks.
    seq.status = SequenceStatus.RUNNING
    scheduler.block_manager.allocate(seq, 0)
    assert seq.block_table, "precondition: blocks were allocated"
    scheduler.running.append(seq)
    assert seq in scheduler.running

    scheduler._pause(seq)

    # Pause flips status, drops it from the running queue ...
    assert seq.status == SequenceStatus.PAUSED
    assert seq.is_paused is True
    assert seq not in scheduler.running
    # ... but RETAINS the KV-cache blocks for a later resume().
    assert len(seq.block_table) >= 1


def test_scheduler_resume_moves_to_waiting():
    scheduler = _scheduler()

    seq = Sequence([1, 2, 3], SamplingParams(pause=["X"], pause_token_ids=[1]))
    seq.status = SequenceStatus.RUNNING
    scheduler.block_manager.allocate(seq, 0)
    scheduler.running.append(seq)
    scheduler._pause(seq)
    assert seq.is_paused

    scheduler.resume(seq)

    assert seq.status == SequenceStatus.WAITING
    assert seq.is_paused is False
    assert seq in scheduler.waiting
    assert seq.is_prefill is True
    assert seq.pause_reason is None


def test_pause_token_ids_kv_consistency():
    """pause_token_ids keeps the KV cache and seq.token_ids in sync (Phase 4 fix).

    Regression: the pause token used to be counted in num_cached_tokens (via
    num_scheduled_tokens) but NOT appended to token_ids, so num_cached_tokens ==
    num_tokens + 1 after pausing and the resume() prefill accounting
    (num_tokens - num_cached_tokens) underflowed. The fix appends the token first
    and hides it from the displayed text via pause_index.
    """
    scheduler = _scheduler()
    pause_id = scheduler.tokenizer.encode(".", add_special_tokens=False)[-1]

    seq = Sequence([101, 202, 303], SamplingParams(pause_token_ids=[pause_id]))
    seq.status = SequenceStatus.RUNNING
    scheduler.block_manager.allocate(seq, 0)
    scheduler.running.append(seq)
    seq.num_cached_tokens = seq.num_tokens  # prompt already prefilled / cached

    # decode two ordinary tokens, then the pause token
    for tid in (11, 22, pause_id):
        seq.num_scheduled_tokens = 1
        scheduler.postprocess([seq], [tid], is_prefill=False)
        if seq.is_paused:
            break

    assert seq.is_paused, "the pause token should have paused the sequence"
    assert seq.token_ids[-1] == pause_id, "pause token must be retained as last token"
    assert seq.num_cached_tokens == seq.num_tokens, (
        f"postprocess desynced bookkeeping: num_cached_tokens={seq.num_cached_tokens} "
        f"!= num_tokens={seq.num_tokens}")
    assert seq.pause_index >= 0, "pause token must be hidden via pause_index"


# --------------------------------------------------------------------------- #
# End-to-end through the LLM engine (needs CUDA + GPT-2 124M)
# --------------------------------------------------------------------------- #
def test_generate_stream_pause_string(llm):
    """A pause string yields a finish_reason='pause' event and never 'stop'."""
    sp = SamplingParams(temperature=0, max_tokens=50, ignore_eos=True, pause=["."])
    events = list(llm.generate_stream("The cat sat on the mat and then", sp))

    pause_events = [e for e in events if e["finish_reason"] == "pause"]
    assert pause_events, "expected at least one pause event from generate_stream"

    # Hitting a pause condition must not also report a stop finish.
    assert not any(e["finish_reason"] == "stop" for e in events)
    # Every event carries its seq_id.
    assert all("seq_id" in e for e in events)


def test_pause_resume_continuation(llm):
    """After a pause, resume() appends context and generation continues.

    pause conditions stay active across resume(), so the model typically pauses
    again at the next occurrence -- both outcomes (finish, or re-pause) prove the
    retained KV cache was successfully re-entered.
    """
    sp = SamplingParams(temperature=0, max_tokens=50, ignore_eos=True, pause=["."])
    events = list(llm.generate_stream("The cat sat on the mat and then", sp))

    pause_events = [e for e in events if e["finish_reason"] == "pause"]
    assert pause_events, "expected a pause before attempting resume"
    seq_id = pause_events[-1]["seq_id"]

    # The sequence is parked PAUSED with its KV cache retained.
    assert llm.is_paused(seq_id)

    # Resume by appending fresh context (e.g. a tool / user follow-up).
    llm.resume(seq_id, " After that")
    seq = llm._active[seq_id]
    assert not seq.is_paused  # moved out of PAUSED back to WAITING

    # Step the engine manually. Stop when the sequence terminates OR re-pauses
    # (the pause condition is still live, so "." triggers another pause).
    tokens_at_resume = seq.num_tokens
    steps = 0
    while not seq.is_finished and not seq.is_paused and steps < 200:
        llm.step()
        steps += 1

    assert seq.is_finished or seq.is_paused, "resumed sequence stalled"
    # The model produced brand-new tokens after resume (not just the prefill).
    assert seq.num_tokens > tokens_at_resume, "model did not generate new tokens after resume"


def test_generate_agentic_tool_call(llm):
    """The agentic loop emits tool_call -> tool_result -> done events."""
    sp = SamplingParams(temperature=0, max_tokens=50, ignore_eos=True, pause=["."])

    seen_calls = []

    def tool_executor(call_text):
        seen_calls.append(call_text)
        return "OK"

    events = list(llm.generate_agentic(
        "The cat sat on the mat and then", sp, tool_executor, max_turns=3))

    types = [e["type"] for e in events]

    # A pause must have triggered at least one tool round-trip.
    assert "tool_call" in types, f"no tool_call event; got types={types}"
    assert "tool_result" in types
    tool_results = [e["text"] for e in events if e["type"] == "tool_result"]
    assert "OK" in tool_results
    # The loop always terminates with a 'done' event.
    assert types[-1] == "done"
    # The executor was actually invoked.
    assert seen_calls


def test_generate_rejects_pause(llm):
    """generate() refuses pause parameters (use generate_stream/agentic)."""
    with pytest.raises(ValueError):
        llm.generate(["hello"], SamplingParams(pause=["x"]), use_tqdm=False)


def test_pause_token_ids_e2e(llm):
    """End-to-end pause_token_ids: pause -> KV consistent -> resume generates.

    Mirrors test_pause_resume_continuation but drives the pause via a token id
    (the code path fixed in the KV/append bug) instead of a pause string.
    """
    period = llm.tokenizer.encode(".", add_special_tokens=False)[-1]
    sp = SamplingParams(temperature=0, max_tokens=50, ignore_eos=True,
                        pause_token_ids=[period])
    events = list(llm.generate_stream("The cat sat on the mat and then", sp))

    pause_events = [e for e in events if e["finish_reason"] == "pause"]
    assert pause_events, "expected a pause_token pause event"
    seq_id = pause_events[-1]["seq_id"]
    seq = llm._active[seq_id]

    # THE FIX: the pause token is RETAINED as the last token in the sequence
    # (before the fix it was dropped from token_ids, silently losing it from the
    # resumed context). In real decode the KV cache lags one token, so at pause
    # time num_cached_tokens == num_tokens - 1 -- the pause token's KV is cached
    # on the resume() prefill, which is exactly correct.
    assert seq.token_ids[-1] == period, (
        f"pause token not retained: last token={seq.token_ids[-1]} != {period}")
    assert seq.num_cached_tokens == seq.num_tokens - 1, (
        f"bad KV/token invariant at pause: num_cached={seq.num_cached_tokens} "
        f"num_tokens={seq.num_tokens}")
    assert llm.is_paused(seq_id)

    # Resume and confirm generation actually continues (new tokens produced).
    tokens_at_resume = seq.num_tokens
    llm.resume(seq_id, " After that")
    steps = 0
    while not seq.is_finished and not seq.is_paused and steps < 200:
        llm.step()
        steps += 1
    assert seq.is_finished or seq.is_paused, "resumed sequence stalled"
    assert seq.num_tokens > tokens_at_resume, "no new tokens after resume"


def test_pause_resume_errors(llm):
    """resume() rejects unknown / non-paused sequences; pause() is idempotent."""
    # 1) unknown seq_id -> KeyError
    with pytest.raises(KeyError):
        llm.resume(999999, "x")

    # 2) a freshly-added (WAITING, not paused) sequence cannot be resumed
    sid = llm.add_request("hello", SamplingParams(temperature=0, max_tokens=4))
    with pytest.raises(AssertionError):
        llm.resume(sid, "x")

    # 3) pause() on a finished sequence is a no-op; resume() on it is rejected.
    while not llm.is_finished():
        llm.step()
    seq = llm._active[sid]
    assert seq.is_finished
    llm.pause(sid)              # no-op on a finished seq, must not raise
    with pytest.raises(AssertionError):
        llm.resume(sid, "x")


if __name__ == "__main__":
    import sys

    # Pure unit tests (no model, no GPU).
    test_sampling_params_pause_fields()
    test_sampling_params_pause_normalization()
    test_sequence_pause_fields()
    test_sequence_append_tokens()
    test_sequence_paused_status()
    print("sampling/sequence unit tests OK")

    # Scheduler unit tests (Config from disk, no GPU needed).
    test_scheduler_pause_retains_blocks()
    test_scheduler_resume_moves_to_waiting()
    test_pause_token_ids_kv_consistency()
    print("scheduler unit tests OK")

    # End-to-end (needs CUDA + GPT-2 124M).
    try:
        import torch
    except ImportError:
        print("torch not installed; skipping end-to-end.")
        sys.exit(0)
    if not torch.cuda.is_available():
        print("CUDA required; skipping end-to-end.")
        sys.exit(0)

    from nanovllm import LLM
    _llm = LLM(MODEL_PATH, tensor_parallel_size=1,
               enforce_eager=True, gpu_memory_utilization=0.7)
    test_generate_stream_pause_string(_llm)
    test_pause_resume_continuation(_llm)
    test_pause_token_ids_e2e(_llm)
    test_generate_agentic_tool_call(_llm)
    test_generate_rejects_pause(_llm)
    test_pause_resume_errors(_llm)
    _llm.exit()
    print("ALL PHASE 4 TESTS PASSED")
