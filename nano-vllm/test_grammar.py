"""Tests for xgrammar guided decoding (Phase 3).

Unit tests for the bridge (nanovllm/grammar.py) plus end-to-end tests through the
LLM engine. Run directly (`python test_grammar.py`) or via pytest. The end-to-end
tests load GPT-2 124M and need a CUDA GPU.
"""
import re

import pytest

from nanovllm.sampling_params import SamplingParams


# --------------------------------------------------------------------------- #
# SamplingParams spec validation (pure, no model / no GPU)
# --------------------------------------------------------------------------- #
def test_sampling_params_accepts_one_guided_spec():
    SamplingParams(json_schema='{"type":"object"}')
    SamplingParams(regex=r"[0-9]+")
    SamplingParams(grammar='root ::= "a"+')
    SamplingParams(choice=["a", "b"])
    SamplingParams(json_object=True)



def test_sampling_params_rejects_multiple_guided_specs():
    with pytest.raises(AssertionError):
        SamplingParams(regex=r"[0-9]+", choice=["a", "b"])
    with pytest.raises(AssertionError):
        SamplingParams(json_schema="{}", json_object=True)


def test_has_guidance_flag():
    assert not SamplingParams().has_guidance
    assert SamplingParams(regex="x").has_guidance
    assert SamplingParams(json_object=True).has_guidance


# --------------------------------------------------------------------------- #
# Processor unit tests (need xgrammar; no model load)
# --------------------------------------------------------------------------- #
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("/root/models/gpt2-nanovllm", use_fast=True)


def test_json_schema_processor_masks_forbidden_token():
    import torch
    from nanovllm.grammar import compile_guided_decoding, XGrammarLogitsProcessor
    tok = _tok()
    V = tok.vocab_size
    compiled = compile_guided_decoding(tok, json_schema={
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    })
    proc = XGrammarLogitsProcessor(compiled, num_prompt_tokens=0)

    # Make a clearly-forbidden token ('Z') the argmax; masking must override it.
    logits = torch.full((V,), -10.0, dtype=torch.float32)
    Z = tok.encode("Z", add_special_tokens=False)[0]
    logits[Z] = 10.0
    assert logits.argmax().item() == Z
    out = proc([], logits.cuda())
    assert out.argmax().item() != Z            # forbidden token blocked
    # JSON object must start with '{' (token 90 for GPT-2) -> that should be allowed
    brace = tok.encode("{", add_special_tokens=False)[0]
    assert out[brace].item() > -float("inf")


def test_choice_processor_only_allows_a_literal():
    import torch
    from nanovllm.grammar import compile_guided_decoding, XGrammarLogitsProcessor
    tok = _tok()
    V = tok.vocab_size
    compiled = compile_guided_decoding(tok, choice=["yes", "no"])
    proc = XGrammarLogitsProcessor(compiled, num_prompt_tokens=0)

    # Uniform logits -> the mask should restrict to tokens that start "yes"/"no".
    logits = torch.zeros(V, dtype=torch.float32, device="cuda")
    out = proc([], logits)
    # At least one allowed token survives the mask, and the max is a valid starter.
    n_allowed = int((out > -float("inf")).sum())
    assert n_allowed >= 1 and n_allowed < V
    assert tok.decode([out.argmax().item()]).startswith(("y", "n"))


def test_terminated_matcher_does_not_raise():
    """After accepting EOS (terminate), further calls must no-op, not raise."""
    import torch
    from nanovllm.grammar import compile_guided_decoding, XGrammarLogitsProcessor
    tok = _tok()
    V = tok.vocab_size
    compiled = compile_guided_decoding(tok, json_object=True)
    proc = XGrammarLogitsProcessor(compiled, num_prompt_tokens=0)
    # First call produces a mask (allows a JSON starter).
    proc([], torch.zeros(V, device="cuda"))
    # Force termination by accepting the stop token (EOS), then call again.
    eos = tok.eos_token_id
    proc([eos], torch.zeros(V, device="cuda"))
    # Repeated post-termination calls should be cheap and never raise.
    proc([eos, 1, 2, 3], torch.randn(V, device="cuda"))


# --------------------------------------------------------------------------- #
# End-to-end through the LLM engine (needs CUDA + GPT-2 124M)
# --------------------------------------------------------------------------- #
def _eos(llm, text):
    eos = llm.tokenizer.eos_token or ""
    return text[: -len(eos)] if eos and text.endswith(eos) else text


def test_e2e_json_schema(llm):
    import json
    schema = {"type": "object",
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
              "required": ["name", "age"]}
    out = llm.generate(["Record:"], SamplingParams(
        temperature=0, max_tokens=64, json_schema=schema), use_tqdm=False)[0]
    payload = _eos(llm, out["text"]).strip()
    obj = json.loads(payload)
    assert set(obj) == {"name", "age"}
    assert isinstance(obj["name"], str) and isinstance(obj["age"], int)


def test_e2e_regex(llm):
    out = llm.generate(["N:"], SamplingParams(
        temperature=0, max_tokens=24, regex=r"[0-9]+"), use_tqdm=False)[0]
    payload = _eos(llm, out["text"]).strip()
    assert re.fullmatch(r"[0-9]+", payload), repr(payload)


def test_e2e_choice(llm):
    choices = ["yes", "no", "maybe"]
    out = llm.generate(["Q:"], SamplingParams(
        temperature=0, max_tokens=16, choice=choices), use_tqdm=False)[0]
    assert _eos(llm, out["text"]).strip() in set(choices)


def test_e2e_grammar_ebnf(llm):
    # EBNF: one or more 'a'
    out = llm.generate(["x:"], SamplingParams(
        temperature=0, max_tokens=16, grammar='root ::= "a"+'), use_tqdm=False)[0]
    payload = _eos(llm, out["text"]).strip()
    assert re.fullmatch(r"a+", payload), repr(payload)


def test_e2e_custom_logits_processor(llm):
    """A user-supplied logits_processor is still applied alongside guidance."""
    import torch

    def ban_first_500(token_ids, logits):
        logits[:500] = float("-inf")
        return logits

    out = llm.generate(["The"], SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True,
        logits_processors=[ban_first_500]), use_tqdm=False)[0]
    for tid in out["token_ids"]:
        assert tid >= 500


if __name__ == "__main__":
    import sys, torch
    from nanovllm import LLM
    if not torch.cuda.is_available():
        print("CUDA required; skipping end-to-end.")
        sys.exit(0)
    # Pure unit tests.
    test_sampling_params_accepts_one_guided_spec()
    test_sampling_params_rejects_multiple_guided_specs()
    test_has_guidance_flag()
    test_json_schema_processor_masks_forbidden_token()
    test_choice_processor_only_allows_a_literal()
    test_terminated_matcher_does_not_raise()
    print("unit tests OK")
    # End-to-end through the engine.
    _llm = LLM("/root/models/gpt2-nanovllm", tensor_parallel_size=1,
               enforce_eager=True, gpu_memory_utilization=0.7)
    test_e2e_json_schema(_llm)
    test_e2e_regex(_llm)
    test_e2e_choice(_llm)
    test_e2e_grammar_ebnf(_llm)
    test_e2e_custom_logits_processor(_llm)
    print("ALL GRAMMAR TESTS PASSED")
