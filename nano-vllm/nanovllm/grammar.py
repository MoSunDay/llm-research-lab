"""xgrammar-based guided (structured) decoding for nano-vllm.

Bridges xgrammar's grammar matching to nano-vllm's logits-processor hook
(SamplingParams.logits_processors), which ModelRunner._apply_logits_processors
runs on rank 0 only. The Sequence.__getstate__ strips logits_processors before
the seq is pickled to TP workers, so matchers never leave rank 0.

Public API:
    compile_guided_decoding(tokenizer, *, json_schema=None, regex=None,
                            grammar=None, choice=None, json_object=False)
        -> xgrammar.CompiledGrammar (cached per tokenizer; cheap to reuse)
    XGrammarLogitsProcessor(compiled_grammar, num_prompt_tokens=0)
        -> callable (token_ids: list[int], logits: torch.Tensor) -> torch.Tensor

This module is imported lazily by the engine only when a request uses guided
decoding, so xgrammar remains an optional dependency for the rest of nano-vllm.
"""
import threading

import torch

# import xgrammar lazily inside functions so `import nanovllm` works without it;
# the engine surfaces a clear error if a user requests guided decoding without it.
xgrammar = None


def _ensure_xgrammar():
    global xgrammar
    if xgrammar is None:
        import xgrammar as _xg
        xgrammar = _xg
    return xgrammar


# GrammarCompiler (and its TokenizerInfo) is expensive to build:
# TokenizerInfo.from_huggingface scans the whole vocab to precompute token
# categories. Cache one compiler per tokenizer object so repeated requests reuse it.
_compiler_cache: "dict[int, object]" = {}
_cache_lock = threading.Lock()


def _get_compiler(tokenizer):
    xg = _ensure_xgrammar()
    key = id(tokenizer)
    with _cache_lock:
        compiler = _compiler_cache.get(key)
        if compiler is None:
            tokenizer_info = xg.TokenizerInfo.from_huggingface(tokenizer)
            compiler = xg.GrammarCompiler(tokenizer_info)
            _compiler_cache[key] = compiler
    return compiler


def compile_guided_decoding(tokenizer, *, json_schema=None, regex=None,
                            grammar=None, choice=None, json_object=False):
    """Compile an xgrammar grammar from exactly one guidance spec.

    Returns an xgrammar.CompiledGrammar; wrap it in many XGrammarLogitsProcessor
    instances (one per sequence). ``json_schema`` may be a dict or a JSON string.
    ``grammar`` is an EBNF string (or an xgrammar.Grammar). ``choice`` is a list
    of literal strings; the output is constrained to be exactly one of them.
    """
    import json as _json
    import re as _re

    xg = _ensure_xgrammar()
    spec_count = sum(s is not None for s in (json_schema, regex, grammar, choice)) + bool(json_object)
    if spec_count != 1:
        raise ValueError(
            "Exactly one of json_schema / regex / grammar / choice / json_object "
            f"must be set (got {spec_count})."
        )

    compiler = _get_compiler(tokenizer)
    if json_schema is not None:
        if not isinstance(json_schema, str):
            json_schema = _json.dumps(json_schema)
        return compiler.compile_json_schema(json_schema)
    if regex is not None:
        return compiler.compile_regex(regex)
    if grammar is not None:
        if not isinstance(grammar, xg.Grammar):
            grammar = xg.Grammar.from_ebnf(grammar)
        return compiler.compile_grammar(grammar)
    if choice is not None:
        pattern = "|".join(f"({_re.escape(c)})" for c in choice)
        return compiler.compile_regex(pattern)
    # json_object
    return compiler.compile_builtin_json_grammar()


class XGrammarLogitsProcessor:
    """A logits-processor callable backed by one xgrammar GrammarMatcher.

    Signature matches the hook in ModelRunner._apply_logits_processors:
        (token_ids: list[int], logits: torch.Tensor) -> torch.Tensor

    ``token_ids`` is the FULL sequence (prompt + generated tokens). We skip the
    first ``num_prompt_tokens`` so the matcher only consumes completion tokens.
    Edits ``logits`` IN PLACE (callers already clone before invoking).
    """

    def __init__(self, compiled_grammar, num_prompt_tokens: int = 0):
        xg = _ensure_xgrammar()
        self.compiled_grammar = compiled_grammar
        self.matcher = xg.GrammarMatcher(compiled_grammar)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self._accepted = 0          # completion tokens already fed to the matcher
        self._bitmask = None        # allocated lazily once vocab size is known

    def __call__(self, token_ids, logits):
        xg = _ensure_xgrammar()
        # Feed any newly generated completion tokens to the matcher. Stop as soon
        # as it terminates (after the stop token is accepted) so we never call
        # accept_token() on a terminated matcher (which xgrammar warns about).
        completion = token_ids[self.num_prompt_tokens:]
        for tid in completion[self._accepted:]:
            if self.matcher.is_terminated():
                break
            self.matcher.accept_token(int(tid))
        self._accepted = len(completion)

        # Once terminated (grammar fully satisfied + stop token accepted) the
        # matcher can no longer produce a mask. Leave logits unconstrained from
        # then on; with ignore_eos=False the sequence has already finished.
        if self.matcher.is_terminated():
            return logits

        if self._bitmask is None:
            vocab = logits.size(-1)
            self._bitmask = xg.allocate_token_bitmask(1, vocab)
        xg.reset_token_bitmask(self._bitmask)
        self.matcher.fill_next_token_bitmask(self._bitmask, index=0)

        # xgrammar requires logits and bitmask on the same device; the mask is
        # small (ceil(vocab/32) int32, ~6 KB for GPT-2) so the copy is cheap.
        bitmask = self._bitmask.to(logits.device)
        # apply_token_bitmask_inplace expects a [batch, vocab] logits tensor.
        if logits.dim() == 1:
            xg.apply_token_bitmask_inplace(logits.view(1, -1), bitmask)
        else:
            xg.apply_token_bitmask_inplace(logits, bitmask)
        return logits
