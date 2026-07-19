import atexit
from typing import Callable, Iterator
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.scheduler.tokenizer = self.tokenizer
        self._active = {}
        atexit.register(self.exit)

    def exit(self):
        # Idempotent: safe to call manually AND via the atexit hook.
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        # Compile guided-decoding (xgrammar) matchers here: the tokenizer lives
        # on the engine and the matcher needs num_prompt_tokens of this request.
        proc = self._build_guided_processor(seq, sampling_params)
        if proc is not None:
            seq.logits_processors = [proc] + list(seq.logits_processors)
        self._active[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    def _build_guided_processor(self, seq: Sequence, sampling_params: SamplingParams):
        """Return an xgrammar logits processor for this request, or None."""
        if not sampling_params.has_guidance:
            return None
        backend = sampling_params.guided_decoding_backend or "xgrammar"
        if backend != "xgrammar":
            raise ValueError(
                f"unsupported guided_decoding_backend {backend!r}; only 'xgrammar' is supported")
        try:
            from nanovllm.grammar import compile_guided_decoding, XGrammarLogitsProcessor
        except ImportError as e:  # pragma: no cover - xgrammar is optional
            raise ImportError(
                "guided decoding requires the 'xgrammar' package: pip install xgrammar"
            ) from e
        compiled = compile_guided_decoding(
            self.tokenizer,
            json_schema=sampling_params.json_schema,
            regex=sampling_params.regex,
            grammar=sampling_params.grammar,
            choice=sampling_params.choice,
            json_object=sampling_params.json_object,
        )
        return XGrammarLogitsProcessor(compiled, num_prompt_tokens=seq.num_prompt_tokens)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # ---- pause / resume (Phase 4) ------------------------------------------
    def pause(self, seq_id: int):
        """Manually pause a running sequence. KV-cache blocks are retained.

        The sequence stops generating immediately and can be resumed later via
        ``resume()``.  No-op if already paused/finished.
        """
        seq = self._active.get(seq_id)
        if seq is None:
            raise KeyError(f"unknown seq_id {seq_id}")
        if seq.is_paused or seq.is_finished:
            return
        seq.pause_reason = "manual"
        self.scheduler._pause(seq)

    def resume(self, seq_id: int, prompt: str | list[int]):
        """Resume a PAUSED sequence by appending new tokens (e.g. a tool result).

        The appended tokens are prefilled -- reusing the retained KV cache for
        everything generated so far -- then decode continues from there.  Only
        the *new* tokens are processed; previously cached keys/values are not
        recomputed (prefix-cache fast path).
        """
        seq = self._active.get(seq_id)
        if seq is None:
            raise KeyError(f"unknown seq_id {seq_id}")
        assert seq.is_paused, f"seq {seq_id} is {seq.status.name}, not PAUSED"
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            token_ids = list(prompt)
        if token_ids:
            seq.append_tokens(token_ids)
            # After resume, the pause-string check should only look at text
            # generated from here onward (the old pause delimiter is still in
            # the decoded buffer and would otherwise re-trigger immediately).
            seq.pause_search_offset = len(self.tokenizer.decode(seq.completion_token_ids))
            seq.pause_index = -1
            self.scheduler.block_manager.ensure_blocks(seq)
            self.scheduler.resume(seq)
        else:
            # Nothing to prefill: jump straight back to decode.
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            seq.pause_reason = None
            seq.pause_index = -1
            self.scheduler.running.append(seq)
        return seq_id

    def is_paused(self, seq_id: int) -> bool:
        seq = self._active.get(seq_id)
        return seq is not None and seq.is_paused

    def _display_text(self, seq) -> str:
        """Completion text, truncated at stop/pause index if set."""
        text = self.tokenizer.decode(seq.completion_token_ids)
        if seq.stop_index >= 0:
            return text[:seq.stop_index]
        if seq.pause_index >= 0:
            return text[:seq.pause_index]
        return text

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for sp in sampling_params:
            if sp.has_pause:
                raise ValueError(
                    "pause / pause_token_ids are not supported in generate(); "
                    "use generate_stream() or generate_agentic() instead.")
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, _ in output:
                outputs[seq_id] = self._finalize_output(self._active[seq_id])
                pbar.update(1)
        pbar.close()
        return [outputs[sid] for sid in sorted(outputs)]

    def _finalize_output(self, seq):
        token_ids = seq.completion_token_ids
        text = self.tokenizer.decode(token_ids)
        if seq.stop_index >= 0:
            text = text[:seq.stop_index]
        return {"text": text, "token_ids": token_ids,
                "finish_reason": seq.finish_reason or "stop",
                "prompt_token_ids": seq.prompt_token_ids}

    def generate_stream(self, prompts, sampling_params):
        """Yield incremental output dicts as tokens are produced.

        Each dict: {"seq_id", "text" (delta since the last yield for that seq),
        "finish_reason"}.  finish_reason is ``None`` while still generating,
        ``"stop"`` / ``"length"`` / ``"eos"`` on finish, or ``"pause"`` when the
        sequence hits a pause condition (KV cache retained -- resume externally
        via ``resume()`` and re-invoke ``generate_stream`` for the continuation).
        Order is interleaved across sequences by engine step.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        live = [self._active[self.add_request(p, sp)] for p, sp in zip(prompts, sampling_params)]
        emitted = {s.seq_id: 0 for s in live}
        while live:
            self.step()
            still = []
            for s in live:
                disp = self._display_text(s)
                delta = disp[emitted[s.seq_id]:]
                emitted[s.seq_id] = len(disp)
                if s.is_finished:
                    yield {"seq_id": s.seq_id, "text": delta,
                           "finish_reason": s.finish_reason or "stop"}
                elif s.is_paused:
                    yield {"seq_id": s.seq_id, "text": delta, "finish_reason": "pause"}
                else:
                    if delta:
                        yield {"seq_id": s.seq_id, "text": delta, "finish_reason": None}
                    still.append(s)
            live = still

    def generate_agentic(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        tool_executor: Callable[[str], str],
        max_turns: int = 5,
    ) -> Iterator[dict]:
        """Agentic generation loop with tool-calling via pause / resume.

        Generates model output, pausing whenever a pause condition (``pause``
        string or ``pause_token_ids`` from *sampling_params*) is hit.  At each
        pause the generated text is passed to *tool_executor*; its return value
        is appended as context and generation resumes.  Repeats up to
        *max_turns* tool calls.

        Yields event dicts::

            {"type": "text",        "text": <delta>}        model output chunk
            {"type": "tool_call",   "text": <call text>}    text up to pause point
            {"type": "tool_result", "text": <result>}       tool_executor return value
            {"type": "done",        "finish_reason": ...}   terminal event

        KV-cache blocks are retained across pauses: only the *new* (tool-result)
        tokens are prefilled on resume -- everything generated so far is reused
        via the prefix-cache fast path.
        """
        seq_id = self.add_request(prompt, sampling_params)
        seq = self._active[seq_id]
        emitted = 0  # char offset into decoded completion text

        for turn in range(max_turns + 1):
            # --- generate until the sequence pauses or finishes ---
            while not seq.is_finished and not seq.is_paused:
                self.step()
                text = self.tokenizer.decode(seq.completion_token_ids)
                if len(text) > emitted:
                    yield {"type": "text", "text": text[emitted:]}
                    emitted = len(text)

            if seq.is_finished:
                text = self.tokenizer.decode(seq.completion_token_ids)
                if len(text) > emitted:
                    yield {"type": "text", "text": text[emitted:]}
                yield {"type": "done", "finish_reason": seq.finish_reason or "stop"}
                return

            if turn >= max_turns:
                # Paused but we've exhausted the turn budget.
                yield {"type": "done", "finish_reason": "length"}
                return

            # --- paused: extract tool-call text, execute tool, resume ---
            text = self.tokenizer.decode(seq.completion_token_ids)
            call_end = seq.pause_index if seq.pause_index >= 0 else len(text)
            call_text = text[:call_end]
            yield {"type": "tool_call", "text": call_text}

            result = tool_executor(call_text)
            yield {"type": "tool_result", "text": result}

            self.resume(seq_id, result)
            # Skip the tool-result text (and any pause delimiter) in subsequent
            # streaming -- only model-generated tokens should appear as "text".
            text = self.tokenizer.decode(seq.completion_token_ids)
            emitted = len(text)
