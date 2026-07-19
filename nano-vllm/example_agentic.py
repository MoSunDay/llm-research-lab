"""nano-vllm PAUSED state + agentic-loop demo (Phase 4).

Shows the three pause/resume surfaces the engine exposes:

  1. generate_stream() with pause=['.'] -- the sequence PAUSES at the delimiter
     (KV cache retained) instead of stopping; we capture that event, inject fresh
     context via resume(), and continue generating.
  2. Manual pause()/resume() -- force a RUNNING sequence to halt mid-generation,
     then resume() it with appended context.
  3. generate_agentic() -- a turn-based tool-calling loop built on pause/resume:
     each pause hands the generated text to a tool_executor whose return value is
     appended before generation continues, up to max_turns times.

LLM subclasses LLMEngine directly (see nanovllm/llm.py), so generate_stream,
pause, resume, generate_agentic, step, tokenizer and the _active dict are all
available on the `llm` instance -- there is no separate `_engine` attribute.

The GPT-2 124M model is a *base* (not instruction-tuned) model, so the "tool
calls" the agentic loop produces are semantically random: they exercise the full
pause -> tool -> resume wiring but carry no real meaning. That is expected and is
what proves the loop is hooked up correctly (cf. example_grammar.py).
"""
from nanovllm import LLM, SamplingParams


def _drain(llm, seq):
    """Step the engine, printing one line per emitted text delta, until *seq*
    pauses or finishes. Returns the final SequenceStatus name.

    `emitted` is seeded from the *full* decoded completion so that previously
    generated tokens (and any context appended by resume()) are not re-printed
    -- only newly produced tokens are shown.
    """
    emitted = len(llm.tokenizer.decode(seq.completion_token_ids))
    while not seq.is_finished and not seq.is_paused:
        llm.step()
        text = llm.tokenizer.decode(seq.completion_token_ids)
        if len(text) > emitted:
            print(f"      + {text[emitted:]!r}")
            emitted = len(text)
    return seq.status.name


def main():
    path = "/root/models/gpt2-nanovllm"
    # TP=1 is throughput-optimal for 124M; see BENCH.md.
    llm = LLM(path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.7)

    # ------------------------------------------------------------------
    # 1) PAUSE/RESUME via generate_stream
    #    pause=['.'] makes the sequence PAUSE (not stop) when it emits a '.',
    #    keeping its KV cache. We capture that pause event, inject new context
    #    with resume(), and continue generating from the retained cache.
    # ------------------------------------------------------------------
    sp = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True, pause=["."])
    prompt = "The cat sat on the mat"
    print("=" * 60, "\nPAUSE/RESUME via generate_stream  (pause=['.'])")
    print(f"   prompt: {prompt!r}")

    paused_seq_id = None
    so_far = ""
    print("   stream:", end="", flush=True)
    for ev in llm.generate_stream([prompt], sp):
        so_far += ev["text"]
        if ev["finish_reason"] is None:
            print(ev["text"], end="", flush=True)
        elif ev["finish_reason"] == "pause":
            paused_seq_id = ev["seq_id"]
            print(f"\n   [PAUSED at '.'  seq_id={paused_seq_id}]")
            print(f"   text-so-far: {so_far!r}")
        else:
            print(f"\n   [finished: {ev['finish_reason']}]")

    # Only the PAUSED path demonstrates resume(). On a base model the delimiter
    # is near-certain to appear within 40 tokens, but guard anyway.
    if paused_seq_id is not None:
        ctx = " Then it jumped over the fence"
        print(f"   resume(seq_id={paused_seq_id}, {ctx!r})")
        llm.resume(paused_seq_id, ctx)
        # _active maps seq_id -> Sequence; needed to drive the continuation.
        seq = llm._active[paused_seq_id]
        status = _drain(llm, seq)
        print(f"   [continuation ended: status={status}]")
        # NOTE: the original '.' is still in the completion buffer, so the engine
        # re-triggers the pause condition almost immediately after resuming.
        # generate_agentic() (section 3) automates exactly this resume cycle.
    else:
        print("   (sequence finished without emitting '.'; resume() skipped)")

    # ------------------------------------------------------------------
    # 2) MANUAL pause()/resume()
    #    No pause string here -- we step the engine ourselves and call pause()
    #    to halt a RUNNING sequence mid-generation, then resume() with context.
    # ------------------------------------------------------------------
    sp = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)
    prompt = "The quick brown fox"
    print("\n" + "=" * 60, "\nMANUAL pause()/resume()")
    seq_id = llm.add_request(prompt, sp)
    seq = llm._active[seq_id]
    print(f"   prompt: {prompt!r}  seq_id={seq_id}")

    # Decode a few tokens, then force the pause while the sequence is RUNNING.
    steps = 0
    while steps < 4 and not seq.is_finished and not seq.is_paused:
        llm.step()
        steps += 1
    print(f"   after {steps} step(s): generated "
          f"{llm.tokenizer.decode(seq.completion_token_ids)!r}")

    llm.pause(seq_id)
    print(f"   pause({seq_id}) -> status={seq.status.name}, KV cache retained")

    ctx = " and kept on running"
    print(f"   resume({seq_id}, {ctx!r})")
    llm.resume(seq_id, ctx)
    # No pause string is set, so this drains until the max_tokens length cap.
    status = _drain(llm, seq)
    print(f"   [continuation ended: status={status}]")

    # ------------------------------------------------------------------
    # 3) AGENTIC loop via generate_agentic()
    #    pause=['.'] turns each '.' into a "tool call" boundary: the engine
    #    pauses, hands the generated text to tool_executor, appends its return
    #    value, and resumes -- repeating up to max_turns times.
    # ------------------------------------------------------------------
    def echo_tool(call_text: str) -> str:
        """Mock tool: pretend to 'run' the call and return a fake result.

        With a base model call_text is just whatever GPT-2 generated up to the
        '.', so we echo a short snippet back as the (meaningless) result."""
        snippet = " ".join(call_text.split())[:24]
        return f" [tool result: {snippet!r}] "

    sp = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True, pause=["."])
    prompt = "The calculator says"
    print("\n" + "=" * 60,
          "\nAGENTIC LOOP via generate_agentic()  (pause=['.'], max_turns=3)")
    print(f"   prompt: {prompt!r}")
    print("   tool : echo_tool (mock)")
    for ev in llm.generate_agentic(prompt, sp, echo_tool, max_turns=3):
        etype = ev["type"]
        if etype == "text":
            print(f"   [text]    {ev['text']!r}")
        elif etype == "tool_call":
            print(f"   [call]    {ev['text']!r}")
        elif etype == "tool_result":
            print(f"   [result]  {ev['text']!r}")
        elif etype == "done":
            print(f"   [done]    finish_reason={ev['finish_reason']}")

    print("\nPhase 4 (PAUSED state + agentic loop) demo complete.")


if __name__ == "__main__":
    main()
