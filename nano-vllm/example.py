"""nano-vllm GPT-2 (124M) demo: greedy, stop, and streaming (Phase 1 features)."""
import os
from nanovllm import LLM, SamplingParams


def main():
    path = "/root/models/gpt2-nanovllm"
    # TP=1 is the throughput-optimal default for 124M (see BENCH.md).
    llm = LLM(path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.7)

    # 1) TRUE greedy (temperature=0) -> deterministic argmax, matches nanoGPT exactly.
    greedy = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time, in a deep forest, there lived",
    ]
    outputs = llm.generate(prompts, greedy, use_tqdm=False)
    print("=" * 60, "\nGREEDY (temperature=0)")
    for p, o in zip(prompts, outputs):
        print(f"  {p!r}\n   -> {(p + o['text'])!r}  [finish={o['finish_reason']}]")

    # 2) Stop string -> generation stops and the stop text is truncated out.
    stopped = SamplingParams(temperature=0, max_tokens=60, ignore_eos=True, stop=[" the"])
    o = llm.generate(["Alan Turing was a pioneer in the field of"], stopped, use_tqdm=False)[0]
    print("\n" + "=" * 60, "\nSTOP=[' the']")
    print(f"   {o['text']!r}  [finish={o['finish_reason']}, has ' the'? {' the' in o['text']}]")

    # 3) Streaming -> yield token deltas as they are produced.
    print("\n" + "=" * 60, "\nSTREAMING (greedy, 30 tokens)")
    sp = SamplingParams(temperature=0, max_tokens=30, ignore_eos=True)
    print("   ", end="", flush=True)
    for chunk in llm.generate_stream(["To be, or not to be"], sp):
        print(chunk["text"], end="", flush=True)
    print("  [done]")


if __name__ == "__main__":
    main()
