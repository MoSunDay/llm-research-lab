"""nano-vllm offline benchmark.

Measures prefill + decode throughput for an offline batch of random prompts.
Works for any nano-vllm backend (GPT-2 124M, Qwen3, ...).

Examples:
    python bench.py                                     # GPT-2 124M, TP=1
    python bench.py --tensor-parallel-size 3            # GPT-2 124M, TP=3
    python bench.py --model /path/to/Qwen3-0.6B --max-input-len 1024 --max-output-len 1024
    python bench.py --eager                             # disable CUDA graph

The engine is driven directly (add_request + step loop) instead of generate()
so that prefill and decode phases can be timed separately. Each step() ends with
a GPU->CPU sync (the sampled token .tolist()), so per-step wall time already
includes GPU compute.
"""
import argparse
import os
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser(description="nano-vllm offline throughput benchmark")
    p.add_argument("--model", default="/root/models/gpt2-nanovllm",
                   help="path to the nanovllm-format checkpoint")
    p.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--max-input-len", type=int, default=256)
    p.add_argument("--max-output-len", type=int, default=256)
    p.add_argument("--min-len", type=int, default=16, help="min length for randomized seq lens")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--enforce-eager", "--eager", dest="enforce_eager", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-warmup", dest="warmup", action="store_false", default=True)
    return p.parse_args()


def run(llm, prompt_token_ids, sampling_params):
    """Drive the engine manually and time prefill vs decode phases."""
    for ids, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(ids, sp)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    prefill_tokens, prefill_time = 0, 0.0
    decode_time, decode_steps = 0.0, 0
    total_output_tokens = 0

    t_start = time.perf_counter()
    while not llm.is_finished():
        t = time.perf_counter()
        output, num_tokens = llm.step()
        dt = time.perf_counter() - t
        if num_tokens > 0:                       # prefill step: +tokens prefilled
            prefill_tokens += num_tokens
            prefill_time += dt
        else:                                    # decode step: -batch_size
            decode_steps += 1
            decode_time += dt
            total_output_tokens += -num_tokens
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    return {
        "total_time": total_time,
        "prefill_tokens": prefill_tokens,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "decode_steps": decode_steps,
        "total_output_tokens": total_output_tokens,
        "peak_mem_gib": peak_mem,
    }


def fmt(stats):
    prefill_tps = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0.0
    decode_tps = stats["total_output_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0.0
    overall_tps = stats["total_output_tokens"] / stats["total_time"]
    return (f"output={stats['total_output_tokens']}tok prefill={stats['prefill_tokens']}tok "
            f"| total={stats['total_time']:.2f}s prefill={stats['prefill_time']:.2f}s decode={stats['decode_time']:.2f}s "
            f"| prefill={prefill_tps:.0f}tok/s decode={decode_tps:.0f}tok/s overall={overall_tps:.0f}tok/s "
            f"| peak={stats['peak_mem_gib']:.2f}GiB")
    # also return numbers programmatically below


def main():
    args = parse_args()
    seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[bench] model={args.model} tp={args.tensor_parallel_size} "
          f"num_seqs={args.num_seqs} in<={args.max_input_len} out<={args.max_output_len} "
          f"eager={args.enforce_eager} gmu={args.gpu_memory_utilization}", flush=True)

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    vocab_size = llm.tokenizer.vocab_size

    # pre-tokenized random prompts -> timing excludes tokenization cost
    prompt_token_ids = [
        [randint(0, vocab_size - 1) for _ in range(randint(args.min_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True,
                       max_tokens=randint(args.min_len, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    avg_in = sum(len(p) for p in prompt_token_ids) / len(prompt_token_ids)
    avg_out = sum(sp.max_tokens for sp in sampling_params) / len(sampling_params)

    if args.warmup:
        t0 = time.perf_counter()
        llm.generate(["Warmup. "],
                     SamplingParams(temperature=args.temperature, max_tokens=8, ignore_eos=True),
                     use_tqdm=False)
        print(f"[bench] warmup done in {time.perf_counter()-t0:.2f}s", flush=True)

    stats = run(llm, prompt_token_ids, sampling_params)
    line = fmt(stats)
    print(f"[bench] avg_in={avg_in:.0f} avg_out={avg_out:.0f} {line}", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
