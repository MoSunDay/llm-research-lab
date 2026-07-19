import os
from nanovllm import LLM, SamplingParams


def main():
    path = "/root/models/gpt2-nanovllm"
    llm = LLM(path, tensor_parallel_size=3, enforce_eager=True, gpu_memory_utilization=0.7)

    # temperature very small -> effectively greedy (argmax), for deterministic output.
    sampling_params = SamplingParams(temperature=0.001, max_tokens=40, ignore_eos=True)
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time, in a deep forest, there lived",
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    for prompt, output in zip(prompts, outputs):
        print("\n" + "=" * 60)
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {(prompt + output['text'])!r}")


if __name__ == "__main__":
    main()
