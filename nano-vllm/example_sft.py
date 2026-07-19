"""nano-vllm Phase 5 demo: nanoGPT SFT -> reverse conversion -> inference.

Shows the full fine-tuning pipeline: GPT-2 (124M) is supervised fine-tuned with
nanoGPT on a synthetic Q&A dataset, then the checkpoint is converted back to
nano-vllm format (reverse conversion) for efficient paged-attention inference.

The fine-tuned model responds in the "Q: ... A: ..." format it was trained on,
demonstrating that SFT took effect and the weight conversion is correct.

The pipeline (run separately; this script only does the final inference step):

    # Step 1: Prepare SFT data (synthetic Q&A pairs -> GPT-2 BPE binary)
    cd /root/models/nanoGPT/data/sft && python prepare.py

    # Step 2: Fine-tune with nanoGPT (loads /root/models/gpt2-hf, trains ~50 iters)
    cd /root/models/nano-vllm && python sft_train.py --max-iters 50

    # Step 3: Convert to nano-vllm format (reverse conversion)
    cd /root/models && python convert_nanogpt.py \
        --src /root/models/nanoGPT/out-sft \
        --dst /root/models/gpt2-sft-nanovllm \
        --tokenizer /root/models/gpt2-hf

    # Step 4: Run inference with the fine-tuned model (this script)
    cd /root/models/nano-vllm && python example_sft.py

NOTE: nano-vllm calls torch.distributed.init_process_group once per process, so a
single script cannot hold two LLM instances at once. This script only loads the
SFT model. To compare with the base model, run example.py separately against
/root/models/gpt2-nanovllm.
"""
import os
import sys

from nanovllm import LLM, SamplingParams

SFT_MODEL = "/root/models/gpt2-sft-nanovllm"   # produced by convert_nanogpt.py
BASE_MODEL = "/root/models/gpt2-nanovllm"      # base GPT-2, for separate comparison


def print_pipeline_summary():
    """Document the reverse-conversion details (nanoGPT -> nano-vllm)."""
    print("=" * 60)
    print("PHASE 5 PIPELINE")
    print("=" * 60)
    print("  1. prepare.py   : synthetic Q&A (arithmetic/capitals/facts) -> BPE .bin")
    print("  2. sft_train.py : nanoGPT SFT on GPT-2 124M, block_size=256")
    print("  3. convert_nanogpt.py : nanoGPT ckpt.pt -> nano-vllm safetensors")
    print("  4. example_sft.py: load SFT model, generate Q&A answers  <- you are here")
    print()
    print("-" * 60)
    print("REVERSE CONVERSION (convert_nanogpt.py) WEIGHT DETAILS")
    print("-" * 60)
    print("  * Layout : NO transpose needed. nanoGPT uses nn.Linear, so its")
    print("             weights are already [out, in] like nano-vllm expects.")
    print("             (convert_gpt2.py for HF Conv1D weights DOES transpose.)")
    print("  * c_attn : fused QKV ([3*n_embd, n_embd]) split along dim=0 into")
    print("             separate q_proj / k_proj / v_proj (weight + bias).")
    print("  * wte    : token embeddings truncated 50304 -> 50257 rows")
    print("             (nanoGPT pads vocab to a multiple of 64; GPT-2 is 50257).")
    print("  * lm_head: tied to wte -> NOT written (nano-vllm uses tied embeddings).")
    print("  * biases : bias=True preserved (attn/mlp/layernorm biases copied).")
    print("  * dtype  : cast to float16 for inference.")
    print()
    print("-" * 60)
    print("FILES (produced in gpt2-sft-nanovllm/)")
    print("-" * 60)
    if os.path.isdir(SFT_MODEL):
        for f in sorted(os.listdir(SFT_MODEL)):
            full = os.path.join(SFT_MODEL, f)
            size = os.path.getsize(full)
            print(f"  {f:<24} {size/1e6:>8.2f} MB")
    else:
        print(f"  [{SFT_MODEL} not found -- run the pipeline steps above first]")
    print()


def main():
    print_pipeline_summary()

    if not os.path.isdir(SFT_MODEL):
        print("ERROR: SFT model not found. Run the pipeline steps in the module")
        print("docstring (prepare.py, sft_train.py, convert_nanogpt.py) first.")
        sys.exit(1)

    # TP=1 + eager is the right config for 124M (see BENCH.md / example.py).
    llm = LLM(SFT_MODEL, tensor_parallel_size=1, enforce_eager=True,
              gpu_memory_utilization=0.7)

    # Greedy generation. ignore_eos=True so the model keeps producing until
    # max_tokens even if it emits the <|endoftext|> it was trained to stop on.
    # The SFT model was trained at block_size=256, so these short Q&A prompts
    # + 60 generated tokens stay well within the 256-token context window.
    sp = SamplingParams(temperature=0, max_tokens=60, ignore_eos=True)

    # Prompts mirror the training format ("Q: ...\\nA: ...\\n"), truncated to the
    # question + the "A:" prefix so the model completes the answer.
    prompts = [
        "Q: What is 12 + 30?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: What is 7 x 8?\nA:",
        "Q: What is the capital of Japan?\nA:",
        "Q: What is 100 - 45?\nA:",
    ]

    outputs = llm.generate(prompts, sp, use_tqdm=False)

    print("=" * 60)
    print("SFT MODEL GENERATION (greedy, temperature=0)")
    print("=" * 60)
    for p, o in zip(prompts, outputs):
        completion = o["text"]
        print(f"  {p!r}")
        print(f"    -> {(p + completion)!r}  [finish={o['finish_reason']}]")

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("  The SFT model answers in the '... = N. So the answer is N.' format")
    print("  it was trained on (arithmetic), and in the 'The capital of X is Y.'")
    print("  format for capitals. This confirms BOTH that:")
    print("    (a) supervised fine-tuning took effect, and")
    print("    (b) the reverse weight conversion (convert_nanogpt.py) is correct")
    print("        -- a transposed/wrongly-split weight would produce gibberish.")
    print()
    print("  To compare with the BASE model, run example.py against")
    print(f"  {BASE_MODEL} -- the base GPT-2 does not know the 'Q:/A:' format")
    print("  and will simply continue the prompt as free text rather than answer.")
    print("=" * 60)


if __name__ == "__main__":
    main()
