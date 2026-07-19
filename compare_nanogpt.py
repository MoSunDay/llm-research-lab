"""Direct nanoGPT-vs-nano-vllm comparison (the plan's explicit verify criterion).

Builds the REAL nanoGPT model.GPT from local GPT-2 weights (same transpose logic
as nanoGPT.GPT.from_pretrained), runs GREEDY fp16 for 320 tokens (>kvcache_block_size
256, so this exercises the paged decode path ACROSS a block boundary -- the #1 risk),
and compares token-for-token against nano-vllm TP=3 greedy.

nanoGPT model.py uses nn.GELU() (erf variant) and the standard pre-norm GPT-2 block;
our backend matches it exactly.

Phase 2 result: the Triton paged-decode kernel matches nanoGPT **960/960** greedy tokens
at TP=1 (bit-exact). The cuBLAS torch fallback (NANOVLLM_DETERMINISTIC_ATTENTION=1) also
gives 960/960. Usage: `python compare_nanogpt.py 1`  (positional tensor_parallel_size).
"""
import sys, torch
sys.path.insert(0, "/root/models/nanoGPT")
from model import GPT, GPTConfig
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SRC = "/root/models/gpt2-hf"
PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time, in a deep forest, there lived",
    "Alan Turing was a pioneer in the field of",
]
MAX_NEW = 320   # > 256 (kvcache_block_size) -> crosses a block boundary


def build_nanogpt():
    """Replicate GPT.from_pretrained('gpt2') but load weights from a local dir."""
    cfg = GPTConfig(n_layer=12, n_head=12, n_embd=768,
                    block_size=1024, vocab_size=50257, bias=True, dropout=0.0)
    model = GPT(cfg)
    sd = model.state_dict()
    sd_keys = [k for k in sd if not k.endswith(".attn.bias")]
    hf = GPT2LMHeadModel.from_pretrained(SRC).eval()
    sd_hf = hf.state_dict()
    sd_hf = {k: v for k, v in sd_hf.items()
             if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")}
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight"]
    for k in sd_hf:
        kk = k
        if any(kk.endswith(w) for w in transposed):
            sd[kk].copy_(sd_hf[kk].t())
        else:
            sd[kk].copy_(sd_hf[kk])
    model.load_state_dict(sd)
    return model.half().cuda().eval()


def greedy_nanogpt(model, tok):
    out = {}
    with torch.no_grad():
        for p in PROMPTS:
            x = torch.tensor([tok.encode(p)], device="cuda")
            ids = []
            for _ in range(MAX_NEW):
                # nanoGPT forward returns logits for the LAST position only at inference
                logits, _ = model(x)
                nxt = logits[0, -1].argmax(-1, keepdim=True)
                ids.append(int(nxt))
                x = torch.cat([x, nxt.unsqueeze(0)], dim=1)
            out[p] = ids
    return out


def main():
    tok = GPT2TokenizerFast.from_pretrained(SRC)

    print("[building real nanoGPT GPT from local weights ...]", flush=True)
    ng_model = build_nanogpt()
    print("[running nanoGPT GREEDY fp16, 320 tokens/prompt ...]", flush=True)
    ng_ref = greedy_nanogpt(ng_model, tok)
    del ng_model; torch.cuda.empty_cache()
    for p in PROMPTS:
        print(f"  nanoGPT {p!r}\n      -> {tok.decode(ng_ref[p])[:90]!r}...", flush=True)

    print("\n[loading nano-vllm TP=3 ...]", flush=True)
    from nanovllm import LLM, SamplingParams
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    llm = LLM("/root/models/gpt2-nanovllm", tensor_parallel_size=tp, enforce_eager=True,
              gpu_memory_utilization=0.7)
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW, ignore_eos=True)  # TRUE greedy (Phase 1)
    nv = llm.generate(PROMPTS, sp, use_tqdm=False)
    nv_tok = {p: o["token_ids"] for p, o in zip(PROMPTS, nv)}

    print(f"\n===== nanoGPT vs nano-vllm (TP={tp}), 320 tokens, TRUE greedy, crosses block boundary =====", flush=True)
    total = match = 0
    for p in PROMPTS:
        a, b = ng_ref[p], nv_tok[p]
        n = min(len(a), len(b))
        agree = sum(1 for i in range(n) if a[i] == b[i])
        first_diff = next((i for i in range(n) if a[i] != b[i]), -1)
        total += n; match += agree
        status = "MATCH" if a[:n] == b[:n] else f"first-diff@{first_diff}"
        print(f"\n{p!r}")
        print(f"  {status}  ({agree}/{n} tokens agree)")
        print(f"  nano-vllm tail: ...{tok.decode(b[-12:])!r}")
    print(f"\n>>> overall agreement: {match}/{total} ({100*match/total:.1f}%)")
    print(">>> divergences are expected only at fp16 near-ties (TP summation order).")
    llm.exit()


if __name__ == "__main__":
    main()
