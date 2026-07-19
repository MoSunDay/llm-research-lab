"""Verify nano-vllm GPT-2 (TP=3) against a faithful erf-GELU reference.

Reference = HF GPT2LMHeadModel with activation_function="gelu" (erf variant),
mathematically identical to nanoGPT's model.py. Both fp16. Greedy argmax compared
token-for-token.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config

SRC = "/root/models/gpt2-hf"
PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time, in a deep forest, there lived",
    "Alan Turing was a pioneer in the field of",
]
MAX_NEW = 40


def greedy_ref(act_fn, tok):
    cfg = GPT2Config.from_pretrained(SRC)
    cfg.activation_function = act_fn            # "gelu" -> erf, "gelu_new" -> tanh
    m = GPT2LMHeadModel.from_pretrained(SRC, config=cfg).half().cuda().eval()
    refs = {}
    with torch.no_grad():
        for p in PROMPTS:
            ids = torch.tensor([tok.encode(p)], device="cuda")
            attn = torch.ones_like(ids)
            out = m.generate(ids, attention_mask=attn, do_sample=False,
                             max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
            refs[p] = out[0, ids.shape[1]:].tolist()
    del m; torch.cuda.empty_cache()
    return refs


def main():
    tok = GPT2TokenizerFast.from_pretrained(SRC)
    print(f"tokenizer eos={tok.eos_token_id} bos={tok.bos_token_id}\n", flush=True)

    print("[computing reference greedy (erf = nanoGPT-equivalent) ...]", flush=True)
    ref_erf = greedy_ref("gelu", tok)
    print("[computing reference greedy (tanh = HF default) ...]", flush=True)
    ref_tanh = greedy_ref("gelu_new", tok)
    for p in PROMPTS:
        print(f"  REF erf  {p!r}\n      -> {tok.decode(ref_erf[p])!r}", flush=True)

    print("\n[loading nano-vllm TP=3 ...]", flush=True)
    from nanovllm import LLM, SamplingParams
    llm = LLM("/root/models/gpt2-nanovllm", tensor_parallel_size=3, enforce_eager=True,
              gpu_memory_utilization=0.7)
    sp = SamplingParams(temperature=0.001, max_tokens=MAX_NEW, ignore_eos=True)
    nv = llm.generate(PROMPTS, sp, use_tqdm=False)
    nv_tok = {p: out["token_ids"] for p, out in zip(PROMPTS, nv)}
    llm.exit()

    print("\n===== TOKEN-FOR-TOKEN COMPARISON =====", flush=True)
    all_match = True
    for p in PROMPTS:
        e, t, n = ref_erf[p], ref_tanh[p], nv_tok[p]
        me = "MATCH" if n == e else "DIFF"
        mt = "MATCH" if n == t else "DIFF"
        ed = sum(1 for a, b in zip(e, n) if a == b)
        td = sum(1 for a, b in zip(t, n) if a == b)
        all_match &= (n == e)
        print(f"\n{p!r}")
        print(f"  nano-vllm vs erf (nanoGPT-equivalent): {me}  ({ed}/{len(e)} tokens agree)")
        print(f"  nano-vllm vs tanh (HF default)       : {mt}  ({td}/{len(t)} tokens agree)")
        print(f"  NV OUT: {tok.decode(n)!r}")
        if me == "DIFF":
            print(f"    nv : {n}")
            print(f"    erf: {e}")
    print("\n>>> RESULT: ALL prompts MATCH erf reference greedily." if all_match
          else "\n>>> RESULT: some divergence (see above).")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
