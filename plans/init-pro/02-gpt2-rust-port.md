# 02 -- GPT-2 Rust Port: Architecture, Weights, and Verification

> Reference source: link_repos/nanoGPT/ (READ-ONLY reference -- for understanding only; never modified).

The authoritative architecture is `link_repos/nanoGPT/model.py` (330 lines).
Every item below is grounded by reading that file. The cross-reference backend
in the working copy `nano-vllm/nanovllm/models/gpt2.py` is consistent with it.

## Architecture checklist (from model.py)

- LayerNorm: weight (ones), optional bias (zeros when bias=True). Uses
  `F.layer_norm(..., 1e-5)`, so eps = 1e-5, fp32 variance reduction (PyTorch
  default). (model.py, LayerNorm.)
- CausalSelfAttention: fused `c_attn = Linear(n_embd, 3*n_embd, bias)` then
  `split(n_embd)` into q, k, v, each reshaped to `(B, nh, T, hs)` where
  `hs = n_embd // n_head`. Causal scaled dot-product attention, scale
  `1/sqrt(hs)`. Output projection `c_proj = Linear(n_embd, n_embd, bias)`.
  (model.py, CausalSelfAttention.)
- MLP: `c_fc = Linear(n_embd, 4*n_embd, bias)`, then `nn.GELU()` (the EXACT erf
  variant, i.e. `approximate='none'`), then `c_proj = Linear(4*n_embd, n_embd,
  bias)`. (model.py, MLP.)
- Block: pre-norm. `x = x + attn(ln_1(x))`; `x = x + mlp(ln_2(x))`.
  (model.py, Block.)
- GPT: token embedding `wte = Embedding(vocab_size, n_embd)`; learned absolute
  position embedding `wpe = Embedding(block_size, n_embd)`; dropout; a stack of
  Blocks; final `ln_f = LayerNorm(n_embd, bias=True)`; `lm_head` whose weight is
  TIED to `wte` (`lm_head.weight = wte.weight`). Logits = `lm_head(x)`.
  (model.py, GPT.)
- GPTConfig defaults: `block_size=1024`, `vocab_size=50304`, `n_layer=12`,
  `n_head=12`, `n_embd=768`, `dropout=0.0`, `bias=True`. (GPTConfig.) GPT-2 base
  (124M) per `from_pretrained`: n_layer=12, n_head=12, n_embd=768, block_size
  1024, vocab_size forced to 50257 for checkpoint loading, bias=True.

## Numeric-exactness requirements (must honor)

1. erf GELU, NOT the tanh approximation. model.py uses plain `nn.GELU()` whose
   default is the erf variant. The working-copy backend notes the same: exact
   erf to match nanoGPT. In candle use the erf gelu, not a fast approx.
2. Exact causal masking: an upper-triangular `-inf` mask (model.py registers a
   triangular `bias` buffer for the non-flash path). On CPU there is no flash
   kernel, so implement the explicit causal mask.
3. fp32 LayerNorm: reduce variance/mean in fp32 (PyTorch `F.layer_norm` default).
   Keep LayerNorm statistics in fp32 even if weights are fp16.
4. Weight layout: nanoGPT stores `nn.Linear` weights as `[out, in]`.
   `convert_nanogpt.py` performs NO transpose (this is the difference from
   `convert_gpt2.py`, which transposes HF Conv1D `[in, out]`). candle `Linear`
   also expects `[out, in]`, so load directly.
5. Tied lm_head: do not load a separate head weight; compute logits as
   `x @ wte.weight.transpose()`. `convert_nanogpt.py` intentionally does NOT
   write an lm_head tensor.
6. Padding: nanoGPT pads the embedding to 50304 (next multiple of 64 past 50257)
   for matmul efficiency. `convert_nanogpt.py` truncates `wte` to 50257 rows on
   output and writes `vocab_size: 50257`. Only the first 50257 rows participate
   in embedding lookup and in the softmax over logits. (The 47 free slots
   50257..50303 are reclaimed for special tokens; see doc 03.)

## Weight loading plan (from convert_nanogpt.py output)

`convert_nanogpt.py` writes a single `model.safetensors` (dtype float16,
metadata `{"format": "pt"}`) plus `config.json` (native GPT-2 fields and
nano-vllm aliases) plus tokenizer files. The Rust loader maps these tensor keys
to the GPT-2 structs. Key names (verified by reading convert_nanogpt.py):

| safetensors key (per layer i) | nanoGPT source | Rust struct field |
|---|---|---|
| model.embed_tokens.weight            | transformer.wte.weight (truncated to 50257) | wte |
| model.pos_embeddings.weight          | transformer.wpe.weight                       | wpe |
| model.norm.weight / .bias            | transformer.ln_f.weight / .bias              | ln_f |
| model.layers.{i}.input_layernorm.{w,b}            | h.{i}.ln_1            | Block.ln_1 |
| model.layers.{i}.post_attention_layernorm.{w,b}   | h.{i}.ln_2            | Block.ln_2 |
| model.layers.{i}.self_attn.q_proj.{weight,bias}   | h.{i}.attn.c_attn (q split) | Attn.q_proj |
| model.layers.{i}.self_attn.k_proj.{weight,bias}   | h.{i}.attn.c_attn (k split) | Attn.k_proj |
| model.layers.{i}.self_attn.v_proj.{weight,bias}   | h.{i}.attn.c_attn (v split) | Attn.v_proj |
| model.layers.{i}.self_attn.o_proj.{weight,bias}   | h.{i}.attn.c_proj          | Attn.o_proj |
| model.layers.{i}.mlp.fc.{weight,bias}             | h.{i}.mlp.c_fc             | Mlp.fc |
| model.layers.{i}.mlp.proj.{weight,bias}           | h.{i}.mlp.c_proj           | Mlp.proj |
| (lm_head) -- not written            | tied to wte                                  | none; use wte |

Notes: q/k/v are already split in the safetensors file (no need to split a fused
tensor in Rust). Bias tensors are present because GPT-2 uses `bias=True`; the
converter falls back to zeros only if a bias key is absent (bias=False). `wte`
is truncated from 50304 to 50257 rows on write; for special tokens the Rust side
will widen the effective vocab to 50304 (see doc 03) by padding rows back.

config.json fields read by the loader (verified): `model_type=gpt2`,
`n_embd`, `n_head`, `n_layer`, `n_positions` (= block_size), `vocab_size=50257`,
`layer_norm_epsilon=1e-5`, `bos_token_id=50256`, `eos_token_id=50256`,
`torch_dtype=float16`, `head_dim=n_embd/n_head`, plus aliases
`max_position_embeddings`, `num_hidden_layers`, `num_attention_heads`,
`hidden_size`, `num_key_value_heads`.

## Verification protocol (the gate)

The hard correctness gate is bit-exact greedy agreement with nanoGPT.

- Build nanoGPT `GPT` with `GPTConfig(n_layer=12, n_head=12, n_embd=768,
  block_size=1024, vocab_size=50257, bias=True, dropout=0.0)` (matches
  `compare_nanogpt.py`).
- Run identical greedy decoding (temperature 0 / argmax) in both nanoGPT and the
  Rust GPT-2 from the same tokenized prompts.
- Require 100% token-id match for >= 257 generated tokens per prompt. 257 is
  chosen to exceed the KV cache block boundary used by the reference engine
  (`compare_nanogpt.py` runs 320 new tokens and notes 320 > 256
  `kvcache_block_size`, crossing a paged-decode block boundary).
- Tolerance: greedy argmax can diverge only at near-ties. If divergence occurs,
  inspect logits diff at the first mismatch; fp16 accumulation order may shift
  ulps. If nanoGPT and nano-vllm already agree 960/960 (as `compare_nanogpt.py`
  reports) but the Rust port differs, treat it as a port bug, not a tolerance.

STATUS: BLOCKED until GPT-2 weights are present (the safetensors loader needs
real data; `/root/models` is currently absent). Structural forward and shape
checks (Phase A) can proceed without weights.

## Rust struct sketch (functional style)

```rust
// nvllm-models/src/models/gpt2.rs  (sketch; shapes/verify on implementation)
use candle_core::{Tensor, Device};
use candle_nn::{VarBuilder, LayerNorm, Linear};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,        // base 50257; effective 50304 with special rows
    pub n_positions: usize,       // block_size, 1024
    pub n_embd: usize,            // 768
    pub n_layer: usize,           // 12
    pub n_head: usize,            // 12
    pub layer_norm_epsilon: f64,  // 1e-5
}

pub struct Gpt2 {
    pub cfg: Gpt2Config,
    pub wte: candle_nn::Embedding,
    pub wpe: candle_nn::Embedding,
    pub h: Vec<Block>,
    pub ln_f: LayerNorm,
    // lm_head is tied to wte: logits = x @ wte.weights().t()
}

pub struct Block { pub ln_1: LayerNorm, pub attn: Attn, pub ln_2: LayerNorm, pub mlp: Mlp }
pub struct Attn { pub q_proj: Linear, pub k_proj: Linear,
                  pub v_proj: Linear, pub o_proj: Linear, pub n_head: usize }
pub struct Mlp { pub fc: Linear, pub proj: Linear }

pub fn forward(m: &Gpt2, ids: &Tensor, pos: &Tensor) -> anyhow::Result<Tensor> {
    // tok = wte(ids) + wpe(pos); for each block: pre-norm attn + pre-norm mlp;
    // ln_f; logits = final @ wte.weights().t().  erf gelu in mlp; causal mask.
    todo!("Phase A")
}
```

No classes: structs hold `VarBuilder`-backed tensors; `forward` is a free
function taking `&Gpt2` by reference and returning a new `Tensor`.
