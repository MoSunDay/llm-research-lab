# nano-vllm-rs GPT-2 Snake weights pipeline

Char-level GPT-2 small "brain" weights for the Rust Snake agent, produced from
vendored [nanoGPT](../../../link_repos/nanoGPT/) and converted into the exact
tensor layout the `nvllm-models` GPT-2 loader (`Gpt2::load`) consumes.

## What & why

- **Model**: standard GPT-2 small — 12 layers, 12 heads, `n_embd=768`, char
  vocabulary (65 Tiny Shakespeare chars), `block_size=256`, **`bias=True`**
  (mandatory — the Rust loader always reads LayerNorm/Linear biases).
- **Data**: Tiny Shakespeare, char-level (no BPE / `tiktoken`). Same recipe as
  `link_repos/nanoGPT/data/shakespeare_char/prepare.py`, but `meta.json`
  (plain JSON, not pickle) is what the Rust side reads.
- **Offline**: downloads use only stdlib `urllib` (resume-capable). No internet
  to HuggingFace is needed or used. nanoGPT itself only needs `torch`.
- **No `torch.compile`**: so checkpoint keys are clean nanoGPT names with no
  `_orig_mod.` prefix.

## Environment

Use the bundled interpreter (has torch 2.7 +cu126, CUDA works on the 2080 Ti,
plus `safetensors` and `numpy`):

```
PY=/root/ComfyUI/env/bin/python
```

## Steps

### 1) Train

```
$PY nano-vllm-rs/scripts/train_gpt2_snake.py \
    --out_dir out-gpt2-snake --max_iters 3000 --batch_size 32 \
    --block_size 256 --grad_accum 4
```

This downloads Tiny Shakespeare into `nano-vllm-rs/scripts/data-shakespeare-char/`
(`input.txt`, `train.bin`, `val.bin`, `meta.json`) on first run, then trains and
writes `out-gpt2-snake/ckpt.pt`. Defaults follow the spec; see
`--help` for all flags (`--device auto` picks CUDA if available).

Quick smoke test (tiny, ~1 min on GPU):

```
$PY nano-vllm-rs/scripts/train_gpt2_snake.py \
    --max_iters 20 --batch_size 8 --eval_iters 10 --out_dir /tmp/snake_smoke
```

### 2) Export to Rust format

```
$PY nano-vllm-rs/scripts/export_safetensors.py \
    --in_dir out-gpt2-snake --data_dir nano-vllm-rs/scripts/data-shakespeare-char
```

Produces `model.safetensors` (only whitelisted `transformer.*` keys, fused
`c_attn`, **no** `lm_head.weight`, **no** mask buffers), `config.json`, and
copies `meta.json` into the out dir. Add `--dtype fp16` for half precision.

### 3) Run Rust Snake

Point the Rust binary at the produced directory (see the `snake-gpt2` crate for
the exact flag). Without weights, the engine falls back to `--random` moves.

## Output layout in `out-gpt2-snake/`

```
ckpt.pt             # nanoGPT checkpoint (model_args + state_dict + config)
model.safetensors   # Rust weights: transformer.* keys only, fp32
config.json         # {block_size, vocab_size, n_layer, n_head, n_embd}
meta.json           # {vocab_size, itos:[...], stoi:{...}}  char<->id map
```

For numerical Rust-vs-PyTorch parity, optionally run:

```
$PY nano-vllm-rs/scripts/compare_logits.py \
    --in_dir out-gpt2-snake --data_dir nano-vllm-rs/scripts/data-shakespeare-char
```

which writes `ref_logits.npy`, `ref_tokens.npy`, `ref_meta.json` (CPU, fully
reproducible) for a Rust test to diff against.

## Files

| file | purpose |
|------|---------|
| `train_gpt2_snake.py` | download + dataset + train -> nanoGPT `ckpt.pt` |
| `export_safetensors.py` | `ckpt.pt` -> `model.safetensors` + `config.json` + `meta.json` |
| `compare_logits.py` | dump CPU reference logits for Rust parity tests |
