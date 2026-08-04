# llm-research-lab

> A hands-on workbench for large-model inference and experimentation: a full
> **nanoGPT training → weight export → Rust inference** pipeline, backed by a
> Python reference engine for cross-validation.

[English](README.md) · [简体中文](README.zh-CN.md)

---

## What is this?

This repository explores LLM inference engineering through two parallel tracks:

| Track | Path | Status |
|-------|------|--------|
| **Rust workspace** | [`nano-vllm-rs/`](nano-vllm-rs/) | 🟢 Active development |
| **Python reference engine** | [`nano-vllm/`](nano-vllm/) | 📖 Read-only reference |

The Rust track is the primary focus: a GPT-2 model ported to
[candle](https://github.com/huggingface/candle) 0.11, trained on nanoGPT,
exported to safetensors, and consumed by a terminal Snake game whose decisions
are driven by the model — either via **constrained decoding** or a full
**agent loop** (generate → bash → inject result → repeat).

The Python track (`nano-vllm/`) is a from-scratch inference engine providing
constrained decoding, agentic pause/resume, and SFT round-trip capabilities,
serving as the correctness reference and design inspiration for the Rust work.

## Architecture

```
                    ┌─── Rust Track (active) ──────────────────────┐
                    │                                              │
  train_gpt2_snake.py ──► ckpt.pt                                 │
        (nanoGPT / torch)      │                                  │
                              ▼                                   │
  export_safetensors.py ──► model.safetensors (transformer.*)     │
                              │                                   │
               ┌──────────────┼──────────────┐                   │
               ▼              ▼              ▼                    │
         nvllm-models    snake-gpt2     nvllm-engine              │
         (GPT-2 fwd)    (game+brain)   (scheduler WIP)            │
                    │                                              │
                    └──────────────────────────────────────────────┘

                    ┌─── Python Track (reference) ────────────────┐
                    │                                              │
  nano-vllm/ ── engine (scheduler, block_manager, sampler)       │
     ├── grammar.py        constrained decoding (xgrammar)       │
     ├── engine/           pause / resume / agentic generate     │
     └── sft_train.py      nanoGPT SFT → reverse-convert → verify│
                    │                                              │
                    └──────────────────────────────────────────────┘
```

## Key demos

### 1 · GPT-2 Snake (Rust, constrained decoding)

Train a char-level GPT-2 on Tiny Shakespeare, export to safetensors, and let
the model play terminal Snake via constrained (greedy) decoding.

```
train_gpt2_snake.py → ckpt.pt → export_safetensors.py → model.safetensors
                                                           │
                          snake-gpt2 (constrained decoding mode) ◄┘
```

- **Vocab**: 65 chars (Tiny Shakespeare), `block_size=256`, `bias=True`
- **Pipeline**: nanoGPT train → safetensors export → Rust candle forward
- **Parity**: `compare_logits.py` dumps CPU reference logits for Rust diff

### 2 · GPT-2 Agent SFT (Rust, agent loop)

Warm-start from the Shakespeare checkpoint, fine-tune on synthetic agent
traces (game decisions + code tasks), and run an agent loop in Rust:

```
sft_train_gpt2_snake.py → ckpt.pt (vocab=73) → export_agent_safetensors.py
                                                        │
                    snake-gpt2 --agent (generate → bash → inject → repeat) ◄┘
```

- **Vocab**: 73 tokens = 65 base chars + 8 control tokens
  (`<bash>`, `</bash>`, `<pause>`, `<result>`, `</result>`,
  `<resume>`, `<answer>`, `</answer>`)
- **Training**: 840 loss-masked traces, val_loss 5.08 → 0.066
- **Agent loop**: generates `<bash>` commands → sandbox executes → result
  injected as `<result>` → loop until `</answer>`

### 3 · Python nano-vllm engine (reference)

| Phase | Capability |
|-------|-----------|
| Phase 3 | Guided decoding via xgrammar (JSON schema, regex, grammar, choice) |
| Phase 4 | Agentic pause/resume (`PAUSED` status, KV-cache fix, V100 attention rewrite) |
| Phase 5 | SFT round-trip: nanoGPT fine-tune → reverse-convert → bit-exact parity |

## Repository layout

```
nano-vllm-rs/                  Rust workspace (active)
  crates/
    nvllm-models/              GPT-2 model definition (candle 0.11)
    snake-gpt2/                Terminal Snake + GPT-2 brain + agent loop
    nvllm-engine/              Inference scheduler (placeholder)
  scripts/
    train_gpt2_snake.py        nanoGPT training → ckpt.pt
    export_safetensors.py      ckpt → safetensors (Rust format)
    compare_logits.py          Rust-vs-PyTorch parity
    agent_special_tokens.py    vocab=73 contract + encode/decode
    prepare_agent_snake.py     generate 840 agent traces
    sft_train_gpt2_snake.py    masked-loss SFT trainer
    export_agent_safetensors.py ckpt → safetensors (vocab=73)
nano-vllm/                     Python reference engine (read-only)
  nanovllm/                    core package (engine, layers, grammar)
  test_*.py                    phase tests
  example*.py                  runnable demos
  sft_train.py                 nanoGPT SFT driver
convert_gpt2.py                HF GPT-2 → nano-vllm weights (Python track)
convert_nanogpt.py             nanoGPT → nano-vllm weights (Python track)
verify_gpt2.py                 conversion parity check
compare_nanogpt.py             nanoGPT-vs-nano-vllm greedy parity
BENCH.md                       Python engine benchmark results
```

## Quick start

### Rust Snake demo

```bash
# 1 · Train (needs torch + CUDA)
PY=/root/ComfyUI/env/bin/python
$PY nano-vllm-rs/scripts/train_gpt2_snake.py \
    --out_dir out-gpt2-snake --max_iters 3000

# 2 · Export to Rust format
$PY nano-vllm-rs/scripts/export_safetensors.py \
    --in_dir out-gpt2-snake \
    --data_dir nano-vllm-rs/scripts/data-shakespeare-char

# 3 · Run
cd nano-vllm-rs && cargo run -p snake-gpt2 --release -- \
    --weights ../out-gpt2-snake
```

### Python engine

```bash
cd nano-vllm
python example.py           # base GPT-2 demo
python example_grammar.py   # constrained decoding (Phase 3)
python example_agentic.py   # pause/resume tool loop (Phase 4)
python example_sft.py       # SFT model demo (Phase 5)
```

## Environment

| Component | Spec |
|-----------|------|
| **GPU** | RTX 2080 Ti (Rust dev) / 3× Tesla V100-32GB (Python engine) |
| **CUDA** | 12.x |
| **Python** | 3.10.12, torch 2.7+cu126, transformers 4.51.3, safetensors 0.5.3 |
| **Rust** | edition 2021, candle-core/nn 0.11, crossterm 0.27, clap 4.6 |
| **Offline** | HuggingFace network blocked; GPT-2 BPE unavailable → char-level vocab used |

> The Rust demos use **char-level tokenization** (vocab 65/73) precisely
> because the BPE tokenizer (`tiktoken`) cannot be installed offline.

## Known limitations

- **Rust inference is CPU-only** (~7 s/forward): no KV cache, no GPU offload.
- **BPE tokenizer unavailable** offline; both demos use char-level vocab.
- **`nvllm-engine` scheduler is a placeholder** — only `version()` is implemented.
- **Top-level `convert_*.py`** produces `model.*` keys that are **incompatible**
  with the Rust `Gpt2::load` (which expects nanoGPT-native `transformer.*` with
  fused `c_attn`). Use `export_safetensors.py` for the Rust track.
- Weights and checkpoints are gitignored.

## Documentation

- [Agent memory docs](agents/) — detailed module notes
- [Feature docs](features/) — capability index + changelog
- [Benchmark results](BENCH.md) — Python engine performance
- [Rust scripts README](nano-vllm-rs/scripts/README.md) — weights pipeline

## License

Research / educational project. See vendored upstream licenses in
[`nano-vllm/`](nano-vllm/) and [`link_repos/`](link_repos/).
