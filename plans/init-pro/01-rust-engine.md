# 01 -- Rust Engine: Cargo Workspace and Scope Phasing

> Reference source: link_repos/nano-vllm/ (READ-ONLY reference -- for understanding only; never modified).

## Workspace design (already scaffolded)

The `nano-vllm-rs/` workspace already exists with this layout (verified):

    nano-vllm-rs/
    |-- Cargo.toml               workspace, resolver="2", members below
    `-- crates/
        |-- nvllm-models/        model definitions (gpt2); src/ empty
        `-- nvllm-engine/        runtime; src/ empty

Root `Cargo.toml` facts (read directly): `members = ["crates/nvllm-models",
"crates/nvllm-engine"]`; `version = "0.0.1"`, `edition = "2021"`,
`rust-version = "1.96"`; workspace dependencies `candle-core = "0.11"`,
`candle-nn = "0.11"`, `safetensors = "0.4"`, `serde`, `serde_json`, `anyhow`.

Proposed third crate (not yet present):

    crates/nvllm-serving/        binary / cli entry point (depends on engine + models)

Per-module files stay under 400 lines (new-file limit). Each crate's `src/` is
split by responsibility so no single file grows toward 800.

## Module mapping: link_repos/nano-vllm/nanovllm/ -> nvllm-* crate

The `link_repos/nano-vllm/nanovllm/` tree is the reference. Confirmed contents:
`engine/{block_manager,llm_engine,model_runner,scheduler,sequence}.py`,
`layers/{activation,attention,embed_head,layernorm,linear,rotary_embedding,
sampler}.py`, `models/qwen3.py`, `utils/{context,loader}.py`, plus top-level
`config.py`, `llm.py`, `sampling_params.py`.

| Python reference (link_repos/nano-vllm/nanovllm/) | Rust target | Notes |
|---|---|---|
| engine/sequence.py            | nvllm-engine/src/sequence.rs     | request/token bookkeeping |
| engine/scheduler.py           | nvllm-engine/src/scheduler.rs    | continuous batching |
| engine/block_manager.py       | nvllm-engine/src/block_manager.rs| paged KV blocks |
| engine/model_runner.py        | nvllm-engine/src/model_runner.rs | fwd + KV cache wiring |
| engine/llm_engine.py          | nvllm-engine/src/engine.rs       | orchestration |
| layers/attention.py           | nvllm-models/src/layers/attention.rs | causal SDPA |
| layers/linear.py              | nvllm-models/src/layers/linear.rs    | plain Linear (no TP in v1) |
| layers/layernorm.py           | nvllm-models/src/layers/layernorm.rs | LayerNorm(bias, eps=1e-5) |
| layers/activation.py          | nvllm-models/src/layers/activation.rs| erf GELU |
| layers/embed_head.py          | nvllm-models/src/layers/embed.rs     | wte/wpe, tied lm_head |
| layers/rotary_embedding.py    | nvllm-models/src/layers/rotary.rs   | DEFERRED (GPT-2 has no RoPE) |
| layers/sampler.py             | nvllm-engine/src/sampler.rs         | greedy/top-k/top-p |
| utils/context.py              | nvllm-engine/src/context.rs         | step/prefill-decode context |
| utils/loader.py               | nvllm-models/src/loader.rs          | safetensors reader |
| config.py                     | nvllm-engine/src/config.rs          | config.json -> structs |
| sampling_params.py            | nvllm-engine/src/sampling.rs        | SamplingParams |
| llm.py                        | nvllm-serving/src/main.rs           | cli entry |
| models/qwen3.py               | deferred                           | not in GPT-2 scope |
| models/gpt2.py (working copy) | nvllm-models/src/models/gpt2.rs     | see `02-gpt2-rust-port.md` |
| layers/triton_kernels.py (working copy) | deferred               | GPU-only, see below |

Grounding note: `models/gpt2.py` and `layers/triton_kernels.py` are NOT in
`link_repos/nano-vllm/nanovllm/` (that snapshot has only `models/qwen3.py`). They
exist in the top-level working copy `nano-vllm/nanovllm/`. The architecture
facts for the Rust GPT-2 are grounded in `link_repos/nanoGPT/model.py` instead;
see `02-gpt2-rust-port.md`.

## candle rationale (chosen stack)

candle (huggingface/candle) is the chosen tensor/nn stack because:

- Pure Rust, CPU-verifiable out of the box; deterministic on CPU for the
  bit-exact comparison goal.
- Native safetensors loader (matches the `model.safetensors` written by
  `convert_nanogpt.py`, which sets metadata `{"format": "pt"}`).
- Optional CUDA lives behind a cargo feature (`cuda`), so GPU is opt-in and is
  not a build requirement for the port or for verification.
- `candle-nn` provides `VarBuilder`, `Linear`, `LayerNorm`, `Embedding` building
  blocks that map cleanly onto the reference modules above.

The workspace already pins `candle-core = "0.11"` and `candle-nn = "0.11"`.

## Scope phasing

Phase A -- structural GPT-2 forward (CPU).
  Config + structs in `nvllm-models`; a forward path that runs end-to-end on
  random/zero weights. Smoke test asserts output shapes only (no numerics).
  Deliverable: `nvllm-models/src/models/gpt2.rs` + a shape test.

Phase B -- real weights + numerical verify.
  Load `model.safetensors` from `convert_nanogpt.py` output via a loader in
  `nvllm-models/src/loader.rs`; run greedy decoding and compare to nanoGPT.
  This phase is BLOCKED until weights are present (see environment status in
  `00-overview.md`). Compare at >=257 tokens to cross a KV block boundary.

Phase C -- paged KV cache + continuous-batch scheduler.
  Port `block_manager.py` and `scheduler.py` (paged KV allocation, prefill +
  decode). Reference `model_runner.py` uses `allocate_kv_cache` and reads a
  block size from config (`kvcache_block_size`); `compare_nanogpt.py` runs 320
  new tokens and notes this is `> 256 (kvcache_block_size)`, so 256 is the
  boundary the comparison crosses.

Phase D -- sampler + tokenizer.
  Port `sampler.py` (greedy / top-k / top-p) into `nvllm-engine/src/sampler.rs`
  and add a GPT-2 BPE tokenizer wrapper (tokenizers crate) with special-token
  support from `03-special-tokens-function-calling.md`.

## Explicitly deferred

- Tensor parallelism: the reference uses `torch.distributed`/NCCL and a TP world
  size (e.g. TP=3 in `compare_nanogpt.py`). TP is GPU-only and out of scope for
  the CPU port. Weights load as replicated tensors (no sharding) in v1.
- CUDA graphs: `model_runner.py` captures CUDA graphs (`capture_cudagraph`);
  GPU-only, deferred behind the `cuda` feature.
- Triton kernels: GPU-only, deferred.
- Bit-exact SERVING verification (end-to-end latency/throughput vs nano-vllm) is
  deferred until both weights and (optionally) a GPU are available. The
  bit-exact MODEL verification (Phase B) is the hard gate and only needs CPU +
  weights.
