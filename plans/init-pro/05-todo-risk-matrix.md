# 05 -- TODO List, Risk Matrix, and Milestones

> Reference source: link_repos/nanoGPT/ and link_repos/nano-vllm/ (READ-ONLY reference -- for understanding only; never modified).

## TODO list (T0..T11)

Statuses: todo = ready to start; blocked = needs an external prerequisite.

- T0  Land these six planning docs. status: DONE.
      files: plans/init-pro/00..05.
- T1  Cargo workspace scaffolds and builds with candle. status: DONE.
      `cargo build` + `cargo test` + `cargo clippy` green (candle-core/nn 0.11).
      files: nano-vllm-rs/Cargo.toml, crates/*/Cargo.toml, crates/*/src/lib.rs.
- T2  GPT-2 Config + structs. status: DONE.
      Gpt2Config (defaults = GPT-2 small) + Block/Attn/Mlp/Gpt2 structs.
      files: crates/nvllm-models/src/gpt2.rs (config inline, not a models/ subdir).
- T3  Forward pass, structural implementation (erf gelu, causal mask, pre-norm
      blocks, tied lm_head). status: DONE.
      erf via Tensor::gelu_erf() (NOT tanh gelu); causal mask via U8 where_cond;
      lm_head tied to wte via 2D matmul.
      files: crates/nvllm-models/src/gpt2.rs.
- T4  Safetensors loader from convert_nanogpt.py output (key map in doc 02).
      status: todo.
      files: crates/nvllm-models/src/loader.rs.
- T5  CPU smoke test: forward runs, output shapes correct, on random weights.
      status: DONE.
      forward_shape_matches + causal_mask_is_triangular both pass.
      files: crates/nvllm-models/src/gpt2.rs (#[cfg(test)] mod tests).
- T6  Numerical bit-exact verify vs nanoGPT (>= 257 greedy tokens, 100% match).
      status: BLOCKED on weights (/root/models absent).
      files: crates/nvllm-models/tests/bitexact.rs, top-level compare script.
- T7  Special-token id assignment + tokenizer integration (Python added_tokens
      + Rust tokenizers crate). status: todo.
      files: nanoGPT/data/agent/prepare.py, crates/nvllm-engine/src/tokenizer.rs.
- T8  Agent prepare.py + loss-masked data (train.bin/val.bin/mask.bin/meta.pkl).
      status: todo (depends on T7 for token ids).
      files: nanoGPT/data/agent/prepare.py.
- T9  Bash sandbox subprocess (allowlist, argv, timeout, cwd, output cap).
      status: todo.
      files: crates/nvllm-engine/src/sandbox.rs.
- T10 Pause/resume control loop. status: todo (depends on T3, T9).
      files: crates/nvllm-engine/src/agent.rs, crates/nvllm-serving/src/main.rs.
- T11 Greedy / top-k / top-p sampler in Rust. status: todo.
      files: crates/nvllm-engine/src/sampler.rs.

## Risk matrix

Likelihood and Impact are qualitative (Low / Med / High).

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| R-WGT No model weights present (`/root/models` absent) | High | High | Blocks T6/T4-real. Proceed on random weights (T5) for shapes; gate numerics on weights. Provision HF GPT-2 weights and run convert_nanogpt.py. |
| R-DRIFT candle API drift vs 0.11.0 | Med | Med | Pin `candle-core/candle-nn = "0.11"` (already pinned). Wrap candle calls in thin modules so API changes are localized. |
| R-FP fp16 vs fp32 numeric divergence | Med | Med | erf GELU exact, fp32 LayerNorm stats, identical argmax protocol. Investigate first-mismatch logits before accepting any divergence. |
| R-GPU No GPU; serving perf unverifiable | High | Med | CPU is sufficient for the bit-exact model gate. Defer TP, CUDA graphs, triton kernels (Phase D+). |
| R-SANDBOX Bash sandbox escape | Med | High | Default-deny allowlist, argv form (no shell), cwd confinement, timeout+kill, output cap, cleared env. NO container in v1; v2 adds container/seccomp. |
| R-TOK Tokenizer special-token merge | Low | High | added_tokens with special=True on both Python and Rust sides; symmetric encode/decode; assert ids match. |
| R-SIZE New files exceed 400 lines | Med | Low | Split by responsibility (layers/, models/, engine modules). Check with wc -l per file. |
| R-KV Rust borrow/lifetime complexity in paged KV cache | Med | Med | Model KV blocks as owned Tensor storage indexed by block id; keep the scheduler owning all mutation. Phase C. |

## Milestones (Phase A..D, from doc 01)

- Phase A -- structural GPT-2 forward (CPU, random weights, shape test).
  Covers T1, T2, T3, T5. Exit criterion: forward runs, output shapes correct.
- Phase B -- real weights + numerical bit-exact verify.
  Covers T4, T6. Exit criterion: 100% greedy token match vs nanoGPT over >= 257
  tokens. BLOCKED until weights are present (R-WGT).
- Phase C -- paged KV cache + continuous-batch scheduler.
  Covers block_manager/scheduler/sequence/model_runner ports (R-KV).
  Exit criterion: decode across a block boundary stays correct (>= 257 tokens).
- Phase D -- sampler + tokenizer + agent loop.
  Covers T7, T8, T9, T10, T11. Exit criterion: a prompt drives a full
  call -> sandbox -> result -> answer round trip.

Phases are sequential at the milestone level, but early tasks (T1..T5, T7, T9,
T11) can start in parallel since they do not depend on weights.
