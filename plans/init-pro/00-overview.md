# Plan: nano-vllm-rs -- Rust Port of nano-vllm with GPT-2 and Function Calling

> Reference source: link_repos/nanoGPT/ and link_repos/nano-vllm/ (READ-ONLY reference -- for understanding only; never modified).

## Goals

Three goals, listed in dependency order so each unblocks the next.

1. Rust serving engine. Rewrite the nano-vllm Python serving logic in Rust using
   candle (huggingface/candle) for tensors and nn. CPU-verifiable by default;
   CUDA is kept behind a cargo feature and is not required for the port.
2. GPT-2 in Rust, bit-exact vs nanoGPT. Implement the GPT-2 architecture exactly
   as in `link_repos/nanoGPT/model.py`, load weights produced by the existing
   `convert_nanogpt.py` (which writes `model.safetensors` + `config.json`), and
   verify greedy decoding matches nanoGPT token-for-token.
3. Function calling + bash tool training. Reserve the free token-id slots
   50257..50303 for control/special tokens, generate loss-masked agent traces,
   and run a pause/resume agent loop: the model emits a `<bash>` call, a sandbox
   runs it, a `<result>` is injected back, and generation resumes to a final
   answer.

## Repository map

    /root/llm-research-lab/
    |-- link_repos/                 READ-ONLY references (never modified)
    |   |-- nanoGPT/                model.py (GPT-2 arch), train.py
    |   `-- nano-vllm/              engine/ layers/ utils/ (serving reference)
    |-- convert_nanogpt.py          nanoGPT ckpt.pt -> nano-vllm safetensors
    |-- convert_gpt2.py             HF GPT-2 -> nano-vllm (Conv1D transpose)
    |-- compare_nanogpt.py          nanoGPT-vs-nano-vllm greedy verify (960/960)
    |-- verify_gpt2.py              GPT-2 numeric checks
    |-- nanoGPT/                    working copy: data/sft/prepare.py (+ planned data/agent/)
    |-- nano-vllm/                  Python working copy (nanovllm/models/gpt2.py)
    |-- nano-vllm-rs/               NEW Rust Cargo workspace (this plan)
    |   |-- Cargo.toml              workspace, candle-core/candle-nn 0.11, safetensors 0.4
    |   `-- crates/
    |       |-- nvllm-models/       model defs (gpt2.rs); src/ currently empty
    |       `-- nvllm-engine/       scheduler/sequence/sampler; src/ currently empty
    `-- plans/
        `-- init-pro/               these six documents

Note: the `nano-vllm-rs/` workspace already exists with the two member crates
above (both have empty `src/`). A third crate, `nvllm-serving` (binary/cli), is
proposed in `01-rust-engine.md`.

## Discipline statement

- `link_repos/*` is READ-ONLY. It exists only to understand the reference design.
  Every technical claim about reference code in these docs is grounded by reading
  the cited file; no `link_repos` file is ever modified.
- All integration lands at the top level. The Python `nano-vllm/` and the Rust
  `nano-vllm-rs/` coexist so the same weights can be served by both and the two
  outputs can be cross-checked (this is the whole point of `compare_nanogpt.py`,
  which already reports 960/960 greedy agreement between nanoGPT and nano-vllm).
- New files are kept under 400 lines; iterating files under 800. Modules are
  split by responsibility. Pure functional style in Rust: structs that hold
  `VarBuilder` tensors plus free `forward` functions; no classes / no OOP prose.

## How the three pieces connect

1. Weights flow: a nanoGPT checkpoint (`ckpt.pt`) is converted by
   `convert_nanogpt.py` into `model.safetensors` + `config.json` + tokenizer
   files. The Rust GPT-2 (in `nvllm-models`) reads that safetensors file. The
   converter already splits the fused `c_attn` into q/k/v, uses nn.Linear layout
   (no Conv1D transpose), pads to 50304 rows for matmul efficiency while writing
   `vocab_size: 50257`, and skips `lm_head` because it is tied to `wte`.
2. Tokens: special/control tokens occupy the otherwise-unused slots 50257..50303
   (47 slots). Because nanoGPT pads the embedding to 50304 (the next multiple of
   64 past 50257), those rows already exist and GPT-2 BPE never emits them.
3. Training: `nanoGPT/data/agent/prepare.py` (to be written, modeled on the
   existing `nanoGPT/data/sft/prepare.py`) emits `train.bin`/`val.bin` (uint16)
   plus `meta.pkl`, and is a drop-in for `nanoGPT/train.py`. Its vocab_size is
   50304 so the special-token ids are representable in uint16 and valid rows.

## Current environment status

- Toolchain present: `rustc`/`cargo` 1.96.1; the workspace pins
  `rust-version = "1.96"`. `candle-core`/`candle-nn` 0.11 and `safetensors` 0.4
  are declared as workspace deps and are resolvable.
- Python present: `/usr/bin/python3` has torch 2.4.0+cu121, used by the existing
  convert/compare/prepare scripts.
- Weights ABSENT: `/root/models` does not currently exist on this host. This
  blocks both the safetensors load and the numerical bit-exact verification
  (which need real GPT-2 weights). Structural work (shapes, forward path) can
  proceed on random/zero weights; the numeric gate is marked BLOCKED until
  weights are provisioned. See `05-todo-risk-matrix.md`.

## Document index

- `01-rust-engine.md` -- Cargo workspace design, link_repos module mapping,
  candle rationale, scope phasing (Phase A..D).
- `02-gpt2-rust-port.md` -- architecture checklist, numeric-exactness rules,
  weight-key loading plan, verification protocol, struct sketch.
- `03-special-tokens-function-calling.md` -- 47-slot token budget, special-token
  table, call serialization format, tokenizer integration.
- `04-agent-bash-training.md` -- agent `prepare.py`, pause/resume loop, bash
  sandbox design, loss mask.
- `05-todo-risk-matrix.md` -- T0..T11 task list, risk matrix, phase milestones.
