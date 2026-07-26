Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# GPT-2 Agent 微调与 Rust Agent Loop

## 概述

Upgrade GPT-2 from constrained-decoding into a fine-tuned agent: char-level vocab + 8 control tokens, synthetic agent traces (bash tool-use for snake game + code writing), loss-mask SFT fine-tuning, and a complete Rust agent loop.

## 词表契约 (vocab=73)

- 65 tinyshakespeare chars (IDs 0-64) + 8 control tokens (IDs 65-72)
- Control tokens: `<bash>`, `</bash>`, `<pause>`, `<result>`, `</result>`, `<resume>`, `<answer>`, `</answer>`
- Multi-char tokens via longest-match encode
- Source of truth: `scripts/agent_special_tokens.py`

## 训练链路

```
out-gpt2-snake/ckpt.pt (shakespeare, vocab=65)
    │ warm-start (wte/wpe reinit)
    ▼
prepare_agent_snake.py → data-agent/ (840 traces: 800 game + 40 code)
    │ loss-masked (9.4% supervised)
    ▼
sft_train_gpt2_snake.py → out-agent-snake/ckpt.pt (2500 iters, val_loss 5.08→0.066)
    │
    ▼
export_agent_safetensors.py → out-agent-snake-export/ (148 tensors, vocab=73)
```

## Rust Agent Loop

The `--agent` CLI subcommand runs the full loop:

1. Encode prompt → token IDs
2. Generate autoregressively until `<pause>` or `</answer>`
3. On `<pause>`: parse `<bash>...</bash>`, execute via sandbox
4. Inject `<result>{output}</result><resume>`
5. Repeat until `</answer>` or max_turns

Key files: `agent.rs` (core loop), `sandbox.rs` (bash executor), `runner.rs` (model wiring), `char_tokenizer.rs` (multi-char tokenizer)

## 验证结果

- **44/44 Rust tests pass** (including live e2e with real 85M model)
- **Game task**: correct moves (U/D/L/R), clean `<bash>echo X</bash><pause>` format, proper termination
- **Code task**: format recognized (`<bash>`, EOF heredoc, `<answer>done</answer>`) but content noisy (only 40 training traces)
- **Forward parity**: candle CPU logits match torch CPU to 4 decimal places

## 已知限制

- Rust candle-core: CPU only, no KV cache → ~7s per forward pass for 85M model
- Code task: needs more training data for full code reproduction
- Prompt format sensitivity: trailing `\n` after `MOVE:` is critical (bash `$()` strips trailing newlines)

## 相关文档

- [贪吃蛇 GPT-2 大脑](../../agents/snake-gpt2/index.md) — full module details
- [GPT-2 Rust 移植](../../agents/nvllm-models/index.md) — model forward implementation
- [GPT-2 贪吃蛇演示](../gpt2-snake-demo/index.md) — original constrained-decoding demo
