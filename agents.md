Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# llm-research-lab 逻辑结构

## 概述

大模型推理与实验工作台。核心是一条「nanoGPT 训练 → 权重导出 → Rust 推理」的落地链路，外加 Python 参考引擎、顶层转换脚本与上游 vendored 仓库。Rust 侧 (`nano-vllm-rs/`) 是活跃开发区；Python 参考引擎 (`nano-vllm/`) 与 `link_repos/` 为只读对照。

## 模块索引

| 模块 | 核心路径 | 职责 |
|------|---------|------|
| [GPT-2 Rust 移植](agents/nvllm-models/index.md) | `nano-vllm-rs/crates/nvllm-models/src/gpt2.rs` | GPT-2 模型定义：Config / load / forward，对齐 nanoGPT 权重键名 |
| [贪吃蛇 GPT-2 大脑](agents/snake-gpt2/index.md) | `nano-vllm-rs/crates/snake-gpt2/` | 终端贪吃蛇：游戏模型 + GPT-2 决策大脑 + 渲染。现含**两种模式**：① 约束解码（constrained decoding）；② Agent loop + SFT（生成 → 解析 bash → 沙箱执行 → 注入 result → 循环至 `</answer>`） |
| Agent SFT 微调 | `nano-vllm-rs/scripts/` (agent_*.py) + `crates/snake-gpt2/src/{agent,sandbox,runner}.rs` | char-level vocab=73 + 8控制token，合成agent轨迹，loss-mask SFT，Rust agent loop（生成→解析bash→沙箱执行→注入result→循环至`</answer>`）|
| 推理引擎（占位） | `nano-vllm-rs/crates/nvllm-engine/src/lib.rs` | 仅 `version()`，调度器尚未实现 |
| 训练/导出脚本 | `nano-vllm-rs/scripts/` | train_gpt2_snake / export_safetensors / compare_logits；agent 链路另含 `agent_special_tokens.py` / `prepare_agent_snake.py` / `sft_train_gpt2_snake.py` / `export_agent_safetensors.py` |
| Python 参考引擎 | `nano-vllm/` | nano-vllm 上游 Python 实现，只读对照（含 example_agentic.py） |
| 顶层转换脚本 | `convert_gpt2.py`, `convert_nanogpt.py` | HF/nanoGPT → 自定义格式（`model.*` 键，与 Rust loader **不兼容**，见下） |
| vendored 仓库 | `link_repos/nanoGPT`, `link_repos/nano-vllm` | 上游只读快照 |

## 依赖关系

```
           train_gpt2_snake.py  (torch/CUDA)
                    │ ckpt.pt
                    ▼
           export_safetensors.py
                    │ model.safetensors (transformer.*)
                    ▼
   ┌────────────────────────────────────┐
   │  snake-gpt2 (bin)                  │
   │   └─ gpt2_brain ──► nvllm-models   │  (VarBuilder mmap + Gpt2::forward)
   │   └─ game / prompt / render        │
   └────────────────────────────────────┘
```

snake-gpt2 直接消费 `nvllm-models` 的 `Gpt2`；两者是 `nano-vllm-rs` workspace 的仅两个实质 crate。顶层 `convert_*.py` 产出的 `model.*`（拆分 q/k/v）键文件**不能**喂给 Rust `Gpt2::load`（它要 nanoGPT 原生 `transformer.*` 融合 `c_attn`）——演示链路用 `export_safetensors.py` 而非顶层脚本。

### Agent SFT 链路

char 级 vocab=73 + 8 控制token 的 agent 微调支线，从已训练的 shakespeare ckpt 暖启动，产出可在 Rust 端跑 agent loop 的权重：

```
train_gpt2_snake.py → ckpt.pt (shakespeare, vocab=65)
                    │ warm-start
prepare_agent_snake.py → data-agent/ (840 traces, masked)
                    │
sft_train_gpt2_snake.py → out-agent-snake/ckpt.pt (vocab=73)
                    │
export_agent_safetensors.py → model.safetensors (transformer.*, vocab=73)
                    │
snake-gpt2 --agent → agent_loop (generate → bash → inject → repeat)
```

四个脚本的职责：

- `agent_special_tokens.py` — vocab 契约（73 tokens），longest-match encode/decode，build_trace with mask
- `prepare_agent_snake.py` — 生成 840 条轨迹（800 game + 40 code），loss-masked
- `sft_train_gpt2_snake.py` — masked-loss SFT trainer（val_loss 5.08→0.066）
- `export_agent_safetensors.py` — checkpoint → safetensors（vocab=73，148 tensors）

## 关键环境约束（稳定事实）

- **离线**：HuggingFace 网络被阻断，`/root/models/gpt2-hf` 不存在，tiktoken 未装 → **GPT-2 BPE tokenizer 不可用**。可达源仅 GitHub raw + crates.io 离线缓存。
- **Python**：`/root/ComfyUI/env/bin/python`（3.10.12，torch 2.7+cu126，transformers 4.51.3，safetensors 0.5.3，numpy 2.2.5）。
- **GPU**：RTX 2080 Ti，CUDA 12.x，GPU matmul 正常。
- **Rust**：workspace pin candle-core/candle-nn `0.11`；离线缓存可用 `crossterm 0.27` / `clap 4.6` / `rand 0.8`（勿改版本）。
- **正因 BPE 不可用**，snake demo 用 char 级（vocab=65）规避；详见 [snake-gpt2](agents/snake-gpt2/index.md)。
- **Agent 模型 vocab=73**：65 base chars + 8 控制token（`<bash>`、`</bash>`、`<pause>`、`<result>`、`</result>`、`<resume>`、`<answer>`、`</answer>`）。BPE 仍不可用，但 char 级 + special-token 方案绕过了这一限制，可承载 agent loop 所需的结构化生成。

## 业务能力

参见 [能力索引](features/index.md)。
