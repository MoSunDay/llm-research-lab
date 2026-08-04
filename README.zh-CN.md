# llm-research-lab

> 大模型推理与实验工作台：一条完整的 **nanoGPT 训练 → 权重导出 → Rust 推理**
> 落地链路，辅以 Python 参考引擎做交叉验证。

[English](README.md) · [简体中文](README.zh-CN.md)

---

## 这是什么？

本仓库通过两条并行路线探索 LLM 推理工程：

| 路线 | 路径 | 状态 |
|------|------|------|
| **Rust 工作区** | [`nano-vllm-rs/`](nano-vllm-rs/) | 🟢 活跃开发 |
| **Python 参考引擎** | [`nano-vllm/`](nano-vllm/) | 📖 只读对照 |

Rust 路线是当前重心：将 GPT-2 模型移植到
[candle](https://github.com/huggingface/candle) 0.11，用 nanoGPT 训练，
导出为 safetensors，再由终端贪吃蛇游戏消费——模型的决策可以通过
**约束解码**或完整的 **Agent 循环**（生成 → 执行 bash → 注入结果 → 循环）实现。

Python 路线（`nano-vllm/`）是一个从零实现的推理引擎，提供约束解码、
Agent 暂停/恢复、SFT 全链路等能力，作为 Rust 工作的设计参考与正确性对照。

## 架构

```
                    ┌─── Rust 路线（活跃） ───────────────────────┐
                    │                                              │
  train_gpt2_snake.py ──► ckpt.pt                                 │
       (nanoGPT / torch)      │                                   │
                              ▼                                   │
  export_safetensors.py ──► model.safetensors (transformer.*)     │
                              │                                   │
               ┌──────────────┼──────────────┐                   │
               ▼              ▼              ▼                    │
         nvllm-models    snake-gpt2     nvllm-engine              │
         (GPT-2 前向)   (游戏+大脑)    (调度器，开发中)            │
                    │                                              │
                    └──────────────────────────────────────────────┘

                    ┌─── Python 路线（参考） ─────────────────────┐
                    │                                              │
  nano-vllm/ ── 引擎 (调度器, block_manager, 采样器)              │
     ├── grammar.py        约束解码 (xgrammar)                    │
     ├── engine/           暂停 / 恢复 / Agent 生成               │
     └── sft_train.py      nanoGPT SFT → 逆向转换 → 校验          │
                    │                                              │
                    └──────────────────────────────────────────────┘
```

## 核心演示

### 1 · GPT-2 贪吃蛇（Rust，约束解码）

在 Tiny Shakespeare 字符级语料上训练 GPT-2，导出 safetensors，
让模型通过约束（贪心）解码来下终端贪吃蛇。

```
train_gpt2_snake.py → ckpt.pt → export_safetensors.py → model.safetensors
                                                           │
                        snake-gpt2（约束解码模式）◄─────────┘
```

- **词表**：65 个字符（Tiny Shakespeare），`block_size=256`，`bias=True`
- **流程**：nanoGPT 训练 → safetensors 导出 → Rust candle 前向推理
- **对齐**：`compare_logits.py` 导出 CPU 参考 logits 供 Rust 端逐值比对

### 2 · GPT-2 Agent 微调（Rust，Agent 循环）

从 Shakespeare checkpoint 暖启动，在合成 Agent 轨迹（游戏决策 + 代码任务）
上微调，然后在 Rust 中运行 Agent 循环：

```
sft_train_gpt2_snake.py → ckpt.pt (vocab=73) → export_agent_safetensors.py
                                                        │
                    snake-gpt2 --agent（生成 → bash → 注入 → 循环）◄──┘
```

- **词表**：73 tokens = 65 基础字符 + 8 控制token
  （`<bash>`、`</bash>`、`<pause>`、`<result>`、`</result>`、
  `<resume>`、`<answer>`、`</answer>`）
- **训练**：840 条 loss-masked 轨迹，val_loss 5.08 → 0.066
- **Agent 循环**：生成 `<bash>` 命令 → 沙箱执行 → 结果作为 `<result>` 注入 →
  循环直到 `</answer>`

### 3 · Python nano-vllm 引擎（参考）

| 阶段 | 能力 |
|------|------|
| Phase 3 | 基于 xgrammar 的引导解码（JSON schema、正则、语法、选择题） |
| Phase 4 | Agent 暂停/恢复（`PAUSED` 状态、KV-cache 修复、V100 attention 重写） |
| Phase 5 | SFT 全链路：nanoGPT 微调 → 逆向转换 → bit-exact 对齐验证 |

## 仓库结构

```
nano-vllm-rs/                  Rust 工作区（活跃开发）
  crates/
    nvllm-models/              GPT-2 模型定义 (candle 0.11)
    snake-gpt2/                终端贪吃蛇 + GPT-2 大脑 + Agent 循环
    nvllm-engine/              推理调度器（占位）
  scripts/
    train_gpt2_snake.py        nanoGPT 训练 → ckpt.pt
    export_safetensors.py      ckpt → safetensors (Rust 格式)
    compare_logits.py          Rust-vs-PyTorch 对齐
    agent_special_tokens.py    vocab=73 契约 + 编解码
    prepare_agent_snake.py     生成 840 条 Agent 轨迹
    sft_train_gpt2_snake.py    masked-loss SFT 训练器
    export_agent_safetensors.py ckpt → safetensors (vocab=73)
nano-vllm/                     Python 参考引擎（只读）
  nanovllm/                    核心包 (engine, layers, grammar)
  test_*.py                    各阶段测试
  example*.py                  可运行示例
  sft_train.py                 nanoGPT SFT 驱动
convert_gpt2.py                HF GPT-2 → nano-vllm 权重 (Python 路线)
convert_nanogpt.py             nanoGPT → nano-vllm 权重 (Python 路线)
verify_gpt2.py                 转换正确性检查
compare_nanogpt.py             nanoGPT-vs-nano-vllm 贪心对齐
BENCH.md                       Python 引擎基准测试
```

## 快速开始

### Rust 贪吃蛇演示

```bash
# 1 · 训练（需要 torch + CUDA）
PY=/root/ComfyUI/env/bin/python
$PY nano-vllm-rs/scripts/train_gpt2_snake.py \
    --out_dir out-gpt2-snake --max_iters 3000

# 2 · 导出为 Rust 格式
$PY nano-vllm-rs/scripts/export_safetensors.py \
    --in_dir out-gpt2-snake \
    --data_dir nano-vllm-rs/scripts/data-shakespeare-char

# 3 · 运行
cd nano-vllm-rs && cargo run -p snake-gpt2 --release -- \
    --weights ../out-gpt2-snake
```

### Python 引擎

```bash
cd nano-vllm
python example.py           # 基础 GPT-2 演示
python example_grammar.py   # 约束解码 (Phase 3)
python example_agentic.py   # 暂停/恢复工具循环 (Phase 4)
python example_sft.py       # SFT 模型演示 (Phase 5)
```

## 环境要求

| 组件 | 规格 |
|------|------|
| **GPU** | RTX 2080 Ti（Rust 开发）/ 3× Tesla V100-32GB（Python 引擎） |
| **CUDA** | 12.x |
| **Python** | 3.10.12, torch 2.7+cu126, transformers 4.51.3, safetensors 0.5.3 |
| **Rust** | edition 2021, candle-core/nn 0.11, crossterm 0.27, clap 4.6 |
| **离线** | HuggingFace 网络被阻断；GPT-2 BPE 不可用 → 使用字符级词表 |

> Rust 演示之所以使用**字符级分词**（vocab 65/73），正是因为 BPE 分词器
> （`tiktoken`）在离线环境下无法安装。

## 已知限制

- **Rust 推理仅支持 CPU**（约 7 秒/次前向）：无 KV cache，无 GPU 加速。
- **BPE 分词器离线不可用**；两个演示均使用字符级词表。
- **`nvllm-engine` 调度器仅为占位**——目前只实现了 `version()`。
- **顶层 `convert_*.py`** 产出的 `model.*` 键与 Rust `Gpt2::load`
  **不兼容**（后者需要 nanoGPT 原生 `transformer.*` 融合 `c_attn` 键）。
  Rust 路线请使用 `export_safetensors.py`。
- 权重和 checkpoint 已被 gitignore。

## 文档

- [Agent 记忆文档](agents/) — 各模块详细笔记
- [功能文档](features/) — 能力索引 + 变更日志
- [基准测试](BENCH.md) — Python 引擎性能数据
- [Rust 脚本说明](nano-vllm-rs/scripts/README.md) — 权重训练/导出流程

## 许可

研究 / 教学项目。上游许可证详见
[`nano-vllm/`](nano-vllm/) 和 [`link_repos/`](link_repos/)。
