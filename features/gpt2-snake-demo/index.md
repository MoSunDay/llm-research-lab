Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# GPT-2 贪吃蛇端到端演示

## 能力概述

用 nanoGPT 在 GPU 上训练一个 char 级 GPT-2 small，导出为 safetensors（nanoGPT 原生 `transformer.*` 键），由 Rust `nvllm-models` 的 `Gpt2` 加载推理，驱动终端贪吃蛇。这是本工作台「训练 → 导出 → 推理」链路的落地样例。

## 触发方式

Python 用 `/root/ComfyUI/env/bin/python`。从 `nano-vllm-rs/` 运行：

```bash
# 1) 训练（GPU，产 out-gpt2-snake/ckpt.pt）
/root/ComfyUI/env/bin/python scripts/train_gpt2_snake.py --max_iters 700 --out_dir out-gpt2-snake

# 2) 导出（ckpt.pt → model.safetensors + config.json + meta.json）
/root/ComfyUI/env/bin/python scripts/export_safetensors.py --in_dir out-gpt2-snake --data_dir scripts/data-shakespeare-char

# 3) 数值对拍（产 parity.json / ref_*.npy，供 Rust 测试比）
/root/ComfyUI/env/bin/python scripts/compare_logits.py --in_dir out-gpt2-snake

# 4) Rust 跑贪吃蛇（真实大脑）
cargo run -p snake-gpt2 --release -- --model ../out-gpt2-snake

# 兜底（无需权重，沙箱可跑）
cargo run -p snake-gpt2 --release -- --random --steps 30
cargo run -p snake-gpt2 --release -- --heuristic --steps 30
```

模型规模固定为标准 GPT-2 small：`n_layer=12 n_head=12 n_embd=768`（~85M 参数），`block_size=256`，`vocab_size=65`，`bias=True`。

## 权重契约（模型目录 `--model <dir>` 内容）

| 文件 | 内容 |
|------|------|
| `model.safetensors` | fp32，仅 `transformer.*` 白名单键（融合 `c_attn`，**无** `lm_head.weight`，**无** `*.attn.bias`/`masked_bias`）。见 [权重键契约](../../agents/nvllm-models/index.md#权重键契约) |
| `config.json` | `{block_size, vocab_size, n_layer, n_head, n_embd}`（直接 serde 进 `Gpt2Config`） |
| `meta.json` | `{vocab_size, itos:[chars], stoi:{char:id}}`（char tokenizer 用） |

## 数值对拍（Rust vs PyTorch）

`compare_logits.py` 写 `parity.json`（末位 logits + tokens + action_ids）。Rust 侧 `gpt2_brain.rs` 的测试 `gpt2_logits_parity_vs_pytorch` 在环境变量 `SNAKE_GPT2_MODEL=<dir>` 时加载权重跑 forward 比对：

```bash
SNAKE_GPT2_MODEL=$(pwd)/../out-gpt2-snake cargo test -p snake-gpt2 -- --nocapture
```

- 断言：**动作 argmax 一致**（W/A/S/D 上各自选哪个），而非浮点 bit-exact
- 原因：Rust `gelu_erf()` vs nanoGPT tanh-GELU（[见 nvllm-models](../../agents/nvllm-models/index.md#bias-与-gelu)）。实测差 < 1e-4，可忽略
- 另断言 `max_abs < 5.0` 作 sanity 上限

## 关键状态

| 场景 | 触发 | 效果 |
|------|------|------|
| 真实大脑 | `--model <dir>` 且文件齐 | `Gpt2Brain` 每步 forward，UI 标 `gpt2=true` |
| 加载失败回退 | `--model` 指向坏目录 | 警告 + 退 `WallAvoider`，仍可玩 |
| 启发式 | `--heuristic`（或无任何标志默认） | `WallAvoider`，可玩 |
| 随机 | `--random` | 均匀随机非反向方向 |

## 约束

- **离线**：训练数据走 GitHub raw（tinyshakespeare），断点续传下载；tokenizer 用 char 级（无 HF/tiktoken）
- **语义随机**：基座 GPT-2 不懂棋，下棋接近随机撞墙——预期行为，非 bug
- **block_size**：prompt 超长会 panic，故 `build_prompt` 左截断保末尾（`MOVE:` 必留）
- 大文件（ckpt.pt ~1GB、model.safetensors ~340MB）已 `.gitignore`，**本地再生，勿提交**

## 错误面

- 训练 checkpoint 若 `bias=False` → 导出后 Rust loader 缺 `.bias` 键报错（解法：`bias=True` 重训）
- 喂错键格式（如顶层 `convert_*.py` 的 `model.*`）→ loader 找不到 `transformer.h.0.attn.c_attn.weight`
- 交互式跑需真 TTY；非 TTY 自动降级逐帧 ASCII

## 相关逻辑模块

- [GPT-2 Rust 移植](../../agents/nvllm-models/index.md)：`Gpt2::load` / `forward`，权重键契约
- [贪吃蛇 GPT-2 大脑](../../agents/snake-gpt2/index.md)：brain 接线 / 渲染 / 语义随机性
