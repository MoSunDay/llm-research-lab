Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# 贪吃蛇 GPT-2 大脑：训练→导出→推理端到端打通

## Context

此前 `nvllm-models` 的 GPT-2 仅有结构性单测（随机权重），无端到端真实推理验证；workspace 也无可运行二进制。本任务补全完整落地链路：训练一个 char 级 GPT-2 small，导出 Rust 可加载的权重，并写一个用 GPT-2 当决策大脑的终端贪吃蛇作为「活体推理样例」，验证整条接线无误。

## Change Summary

- 新增 Python 训练/导出/对拍脚本（自包含，仅依赖 ComfyUI 环境 + GitHub raw）
- 新增 Rust crate `snake-gpt2`（游戏模型 + 三种大脑 + crossterm 渲染）
- 用 GPU 训练 char 级 GPT-2 small（iter 500，val_loss 4.25→1.75）并导出 safetensors
- Rust 与 PyTorch 数值对拍通过（动作 argmax 一致）

## Impact Surface

| 路径 | 状态 |
|------|------|
| `nano-vllm-rs/scripts/train_gpt2_snake.py` | 新增（317 行） |
| `nano-vllm-rs/scripts/export_safetensors.py` | 新增（100 行） |
| `nano-vllm-rs/scripts/compare_logits.py` | 新增（151 行） |
| `nano-vllm-rs/scripts/README.md` | 新增 |
| `nano-vllm-rs/crates/snake-gpt2/`（7 文件 + Cargo.toml） | 新增 |
| `nano-vllm-rs/Cargo.toml` | members 加 `crates/snake-gpt2` |
| `agents.md`, `agents/`, `features/` | 新增（本仓库本地记忆脚手架） |
| `.gitignore` | 追加 `out-gpt2-snake/`、`input.txt`、`__pycache__/` |

> 注：上述代码改动**尚未提交**，`Commit:` 指向改动前的 HEAD `336ecd7b`；提交后可 touch 修复本 hash。

## 关键决策

| 决策 | 理由 |
|------|------|
| char 级（vocab=65）而非 GPT-2 BPE | 沙箱 HuggingFace 不可达、无 BPE 文件、无 tiktoken；char 级仅需 GitHub raw，且无需 `tokenizers` crate |
| 标准 GPT-2 small（12/768/12） | 用户选定；与「GPT-2」语义贴合 |
| `bias=True` 强制 | Rust `candle_nn::linear`/`layer_norm` 加载 `.bias`，缺键即报错 |
| 导出保留 `transformer.*` 融合 `c_attn` | 与 Rust `Gpt2::load` 键名对齐；弃用顶层 `convert_*.py`（它产 `model.*` 拆分键，不兼容） |
| 丢 `lm_head.weight` 与 mask buffer | lm_head 绑定 wte 不读；mask buffer Rust 自建 |
| trait 用 `&mut dyn RngCore` | rand 0.8 的 `Rng` 非 dyn 安全 |
| 对拍比「动作 argmax」不比浮点 | Rust erf-GELU vs nanoGPT tanh-GELU 非同一函数 |

## 验证

- `cargo test --workspace`：nvllm-models 3 + snake-gpt2 23（含 parity），全绿
- `cargo clippy --workspace -D warnings`：零告警
- `cargo run -p snake-gpt2 --model out-gpt2-snake`：真实路径跑通，每步出合法方向，不 panic

## Related Docs

- [演示能力](../../gpt2-snake-demo/index.md)
- [贪吃蛇 GPT-2 大脑](../../../agents/snake-gpt2/index.md)
- [GPT-2 Rust 移植](../../../agents/nvllm-models/index.md)
