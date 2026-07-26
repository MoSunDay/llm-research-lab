Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# llm-research-lab 能力索引

## 主要能力

| 能力 | 说明 | 详情 |
|------|------|------|
| GPT-2 贪吃蛇端到端演示 | nanoGPT 训练 → safetensors 导出 → Rust GPT-2 推理 → 终端贪吃蛇下棋 | [演示](gpt2-snake-demo/index.md) |
| GPT-2 前向推理（Rust） | candle 0.11 实现，加载 nanoGPT 原生权重键 | [GPT-2 Rust 移植](../agents/nvllm-models/index.md) |
| 权重训练/导出（Python） | train_gpt2_snake / export_safetensors / compare_logits | 同 [演示](gpt2-snake-demo/index.md) 训练链路 |
| GPT-2 Agent 微调（SFT） | char-level vocab=73 + 8控制token，合成agent轨迹，masked-loss SFT，Rust agent loop（生成→bash→注入→循环）| [Agent SFT](gpt2-agent-sft/index.md) |

## 变更日志

参见 [changelog/](changelog/)。
