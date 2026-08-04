Commit: fa1729a

# 重写 README（英文）+ 新增中文版 README.zh-CN.md

## 背景
旧 `README.md` 只描述 Python `nano-vllm/` 引擎（V100 硬件、xgrammar phases），对整个 Rust 工作区（`nano-vllm-rs/`：GPT-2 移植、贪吃蛇、Agent SFT 链路）只字未提。仓库已无中文 README。

## 变更

### README 重写
- **`README.md`**：完全重写（92% 变更）。新增双轨制概览（Rust 活跃 + Python 参考）、架构流程图、三个核心演示（贪吃蛇约束解码 / Agent SFT 循环 / Python 引擎）、完整仓库结构、快速开始、环境要求、已知限制、文档索引。顶部加语言切换链接。
- **`README.zh-CN.md`**：新增中文版（188 行），忠实对应英文版内容，顶部互链。

## 测试覆盖

本变更仅涉及文档，不影响任何编译/测试目标。

- 全量回归：`cargo test --workspace` → 53 passed, 0 failed
- clippy：`cargo clippy --workspace --all-targets -- -D warnings` → 零警告
- build：`cargo build --workspace` → 编译干净
- 行数：`README.md` 191 行 ≤ 800；`README.zh-CN.md` 188 行 ≤ 800

## Impact Surface
- 对用户/访问者：仓库首页信息从过时的 Python-only 描述更新为双轨全景。
- 不影响：CLI / Rust 代码 / Python 引擎 / session / store / 任何运行时行为。

## Related Docs
- [agents/](../../agents/) — 各模块详细笔记
- [features/index.md](../../index.md) — 能力索引
- [既有 changelog: snake-gpt2-demo](../2026-07-26/snake-gpt2-demo.md)
