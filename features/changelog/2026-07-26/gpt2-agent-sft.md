Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# GPT-2 Agent 微调 + Rust Agent Loop（bash 工具调用 + loss-mask SFT）

## Context

此前 `snake-gpt2` 只有 constrained-decoding 模式（无微调、无工具调用），GPT-2 仅做方向 argmax。
本任务把 GPT-2 升级为真正的 agent：char 级词表 + 8 个控制 token、合成 agent 轨迹、loss-mask
SFT 微调，并在 Rust 侧实现完整 agent loop（生成 → 解析 `<bash>` → 沙箱执行 → 注入
`<result>` → 循环至 `</answer>`）。本文档补全 `gpt2-agent-sft` 能力的 changelog；snake demo
部分见同目录 `snake-gpt2-demo.md`。

## 变更

### 词表契约 (vocab=73)
- **`scripts/agent_special_tokens.py`**：65 tinyshakespeare chars (ID 0-64) + 8 控制token
  (ID 65-72: `<bash>`/`</bash>`/`<pause>`/`<result>`/`</result>`/`<resume>`/`<answer>`/`</answer>`)。
  longest-match encode/decode + `build_trace()` 带损失掩码。

### 数据生成 / 训练 / 导出
- **`scripts/prepare_agent_snake.py`**：从 shakespeare ckpt 暖启动生成 840 条轨迹
  （800 game + 40 code），仅 ~9.4% token 参与 loss（mask 监督）。
- **`scripts/sft_train_gpt2_snake.py`**：masked-loss SFT trainer，2500 iters，
  val_loss 5.08 → 0.066。
- **`scripts/export_agent_safetensors.py`**：ckpt → safetensors（vocab=73，148 tensors）。

### Rust Agent Loop
- **`crates/snake-gpt2/src/agent.rs`**：核心 `agent_loop<S,E>`，对 `step`（logits）与 `exec`
  （命令→stdout）闭包泛型，无需真实权重/Shell 即可单测。`SpecialIds`（8 控制token id）+
  `SPECIAL_LITERALS`（与 Python `SPECIALS` 字节一致）。
- **`crates/snake-gpt2/src/sandbox.rs`**：`run_bash(cmd, dir, timeout)` 执行器 + 危险命令
  deny-list（`rm -rf`/`sudo`/重定向到 `/etc` 等），超时 kill。
- **`crates/snake-gpt2/src/runner.rs`**：把真实 candle `Gpt2` + `sandbox::run_bash` 接到
  `agent_loop`（`run_with_model`）。
- **`crates/snake-gpt2/src/char_tokenizer.rs`**：多字符控制 token 的 longest-match
  encode（`specials` 降序匹配），`meta.json` 反序列化重建 stoi。
- **`crates/snake-gpt2/src/game_agent.rs`**：闭环游戏 agent（模型实际下 UDLR 玩蛇）。
- **`crates/snake-gpt2/src/main.rs`**：`--agent` / `--agent-workdir` / `--agent-max-turns`
  / `--agent-max-new` / `--temp` CLI 子命令。

### 质量门禁修复（本次提交随附）
- clippy `--all-targets -D warnings` 全绿（修复 dead-code / `too_many_arguments` /
  `unnecessary_sort_by`；`sandbox.rs` 一处 `|| true` 空断言 bug 改为真实断言）。
- `.gitignore`：补 `out-agent-snake-export/`、`scripts/out-agent-html-export/`
  （与既有 `out-gpt2-snake/` 一致，忽略可重生成的导出产物 config/meta.json）。

## 测试覆盖

| 功能 | 测试名 | 文件 |
|------|--------|------|
| agent loop 多轮注入终止 | `agent_loop_two_turns_injects_and_terminates` | agent.rs |
| `<pause>` 无 bash 继续 | `agent_loop_pause_without_bash_continues` | agent.rs |
| max_new 终止 | `agent_loop_max_new_terminates` | agent.rs |
| bash 解析（多行/取最后） | `parse_last_bash_*` | agent.rs |
| 危险命令拒绝且不执行 | `run_refuses_dangerous_without_executing` | sandbox.rs |
| deny-list 命中破坏性命令 | `deny_list_matches_destructive` | sandbox.rs |
| 沙箱 stdout 捕获/文件写入 | `run_echo_captures_stdout` / `run_writes_file_in_sandbox` | sandbox.rs |
| 多字符 token longest-match | `encode_longest_match_specials` / `answer_close_vs_answer_distinct` | char_tokenizer.rs |
| 真实模型闭环（可选） | `agent_loop_live_when_model_present` | runner.rs |

- 全量回归：`cargo test --workspace` → **56 passed, 0 failed**（nvllm-models 3 + snake-gpt2 53）
- clippy：`cargo clippy --workspace --all-targets -- -D warnings` → 零告警
- build：`cargo build --workspace` → 干净
- 行数：新增源文件均 ≤ 400（最大 `agent.rs` 390）

## Impact Surface

- **新增**：agent SFT 链路（4 脚本）+ Rust agent loop / sandbox / game_agent / 多字符 tokenizer
  + `--agent` CLI。所有文件均为新增，无既有 API 被改动。
- **不影响**：`nvllm-models` 的 `Gpt2` 接口；snake demo 原有 constrained 模式（`Gpt2Brain`）保持不变。
- 导出权重/数据 `.bin`/`.safetensors`/`input.txt` 仍由 `.gitignore` 排除，不入库。

## Related Docs

- [GPT-2 Agent SFT 能力](../../gpt2-agent-sft/index.md)
- [贪吃蛇 GPT-2 大脑](../../../agents/snake-gpt2/index.md)
- [既有 changelog：snake-gpt2 demo](./snake-gpt2-demo.md)
