Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# 贪吃蛇 GPT-2 大脑 (snake-gpt2)

## 职责

终端贪吃蛇游戏 + GPT-2 bash agent loop。现含**两种模式**：

1. **Constrained decoding（原模式）**：游戏模型 + 三种大脑（随机/启发式/GPT-2）+ crossterm 渲染。GPT-2 路径对 W/A/S/D 四个动作 token 做 constrained argmax，下棋方向语义接近随机（仅验证接线）。
2. **Agent loop（新模式，`--agent`）**：用 SFT 微调后的 GPT-2 自回归生成控制 token（`<bash>`/`<pause>`/`</answer>` 等），执行 bash 工具调用，注入结果，循环至 `</answer>`。模型已微调，输出语义有效。

## 模块组成

| 文件 | 行 | 职责 |
|------|----|------|
| `src/game.rs` | 361 | 纯游戏模型：网格 / `VecDeque` 蛇身 / `step` / 碰撞 / 进食 |
| `src/prompt.rs` | 180 | 棋盘 → 文本 prompt（ASCII 缩略图 + 头/食物/方向 + `MOVE:`） |
| `src/brain.rs` | 159 | `Brain` trait + `RandomBrain` + `WallAvoider`（兜底启发式） |
| `src/gpt2_brain.rs` | 359 | Constrained 模式真实路径：依赖 `nvllm-models`，forward + 动作掩码 argmax |
| `src/agent.rs` | 387 | Agent loop 核心：`generate_until_stop()` / `parse_last_bash()` / `agent_loop()`（泛型于 step+exec 闭包）。10 单测 |
| `src/sandbox.rs` | 180 | `run_bash()`：cwd 锁定 + 超时 + deny-list（rm -rf / sudo / curl 等）。7 单测 |
| `src/runner.rs` | 113 | `AgentModel` / `load()` / `run_with_model()`，把 candle `Gpt2` + sandbox 接进 agent loop |
| `src/char_tokenizer.rs` | 268 | char + 多字符 special-token tokenizer（longest-match encode，`from_meta_dir()`） |
| `src/render.rs` | 199 | crossterm 双缓冲；非 TTY 降级为逐帧 ASCII |
| `src/main.rs` | 220 | clap CLI + 主循环；`--agent`/`--agent-workdir`/`--agent-max-turns`/`--agent-max-new`/`--temp` |

## 游戏模型

`Game { rows, cols, snake: VecDeque<(i32,i32)>, food, dir, score, alive }`（game.rs:68）。`snake[0]` 是头。核心 `step(dir, rng) -> StepOutcome`（game.rs:114）：

- 撞墙或自撞 → `alive=false`
- 自撞判定：除**会移走的尾**外，碰到任意身体格即死（`allowed_to_enter_tail = !ate && new_head==tail`，允许滑进将空出的尾格）
- 吃食物 → `score+=1`，不弹尾（长大），随机空格重生食物
- 食物永不刷在蛇身上（单测保证）

## 大脑（constrained 模式）

```rust
trait Brain { fn decide(&self, game: &Game, rng: &mut dyn RngCore) -> Direction; fn name(&self) -> &'static str; }
```

注意：rand 0.8 的 `Rng` **非 dyn 安全**（泛型方法无 `?Sized` bound），故 trait 用 `&mut dyn RngCore`，靠 blanket impl 仍可调 `gen_range`。

- `RandomBrain`：3 个非反向方向均匀随机
- `WallAvoider`：贪心避撞 + 最小曼哈顿距离趋向食物（可玩兜底）
- `Gpt2Brain`：见下

## GPT-2 接线 (gpt2_brain.rs)

`load(model_dir)`（gpt2_brain.rs:38）：读 `config.json`→`Gpt2Config`，`meta.json`→`CharTokenizer`，解析 W/A/S/D 动作 token id，`VarBuilder::from_mmaped_safetensors` + `Gpt2::load`。

`decide` 流程：
1. `prompt::build_prompt(game, block_size-1)` → 文本
2. encode → u32 ids，左截断保末尾（保 header + `MOVE:`）≤ `block_size`
3. `Tensor::from_vec((1,T) i64)` → `model.forward` → logits `(1,T,vocab)`
4. 取末位 logits，`to_vec1::<f32>()`（vocab 仅 65，CPU argmax 即可）
5. 掩码：屏蔽当前方向的反向动作 token（防 180° 撞自己脖子），在 W/A/S/D 四个动作 id 上 argmax → `Direction`
6. **任何错误**（forward 失败等）→ `eprintln!` 警告并回退 `WallAvoider`，不 panic

## Agent Loop (agent.rs / runner.rs / sandbox.rs)

`--agent "<task>"` 模式：跳过贪吃蛇，跑一个「生成 → 工具调用 → 注入结果 → 再生成」的循环，模型需为 SFT 微调后的 agent 权重（vocab=73）。核心 `agent_loop()`（agent.rs:178）**泛型于 `step`（输入 token ids → 末位 logits）和 `exec`（命令 → stdout）两个闭包**，故无真实权重/shell 也能单测；`runner::run_with_model()` 才把真实 candle `Gpt2` + `sandbox::run_bash` 接进去。

循环流程：

1. `tok.encode(prompt)` → 初始 token ids（上下文）
2. `generate_until_stop()`（agent.rs:138）：自回归贪心（`temp<=0`）或温度采样，每步把新 token 追加进 ctx，超 `block_size` 时左截断保末尾；命中 stop token 或 `max_new` 即停
3. `<pause>`（StopKind::Pause）→ `parse_last_bash()` 抽取最后一个 `<bash>...</bash>` 的命令体 → `sandbox::run_bash()` 执行
4. 把结果注入上下文：`<result>{output}</result><resume>`
5. 回到步骤 2，直到 `</answer>` 或 `max_turns`

### 停止逻辑（易错点）

- `<answer>` **开标签不停**——生成穿过它，继续产出 answer 内容
- 仅 `</answer>` 终止（StopKind::AnswerClose，捕获 answer 文本）
- `<pause>` 触发工具执行 + 注入（StopKind::Pause）
- 都没命中 → MaxNew（给满 `max_new`）

### Sandbox (sandbox.rs)

`run_bash(cmd, workdir, timeout)`：`bash -c` 子进程，cwd **锁定**到 caller 指定目录，硬超时（默认 10s，超时 kill），`is_dangerous()` deny-list（`rm -rf /`、`sudo`、`chmod 777 /`、`mkfs`、`dd if=`、`curl`、`wget`、`ssh`、fork bomb、重定向到 `/etc`/`/usr`/`/` 等，大小写不敏感）。非零退出 / 超时 / 拒绝命令均 `Result::Err`。**这是 agent 唯一触碰真实文件系统/进程表的地方**——是本地 demo 的粗粒度护栏，**不是完整安全边界**。

## Tokenizer 与词表契约

本沙箱离线（HuggingFace 不可达、无 GPT-2 BPE 文件、无 tiktoken）→ 放弃 `tokenizers` crate，用纯 Rust char tokenizer。

`CharTokenizer`（char_tokenizer.rs，268 行，已重写支持多字符 token）：`from_meta_dir()` 读 `meta.json`（`{vocab_size, itos[], stoi{}, special_tokens{}}`）。**itos 索引是 id 的真源**，stoi 由其重建；多字符的 itos 条目成为 longest-match specials。encode 做贪心最长匹配，未知单字符跳过（无 BPE，全离线）。提供 `decode_id()` / `special_id()` / `is_special()`。

**vocab_size=73**：

- IDs 0–64：65 个 tinyshakespeare 字符（按 Unicode 码点排序）
- IDs 65–72：8 个控制 token —— `<bash>`(65) `</bash>`(66) `<pause>`(67) `<result>`(68) `</result>`(69) `<resume>`(70) `<answer>`(71) `</answer>`(72)
- 多字符 token 在 encode 时靠 longest-match 识别为单 token（如 `</answer>` 不会拆成 `<answer>`+…）

> Constrained 模式仍用 vocab=65 的 meta.json（无 special tokens）；agent 模式用 vocab=73 的导出目录。两者共用同一 `CharTokenizer` 实现。

## 语义随机性 vs Agent 微调（重要，避免误判 bug）

**Constrained 模式**：GPT-2 是**基座语言模型**，不懂棋盘 → 下棋方向语义接近随机（可能直撞墙）。这是预期行为（对照 `nano-vllm/example_agentic.py` 的「验证接线，语义随机」注释），不是代码 bug。UI 明确标注 `semantics random — validation wiring`。要可玩用 `--heuristic` 或 `--random`。

**Agent 模式**：模型**已 SFT 微调**，val_loss 5.08→0.066，输出语义有效——能产出结构化的 `<bash>echo X</bash><pause>` 工具调用并给出正确走法（游戏任务上表现良好）。两种模式的语义有效性截然不同，不要混为一谈。

## SFT 微调链路

```
train_gpt2_snake.py → out-gpt2-snake/ckpt.pt (shakespeare, vocab=65)
                    │ warm-start（wte/wpe 因新 vocab 重init）
sft_train_gpt2_snake.py → out-agent-snake/ckpt.pt (vocab=73)
                    │
export_agent_safetensors.py → out-agent-snake-export/ (model.safetensors, 148 tensors)
                    │
snake-gpt2 --agent → agent_loop
```

- **模型**：GPT-2 small（12/768/12），`bias=True`，vocab=73，block_size=256，dropout=0.0
- **Masked loss**：`CE(reduction='none') * mask`，`loss = (ce * w).sum() / w.sum().clamp(min=1)`——mask 与 **target 对齐**（offset i+1+k），只有被监督 token 贡献 loss
- **暖启动**：从 `out-gpt2-snake/ckpt.pt` 复制权重，`wte.weight`/`wpe.weight` 因 vocab 变化而重 init
- **结果**：val_loss **5.08 → 0.066**（~77× 降幅），2500 iters
- **输出**：`out-agent-snake/ckpt.pt` → 导出至 `out-agent-snake-export/`（config.json + meta.json + model.safetensors）

训练相关脚本在 `nano-vllm-rs/scripts/`：`agent_special_tokens.py`（vocab 契约 + longest-match encode/decode + build_trace with mask）、`prepare_agent_snake.py`（生成 800 游戏 + 40 代码轨迹）、`sft_train_gpt2_snake.py`（masked-loss trainer）、`export_agent_safetensors.py`（ckpt → safetensors，148 tensors）。

## 训练数据格式（prompt 构造关键）

游戏 prompt 用此精确格式（bash 会剥掉 `$()` 的尾换行，故必须用 `$'...'` 或写文件）：

```
Task: choose the snake's next move UDLR to reach the food without crashing.\n
State:\n\n
[6 rows × 8 cols of ./o/H/F]\n\n
HEAD, FOOD, DIRR SCORE\n
MOVE:\n
```

`MOVE:` 后的尾 `\n` **至关重要**——缺它，模型首 token 预测会整体偏移（实测：168 vs 167 token 的 prompt 把 argmax 从 `<bash>` 变成 `h`）。

## 游戏任务 vs 代码任务

- **游戏任务（800 条轨迹）**：WORKS——走法正确（U/D/L/R），`<bash>echo X</bash><pause>` 格式干净
- **代码任务（40 条轨迹）**：格式能识别，但内容偏噪——需更多代码训练数据才能完整复现

## 性能与对齐

- Rust candle-core **仅 CPU forward**（无 GPU、无 KV cache）：85M 模型每次 forward ~7s → agent loop 视轮次约 **65–230s**
- Python torch forward 与 candle logits **吻合到小数点后 4 位**（已对拍验证）

## 渲染

`Renderer`（render.rs:51）：检测 `IsTerminal`。TTY 走 alt-screen + raw mode + 'q' 退出；`TerminalGuard` 的 `Drop` **保证任何退出路径都恢复终端**（绝不留 raw mode）。非 TTY（沙箱/管道）降级为逐帧 ASCII + 状态行，靠 `--steps` 限步终止。

## 相关模块

- [GPT-2 Rust 移植](../nvllm-models/index.md)：`Gpt2::load` / `forward` 来源，权重键契约见彼。
- 演示端到端：见 [能力](../../features/gpt2-snake-demo/index.md)。
