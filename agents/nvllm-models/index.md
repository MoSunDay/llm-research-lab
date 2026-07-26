Commit: 336ecd7bc3c6972a98d715ed881a3fba8e241d81

# GPT-2 Rust 移植 (nvllm-models)

## 职责

用 candle 0.11 实现 GPT-2 的前向推理（pre-norm Transformer），并定义与 nanoGPT state_dict 对齐的权重加载契约。供 snake-gpt2 通过 `VarBuilder` mmap 加载真实权重。

## 边界

- 只做前向（`forward`），无训练、无 KV cache、无采样
- `lm_head` 与 `wte` **权重绑定**，loader 不读任何 `lm_head.*` 键
- 不负责 tokenizer（调用方自带）；不负责下载（无 hf-hub 依赖）
- 序列超 `block_size` 会 `assert!` panic（`gpt2.rs:199`），非 `Result::Err`

## 关键抽象

### Gpt2Config (gpt2.rs:23)

| 字段 | 默认 (gpt2.rs:33) | 说明 |
|------|------------------|------|
| `block_size` | 1024 | 最大上下文（= wpe 行数） |
| `vocab_size` | 50304 | 词表（GPT-2 50257 向上取整） |
| `n_layer` | 12 | 层数 |
| `n_head` | 12 | 头数 |
| `n_embd` | 768 | 嵌入维 |

注意：**无 `bias`、无 `ln_eps` 字段**。eps 硬编码 `const LN_EPS = 1e-5`（gpt2.rs:19）；bias 恒为 `true`（见下）。`serde::Deserialize`，可直接从 `config.json` 反序列化。

### Gpt2 (gpt2.rs:163)

```
Gpt2
 ├── wte: Embedding(vocab, n_embd)
 ├── wpe: Embedding(block_size, n_embd)
 ├── blocks: Vec<Block> × n_layer
 │    └── Block: ln_1 → CausalSelfAttention → ln_2 → Mlp（均 pre-norm 残差）
 └── ln_f: LayerNorm        （lm_head = x @ wte.weight.T，不存储）
```

### 接口

| 方法 | 位置 | 签名 |
|------|------|------|
| `Gpt2::load` | gpt2.rs:175 | `(cfg: Gpt2Config, vb: VarBuilder) -> Result<Self>` |
| `Gpt2::forward` | gpt2.rs:197 | `(&self, idx: &Tensor) -> Result<Tensor>`，`idx` 形 `(b,t)` i64，返回 `(b,t,vocab)` logits |
| `Gpt2::config` | gpt2.rs:219 | `-> &Gpt2Config` |

位置与因果 mask 在 `forward` 内部构造（gpt2.rs:202 算 positions），无 positions/mask 参数。

## 权重键契约（最高价值稳定事实）

`Gpt2::load` 经 `vb.pp("transformer").pp(...)` 链构建键名，**期望 nanoGPT 原生 `transformer.*`**，注意力为**融合** `c_attn`（`[3·n_embd, n_embd]`，forward 内 narrow 拆 q/k/v）：

| 组件 | safetensors 键 |
|------|----------------|
| token embed | `transformer.wte.weight` |
| 位置 embed | `transformer.wpe.weight` |
| block ln_1 | `transformer.h.{i}.ln_1.{weight,bias}` |
| block ln_2 | `transformer.h.{i}.ln_2.{weight,bias}` |
| 融合 qkv | `transformer.h.{i}.attn.c_attn.{weight,bias}` |
| attn 输出 | `transformer.h.{i}.attn.c_proj.{weight,bias}` |
| mlp up | `transformer.h.{i}.mlp.c_fc.{weight,bias}` |
| mlp down | `transformer.h.{i}.mlp.c_proj.{weight,bias}` |
| 终 norm | `transformer.ln_f.{weight,bias}` |

**不要喂**：`lm_head.weight`（绑定，loader 不读）、`*.attn.bias` / `*.attn.masked_bias`（因果 mask buffer）。也不要喂顶层 `convert_*.py` 的 `model.*` 拆分键。

## bias 与 GELU（两个易错点）

- **bias 强制 `true`**：`candle_nn::linear`（非 `linear_no_bias`）+ `layer_norm(eps_f64)`（affine=true）都加载 `.bias`。训练 checkpoint 必须 `bias=True`，否则缺键报错。
- **GELU 是 erf 精确式**（gpt2.rs:131 `gelu_erf()`），而 nanoGPT `NewGELU` 是 tanh 近似。→ Rust 与 PyTorch logits 仅**近似**一致，不是 bit-exact；对拍时比「动作 argmax」而非浮点（见 [演示](../../features/gpt2-snake-demo/index.md)）。

## 加载模式（调用方职责）

crate 本身不含 `from_mmaped_safetensors`，仅测试用 `from_varmap`。真实加载在调用方：

```rust
let vb = unsafe { VarBuilder::from_mmaped_safetensors(
    &[&dir.join("model.safetensors")], DType::F32, &Device::Cpu)? };
let cfg: Gpt2Config = serde_json::from_reader(File::open(dir.join("config.json"))?)?;
let model = Gpt2::load(cfg, vb)?;
```

## 测试

`cargo test -p nvllm-models`：3 个结构性单测（形状 / 因果 mask 三角性 / 超 block_size panic）。仅用随机 `VarMap`，**无数值正确性测试**。

## 相关模块

- [贪吃蛇 GPT-2 大脑](../snake-gpt2/index.md)：唯一消费者，把 forward 接进游戏决策。
