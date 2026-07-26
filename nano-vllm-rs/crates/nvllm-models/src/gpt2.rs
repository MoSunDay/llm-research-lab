//! GPT-2 port for the nano-vllm Rust engine.
//!
//! Reference (READ-ONLY): link_repos/nanoGPT/model.py -- every numeric choice here
//! mirrors that file so the forward pass is numerically compatible:
//!   - LayerNorm with bias, eps = 1e-5 (F.layer_norm)
//!   - fused c_attn nn.Linear(n_embd, 3*n_embd) then split q/k/v
//!   - causal scaled-dot-product attention, scale = head_dim^-0.5
//!   - MLP: c_fc(n_embd, 4*n_embd) -> erf GELU -> c_proj(4*n_embd, n_embd)
//!   - pre-norm blocks, learned wte + wpe, ln_f, lm_head TIED to wte
//!   - vocab 50257 padded to 50304 (next multiple of 64) for matmul efficiency
//!
//! Design is functional: configs are plain structs, "modules" are structs that
//! own candle Var-backed tensors and expose forward methods. No classes.

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

/// LayerNorm epsilon used by GPT-2 / nanoGPT (F.layer_norm default 1e-5).
const LN_EPS: f64 = 1e-5;

/// GPT-2 model config. Defaults are GPT-2 small (124M).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gpt2Config {
    pub block_size: usize,
    /// Padded vocabulary size (50257 -> 50304). Rows 50257..50303 are unused
    /// by the BPE tokenizer and reserved for special/control tokens.
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        // link_repos/nanoGPT/model.py GPTConfig defaults.
        Self {
            block_size: 1024,
            vocab_size: 50304,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
        }
    }
}

/// Causal self-attention: fused qkv projection then scaled-dot-product attention.
pub struct CausalSelfAttention {
    c_attn: Linear,  // n_embd -> 3*n_embd
    c_proj: Linear,  // n_embd -> n_embd
    n_head: usize,
    head_dim: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    pub fn load(cfg: &Gpt2Config, vb: VarBuilder) -> Result<Self> {
        let n_embd = cfg.n_embd;
        let attn_vb = vb.pp("attn");
        let c_attn = linear(n_embd, 3 * n_embd, attn_vb.pp("c_attn"))?;
        let c_proj = linear(n_embd, n_embd, attn_vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
            head_dim: n_embd / cfg.n_head,
            n_embd,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        assert_eq!(c, self.n_embd, "attn input n_embd mismatch");
        let dev = x.device();

        // fused qkv then split into q,k,v along the feature dim.
        let qkv = self.c_attn.forward(x)?; // (b, t, 3c)
        let q = qkv.narrow(D::Minus1, 0, c)?;
        let k = qkv.narrow(D::Minus1, c, c)?;
        let v = qkv.narrow(D::Minus1, 2 * c, c)?;

        // (b, t, nh, hs) -> (b, nh, t, hs)
        let heads = (b, t, self.n_head, self.head_dim);
        // transpose to (b, nh, t, hs); contiguous() because candle's CPU matmul
        // rejects non-contiguous lhs produced by the transpose.
        let q = q.reshape(heads)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(heads)?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(heads)?.transpose(1, 2)?.contiguous()?;

        // attention scores: (b, nh, t, t)
        let mut att = q.matmul(&k.t()?)?;
        let scale = (self.head_dim as f64).recip().sqrt();
        att = att.affine(scale, 0.0)?;

        // additive causal mask: 0 on/below diagonal, -inf above. where_cond needs
        // a U8 condition (candle comparison results are U8), operands are F32.
        let tril = Tensor::tril2(t, DType::U8, dev)?; // lower = 1, upper = 0
        let zeros = Tensor::zeros((t, t), DType::F32, dev)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (t, t), dev)?;
        let add_mask = tril.where_cond(&zeros, &neg_inf)?; // lower->0, upper->-inf
        // explicit broadcast: candle binary ops do not auto-broadcast across ranks.
        let add_mask = add_mask.broadcast_as((b, self.n_head, t, t))?;
        att = (&att + &add_mask)?;
        att = candle_nn::ops::softmax(&att, D::Minus1)?;

        let y = att.matmul(&v)?; // (b, nh, t, hs)
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.c_proj.forward(&y)
    }
}

/// GPT-2 MLP: linear up, erf GELU, linear down.
pub struct Mlp {
    c_fc: Linear,    // n_embd -> 4*n_embd
    c_proj: Linear,  // 4*n_embd -> n_embd
}

impl Mlp {
    pub fn load(cfg: &Gpt2Config, vb: VarBuilder) -> Result<Self> {
        let h = 4 * cfg.n_embd;
        let mlp_vb = vb.pp("mlp");
        Ok(Self {
            c_fc: linear(cfg.n_embd, h, mlp_vb.pp("c_fc"))?,
            c_proj: linear(h, cfg.n_embd, mlp_vb.pp("c_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // nn.GELU() approximate='none' == exact erf variant. candle's gelu()
        // is the tanh approximation; gelu_erf() is the exact erf form nanoGPT uses.
        let x = self.c_fc.forward(x)?;
        let x = x.gelu_erf()?;
        self.c_proj.forward(&x)
    }
}

/// Pre-norm transformer block: x + attn(ln_1(x)); x + mlp(ln_2(x)).
pub struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    pub fn load(cfg: &Gpt2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: layer_norm(cfg.n_embd, LN_EPS, vb.pp("ln_1"))?,
            attn: CausalSelfAttention::load(cfg, vb.clone())?,
            ln_2: layer_norm(cfg.n_embd, LN_EPS, vb.pp("ln_2"))?,
            mlp: Mlp::load(cfg, vb.clone())?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.attn.forward(&self.ln_1.forward(x)?)?;
        let x = (x + a)?;
        let m = self.mlp.forward(&self.ln_2.forward(&x)?)?;
        &x + m
    }
}

/// GPT-2 model. lm_head is tied to wte (not stored separately).
pub struct Gpt2 {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    cfg: Gpt2Config,
}

impl Gpt2 {
    /// Build the model from a VarBuilder. Works with a random VarMap (smoke test)
    /// or a safetensors mmap (real weights, T4). Key names follow nanoGPT's
    /// state_dict: transformer.wte/wpe, transformer.h.{i}.*, transformer.ln_f.
    pub fn load(cfg: Gpt2Config, vb: VarBuilder) -> Result<Self> {
        let tvb = vb.pp("transformer");
        let wte = Embedding::new(
            tvb.pp("wte").get((cfg.vocab_size, cfg.n_embd), "weight")?,
            cfg.n_embd,
        );
        let wpe = Embedding::new(
            tvb.pp("wpe").get((cfg.block_size, cfg.n_embd), "weight")?,
            cfg.n_embd,
        );
        let h_vb = tvb.pp("h");
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Block::load(&cfg, h_vb.pp(i))?);
        }
        let ln_f = layer_norm(cfg.n_embd, LN_EPS, tvb.pp("ln_f"))?;
        Ok(Self { wte, wpe, blocks, ln_f, cfg })
    }

    /// Full-sequence forward. `idx` is (b, t) of token ids. Returns logits
    /// (b, t, vocab) via the wte-tied lm_head. Mirrors nanoGPT forward() minus
    /// its last-position-only inference optimization.
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (b, t) = (idx.dim(0)?, idx.dim(1)?);
        assert!(t <= self.cfg.block_size, "sequence longer than block_size");
        let dev = idx.device();

        let pos = Tensor::arange(0i64, t as i64, dev)?;
        let tok_emb = self.wte.forward(idx)?; // (b, t, n_embd)
        let pos_emb = self.wpe.forward(&pos)?.broadcast_as((b, t, self.cfg.n_embd))?;
        let mut x = (&tok_emb + &pos_emb)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        // tied lm_head: logits = x @ wte.weight.T. Reshape to a 2D matmul
        // (candle matmul does not auto-broadcast a 2D rhs vs a batched lhs) and
        // make wte.T contiguous so the CPU backend accepts the striding.
        let wt = self.wte.embeddings().t()?.contiguous()?; // (n_embd, vocab)
        let logits = x.reshape((b * t, self.cfg.n_embd))?.matmul(&wt)?;
        let logits = logits.reshape((b, t, self.cfg.vocab_size))?;
        Ok(logits)
    }

    pub fn config(&self) -> &Gpt2Config {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    /// Structural smoke test (T3): a tiny GPT-2 with random VarMap weights must
    /// run a full forward pass and produce logits of the expected shape on CPU.
    /// This verifies tensor plumbing / reshape / causal-mask plumbing; it is NOT
    /// a numerical-correctness check (that is T6, gated on real weights).
    fn tiny_cfg() -> Gpt2Config {
        Gpt2Config {
            block_size: 32,
            vocab_size: 64,
            n_layer: 2,
            n_head: 4,
            n_embd: 48,
        }
    }

    #[test]
    fn forward_shape_matches() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = Gpt2::load(cfg.clone(), vb)?;

        // config() getter exposes the stored config (closes review gap #4).
        assert_eq!(model.config().block_size, cfg.block_size);
        assert_eq!(model.config().vocab_size, cfg.vocab_size);

        // token ids 0..19 reshaped to (2,10); all < vocab 64.
        let idx = Tensor::arange(0i64, 20i64, &dev)?.reshape((2, 10))?;
        let logits = model.forward(&idx)?;
        assert_eq!(logits.dims(), [2usize, 10, cfg.vocab_size]);
        Ok(())
    }

    #[test]
    fn causal_mask_is_triangular() {
        // The additive mask built in attention must be 0 on/below the diagonal
        // and -inf strictly above it; otherwise future tokens leak.
        let dev = Device::Cpu;
        let n = 4usize;
        let tril = Tensor::tril2(n, DType::U8, &dev).unwrap();
        let zeros = Tensor::zeros((n, n), DType::F32, &dev).unwrap();
        let neg = Tensor::full(f32::NEG_INFINITY, (n, n), &dev).unwrap();
        let m = tril.where_cond(&zeros, &neg).unwrap().to_vec2::<f32>().unwrap();
        for (i, row) in m.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if j <= i {
                    assert_eq!(v, 0.0, "lower tri must add 0 at ({i},{j})");
                } else {
                    assert!(v.is_infinite() && v.is_sign_negative(),
                        "upper tri must be -inf at ({i},{j}), got {v}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "sequence longer than block_size")]
    fn forward_panics_when_seq_exceeds_block_size() {
        // Error/boundary path: the model's hard contract is seq_len <=
        // block_size, because the learned positional embedding only covers
        // block_size positions. A longer sequence must panic up front
        // rather than read the wpe out of bounds.
        let dev = Device::Cpu;
        let cfg = tiny_cfg(); // block_size 32
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = Gpt2::load(cfg, vb).unwrap();
        // 33 tokens, one past block_size; ids all < vocab 64.
        let idx = Tensor::arange(0i64, 33i64, &dev)
            .unwrap()
            .reshape((1, 33))
            .unwrap();
        let _ = model.forward(&idx);
    }
}
