//! GPT-2 snake brain: the real neural path.
//!
//! Loads an exported GPT-2 (`model.safetensors` + `config.json` + `meta.json`)
//! via `nvllm_models::gpt2`, builds a text prompt describing the board, runs a
//! single forward pass, and reads the model's predicted next token. The token
//! is constrained to one of the four action keys (W/A/S/D), with the reverse
//! direction masked out so the model can never instant-reverse into its neck.
//!
//! On ANY error (load-time callers aside), the brain falls back to the
//! `WallAvoider` heuristic and never panics.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rand::RngCore;

use nvllm_models::gpt2::{Gpt2, Gpt2Config};

use crate::brain::{Brain, WallAvoider};
use crate::char_tokenizer::CharTokenizer;
use crate::game::{Direction, Game};
use crate::prompt;

/// GPT-2-backed brain with a heuristic fallback.
pub struct Gpt2Brain {
    model: Gpt2,
    tok: CharTokenizer,
    block_size: usize,
    /// Direction -> the token id the model must emit to choose that direction.
    action_ids: HashMap<Direction, u32>,
    fallback: WallAvoider,
}

/// Load a `Gpt2Brain` from an export directory. Fails (returns Err) only at
/// load time; runtime forward failures are handled inside `decide`.
pub fn load(model_dir: &Path) -> anyhow::Result<Gpt2Brain> {
    let cfg_path = model_dir.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", cfg_path.display()))?;
    let cfg: Gpt2Config = serde_json::from_str(&cfg_str)
        .map_err(|e| anyhow::anyhow!("parse config.json: {e}"))?;

    let tok = CharTokenizer::from_meta_dir(model_dir)?;

    // Resolve the four action-token ids; error loudly if any are missing.
    let mut action_ids = HashMap::new();
    for dir in Direction::all() {
        let ch = prompt::dir_to_action_char(dir);
        let id = tok.action_token_id(ch).ok_or_else(|| {
            anyhow::anyhow!(
                "action token '{}' is missing from the model vocabulary",
                ch
            )
        })?;
        action_ids.insert(dir, id);
    }

    let weights = model_dir.join("model.safetensors");
    // SAFETY: mmap of a local, trusted safetensors file produced by the export.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &Device::Cpu)?
    };
    let block_size = cfg.block_size;
    let model = Gpt2::load(cfg, vb)?;

    Ok(Gpt2Brain {
        model,
        tok,
        block_size,
        action_ids,
        fallback: WallAvoider,
    })
}

impl Gpt2Brain {
    /// Try a real forward pass and map the predicted token to a Direction.
    /// Returns `None` on any failure so the caller can fall back gracefully.
    fn try_decide(&self, game: &Game) -> Option<Direction> {
        // Leave one token of headroom under block_size for safety.
        let budget = self.block_size.saturating_sub(1).max(1);
        let prompt = prompt::build_prompt(game, budget);
        let mut ids = self.tok.encode(&prompt);
        if ids.is_empty() {
            return None;
        }
        // Hard cap at block_size: the model asserts t <= block_size, so
        // left-truncate keeping the END (header + MOVE:) when over budget.
        if ids.len() > self.block_size {
            let start = ids.len() - self.block_size;
            ids = ids[start..].to_vec();
        }
        let t = ids.len();
        let data: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_vec(data, (1, t), &Device::Cpu).ok()?;
        let logits = self.model.forward(&idx).ok()?; // (1, t, vocab)
        let last = logits.get(0).ok()?.get(t - 1).ok()?; // 1-D of len vocab
        let probs: Vec<f32> = last.to_vec1::<f32>().ok()?;

        // Masked argmax over the allowed action tokens. The reverse direction
        // is excluded so the model cannot instant-reverse into its own neck.
        let reverse = game.dir.opposite();
        masked_argmax(&probs, &self.action_ids, reverse)
    }
}

impl Brain for Gpt2Brain {
    fn decide(&self, game: &Game, rng: &mut dyn RngCore) -> Direction {
        match self.try_decide(game) {
            Some(d) => d,
            None => {
                eprintln!(
                    "[gpt2] forward/decode failed; using wall-avoider fallback"
                );
                self.fallback.decide(game, rng)
            }
        }
    }

    fn name(&self) -> &'static str {
        "gpt2-small"
    }
}

/// Pick the action token with the highest logit, ignoring the reverse
/// direction. Returns `None` only if no allowed action id is valid.
fn masked_argmax(
    probs: &[f32],
    action_ids: &HashMap<Direction, u32>,
    reverse: Direction,
) -> Option<Direction> {
    let mut best: Option<(Direction, f32)> = None;
    for dir in Direction::all() {
        if dir == reverse {
            continue;
        }
        let id = match action_ids.get(&dir) {
            Some(v) => *v as usize,
            None => continue,
        };
        if id >= probs.len() {
            continue;
        }
        let score = probs[id];
        match best {
            Some((_, bs)) if score <= bs => {}
            _ => best = Some((dir, score)),
        }
    }
    best.map(|(d, _)| d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use serde::Deserialize;
    use std::env;

    /// Exercises the real model path only when `SNAKE_GPT2_MODEL` points at an
    /// exported dir. Offline `cargo test` skips this (returns Ok) so the suite
    /// passes without weights; CI sets the env var to validate end-to-end.
    #[test]
    fn gpt2_forward_shape_when_model_present() {
        let path = match env::var("SNAKE_GPT2_MODEL") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "[gpt2] SNAKE_GPT2_MODEL unset; skipping live forward test"
                );
                return;
            }
        };
        let brain = load(Path::new(&path)).expect("load GPT-2 export");
        let mut rng = StdRng::seed_from_u64(0);
        let game = Game::new(12, 12, &mut rng);

        let budget = brain.block_size.saturating_sub(1).max(1);
        let prompt = prompt::build_prompt(&game, budget);
        let ids = brain.tok.encode(&prompt);
        assert!(!ids.is_empty(), "prompt must encode to >=1 token");
        let t = ids.len();
        assert!(t <= brain.block_size, "prompt must fit block_size");

        let data: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_vec(data, (1, t), &Device::Cpu).unwrap();
        let logits = brain.model.forward(&idx).expect("forward");

        assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
        assert_eq!(logits.dim(1).unwrap(), t, "seq dim");
        assert_eq!(
            logits.dim(2).unwrap(),
            brain.model.config().vocab_size,
            "vocab dim"
        );

        // decide() must return a valid, non-reversing direction.
        let d = brain.decide(&game, &mut StdRng::seed_from_u64(1));
        assert_ne!(d, game.dir.opposite());
    }

    /// Index of the largest f32 in a slice (total order, NaN-safe). Pure free
    /// function used only by the parity test's action-argmax comparison.
    fn argmax_index(scores: &[f32]) -> usize {
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// nanoGPT-vs-Rust numerical logits parity. The Python side dumps
    /// `parity.json` (tokens + last-position logits + W/A/S/D ids) via
    /// `scripts/compare_logits.py`; this test replays the SAME tokens through
    /// the candle forward and checks the action argmax agrees. The two stacks
    /// differ ONLY in GELU (Rust=erf, nanoGPT=tanh), so raw floats are close
    /// but not bit-exact; the argmax over the 4 action tokens is the real gate.
    #[test]
    fn gpt2_logits_parity_vs_pytorch() {
        let path = match env::var("SNAKE_GPT2_MODEL") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "[parity] SNAKE_GPT2_MODEL unset; skipping parity test"
                );
                return;
            }
        };
        let model_dir = Path::new(&path);
        let parity_path = model_dir.join("parity.json");
        if !parity_path.is_file() {
            eprintln!(
                "[parity] {} missing; run \
                 scripts/compare_logits.py --in_dir {} (skipping parity test)",
                parity_path.display(),
                path
            );
            return;
        }

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Parity {
            tokens: Vec<u32>,
            last_logits: Vec<f32>,
            action_chars: Vec<String>,
            action_ids: Vec<u32>,
        }

        let brain = load(model_dir).expect("load GPT-2 export");
        let raw =
            std::fs::read_to_string(&parity_path).expect("read parity.json");
        let parity: Parity =
            serde_json::from_str(&raw).expect("parse parity.json");

        // Forward the SAME token sequence the Python side forwarded.
        let t = parity.tokens.len();
        assert!(t > 0, "parity.tokens must be non-empty");
        assert!(t <= brain.block_size, "parity tokens exceed block_size");
        let data: Vec<i64> =
            parity.tokens.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_vec(data, (1, t), &Device::Cpu)
            .expect("build idx tensor");
        let logits = brain.model.forward(&idx).expect("rust forward");
        // Last position: (1, t, vocab) -> (vocab,). Mirrors try_decide().
        let last = logits
            .get(0)
            .expect("batch dim")
            .get(t - 1)
            .expect("last seq pos");
        let rust: Vec<f32> = last.to_vec1::<f32>().expect("rust logits vec");
        let ref_logits = &parity.last_logits;

        assert_eq!(
            rust.len(),
            ref_logits.len(),
            "vocab size mismatch: rust={} ref={}",
            rust.len(),
            ref_logits.len()
        );
        assert_eq!(
            rust.len(),
            brain.model.config().vocab_size,
            "rust logits len != config vocab_size"
        );

        // Diagnostics: max/mean abs diff across the whole vocab.
        let max_abs = rust
            .iter()
            .zip(ref_logits.iter())
            .map(|(r, rf)| (r - rf).abs())
            .fold(0.0_f32, f32::max);
        let sum_abs: f32 = rust
            .iter()
            .zip(ref_logits.iter())
            .map(|(r, rf)| (r - rf).abs())
            .sum();
        let mean_abs = sum_abs / rust.len() as f32;
        eprintln!(
            "[parity] logits |diff|: max={max_abs:.6} mean={mean_abs:.6} \
             (erf-vs-tanh GELU is the expected source of non-zero diff)"
        );

        // The meaningful, GELU-robust check: the chosen ACTION must agree.
        // argmax over the 4 action-token logits only (k indexes action_ids).
        let action_scores_rust: Vec<f32> = parity
            .action_ids
            .iter()
            .map(|&id| rust[id as usize])
            .collect();
        let action_scores_ref: Vec<f32> = parity
            .action_ids
            .iter()
            .map(|&id| ref_logits[id as usize])
            .collect();
        let rust_argmax_k = argmax_index(&action_scores_rust);
        let ref_argmax_k = argmax_index(&action_scores_ref);
        eprintln!(
            "[parity] rust chose action '{}' (id={}); ref chose '{}' (id={})",
            parity.action_chars[rust_argmax_k],
            parity.action_ids[rust_argmax_k],
            parity.action_chars[ref_argmax_k],
            parity.action_ids[ref_argmax_k],
        );
        assert_eq!(
            rust_argmax_k, ref_argmax_k,
            "action-direction disagreement -> real wiring bug (not just GELU)"
        );

        // Loose sanity ceiling: erf-vs-tanh GELU drift through the layers stays
        // small; a value >= 5.0 signals a real problem (transposed/mismatched
        // weights, wrong dtype, etc.). Loosen only if a measured checkpoint
        // legitimately exceeds it AND the action-argmax still agrees.
        assert!(
            max_abs < 5.0,
            "max_abs {max_abs:.6} >= 5.0 sanity ceiling; expected erf-vs-tanh \
             GELU drift only"
        );
    }

    #[test]
    fn masked_argmax_excludes_reverse() {
        // vocab of 4: id 0='W'(Up) 1='A'(Left) 2='S'(Down) 3='D'(Right)
        let mut action_ids = HashMap::new();
        action_ids.insert(Direction::Up, 0);
        action_ids.insert(Direction::Left, 1);
        action_ids.insert(Direction::Down, 2);
        action_ids.insert(Direction::Right, 3);

        // Highest raw logit is on id 2 (Down), but Down is the reverse of Up,
        // so the mask must skip it and pick the next best (id 0, Up).
        let probs = vec![0.5, 0.1, 0.9, 0.2];
        let picked = masked_argmax(&probs, &action_ids, Direction::Down).unwrap();
        assert_eq!(picked, Direction::Up);
    }
}
