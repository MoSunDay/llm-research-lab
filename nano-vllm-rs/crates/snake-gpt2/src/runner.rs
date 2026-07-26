//! Model wiring for the agent loop: loads an exported GPT-2 and runs the loop
//! against a real candle `Gpt2`, executing bash via `sandbox::run_bash`.
//!
//! Separated from `agent.rs` (pure loop logic + tests) to keep both files
//! within the line budget and to isolate the candle/tensor concerns.

use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;

use nvllm_models::gpt2::{Gpt2, Gpt2Config};

use crate::agent::{agent_loop, resolve_special_ids, AgentRun};
use crate::char_tokenizer::CharTokenizer;

/// Loaded model + tokenizer ready to drive the agent loop.
pub struct AgentModel {
    pub model: Gpt2,
    pub tok: CharTokenizer,
    pub block_size: usize,
}

/// Load an exported agent model dir (config.json + model.safetensors + meta.json).
pub fn load(dir: &Path) -> anyhow::Result<AgentModel> {
    let cfg_path = dir.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", cfg_path.display()))?;
    let cfg: Gpt2Config = serde_json::from_str(&cfg_str)
        .map_err(|e| anyhow::anyhow!("parse config.json: {e}"))?;
    let tok = CharTokenizer::from_meta_dir(dir)?;
    let weights = dir.join("model.safetensors");
    // SAFETY: mmap of a local, trusted safetensors file produced by the export.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], candle_core::DType::F32, &Device::Cpu)?
    };
    let block_size = cfg.block_size;
    let model = Gpt2::load(cfg, vb)?;
    Ok(AgentModel { model, tok, block_size })
}

/// Run the full agent loop against a real model, executing bash in `workdir`.
pub fn run_with_model(
    am: &AgentModel,
    prompt: &str,
    workdir: &Path,
    max_turns: usize,
    max_new: usize,
    temp: f32,
    rng: &mut StdRng,
) -> anyhow::Result<AgentRun> {
    let special = resolve_special_ids(&am.tok)?;
    let block_size = am.block_size;
    let step = |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
        let t = ids.len().max(1);
        let data: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_vec(data, (1, t), &Device::Cpu)?;
        let logits = am.model.forward(&idx)?;
        let last = logits.get(0)?.get(t - 1)?;
        Ok(last.to_vec1::<f32>()?)
    };
    let timeout = std::time::Duration::from_secs(10);
    let exec = |cmd: &str| crate::sandbox::run_bash(cmd, workdir, timeout);
    agent_loop(
        &step,
        &am.tok,
        &special,
        block_size,
        prompt,
        &exec,
        max_turns,
        max_new,
        temp,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::env;

    /// Live end-to-end run, gated on SNAKE_AGENT_MODEL so offline `cargo test`
    /// skips it (no weights present). CI/interactive use sets the env var.
    #[test]
    fn agent_loop_live_when_model_present() {
        let path = match env::var("SNAKE_AGENT_MODEL") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("[agent] SNAKE_AGENT_MODEL unset; skipping live agent-loop test");
                return;
            }
        };
        let am = load(std::path::Path::new(&path)).expect("load agent model");
        let tmp = std::env::temp_dir().join(format!("snake-agent-live-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let run = run_with_model(
            &am,
            "Task: print the direction R using echo.\n",
            &tmp,
            4,
            64,
            0.0,
            &mut rng,
        )
        .expect("agent loop");
        eprintln!("[agent] live transcript:\n{}", run.transcript);
        assert!(!run.events.is_empty(), "must produce at least one event");
    }
}
