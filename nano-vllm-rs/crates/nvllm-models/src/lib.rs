//! Model definitions for the nano-vllm Rust port.
//!
//! Reference (read-only): link_repos/nanoGPT/model.py for GPT-2 architecture.
//! Pure-functional design: configs are plain structs, modules are structs that
//! own candle Var-tensors and expose free-function `forward` style methods.

pub mod gpt2;

/// Force the candle dependency to be linked so a workspace build verifies that
/// candle-core compiles in this environment. Removed once real code lands.
pub fn _links_candle() -> candle_core::Result<()> {
    let _device = candle_core::Device::Cpu;
    let _dtype = candle_core::DType::F32;
    Ok(())
}
