//! Serving engine for the nano-vllm Rust port.
//!
//! Reference (read-only): link_repos/nano-vllm/nanovllm/engine/ for the
//! scheduler / block-manager / model-runner design this will mirror later.

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
