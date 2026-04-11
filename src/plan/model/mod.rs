#![allow(dead_code)]

pub mod resolver;

/// Resolved properties of an ML model needed for VRAM and duration estimation.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The model identifier or path as supplied by the user.
    pub model_id: String,
    /// Total trainable parameter count in billions.
    pub param_count_b: f64,
    /// Number of transformer layers (hidden blocks).
    pub num_layers: u32,
    /// Width of the hidden state.
    pub hidden_size: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Number of key-value heads (may differ from num_heads in GQA models).
    pub num_kv_heads: u32,
}
