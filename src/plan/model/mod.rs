pub mod resolver;

use serde::Serialize;

/// How the parameter count was obtained (shown to the user).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamSource {
    /// `--params-b` on the command line.
    Override,
    /// Exact total from the Hugging Face Hub safetensors metadata.
    HubSafetensors,
    /// Exact total from local `.safetensors` headers.
    LocalSafetensors,
    /// Computed from `config.json` dimensions.
    ConfigFormula,
}

/// Resolved properties of a model needed for VRAM and duration estimation.
#[derive(Debug, Clone, Serialize)]
pub struct ModelSpec {
    pub model_id: String,
    /// Total parameter count in billions (all experts for MoE).
    pub param_count_b: f64,
    /// Parameters active per token in billions (= total for dense models).
    pub active_param_count_b: f64,
    pub param_source: ParamSource,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub tie_word_embeddings: bool,
    /// True when the architecture was guessed from the parameter count.
    pub architecture_estimated: bool,
}

impl ModelSpec {
    /// Embedding (+ untied LM head) parameter count.
    pub fn embedding_params(&self) -> f64 {
        let e = self.vocab_size as f64 * self.hidden_size as f64;
        if self.tie_word_embeddings {
            e
        } else {
            2.0 * e
        }
    }
}
