use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use super::ModelSpec;

const HF_TIMEOUT_SECS: u64 = 15;

/// Loose deserialization of HuggingFace / local config.json.
///
/// Different model families use different key names for the same concept;
/// every field is `Option` so we can gracefully handle whatever subset is present.
#[derive(Debug, Deserialize)]
struct HfConfig {
    // Parameter count — not always present
    num_parameters: Option<u64>,

    // Layer count (various naming conventions)
    num_hidden_layers: Option<u32>,
    n_layer: Option<u32>,
    n_layers: Option<u32>,

    // Hidden size
    hidden_size: Option<u32>,
    n_embd: Option<u32>,
    d_model: Option<u32>,

    // Attention heads
    num_attention_heads: Option<u32>,
    n_head: Option<u32>,

    // GQA key-value heads — only present on Llama-2/3 style models
    num_key_value_heads: Option<u32>,
}

impl HfConfig {
    fn num_layers(&self) -> u32 {
        self.num_hidden_layers
            .or(self.n_layer)
            .or(self.n_layers)
            .unwrap_or(32)
    }

    fn hidden_size(&self) -> u32 {
        self.hidden_size
            .or(self.n_embd)
            .or(self.d_model)
            .unwrap_or(4096)
    }

    fn num_heads(&self) -> u32 {
        self.num_attention_heads.or(self.n_head).unwrap_or(32)
    }

    fn num_kv_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or_else(|| self.num_heads())
    }

    /// Derive approximate parameter count from architecture when `num_parameters`
    /// is absent. Uses the standard transformer estimate: ≈ 12 × L × H².
    fn param_count_b(&self) -> f64 {
        if let Some(n) = self.num_parameters {
            return n as f64 / 1e9;
        }
        let l = self.num_layers() as f64;
        let h = self.hidden_size() as f64;
        (12.0 * l * h * h) / 1e9
    }

    fn into_spec(self, model_id: &str, param_override: Option<f64>) -> ModelSpec {
        let param_count_b = param_override.unwrap_or_else(|| self.param_count_b());
        ModelSpec {
            model_id: model_id.to_string(),
            param_count_b,
            num_layers: self.num_layers(),
            hidden_size: self.hidden_size(),
            num_heads: self.num_heads(),
            num_kv_heads: self.num_kv_heads(),
        }
    }
}

/// Resolve `model_id` to a `ModelSpec`.
///
/// Resolution order:
/// 1. Local filesystem path — if the string is an existing path, read
///    `config.json` from that directory (or the file itself).
/// 2. Hugging Face Hub — HTTP GET to
///    `https://huggingface.co/{model_id}/resolve/main/config.json`.
/// 3. Params-only fallback — if `params_b_override` is `Some`, build a spec
///    with architecture values inferred from the parameter count.
///
/// On network failure or unknown model, returns `Err` with a clear error
/// message that tells the user to pass `--params-b`.
pub async fn resolve(model_id: &str, params_b_override: Option<f64>) -> Result<ModelSpec> {
    // ── 1. Local path ─────────────────────────────────────────────────────────
    let local = Path::new(model_id);
    if local.exists() {
        let config_path = if local.is_dir() {
            local.join("config.json")
        } else {
            local.to_path_buf()
        };
        if config_path.exists() {
            let json = std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?;
            let cfg: HfConfig = serde_json::from_str(&json)
                .with_context(|| format!("parsing {}", config_path.display()))?;
            return Ok(cfg.into_spec(model_id, params_b_override));
        }
    }

    // ── 2. Hugging Face Hub ───────────────────────────────────────────────────
    let url = format!(
        "https://huggingface.co/{}/resolve/main/config.json",
        model_id
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(HF_TIMEOUT_SECS))
        .user_agent("calibrate/0.1")
        .build()
        .context("building HTTP client")?;

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let cfg: HfConfig = resp
                .json()
                .await
                .with_context(|| format!("parsing config.json for {model_id}"))?;
            Ok(cfg.into_spec(model_id, params_b_override))
        }
        Ok(resp) => {
            if let Some(p) = params_b_override {
                return Ok(infer_spec_from_params(model_id, p));
            }
            bail!(
                "Model '{}' not found on Hugging Face Hub (HTTP {}). \
                 Pass `--params-b <BILLIONS>` to specify the parameter count directly.",
                model_id,
                resp.status()
            )
        }
        Err(e) => {
            if let Some(p) = params_b_override {
                return Ok(infer_spec_from_params(model_id, p));
            }
            bail!(
                "Could not reach Hugging Face Hub for '{}': {}. \
                 Pass `--params-b <BILLIONS>` to specify the parameter count directly.",
                model_id,
                e
            )
        }
    }
}

/// Build a ModelSpec using only the parameter count, approximating architecture
/// values from well-known model family sizes.
fn infer_spec_from_params(model_id: &str, param_count_b: f64) -> ModelSpec {
    let (num_layers, hidden_size, num_heads) = match param_count_b as u32 {
        0..=3 => (18u32, 2048u32, 16u32),
        4..=8 => (32, 4096, 32),
        9..=13 => (40, 5120, 40),
        14..=35 => (60, 8192, 64),
        _ => (80, 8192, 64),
    };
    ModelSpec {
        model_id: model_id.to_string(),
        param_count_b,
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads: num_heads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_spec_7b_gives_expected_arch() {
        let spec = infer_spec_from_params("test/7b", 7.0);
        assert_eq!(spec.num_layers, 32);
        assert_eq!(spec.hidden_size, 4096);
    }

    #[test]
    fn infer_spec_70b_gives_wide_arch() {
        let spec = infer_spec_from_params("test/70b", 70.0);
        assert!(
            spec.num_layers >= 60,
            "expected deep model, got {}",
            spec.num_layers
        );
        assert!(spec.hidden_size >= 8192);
    }

    #[test]
    fn hf_config_param_count_from_explicit_field() {
        let cfg = HfConfig {
            num_parameters: Some(7_000_000_000),
            num_hidden_layers: Some(32),
            hidden_size: Some(4096),
            num_attention_heads: Some(32),
            num_key_value_heads: None,
            n_layer: None,
            n_layers: None,
            n_embd: None,
            d_model: None,
            n_head: None,
        };
        let b = cfg.param_count_b();
        assert!((b - 7.0).abs() < 0.01, "expected 7.0B, got {b}");
    }

    #[test]
    fn hf_config_kv_heads_defaults_to_attention_heads() {
        let cfg = HfConfig {
            num_parameters: None,
            num_hidden_layers: Some(32),
            hidden_size: Some(4096),
            num_attention_heads: Some(32),
            num_key_value_heads: None,
            n_layer: None,
            n_layers: None,
            n_embd: None,
            d_model: None,
            n_head: None,
        };
        assert_eq!(cfg.num_kv_heads(), 32);
    }
}
