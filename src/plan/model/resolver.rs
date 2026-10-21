//! Resolve a model ID or local path to a [`ModelSpec`].
//!
//! Parameter counts come, in order of preference, from `--params-b`, the
//! Hugging Face Hub's safetensors metadata (exact, and public even for gated
//! repositories), local `.safetensors` headers (exact), or a formula over
//! `config.json` that accounts for GQA, gated MLPs, vocabulary embeddings and
//! mixture-of-experts layers.

use std::path::Path;

use anyhow::{bail, Context, Result};
use reqwest::StatusCode;

use super::{ModelSpec, ParamSource};

#[derive(Debug, Default, Clone)]
pub struct HfConfig {
    pub model_type: Option<String>,
    pub num_parameters: Option<u64>,
    pub num_hidden_layers: Option<u32>,
    pub hidden_size: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub vocab_size: Option<u32>,
    pub tie_word_embeddings: Option<bool>,
    pub hidden_act: Option<String>,
    pub num_local_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub moe_intermediate_size: Option<u32>,
    /// Multimodal configs nest the language model here.
    pub text_config: Option<Box<HfConfig>>,
}

impl HfConfig {
    /// Parse tolerantly: the first present key of each alias list wins, so
    /// configs that carry several spellings of a field never fail to parse.
    pub fn from_value(v: &serde_json::Value) -> Self {
        let u = |keys: &[&str]| {
            keys.iter()
                .find_map(|k| v.get(*k).and_then(|x| x.as_u64()))
                .and_then(|x| u32::try_from(x).ok())
        };
        let st = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        Self {
            model_type: st("model_type"),
            num_parameters: v.get("num_parameters").and_then(|x| x.as_u64()),
            num_hidden_layers: u(&["num_hidden_layers", "n_layer", "n_layers", "num_layers"]),
            hidden_size: u(&["hidden_size", "n_embd", "d_model"]),
            intermediate_size: u(&["intermediate_size", "n_inner", "ffn_dim", "d_ff"]),
            num_attention_heads: u(&["num_attention_heads", "n_head", "num_heads"]),
            num_key_value_heads: u(&[
                "num_key_value_heads",
                "num_kv_heads",
                "multi_query_group_num",
            ]),
            head_dim: u(&["head_dim"]),
            vocab_size: u(&["vocab_size"]),
            tie_word_embeddings: v.get("tie_word_embeddings").and_then(|x| x.as_bool()),
            hidden_act: st("hidden_act").or_else(|| st("activation_function")),
            num_local_experts: u(&["num_local_experts", "num_experts", "n_routed_experts"]),
            num_experts_per_tok: u(&["num_experts_per_tok", "num_experts_per_token"]),
            moe_intermediate_size: u(&["moe_intermediate_size"]),
            text_config: v
                .get("text_config")
                .filter(|t| t.is_object())
                .map(|t| Box::new(Self::from_value(t))),
        }
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        if !v.is_object() {
            bail!("config.json is not a JSON object");
        }
        Ok(Self::from_value(&v))
    }
}

impl HfConfig {
    /// Use the nested language-model config for multimodal models.
    fn effective(self) -> HfConfig {
        match self.text_config {
            Some(t) if self.num_hidden_layers.is_none() => *t,
            _ => self,
        }
    }

    fn layers(&self) -> u32 {
        self.num_hidden_layers.unwrap_or(32)
    }
    fn hidden(&self) -> u32 {
        self.hidden_size.unwrap_or(4096)
    }
    fn heads(&self) -> u32 {
        self.num_attention_heads.unwrap_or(32).max(1)
    }
    fn kv_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or_else(|| self.heads())
    }
    fn head_dim(&self) -> u32 {
        self.head_dim.unwrap_or(self.hidden() / self.heads())
    }
    fn ffn(&self) -> u32 {
        self.intermediate_size.unwrap_or(self.hidden() * 4)
    }
    fn vocab(&self) -> u32 {
        self.vocab_size.unwrap_or(32_000)
    }
    fn tied(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }
    /// SwiGLU/GeGLU-style MLPs have three projection matrices.
    fn gated_mlp(&self) -> bool {
        let act = self.hidden_act.as_deref().unwrap_or("").to_lowercase();
        let mt = self.model_type.as_deref().unwrap_or("").to_lowercase();
        let classic = [
            "gpt2", "gpt_neox", "gptj", "opt", "bloom", "falcon", "bert", "roberta",
        ];
        if classic.iter().any(|m| mt == *m) {
            return false;
        }
        act.contains("silu") || act.contains("swiglu") || act.contains("geglu") || !mt.is_empty()
    }

    /// (total, active) parameter counts from the architecture.
    pub fn formula_params(&self) -> (f64, f64) {
        let h = self.hidden() as f64;
        let l = self.layers() as f64;
        let q = self.heads() as f64 * self.head_dim() as f64;
        let kv = self.kv_heads() as f64 * self.head_dim() as f64;
        let attn = h * q + 2.0 * h * kv + q * h;
        let mats = if self.gated_mlp() { 3.0 } else { 2.0 };
        let emb = self.vocab() as f64 * h * if self.tied() { 1.0 } else { 2.0 };
        let norms = 2.0 * h;
        match (self.num_local_experts, self.num_experts_per_tok) {
            (Some(e), k) if e > 1 => {
                let expert_ffn = self.moe_intermediate_size.unwrap_or(self.ffn()) as f64;
                let expert = mats * h * expert_ffn;
                let k = k.unwrap_or(2).min(e) as f64;
                let router = h * e as f64;
                let total = emb + l * (attn + e as f64 * expert + router + norms) + h;
                let active = emb + l * (attn + k * expert + router + norms) + h;
                (total, active)
            }
            _ => {
                let mlp = mats * h * self.ffn() as f64;
                let total = emb + l * (attn + mlp + norms) + h;
                (total, total)
            }
        }
    }

    fn into_spec(self, model_id: &str, total_override: Option<(f64, ParamSource)>) -> ModelSpec {
        let (formula_total, formula_active) = self.formula_params();
        let (total, source) = match total_override {
            Some((t, s)) => (t, s),
            None => match self.num_parameters {
                Some(n) => (n as f64, ParamSource::ConfigFormula),
                None => (formula_total, ParamSource::ConfigFormula),
            },
        };
        // Keep the active/total ratio from the architecture for MoE models.
        let active = if formula_total > 0.0 {
            total * (formula_active / formula_total)
        } else {
            total
        };
        ModelSpec {
            model_id: model_id.to_string(),
            param_count_b: total / 1e9,
            active_param_count_b: active / 1e9,
            param_source: source,
            num_layers: self.layers(),
            hidden_size: self.hidden(),
            intermediate_size: self.ffn(),
            num_heads: self.heads(),
            num_kv_heads: self.kv_heads(),
            head_dim: self.head_dim(),
            vocab_size: self.vocab(),
            tie_word_embeddings: self.tied(),
            architecture_estimated: self.num_hidden_layers.is_none(),
        }
    }
}

/// Sum tensor element counts from a `.safetensors` file header (exact).
pub fn safetensors_param_count(path: &Path) -> Result<u64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > 100 * 1024 * 1024 {
        bail!(
            "{}: implausible safetensors header size {n}",
            path.display()
        );
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)?;
    let v: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("{}: invalid safetensors header", path.display()))?;
    let obj = v
        .as_object()
        .context("safetensors header is not an object")?;
    let mut total = 0u64;
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        let shape = t["shape"].as_array().context("tensor without shape")?;
        total += shape
            .iter()
            .map(|d| d.as_u64().unwrap_or(0))
            .product::<u64>();
    }
    Ok(total)
}

fn local_safetensors_total(dir: &Path) -> Option<u64> {
    let mut total = 0u64;
    let mut any = false;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "safetensors") {
            total += safetensors_param_count(&p).ok()?;
            any = true;
        }
    }
    any.then_some(total)
}

/// Architecture guess when only the parameter count is known.
pub fn infer_spec_from_params(model_id: &str, params_b: f64, source: ParamSource) -> ModelSpec {
    let (l, h, nh, kv, ffn) = match params_b {
        p if p < 2.0 => (24u32, 2048u32, 16u32, 8u32, 5632u32),
        p if p < 5.0 => (28, 3072, 24, 8, 8192),
        p if p < 10.0 => (32, 4096, 32, 8, 14336),
        p if p < 20.0 => (40, 5120, 40, 8, 13824),
        p if p < 40.0 => (64, 5120, 40, 8, 27648),
        _ => (80, 8192, 64, 8, 28672),
    };
    ModelSpec {
        model_id: model_id.to_string(),
        param_count_b: params_b,
        active_param_count_b: params_b,
        param_source: source,
        num_layers: l,
        hidden_size: h,
        intermediate_size: ffn,
        num_heads: nh,
        num_kv_heads: kv,
        head_dim: h / nh,
        vocab_size: 128_256,
        tie_word_embeddings: false,
        architecture_estimated: true,
    }
}

pub async fn resolve(model_id: &str, params_b_override: Option<f64>) -> Result<ModelSpec> {
    let ov = params_b_override.map(|p| (p * 1e9, ParamSource::Override));

    // Local directory or config file.
    let local = Path::new(model_id);
    if local.exists() {
        let (dir, cfg_path) = if local.is_dir() {
            (local.to_path_buf(), local.join("config.json"))
        } else {
            (
                local.parent().unwrap_or(Path::new(".")).to_path_buf(),
                local.to_path_buf(),
            )
        };
        if cfg_path.exists() {
            let cfg = HfConfig::from_json(
                &std::fs::read_to_string(&cfg_path)
                    .with_context(|| format!("reading {}", cfg_path.display()))?,
            )
            .with_context(|| format!("parsing {}", cfg_path.display()))?;
            let exact = ov.or_else(|| {
                local_safetensors_total(&dir).map(|t| (t as f64, ParamSource::LocalSafetensors))
            });
            return Ok(cfg.effective().into_spec(model_id, exact));
        }
        if let Some((t, s)) = ov {
            return Ok(infer_spec_from_params(model_id, t / 1e9, s));
        }
        bail!(
            "{} has no config.json — pass `--params-b <BILLIONS>`",
            local.display()
        );
    }

    let settings = crate::config::get();
    if settings.offline {
        if let Some((t, s)) = ov {
            return Ok(infer_spec_from_params(model_id, t / 1e9, s));
        }
        bail!(
            "--offline is set: pass a local model directory or `--params-b <BILLIONS>` for '{model_id}'"
        );
    }

    let client = crate::net::client(None)?;
    let auth = |rb: reqwest::RequestBuilder| match &settings.hf_token {
        Some(t) => rb.bearer_auth(t),
        None => rb,
    };
    let base = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());

    // 1. Exact parameter count from the Hub API (public even for gated repos).
    let mut exact = ov;
    if exact.is_none() {
        let url = format!("{base}/api/models/{model_id}");
        match auth(client.get(&url)).send().await {
            Ok(r) if r.status().is_success() => {
                if let Ok(v) = r.json::<serde_json::Value>().await {
                    if let Some(t) = v["safetensors"]["total"].as_u64().filter(|t| *t > 0) {
                        exact = Some((t as f64, ParamSource::HubSafetensors));
                    }
                }
            }
            Ok(r) if r.status() == StatusCode::NOT_FOUND => bail!(
                "model '{model_id}' does not exist on the Hugging Face Hub (or is private — \
                 set HF_TOKEN). Check the ID or pass a local path."
            ),
            Ok(r) => tracing::debug!("HF API {url}: {}", r.status()),
            Err(e) => {
                if exact.is_none() {
                    bail!(
                        "could not reach the Hugging Face Hub ({e}). Pass `--params-b <BILLIONS>`, \
                         a local model path, or set HTTPS_PROXY if you are behind a proxy."
                    );
                }
            }
        }
    }

    // 2. Architecture from config.json (needs a token for gated repos).
    let url = format!("{base}/{model_id}/resolve/main/config.json");
    match auth(client.get(&url)).send().await {
        Ok(r) if r.status().is_success() => {
            let text = r.text().await?;
            let cfg = HfConfig::from_json(&text)
                .with_context(|| format!("parsing config.json for {model_id}"))?;
            Ok(cfg.effective().into_spec(model_id, exact))
        }
        Ok(r) if matches!(r.status(), StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => {
            match exact {
                Some((t, s)) => {
                    tracing::warn!(
                        "config.json for '{model_id}' is gated — architecture estimated from the \
                         parameter count. Set HF_TOKEN (and accept the licence) for exact figures."
                    );
                    Ok(infer_spec_from_params(model_id, t / 1e9, s))
                }
                None => bail!(
                    "'{model_id}' is gated or private (HTTP {}). Accept its licence on \
                     huggingface.co and set HF_TOKEN, or pass `--params-b <BILLIONS>`.",
                    r.status()
                ),
            }
        }
        Ok(r) => match exact {
            Some((t, s)) => Ok(infer_spec_from_params(model_id, t / 1e9, s)),
            None => bail!(
                "could not fetch config.json for '{model_id}' (HTTP {}). Pass `--params-b <BILLIONS>`.",
                r.status()
            ),
        },
        Err(e) => match exact {
            Some((t, s)) => Ok(infer_spec_from_params(model_id, t / 1e9, s)),
            None => bail!("could not reach the Hugging Face Hub ({e}). Pass `--params-b <BILLIONS>`."),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(json: &str) -> HfConfig {
        HfConfig::from_json(json).unwrap().effective()
    }

    #[test]
    fn qwen2_5_7b_formula_matches_published_count() {
        // Published: 7.62B parameters.
        let c = cfg(
            r#"{"model_type":"qwen2","hidden_act":"silu","hidden_size":3584,
            "intermediate_size":18944,"num_attention_heads":28,"num_hidden_layers":28,
            "num_key_value_heads":4,"vocab_size":152064,"tie_word_embeddings":false}"#,
        );
        let (total, active) = c.formula_params();
        assert!((total / 1e9 - 7.62).abs() < 0.05, "{}", total / 1e9);
        assert_eq!(total, active);
    }

    #[test]
    fn llama3_8b_formula_matches_published_count() {
        // Published: 8.03B parameters.
        let c = cfg(
            r#"{"model_type":"llama","hidden_act":"silu","hidden_size":4096,
            "intermediate_size":14336,"num_attention_heads":32,"num_hidden_layers":32,
            "num_key_value_heads":8,"vocab_size":128256}"#,
        );
        let (total, _) = c.formula_params();
        assert!((total / 1e9 - 8.03).abs() < 0.05, "{}", total / 1e9);
    }

    #[test]
    fn mixtral_moe_total_and_active() {
        // Published: 46.7B total, ~12.9B active.
        let c = cfg(
            r#"{"model_type":"mixtral","hidden_act":"silu","hidden_size":4096,
            "intermediate_size":14336,"num_attention_heads":32,"num_hidden_layers":32,
            "num_key_value_heads":8,"vocab_size":32000,"num_local_experts":8,
            "num_experts_per_tok":2}"#,
        );
        let (total, active) = c.formula_params();
        assert!((total / 1e9 - 46.7).abs() < 0.3, "{}", total / 1e9);
        assert!((active / 1e9 - 12.9).abs() < 0.3, "{}", active / 1e9);
    }

    #[test]
    fn gpt2_uses_classic_mlp_and_aliases() {
        // GPT-2 small: 124M with tied embeddings.
        let c = cfg(
            r#"{"model_type":"gpt2","n_embd":768,"n_layer":12,"n_head":12,
            "vocab_size":50257,"tie_word_embeddings":true}"#,
        );
        let (total, _) = c.formula_params();
        assert!((total / 1e6 - 124.0).abs() < 2.0, "{}", total / 1e6);
    }

    #[test]
    fn multimodal_config_uses_text_config() {
        let c = cfg(
            r#"{"model_type":"llava","text_config":{"model_type":"llama",
            "hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,
            "intermediate_size":11008,"vocab_size":32000}}"#,
        );
        assert_eq!(c.layers(), 32);
    }

    #[test]
    fn exact_count_overrides_formula_but_keeps_moe_ratio() {
        let c = cfg(
            r#"{"model_type":"mixtral","hidden_size":4096,"intermediate_size":14336,
            "num_attention_heads":32,"num_hidden_layers":32,"num_key_value_heads":8,
            "vocab_size":32000,"num_local_experts":8,"num_experts_per_tok":2}"#,
        );
        let s = c.into_spec("m", Some((46.7e9, ParamSource::HubSafetensors)));
        assert_eq!(s.param_source, ParamSource::HubSafetensors);
        assert!((s.param_count_b - 46.7).abs() < 1e-9);
        assert!(s.active_param_count_b < 14.0 && s.active_param_count_b > 12.0);
    }

    #[test]
    fn safetensors_header_is_counted_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let header = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]},"b":{"dtype":"BF16","shape":[10],"data_offsets":[64,84]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend(vec![0u8; 84]);
        std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":1}"#,
        )
        .unwrap();
        assert_eq!(local_safetensors_total(dir.path()), Some(42));
        let rt = tokio::runtime::Runtime::new().unwrap();
        let spec = rt
            .block_on(resolve(dir.path().to_str().unwrap(), None))
            .unwrap();
        assert_eq!(spec.param_source, ParamSource::LocalSafetensors);
        assert!((spec.param_count_b - 42e-9).abs() < 1e-15);
    }

    #[test]
    fn duplicate_key_spellings_do_not_break_parsing() {
        let c = cfg(r#"{"n_head":71,"num_attention_heads":71,"n_layer":32,"hidden_size":4544}"#);
        assert_eq!(c.heads(), 71);
        assert_eq!(c.layers(), 32);
        assert!(HfConfig::from_json("[1,2]").is_err());
    }

    #[test]
    fn inferred_arch_is_flagged() {
        let s = infer_spec_from_params("x", 7.0, ParamSource::Override);
        assert!(s.architecture_estimated);
        assert_eq!(s.num_layers, 32);
    }
}
