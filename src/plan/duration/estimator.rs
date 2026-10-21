//! Training duration and cost.
//!
//! Hardware FLOPs per token, with gradient checkpointing (one extra forward):
//! * full fine-tune: 2N forward + 2N recompute + 4N backward = **8N**
//! * LoRA / QLoRA:   2N + 2N + 2N (input gradients only; frozen weights get
//!   no weight-gradient matmul) = **6N**
//!
//! plus the attention score/value matmuls, `2·L·s·d_attn` per token per
//! forward pass (scaled by the same pass count), where N is the number of
//! *active* parameters (MoE models only run `k` experts per token).
//!
//! Duration = FLOPs ÷ (GPUs × peak × utilization × scaling efficiency). The
//! utilization is a hardware-level figure (it includes recomputation), which
//! is what `calibrate watch` reports from tensor-core activity.

use crate::cli::FinetuneMethod;
use crate::plan::model::ModelSpec;
use crate::plan::{CostRange, DurationRange};

/// Utilization assumed when the user gives none.
pub const DEFAULT_UTILIZATION: f64 = 0.30;
/// Uncertainty band around the point estimate.
const LOW_FACTOR: f64 = 0.8;
const HIGH_FACTOR: f64 = 1.5;

/// Forward-pass equivalents per token (each = 2 FLOPs per active parameter).
fn passes(method: FinetuneMethod) -> f64 {
    match method {
        FinetuneMethod::Full => 4.0,
        FinetuneMethod::Lora | FinetuneMethod::Qlora => 3.0,
    }
}

pub fn flops_per_token(spec: &ModelSpec, method: FinetuneMethod, seq_len: u32) -> f64 {
    let n = spec.active_param_count_b * 1e9;
    let d_attn = spec.num_heads as f64 * spec.head_dim as f64;
    let attn_fwd = 2.0 * spec.num_layers as f64 * seq_len as f64 * d_attn;
    passes(method) * (2.0 * n + attn_fwd)
}

pub fn total_tokens(dataset_rows: u64, seq_len: u32, epochs: u32) -> f64 {
    dataset_rows as f64 * seq_len as f64 * epochs.max(1) as f64
}

/// Data-parallel scaling efficiency within one instance.
pub fn scaling_efficiency(gpu_count: u32) -> f64 {
    match gpu_count {
        0 | 1 => 1.0,
        2..=8 => 0.90,
        _ => 0.85,
    }
}

pub fn estimate_duration_range(
    total_flops: f64,
    peak_tflops_per_gpu: f64,
    gpu_count: u32,
    utilization: f64,
) -> Option<DurationRange> {
    let u = utilization.clamp(0.01, 1.0);
    let n = gpu_count.max(1) as f64;
    let rate = n * peak_tflops_per_gpu * 1e12 * u * scaling_efficiency(gpu_count);
    if rate.is_nan() || rate <= 0.0 || total_flops.is_nan() || total_flops <= 0.0 {
        return None;
    }
    let point = total_flops / rate;
    Some(DurationRange {
        low_secs: point * LOW_FACTOR,
        high_secs: point * HIGH_FACTOR,
    })
}

pub fn cost_range(d: &DurationRange, hourly_usd: f64) -> CostRange {
    CostRange {
        low_usd: d.low_secs / 3600.0 * hourly_usd,
        high_usd: d.high_secs / 3600.0 * hourly_usd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::model::ParamSource;

    fn qwen7b() -> ModelSpec {
        ModelSpec {
            model_id: "Qwen/Qwen2.5-7B".into(),
            param_count_b: 7.62,
            active_param_count_b: 7.62,
            param_source: ParamSource::HubSafetensors,
            num_layers: 28,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 152_064,
            tie_word_embeddings: false,
            architecture_estimated: false,
        }
    }

    #[test]
    fn sequence_length_is_part_of_the_work() {
        // Regression: this used to treat each row as one token, giving
        // "LoRA on a 7B model, 50k rows: under a minute, $0.00".
        let flops =
            flops_per_token(&qwen7b(), FinetuneMethod::Lora, 2048) * total_tokens(50_000, 2048, 1);
        let d = estimate_duration_range(flops, 165.2, 1, 0.30).unwrap();
        let mid_h = (d.low_secs + d.high_secs) / 2.0 / 3600.0;
        // ~1e8 tokens × ~5e10 FLOPs/token ÷ (165 TFLOPS × 0.3) ≈ 30 h
        assert!(mid_h > 15.0 && mid_h < 80.0, "{mid_h} h");
    }

    #[test]
    fn full_costs_a_third_more_than_lora() {
        let s = qwen7b();
        let ratio = flops_per_token(&s, FinetuneMethod::Full, 2048)
            / flops_per_token(&s, FinetuneMethod::Lora, 2048);
        assert!((ratio - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn moe_uses_active_params() {
        let mut s = qwen7b();
        s.param_count_b = 46.7;
        s.active_param_count_b = 12.9;
        let dense = {
            let mut d = s.clone();
            d.active_param_count_b = 46.7;
            flops_per_token(&d, FinetuneMethod::Lora, 2048)
        };
        assert!(flops_per_token(&s, FinetuneMethod::Lora, 2048) < dense / 3.0);
    }

    #[test]
    fn more_gpus_and_higher_utilization_are_faster() {
        let f = 1e18;
        let one = estimate_duration_range(f, 989.0, 1, 0.3).unwrap();
        let eight = estimate_duration_range(f, 989.0, 8, 0.3).unwrap();
        assert!((one.low_secs / eight.low_secs - 8.0 * 0.9).abs() < 1e-6);
        let fast = estimate_duration_range(f, 989.0, 1, 0.6).unwrap();
        assert!((one.low_secs / fast.low_secs - 2.0).abs() < 1e-6);
    }

    #[test]
    fn band_and_cost() {
        let d = estimate_duration_range(1e18, 100.0, 1, 0.5).unwrap();
        assert!((d.high_secs / d.low_secs - HIGH_FACTOR / LOW_FACTOR).abs() < 1e-9);
        let c = cost_range(
            &DurationRange {
                low_secs: 3600.0,
                high_secs: 7200.0,
            },
            2.0,
        );
        assert_eq!((c.low_usd, c.high_usd), (2.0, 4.0));
        assert!(estimate_duration_range(1e18, 0.0, 1, 0.5).is_none());
    }
}
