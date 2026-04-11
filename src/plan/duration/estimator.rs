use crate::gpu_specs::fallback::FallbackRepository;
use crate::gpu_specs::SpecsRepository;
use crate::plan::model::ModelSpec;
use crate::plan::{CostRange, DurationRange};

/// Conservative default MFU when the user hasn't provided a measured value.
/// 30% reflects typical real-world training efficiency on consumer and cloud GPUs.
const DEFAULT_MFU: f64 = 0.30;

/// Estimate training duration for one GPU, expressed as a low/high range.
///
/// Returns `None` when:
/// - `dataset_rows` is not provided (can't estimate steps without it), or
/// - The GPU model is not in the spec catalog.
///
/// The range accounts for real-world variability:
/// - `low_secs` = 80% of the point estimate (GPU slightly above average MFU).
/// - `high_secs` = 150% (data-loading stalls, scheduling overhead, cold starts).
pub fn estimate_duration_range(
    spec: &ModelSpec,
    gpu_model: &str,
    dataset_rows: Option<u64>,
    batch_size: u32,
    epochs: u32,
    mfu_override: Option<f64>,
) -> Option<DurationRange> {
    let dataset_rows = dataset_rows?;
    let mfu = mfu_override.unwrap_or(DEFAULT_MFU).clamp(0.01, 1.0);

    // Look up GPU peak TFLOPS from the baked-in catalog.
    // Uses the same `find_best_match` fuzzy lookup used by `calibrate watch`.
    let gpu_spec = FallbackRepository.get_by_name(gpu_model)?;
    let peak_tflops = gpu_spec.bf16_tflops;

    // Standard FLOP estimate for a transformer training step:
    //   forward pass  ≈ 2 × params   (one multiply-add per weight)
    //   backward pass ≈ 4 × params   (two passes: input grad + weight grad)
    //   total         = 6 × params
    let params = spec.param_count_b * 1e9;
    let flops_per_step = 6.0 * params * batch_size as f64;

    let steps_per_epoch = (dataset_rows as f64 / batch_size as f64).ceil();
    let total_steps = steps_per_epoch * epochs as f64;
    let total_flops = flops_per_step * total_steps;

    let effective_flops_per_sec = peak_tflops * 1e12 * mfu;
    if effective_flops_per_sec <= 0.0 {
        return None;
    }

    let point_estimate_secs = total_flops / effective_flops_per_sec;

    Some(DurationRange {
        low_secs: point_estimate_secs * 0.8,
        high_secs: point_estimate_secs * 1.5,
    })
}

/// Convert a duration range and hourly price to an estimated cost range.
pub fn cost_range(duration: &DurationRange, hourly_usd: f64) -> CostRange {
    CostRange {
        low_usd: duration.low_secs / 3600.0 * hourly_usd,
        high_usd: duration.high_secs / 3600.0 * hourly_usd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::model::ModelSpec;

    fn llama_7b() -> ModelSpec {
        ModelSpec {
            model_id: "llama-7b".to_string(),
            param_count_b: 7.0,
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
        }
    }

    #[test]
    fn none_without_dataset_rows() {
        let r = estimate_duration_range(&llama_7b(), "RTX 3090", None, 1, 1, None);
        assert!(r.is_none());
    }

    #[test]
    fn unknown_gpu_returns_none() {
        let r = estimate_duration_range(&llama_7b(), "NonExistentGPU9999", Some(10_000), 8, 1, None);
        assert!(r.is_none());
    }

    #[test]
    fn known_gpu_produces_finite_positive_range() {
        let r = estimate_duration_range(&llama_7b(), "RTX 3090", Some(10_000), 8, 1, None)
            .expect("RTX 3090 is in the fallback catalog");
        assert!(r.low_secs > 0.0);
        assert!(r.high_secs > r.low_secs);
    }

    #[test]
    fn high_to_low_ratio_is_correct() {
        let r = estimate_duration_range(&llama_7b(), "RTX 3090", Some(10_000), 8, 1, None)
            .unwrap();
        // high = 1.5 × point, low = 0.8 × point  →  ratio = 1.5 / 0.8 = 1.875
        let ratio = r.high_secs / r.low_secs;
        assert!((ratio - 1.875).abs() < 0.001, "expected ratio 1.875, got {ratio}");
    }

    #[test]
    fn higher_mfu_gives_shorter_duration() {
        let low = estimate_duration_range(&llama_7b(), "RTX 3090", Some(10_000), 8, 1, Some(0.20)).unwrap();
        let high = estimate_duration_range(&llama_7b(), "RTX 3090", Some(10_000), 8, 1, Some(0.60)).unwrap();
        assert!(high.low_secs < low.low_secs, "Higher MFU must produce shorter duration");
    }

    #[test]
    fn more_epochs_scales_duration_linearly() {
        let one = estimate_duration_range(&llama_7b(), "RTX 3090", Some(10_000), 8, 1, None).unwrap();
        let three = estimate_duration_range(&llama_7b(), "RTX 3090", Some(10_000), 8, 3, None).unwrap();
        assert!((three.low_secs / one.low_secs - 3.0).abs() < 0.001, "3 epochs should take 3× as long");
    }

    #[test]
    fn cost_range_one_hour_at_one_dollar() {
        let dr = DurationRange { low_secs: 3600.0, high_secs: 7200.0 };
        let cr = cost_range(&dr, 1.0);
        assert!((cr.low_usd - 1.0).abs() < 0.001);
        assert!((cr.high_usd - 2.0).abs() < 0.001);
    }
}
