use crate::cli::{FinetuneMethod, OptimizerLib, QuantLevel};
use crate::plan::model::ModelSpec;
use crate::plan::VramBreakdown;

const GIB: f64 = 1_073_741_824.0; // 2^30 bytes

/// Default training sequence length used for activation/KV-cache estimates.
/// 2048 is a conservative pick that covers most fine-tuning scenarios.
const DEFAULT_SEQ_LEN: u32 = 2048;

/// LoRA adapter parameters as a fraction of total model parameters (~1%).
const LORA_ADAPTER_FRACTION: f64 = 0.01;

/// Factor by which gradient checkpointing reduces activation memory.
/// Checkpointing stores only O(√L) activations rather than all L layers.
const GRAD_CKPT_FACTOR: f64 = 4.0;

/// Estimate the full VRAM breakdown for a training run.
///
/// All returned values are in GiB. The caller should add a safety margin
/// (e.g. 5%) on top of `total_gib` before comparing against GPU VRAM.
///
/// Assumptions:
/// - Gradient checkpointing is always enabled (reduces activation memory ~4×).
/// - Sequence length is `DEFAULT_SEQ_LEN` (2048 tokens).
/// - Loss scaling for mixed precision is not counted (negligible).
pub fn estimate(
    spec: &ModelSpec,
    method: FinetuneMethod,
    optimizer_lib: OptimizerLib,
    quant: QuantLevel,
    batch_size: u32,
) -> VramBreakdown {
    let params = spec.param_count_b * 1e9;
    let bytes_per_weight = bytes_per_param(quant);

    // ── Model weights──
    let weights_gib = params * bytes_per_weight / GIB;

    // ── Trainable parameter count ──────────────────────────────────────────────
    // Full fine-tuning: all parameters have gradients.
    // LoRA / QLoRA: only the small adapter matrices are trained (~1% of total).
    let trainable_params = match method {
        FinetuneMethod::Full => params,
        FinetuneMethod::Lora | FinetuneMethod::Qlora => params * LORA_ADAPTER_FRACTION,
    };

    // ── Gradients (bf16 — 2 bytes per trainable parameter)────────
    let gradients_gib = trainable_params * 2.0 / GIB;

    // ── Optimizer states (fp32 Adam: first + second moment = 8 bytes/param)
    let optimizer_gib = trainable_params * 8.0 / GIB;

    // ── Activations (gradient checkpointing assumed)─────────
    // Without checkpointing: batch × seq × hidden × layers × 2 bytes.
    // With checkpointing:    divide by GRAD_CKPT_FACTOR (~4× reduction).
    let seq_len = DEFAULT_SEQ_LEN as f64;
    let activations_gib = batch_size as f64
        * seq_len
        * spec.hidden_size as f64
        * spec.num_layers as f64
        * 2.0 // bf16
        / GRAD_CKPT_FACTOR
        / GIB;

    // ── KV cache────────────
    // 2 (K+V) × layers × kv_heads × head_dim × seq_len × batch × 2 bytes (bf16)
    let head_dim = spec.hidden_size as f64 / spec.num_heads.max(1) as f64;
    let kv_cache_gib = 2.0
        * spec.num_layers as f64
        * spec.num_kv_heads as f64
        * head_dim
        * seq_len
        * batch_size as f64
        * 2.0 // bf16
        / GIB;

    // ── Sub-total before library savings─────────
    let subtotal = weights_gib + gradients_gib + optimizer_gib + activations_gib + kv_cache_gib;

    // ── Library efficiency savings───────────
    // Negative value — represents memory freed relative to the sub-total.
    let library_savings_gib = match optimizer_lib {
        // Unsloth: memory-efficient attention + optimized quantization, ~45% reduction.
        OptimizerLib::Unsloth => -subtotal * 0.45,
        // DeepSpeed ZeRO-2: shards optimizer states + gradients — conservative 20% on
        // single GPU (most benefit comes from multi-GPU sharding, not counted here).
        OptimizerLib::DeepSpeed => -subtotal * 0.20,
        OptimizerLib::None => 0.0,
    };

    let total_gib = subtotal + library_savings_gib;

    VramBreakdown {
        weights_gib,
        gradients_gib,
        optimizer_gib,
        activations_gib,
        kv_cache_gib,
        library_savings_gib,
        total_gib,
    }
}

/// Return the bytes required to store one model parameter at the given quantization.
fn bytes_per_param(quant: QuantLevel) -> f64 {
    match quant {
        QuantLevel::None => 2.0,     // bfloat16
        QuantLevel::EightBit => 1.0, // int8
        QuantLevel::FourBit => 0.5,  // 4-bit NF4
    }
}

/// Standard GPU VRAM tiers in GiB.
const VRAM_TIERS: &[u32] = &[8, 10, 12, 16, 20, 24, 40, 48, 80];

/// Return all standard VRAM tiers (as "NGB" strings) that are large enough to
/// fit `required_gib` with a 5% safety margin.
pub fn fitting_tiers(required_gib: f64) -> Vec<String> {
    let with_margin = required_gib * 1.05;
    VRAM_TIERS
        .iter()
        .filter(|&&t| t as f64 >= with_margin)
        .map(|&t| format!("{t}GB"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::model::ModelSpec;

    fn llama_7b() -> ModelSpec {
        ModelSpec {
            model_id: "meta-llama/Llama-2-7b".to_string(),
            param_count_b: 7.0,
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
        }
    }

    #[test]
    fn full_precision_7b_weights_approx_13gib() {
        // 7 × 10^9 × 2 bytes / 2^30 ≈ 13.04 GiB
        let bd = estimate(
            &llama_7b(),
            FinetuneMethod::Full,
            OptimizerLib::None,
            QuantLevel::None,
            1,
        );
        assert!(
            (bd.weights_gib - 13.04).abs() < 0.1,
            "expected ~13.0 GiB, got {:.2}",
            bd.weights_gib
        );
    }

    #[test]
    fn fourbit_weights_half_of_eightbit() {
        let bd8 = estimate(
            &llama_7b(),
            FinetuneMethod::Lora,
            OptimizerLib::None,
            QuantLevel::EightBit,
            1,
        );
        let bd4 = estimate(
            &llama_7b(),
            FinetuneMethod::Lora,
            OptimizerLib::None,
            QuantLevel::FourBit,
            1,
        );
        assert!(
            (bd8.weights_gib / bd4.weights_gib - 2.0).abs() < 0.01,
            "4-bit weights should be exactly half of 8-bit"
        );
    }

    #[test]
    fn lora_gradients_are_about_1pct_of_full() {
        let bd_full = estimate(
            &llama_7b(),
            FinetuneMethod::Full,
            OptimizerLib::None,
            QuantLevel::None,
            1,
        );
        let bd_lora = estimate(
            &llama_7b(),
            FinetuneMethod::Lora,
            OptimizerLib::None,
            QuantLevel::None,
            1,
        );
        let ratio = bd_lora.gradients_gib / bd_full.gradients_gib;
        assert!(
            (ratio - LORA_ADAPTER_FRACTION).abs() < 0.001,
            "LoRA gradient ratio should equal LORA_ADAPTER_FRACTION, got {ratio:.4}"
        );
    }

    #[test]
    fn unsloth_reduces_total_by_40_to_60_pct() {
        let bd_none = estimate(
            &llama_7b(),
            FinetuneMethod::Lora,
            OptimizerLib::None,
            QuantLevel::FourBit,
            1,
        );
        let bd_unsloth = estimate(
            &llama_7b(),
            FinetuneMethod::Lora,
            OptimizerLib::Unsloth,
            QuantLevel::FourBit,
            1,
        );
        let reduction = 1.0 - bd_unsloth.total_gib / bd_none.total_gib;
        assert!(
            (0.40..=0.60).contains(&reduction),
            "Unsloth should reduce by 40–60%, got {:.1}%",
            reduction * 100.0
        );
    }

    #[test]
    fn library_savings_is_negative() {
        let bd = estimate(
            &llama_7b(),
            FinetuneMethod::Lora,
            OptimizerLib::Unsloth,
            QuantLevel::FourBit,
            1,
        );
        assert!(
            bd.library_savings_gib < 0.0,
            "savings should be expressed as a negative delta"
        );
    }

    #[test]
    fn total_equals_sum_of_components() {
        let bd = estimate(
            &llama_7b(),
            FinetuneMethod::Full,
            OptimizerLib::None,
            QuantLevel::None,
            1,
        );
        let computed = bd.weights_gib
            + bd.gradients_gib
            + bd.optimizer_gib
            + bd.activations_gib
            + bd.kv_cache_gib
            + bd.library_savings_gib;
        assert!((computed - bd.total_gib).abs() < 1e-9, "sum mismatch");
    }

    #[test]
    fn fitting_tiers_excludes_insufficient_tiers() {
        let tiers = fitting_tiers(11.0); // 11 × 1.05 = 11.55 — needs ≥ 12 GB
        assert!(tiers.contains(&"12GB".to_string()));
        assert!(!tiers.contains(&"10GB".to_string()));
        assert!(!tiers.contains(&"8GB".to_string()));
    }

    #[test]
    fn fitting_tiers_all_for_tiny_requirement() {
        let tiers = fitting_tiers(1.0);
        assert_eq!(
            tiers.len(),
            VRAM_TIERS.len(),
            "all tiers fit a 1 GiB requirement"
        );
    }
}
