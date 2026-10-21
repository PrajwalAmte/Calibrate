//! Per-GPU VRAM estimate for fine-tuning.
//!
//! Model (bytes):
//! * weights: 2/param bf16, 1 for int8, ~0.53 for NF4 (incl. quantisation
//!   constants); embeddings and LM head stay 16-bit when quantised.
//! * full fine-tune, mixed-precision AdamW: grads 2 + fp32 master/m/v 12
//!   bytes per param (the "16 bytes/param" rule). LoRA adapters (rank 16 on
//!   every linear layer) are trained in fp32: 4 + 12 bytes per adapter param.
//! * activations with gradient checkpointing: one bf16 checkpoint per layer
//!   plus one layer's recompute working set, `b·s·(16h + 6f)` bytes with
//!   FlashAttention/SDPA (no materialised attention scores).
//! * logits: `b·s·V` in bf16 plus the fp32 copy used by cross-entropy.
//! * overhead: CUDA context + workspaces (~0.75 GiB) and 5 % fragmentation.

use anyhow::{bail, Result};

use crate::cli::{FinetuneMethod, OptimizerLib, QuantLevel};
use crate::plan::model::ModelSpec;
use crate::plan::{TrainingSetup, VramBreakdown};

const GIB: f64 = 1_073_741_824.0;
pub const LORA_RANK: f64 = 16.0;
const CUDA_CONTEXT_GIB: f64 = 0.75;
const FRAGMENTATION: f64 = 0.05;

/// Reject combinations no framework supports.
pub fn validate(setup: &TrainingSetup) -> Result<()> {
    if setup.method == FinetuneMethod::Full && setup.quantization != QuantLevel::None {
        bail!(
            "full fine-tuning of a quantized model is not supported by mainstream frameworks — \
             use `--method qlora` (4-bit base + adapters) or drop `--quantization`"
        );
    }
    if setup.method == FinetuneMethod::Qlora && setup.quantization == QuantLevel::EightBit {
        bail!("QLoRA uses a 4-bit base model; use `--method lora --quantization 8bit` for 8-bit");
    }
    Ok(())
}

/// Effective weight quantisation (QLoRA implies 4-bit).
pub fn effective_quant(setup: &TrainingSetup) -> QuantLevel {
    if setup.method == FinetuneMethod::Qlora {
        QuantLevel::FourBit
    } else {
        setup.quantization
    }
}

/// LoRA adapter parameters for rank `LORA_RANK` on q,k,v,o and the MLP projections.
pub fn lora_params(spec: &ModelSpec) -> f64 {
    let h = spec.hidden_size as f64;
    let q = spec.num_heads as f64 * spec.head_dim as f64;
    let kv = spec.num_kv_heads as f64 * spec.head_dim as f64;
    let f = spec.intermediate_size as f64;
    let per_layer = (h + q) + 2.0 * (h + kv) + (q + h) + 3.0 * (h + f);
    LORA_RANK * per_layer * spec.num_layers as f64
}

pub fn estimate(spec: &ModelSpec, setup: &TrainingSetup, gpu_count: u32) -> VramBreakdown {
    let n = gpu_count.max(1) as f64;
    let params = spec.param_count_b * 1e9;
    let emb = spec.embedding_params().min(params);
    let quant = effective_quant(setup);
    let mut notes = Vec::new();

    let weights_bytes = match quant {
        QuantLevel::None => params * 2.0,
        QuantLevel::EightBit => (params - emb) * 1.0 + emb * 2.0,
        QuantLevel::FourBit => (params - emb) * 0.53 + emb * 2.0,
    };

    let (grad_bytes, opt_bytes) = match setup.method {
        FinetuneMethod::Full => (params * 2.0, params * 12.0),
        FinetuneMethod::Lora | FinetuneMethod::Qlora => {
            let a = lora_params(spec);
            notes.push(format!(
                "LoRA rank {LORA_RANK:.0} on all linear layers ({:.0}M trainable params)",
                a / 1e6
            ));
            (a * 4.0, a * 12.0)
        }
    };

    let b = setup.micro_batch as f64;
    let s = setup.seq_len as f64;
    let h = spec.hidden_size as f64;
    let f = spec.intermediate_size as f64;
    let l = spec.num_layers as f64;
    let mut act_bytes = l * b * s * h * 2.0 + b * s * (16.0 * h + 6.0 * f);
    let mut logit_bytes = b * s * spec.vocab_size as f64 * (2.0 + 4.0);
    notes.push(format!(
        "gradient checkpointing + FlashAttention, {} tokens × micro-batch {}",
        setup.seq_len, setup.micro_batch
    ));

    let (mut w, mut g, mut o) = (weights_bytes, grad_bytes, opt_bytes);
    match setup.optimizer_lib {
        OptimizerLib::DeepSpeed if n > 1.0 => {
            w /= n;
            g /= n;
            o /= n;
            notes.push(format!(
                "DeepSpeed ZeRO-3: weights, gradients and optimizer state sharded over {n:.0} GPUs"
            ));
        }
        OptimizerLib::DeepSpeed => {
            o = 0.0;
            notes.push("DeepSpeed ZeRO-Offload: optimizer state in CPU RAM".into());
        }
        OptimizerLib::Unsloth => {
            act_bytes *= 0.5;
            logit_bytes *= 0.25;
            notes.push(
                "Unsloth: activation offloading and chunked cross-entropy (≈−50 % activations, \
                 −75 % logits)"
                    .into(),
            );
        }
        OptimizerLib::None => {}
    }

    let subtotal = w + g + o + act_bytes + logit_bytes;
    let overhead = CUDA_CONTEXT_GIB * GIB + subtotal * FRAGMENTATION;
    VramBreakdown {
        gpu_count: gpu_count.max(1),
        weights_gib: w / GIB,
        gradients_gib: g / GIB,
        optimizer_gib: o / GIB,
        activations_gib: act_bytes / GIB,
        logits_gib: logit_bytes / GIB,
        overhead_gib: overhead / GIB,
        total_gib: (subtotal + overhead) / GIB,
        notes,
    }
}

/// Common GPU memory sizes (GiB).
const VRAM_TIERS: &[u32] = &[8, 12, 16, 24, 32, 40, 48, 80, 94, 141, 180];

pub fn fitting_tiers(required_gib: f64) -> Vec<String> {
    VRAM_TIERS
        .iter()
        .filter(|&&t| t as f64 >= required_gib)
        .map(|&t| format!("{t}GB"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::model::ParamSource;

    fn llama8b() -> ModelSpec {
        ModelSpec {
            model_id: "meta-llama/Llama-3.1-8B".into(),
            param_count_b: 8.03,
            active_param_count_b: 8.03,
            param_source: ParamSource::HubSafetensors,
            num_layers: 32,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 128_256,
            tie_word_embeddings: false,
            architecture_estimated: false,
        }
    }

    fn setup(method: FinetuneMethod, lib: OptimizerLib, quant: QuantLevel) -> TrainingSetup {
        TrainingSetup {
            method,
            optimizer_lib: lib,
            quantization: quant,
            micro_batch: 1,
            seq_len: 2048,
        }
    }

    #[test]
    fn full_finetune_follows_16_bytes_per_param() {
        let v = estimate(
            &llama8b(),
            &setup(FinetuneMethod::Full, OptimizerLib::None, QuantLevel::None),
            1,
        );
        let states = v.weights_gib + v.gradients_gib + v.optimizer_gib;
        assert!((states - 8.03e9 * 16.0 / GIB).abs() < 0.01);
        assert!(
            v.total_gib > 120.0,
            "8B full FT needs >1 80 GB GPU: {}",
            v.total_gib
        );
    }

    #[test]
    fn lora_8b_fits_24gb_and_qlora_fits_less() {
        let lora = estimate(
            &llama8b(),
            &setup(FinetuneMethod::Lora, OptimizerLib::None, QuantLevel::None),
            1,
        );
        assert!(
            lora.total_gib > 16.0 && lora.total_gib < 24.0,
            "{}",
            lora.total_gib
        );
        let q = estimate(
            &llama8b(),
            &setup(FinetuneMethod::Qlora, OptimizerLib::None, QuantLevel::None),
            1,
        );
        assert!(q.total_gib < 12.0 && q.total_gib > 5.0, "{}", q.total_gib);
        assert!(q.weights_gib < lora.weights_gib / 2.5);
    }

    #[test]
    fn logits_matter_for_large_vocab() {
        let v = estimate(
            &llama8b(),
            &setup(FinetuneMethod::Lora, OptimizerLib::None, QuantLevel::None),
            1,
        );
        // 2048 × 128256 × 6 bytes ≈ 1.47 GiB
        assert!((v.logits_gib - 1.47).abs() < 0.02, "{}", v.logits_gib);
    }

    #[test]
    fn deepspeed_shards_across_gpus() {
        let plain = estimate(
            &llama8b(),
            &setup(FinetuneMethod::Full, OptimizerLib::None, QuantLevel::None),
            8,
        );
        let s = setup(
            FinetuneMethod::Full,
            OptimizerLib::DeepSpeed,
            QuantLevel::None,
        );
        let one = estimate(&llama8b(), &s, 1);
        let eight = estimate(&llama8b(), &s, 8);
        assert!(
            eight.total_gib < plain.total_gib / 5.0,
            "replicated DDP does not shard"
        );
        assert!(eight.total_gib < one.total_gib);
        assert!(
            one.optimizer_gib == 0.0,
            "single-GPU DeepSpeed offloads optimizer state"
        );
        assert!(eight.total_gib < 40.0, "{}", eight.total_gib);
    }

    #[test]
    fn unsloth_reduces_activation_memory() {
        let base = estimate(
            &llama8b(),
            &setup(FinetuneMethod::Lora, OptimizerLib::None, QuantLevel::None),
            1,
        );
        let un = estimate(
            &llama8b(),
            &setup(
                FinetuneMethod::Lora,
                OptimizerLib::Unsloth,
                QuantLevel::None,
            ),
            1,
        );
        assert!(un.total_gib < base.total_gib);
        assert_eq!(un.weights_gib, base.weights_gib);
    }

    #[test]
    fn invalid_combinations_are_rejected() {
        assert!(validate(&setup(
            FinetuneMethod::Full,
            OptimizerLib::None,
            QuantLevel::FourBit
        ))
        .is_err());
        assert!(validate(&setup(
            FinetuneMethod::Qlora,
            OptimizerLib::None,
            QuantLevel::EightBit
        ))
        .is_err());
        assert!(validate(&setup(
            FinetuneMethod::Lora,
            OptimizerLib::None,
            QuantLevel::EightBit
        ))
        .is_ok());
    }

    #[test]
    fn tiers() {
        assert_eq!(fitting_tiers(20.0)[0], "24GB");
        assert!(fitting_tiers(500.0).is_empty());
    }
}
