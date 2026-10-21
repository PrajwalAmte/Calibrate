//! One actionable recommendation per detected bottleneck.
//!
//! Gains are only quoted where they follow from the measurements (removing
//! input-pipeline wait scales throughput by 1 / (1 − wait)); elsewhere none is
//! promised.

use serde::Serialize;

use crate::analysis::bottleneck::Bottleneck;

/// Which device API the workload uses — affects wording only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    Cuda,
    Mps,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recommendation {
    pub kind: Bottleneck,
    pub title: String,
    pub action: String,
    /// Expected MFU gain in percentage points, when it can be derived.
    pub expected_mfu_gain_ppt: Option<f32>,
}

pub struct RecommendationEngine;

impl RecommendationEngine {
    pub fn recommend(
        b: Bottleneck,
        accel: Accelerator,
        mfu_pct: Option<f32>,
        data_wait_pct: f32,
        target_mfu: f32,
        hook_active: bool,
    ) -> Recommendation {
        let sync_call = match accel {
            Accelerator::Mps => "`torch.mps.synchronize()`",
            _ => "`torch.cuda.synchronize()`",
        };
        let hook_hint = if hook_active {
            ""
        } else {
            " For exact step timings, add the calibrate Python hook (see docs/hook.md)."
        };
        let (title, action, gain): (String, String, Option<f32>) = match b {
            Bottleneck::WarmingUp => (
                "Collecting data".into(),
                "Gathering the first samples — figures stabilise after ~30 s.".into(),
                None,
            ),
            Bottleneck::NoGpuMetrics => (
                "GPU metrics unavailable".into(),
                "No GPU backend could be initialised, so only CPU usage is shown. On NVIDIA \
                 systems run `calibrate diagnose` to find the missing driver component."
                    .into(),
                None,
            ),
            Bottleneck::NoGpuActivity => (
                "Target is not using a GPU".into(),
                "None of the target's processes holds a GPU context. Check the device \
                 placement in your script (e.g. `model.to(\"cuda\")`), wait until the job \
                 starts training, or — inside containers, where NVML cannot map PIDs — use \
                 `--gpus all`."
                    .into(),
                None,
            ),
            Bottleneck::ThermalThrottle => (
                "GPU is thermal throttling".into(),
                "Clocks are being reduced for temperature. Check cooling and fan curves; on \
                 cloud instances ask for a different host. Throttling lowers every other \
                 figure until it is fixed."
                    .into(),
                None,
            ),
            Bottleneck::PowerCapped => (
                "GPU is power capped".into(),
                "Clocks are limited by the board power limit. Check `nvidia-smi -q -d POWER`; \
                 if the enforced limit is below the default, raise it with \
                 `sudo nvidia-smi -pl <watts>` or ask your provider."
                    .into(),
                None,
            ),
            Bottleneck::HardwareSlowdown => (
                "Hardware slowdown signalled".into(),
                "The GPU reports HW slowdown or power-brake throttling (often PSU or \
                 external power-brake signals). This is a host problem — report it to the \
                 operator or move to another node."
                    .into(),
                None,
            ),
            Bottleneck::GpuImbalance => (
                "GPUs are unevenly loaded — straggler rank".into(),
                "One device does noticeably less work than the others, and in data-parallel \
                 training every rank waits for the slowest. Check the per-GPU table for a \
                 throttled or slower card, uneven data sharding or sequence lengths per rank, \
                 and interconnect topology (`nvidia-smi topo -m`); pin ranks to the right \
                 NUMA node and NIC."
                    .into(),
                None,
            ),
            Bottleneck::DataLoader => (
                "Input pipeline is the bottleneck".into(),
                format!(
                    "The GPU is waiting for batches ~{data_wait_pct:.0}% of the time. Use \
                     `DataLoader(num_workers=N, pin_memory=True, persistent_workers=True, \
                     prefetch_factor=4)`, pre-tokenise the dataset, and move augmentation off \
                     the critical path.{hook_hint}"
                ),
                mfu_pct.map(|m| {
                    let d = (data_wait_pct / 100.0).clamp(0.0, 0.95);
                    (m / (1.0 - d) - m).clamp(0.0, 100.0 - m)
                }),
            ),
            Bottleneck::SyncStall => (
                "GPU and CPU are both idle — synchronisation stalls".into(),
                format!(
                    "Look for host-device syncs in the hot loop: `.item()`, `.cpu()`, \
                     `print(loss)`, {sync_call}, and per-step logging or checkpointing. In \
                     multi-GPU jobs this pattern also appears when ranks wait on a slow \
                     collective. Log every N steps and accumulate metrics on-device.{hook_hint}"
                ),
                None,
            ),
            Bottleneck::MemoryBound => (
                "Memory-bandwidth bound".into(),
                "DRAM is saturated while compute is low. Use fused kernels \
                 (`torch.compile`, FlashAttention, fused optimizers), bf16 activations, and \
                 larger batches to raise arithmetic intensity."
                    .into(),
                None,
            ),
            Bottleneck::OptimizerHeavy => (
                "Optimizer step is expensive".into(),
                "Use a fused or foreach optimizer (`torch.optim.AdamW(..., fused=True)`), \
                 increase gradient accumulation so the step runs less often, or shard \
                 optimizer state (ZeRO / FSDP)."
                    .into(),
                None,
            ),
            Bottleneck::ClockUnderspeed => (
                "GPU clock below boost".into(),
                "The SM clock is well below the rated boost clock with no throttle reason. \
                 Check application clocks (`nvidia-smi -q -d CLOCK`), enable persistence \
                 mode (`sudo nvidia-smi -pm 1`), and confirm the provider is not capping \
                 clocks."
                    .into(),
                None,
            ),
            Bottleneck::ErraticSteps => (
                "Step times are erratic".into(),
                "Step duration varies widely. Common causes: variable sequence lengths \
                 (bucket or pack sequences), periodic evaluation/checkpointing inside the \
                 loop, memory-allocator churn (set \
                 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), or a noisy neighbour."
                    .into(),
                None,
            ),
            Bottleneck::BelowTargetMfu => (
                format!("MFU below {target_mfu:.0}% — no single dominant cause"),
                "Try, in order: bf16 autocast, larger micro-batch (or gradient accumulation), \
                 `torch.compile`, FlashAttention / SDPA, and fused optimizers. Re-run \
                 calibrate after each change to see which one moved the number."
                    .into(),
                // The gap to target is not an expected gain from any one change.
                None,
            ),
            Bottleneck::Unverified => (
                "Real efficiency cannot be verified".into(),
                format!(
                    "The GPU is busy most of the time, but NVML busy-time only bounds MFU from \
                     above — a fully busy GPU can still be at 30% MFU. Use a Hopper+ GPU (GPM), \
                     run DCGM (`--metrics-source dcgm`), or add the calibrate Python hook to \
                     measure it.{hook_hint}"
                ),
                None,
            ),
            Bottleneck::None => (
                "No significant bottleneck".into(),
                format!(
                    "Measured efficiency is at or above the {target_mfu:.0}% target. Further \
                     gains come from model-level work (kernel fusion, parallelism strategy)."
                ),
                None,
            ),
        };
        Recommendation {
            kind: b,
            title,
            action,
            expected_mfu_gain_ppt: gain.filter(|g| *g > 0.05),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(b: Bottleneck, accel: Accelerator) -> Recommendation {
        RecommendationEngine::recommend(b, accel, Some(30.0), 25.0, 45.0, false)
    }

    #[test]
    fn data_loader_gain_follows_from_wait_fraction() {
        // 25 % wait at 30 % MFU → 30 / 0.75 = 40 % → +10 ppt.
        let r = rec(Bottleneck::DataLoader, Accelerator::Cuda);
        assert!((r.expected_mfu_gain_ppt.unwrap() - 10.0).abs() < 0.01);
        assert!(r.action.contains("num_workers"));
    }

    #[test]
    fn mps_wording_does_not_mention_cuda() {
        let r = rec(Bottleneck::SyncStall, Accelerator::Mps);
        assert!(r.action.contains("torch.mps.synchronize"));
        assert!(!r.action.contains("cuda"));
    }

    #[test]
    fn no_made_up_gains() {
        for b in [
            Bottleneck::SyncStall,
            Bottleneck::MemoryBound,
            Bottleneck::ThermalThrottle,
            Bottleneck::Unverified,
            Bottleneck::BelowTargetMfu,
            Bottleneck::None,
        ] {
            assert_eq!(
                rec(b, Accelerator::Cuda).expected_mfu_gain_ppt,
                None,
                "{b:?}"
            );
        }
    }

    #[test]
    fn hook_hint_disappears_when_hook_is_active() {
        let without = rec(Bottleneck::SyncStall, Accelerator::Cuda);
        let with = RecommendationEngine::recommend(
            Bottleneck::SyncStall,
            Accelerator::Cuda,
            None,
            0.0,
            45.0,
            true,
        );
        assert!(without.action.contains("Python hook"));
        assert!(!with.action.contains("Python hook"));
    }

    #[test]
    fn every_variant_has_text() {
        use Bottleneck::*;
        for b in [
            WarmingUp,
            NoGpuMetrics,
            NoGpuActivity,
            ThermalThrottle,
            PowerCapped,
            HardwareSlowdown,
            GpuImbalance,
            DataLoader,
            SyncStall,
            MemoryBound,
            OptimizerHeavy,
            ClockUnderspeed,
            ErraticSteps,
            BelowTargetMfu,
            Unverified,
            None,
        ] {
            let r = rec(b, Accelerator::Unknown);
            assert!(!r.title.is_empty() && !r.action.is_empty(), "{b:?}");
        }
    }
}
