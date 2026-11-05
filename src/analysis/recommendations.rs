//! One actionable recommendation per detected bottleneck.
//!
//! Advice uses what the session knows: the device API, the training framework
//! the hook reports, and step statistics. PyTorch-specific settings are never
//! suggested to a job known to use something else, and causes the data rules
//! out are not listed. Gains are only quoted where they follow from the
//! measurements (removing input-pipeline wait scales throughput by
//! 1 / (1 − wait)); elsewhere none is promised.

use serde::Serialize;

use crate::analysis::bottleneck::Bottleneck;
use crate::metrics::steps::StepStats;

/// Which device API the workload uses.
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

/// What the session knows when choosing the wording.
#[derive(Debug, Clone, Copy)]
pub struct RecContext<'a> {
    pub accel: Accelerator,
    /// Framework reported by the hook's hello message, if any.
    pub framework: Option<&'a str>,
    pub mfu_pct: Option<f32>,
    pub data_wait_pct: f32,
    pub target_mfu: f32,
    pub hook_active: bool,
    pub steps: Option<&'a StepStats>,
}

impl RecContext<'_> {
    /// PyTorch advice applies unless the hook says another framework runs.
    fn pytorch_likely(&self) -> bool {
        self.framework.is_none_or(|f| {
            let f = f.to_lowercase();
            ["pytorch", "torch", "transformers", "lightning"]
                .iter()
                .any(|k| f.contains(k))
        })
    }
}

pub struct RecommendationEngine;

fn erratic_action(c: &RecContext) -> String {
    let mut causes: Vec<String> = Vec::new();
    let mut lead = String::from("Step duration varies widely.");
    if let Some(s) = c.steps {
        if s.stall_time_pct > 20.0 {
            lead = format!(
                "Occasional long steps take {:.0}% of step time.",
                s.stall_time_pct
            );
        }
    }
    let tokens_cv = c.steps.and_then(|s| s.tokens_cv);
    match tokens_cv {
        Some(cv) if cv > 0.05 => causes.push(format!(
            "tokens per step vary (CV {cv:.2}), so try bucketing or packing sequences"
        )),
        Some(_) => {
            lead.push_str(" Tokens per step are constant, so sequence length is not the cause.")
        }
        None => causes.push("variable sequence lengths (bucket or pack sequences)".into()),
    }
    causes.push("periodic evaluation or checkpointing inside the loop".into());
    causes.push(match (c.accel, c.pytorch_likely()) {
        (Accelerator::Cuda, true) => {
            "memory-allocator churn (set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)".into()
        }
        _ => "memory-allocator churn".into(),
    });
    causes.push(match c.accel {
        Accelerator::Mps => {
            "thermal throttling, which macOS does not report here (likely on fanless Macs), or \
             other apps using the GPU"
                .into()
        }
        _ => "a noisy neighbour".into(),
    });
    let last = causes.pop().unwrap_or_default();
    format!("{lead} Check: {}, or {last}.", causes.join(", "))
}

impl RecommendationEngine {
    pub fn recommend(b: Bottleneck, c: &RecContext) -> Recommendation {
        let pytorch = c.pytorch_likely();
        let sync_call = match c.accel {
            Accelerator::Mps => "`torch.mps.synchronize()`",
            _ => "`torch.cuda.synchronize()`",
        };
        let device = match c.accel {
            Accelerator::Mps => "\"mps\"",
            _ => "\"cuda\"",
        };
        let hook_hint = if c.hook_active {
            ""
        } else {
            " For exact step timings, add the calibrate training hook (see docs/hook.md)."
        };
        let target_mfu = c.target_mfu;
        let data_wait_pct = c.data_wait_pct;
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
                format!(
                    "None of the target's processes holds a GPU context. Check the device \
                     placement in your script (e.g. `model.to({device})`), wait until the job \
                     starts training, or — inside containers, where NVML cannot map PIDs — use \
                     `--gpus all`."
                ),
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
                if pytorch {
                    format!(
                        "The GPU is waiting for batches ~{data_wait_pct:.0}% of the time. Use \
                         `DataLoader(num_workers=N, pin_memory=True, persistent_workers=True, \
                         prefetch_factor=4)`, pre-tokenise the dataset, and move augmentation \
                         off the critical path.{hook_hint}"
                    )
                } else {
                    format!(
                        "The GPU is waiting for batches ~{data_wait_pct:.0}% of the time. Load \
                         and tokenise batches in parallel ahead of time, pre-tokenise the \
                         dataset, and move augmentation off the critical path.{hook_hint}"
                    )
                },
                c.mfu_pct.map(|m| {
                    let d = (data_wait_pct / 100.0).clamp(0.0, 0.95);
                    (m / (1.0 - d) - m).clamp(0.0, 100.0 - m)
                }),
            ),
            Bottleneck::SyncStall => (
                "GPU and CPU are both idle — synchronisation stalls".into(),
                if pytorch {
                    format!(
                        "Look for host-device syncs in the hot loop: `.item()`, `.cpu()`, \
                         `print(loss)`, {sync_call}, and per-step logging or checkpointing. In \
                         multi-GPU jobs this pattern also appears when ranks wait on a slow \
                         collective. Log every N steps and accumulate metrics on-device.{hook_hint}"
                    )
                } else {
                    format!(
                        "Look for host-device transfers in the hot loop: reading tensor values \
                         back, per-step logging or checkpointing. In multi-GPU jobs this also \
                         appears when ranks wait on a slow collective. Log every N steps.{hook_hint}"
                    )
                },
                None,
            ),
            Bottleneck::MemoryBound => (
                "Memory-bandwidth bound".into(),
                if pytorch {
                    "DRAM is saturated while compute is low. Use fused kernels \
                     (`torch.compile`, FlashAttention, fused optimizers), bf16 activations, and \
                     larger batches to raise arithmetic intensity."
                        .into()
                } else {
                    "DRAM is saturated while compute is low. Use fused kernels and fused \
                     optimizers, 16-bit activations, and larger batches to raise arithmetic \
                     intensity."
                        .into()
                },
                None,
            ),
            Bottleneck::OptimizerHeavy => (
                "Optimizer step is expensive".into(),
                if pytorch {
                    "Use a fused or foreach optimizer (`torch.optim.AdamW(..., fused=True)`), \
                     increase gradient accumulation so the step runs less often, or shard \
                     optimizer state (ZeRO / FSDP)."
                        .into()
                } else {
                    "Use a fused optimizer implementation if your framework has one, or increase \
                     gradient accumulation so the step runs less often."
                        .into()
                },
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
            Bottleneck::ErraticSteps => ("Step times are erratic".into(), erratic_action(c), None),
            Bottleneck::BelowTargetMfu => (
                format!("MFU below {target_mfu:.0}% — no single dominant cause"),
                if pytorch {
                    "Try, in order: bf16 autocast, larger micro-batch (or gradient accumulation), \
                     `torch.compile`, FlashAttention / SDPA, and fused optimizers. Re-run \
                     calibrate after each change to see which one moved the number."
                        .into()
                } else {
                    "Try, in order: 16-bit mixed precision, a larger micro-batch, your \
                     framework's graph compilation or kernel fusion, and an efficient attention \
                     kernel. Re-run calibrate after each change to see which one moved the number."
                        .into()
                },
                // The gap to target is not an expected gain from any one change.
                None,
            ),
            Bottleneck::Unverified => (
                "Real efficiency cannot be verified".into(),
                format!(
                    "The GPU is busy most of the time, but busy time only bounds MFU from \
                     above — a fully busy GPU can still be at 30% MFU. Use a Hopper+ GPU (GPM), \
                     run DCGM (`--metrics-source dcgm`), or add the calibrate training hook to \
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

    fn ctx(accel: Accelerator) -> RecContext<'static> {
        RecContext {
            accel,
            framework: None,
            mfu_pct: Some(30.0),
            data_wait_pct: 25.0,
            target_mfu: 45.0,
            hook_active: false,
            steps: None,
        }
    }

    fn rec(b: Bottleneck, accel: Accelerator) -> Recommendation {
        RecommendationEngine::recommend(b, &ctx(accel))
    }

    fn stats(tokens_cv: Option<f64>, stall: f64) -> StepStats {
        StepStats {
            steps_in_window: 100,
            mean_ms: 500.0,
            p50_ms: 450.0,
            p95_ms: 900.0,
            cv: 0.5,
            robust_cv: 0.4,
            stall_time_pct: stall,
            erratic: true,
            tokens_cv,
            steps_per_sec: 2.0,
            tokens_per_sec: None,
            samples_per_sec: None,
            ranks: 1,
        }
    }

    const ALL: [Bottleneck; 16] = [
        Bottleneck::WarmingUp,
        Bottleneck::NoGpuMetrics,
        Bottleneck::NoGpuActivity,
        Bottleneck::ThermalThrottle,
        Bottleneck::PowerCapped,
        Bottleneck::HardwareSlowdown,
        Bottleneck::GpuImbalance,
        Bottleneck::DataLoader,
        Bottleneck::SyncStall,
        Bottleneck::MemoryBound,
        Bottleneck::OptimizerHeavy,
        Bottleneck::ClockUnderspeed,
        Bottleneck::ErraticSteps,
        Bottleneck::BelowTargetMfu,
        Bottleneck::Unverified,
        Bottleneck::None,
    ];

    #[test]
    fn data_loader_gain_follows_from_wait_fraction() {
        // 25 % wait at 30 % MFU → 30 / 0.75 = 40 % → +10 ppt.
        let r = rec(Bottleneck::DataLoader, Accelerator::Cuda);
        assert!((r.expected_mfu_gain_ppt.unwrap() - 10.0).abs() < 0.01);
        assert!(r.action.contains("num_workers"));
    }

    #[test]
    fn apple_advice_never_mentions_cuda() {
        for b in ALL {
            if matches!(
                b,
                Bottleneck::PowerCapped | Bottleneck::ClockUnderspeed | Bottleneck::GpuImbalance
            ) {
                continue; // NVIDIA-only findings: never raised on macOS
            }
            let a = rec(b, Accelerator::Mps).action.to_lowercase();
            assert!(!a.contains("cuda"), "{b:?}: {a}");
        }
        assert!(rec(Bottleneck::SyncStall, Accelerator::Mps)
            .action
            .contains("torch.mps.synchronize"));
    }

    #[test]
    fn non_pytorch_jobs_get_no_pytorch_advice() {
        // Regression (real run): a Burn job on Metal was told to set
        // PYTORCH_CUDA_ALLOC_CONF.
        for accel in [Accelerator::Cuda, Accelerator::Mps] {
            let mut c = ctx(accel);
            c.framework = Some("burn");
            c.hook_active = true;
            for b in ALL {
                let a = RecommendationEngine::recommend(b, &c).action;
                for word in ["PYTORCH", "torch.", "DataLoader", ".item()", "ZeRO"] {
                    assert!(!a.contains(word), "{b:?} mentions {word}: {a}");
                }
            }
        }
    }

    #[test]
    fn pytorch_allocator_tip_only_on_cuda_pytorch() {
        let mut c = ctx(Accelerator::Cuda);
        c.framework = Some("transformers");
        assert!(
            RecommendationEngine::recommend(Bottleneck::ErraticSteps, &c)
                .action
                .contains("PYTORCH_CUDA_ALLOC_CONF")
        );
    }

    #[test]
    fn erratic_advice_follows_the_evidence() {
        let mut c = ctx(Accelerator::Cuda);
        let constant = stats(Some(0.0), 0.0);
        c.steps = Some(&constant);
        let a = RecommendationEngine::recommend(Bottleneck::ErraticSteps, &c).action;
        assert!(a.contains("sequence length is not the cause"), "{a}");
        assert!(!a.contains("bucket"), "{a}");

        let varying = stats(Some(0.4), 0.0);
        c.steps = Some(&varying);
        let a = RecommendationEngine::recommend(Bottleneck::ErraticSteps, &c).action;
        assert!(a.contains("tokens per step vary (CV 0.40)"), "{a}");

        let stalls = stats(Some(0.0), 45.0);
        c.steps = Some(&stalls);
        let a = RecommendationEngine::recommend(Bottleneck::ErraticSteps, &c).action;
        assert!(a.contains("take 45% of step time"), "{a}");
    }

    #[test]
    fn no_made_up_gains() {
        for b in ALL {
            if b != Bottleneck::DataLoader {
                assert_eq!(
                    rec(b, Accelerator::Cuda).expected_mfu_gain_ppt,
                    None,
                    "{b:?}"
                );
            }
        }
    }

    #[test]
    fn hook_hint_disappears_when_hook_is_active() {
        let without = rec(Bottleneck::SyncStall, Accelerator::Cuda);
        let mut c = ctx(Accelerator::Cuda);
        c.hook_active = true;
        let with = RecommendationEngine::recommend(Bottleneck::SyncStall, &c);
        assert!(without.action.contains("training hook"));
        assert!(!with.action.contains("training hook"));
    }

    #[test]
    fn every_variant_has_text() {
        for b in ALL {
            let r = rec(b, Accelerator::Unknown);
            assert!(!r.title.is_empty() && !r.action.is_empty(), "{b:?}");
        }
    }
}
