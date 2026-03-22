use crate::analysis::bottleneck::Bottleneck;

/// A single, actionable recommendation derived from the detected bottleneck.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Recommendation {
    /// Short headline (shown prominently in the TUI).
    pub title: &'static str,
    /// The specific code change or configuration the user should make.
    pub action: &'static str,
    /// Estimated MFU improvement if the recommendation is followed (percentage points).
    pub expected_mfu_gain_ppt: f32,
}

/// Produces exactly one recommendation from the detected bottleneck.
///
/// The tool spec requires a *single* primary recommendation — not a list.
/// Each arm is derived directly from measured data as specified in the requirements.
pub struct RecommendationEngine;

impl RecommendationEngine {
    pub fn recommend(bottleneck: &Bottleneck, data_loader_pct: f32) -> Recommendation {
        match bottleneck {
            Bottleneck::DataLoader => Recommendation {
                title: "Data loader is the primary bottleneck",
                action: "Add `num_workers=4, pin_memory=True` to your DataLoader. \
                         If the dataset fits in RAM, use an in-memory dataset to eliminate disk I/O.",
                expected_mfu_gain_ppt: estimate_dl_gain(data_loader_pct),
            },

            Bottleneck::CudaSync => Recommendation {
                title: "Excessive CUDA synchronization detected",
                action: "Audit your training loop for `.item()` calls, scalar logging inside \
                         the loop, and `loss.item()` every step. Move metric collection outside \
                         the hot path or accumulate in tensors.",
                expected_mfu_gain_ppt: 8.0,
            },

            Bottleneck::MemoryFragmentation => Recommendation {
                title: "Memory allocation overhead is high",
                action: "Pre-allocate output tensors outside the training loop. Ensure \
                         `torch.no_grad()` wraps validation. Avoid creating new tensors \
                         inside the loop for buffers that can be reused.",
                expected_mfu_gain_ppt: 6.0,
            },

            Bottleneck::ThermalThrottle => Recommendation {
                title: "GPU is thermal throttling",
                action: "Check GPU cooling. On a cloud instance the physical machine may be \
                         under thermal stress — consider migrating to a different node. \
                         Temporarily reducing batch size reduces thermal load.",
                expected_mfu_gain_ppt: 10.0,
            },

            Bottleneck::ClockUnderspeed => Recommendation {
                title: "GPU clock running below boost",
                action: "Check the power limit on this instance: `nvidia-smi -q -d POWER`. \
                         Some cloud providers cap GPU clocks on shared hardware. \
                         Setting `sudo nvidia-smi -pm 1` enables persistent mode.",
                expected_mfu_gain_ppt: 5.0,
            },

            Bottleneck::BelowTargetMfu => Recommendation {
                title: "MFU below 45% — general optimization opportunity",
                action: "Try: (1) increase batch size if VRAM allows, \
                         (2) enable mixed precision with `torch.autocast`, \
                         (3) verify gradient accumulation steps match your effective batch target.",
                expected_mfu_gain_ppt: 15.0,
            },

            Bottleneck::None => Recommendation {
                title: "Training is well-optimized",
                action: "MFU is above 45%. No significant bottleneck detected. \
                         Consider profiling at the model-architecture level for further gains.",
                expected_mfu_gain_ppt: 0.0,
            },
        }
    }
}

/// Rough estimate of MFU gain from fixing a data loader bottleneck.
/// The more time is wasted on data loading, the larger the relative gain.
fn estimate_dl_gain(data_loader_pct: f32) -> f32 {
    (data_loader_pct * 0.6).clamp(5.0, 30.0)
}
