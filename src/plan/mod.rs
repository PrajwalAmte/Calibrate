pub mod duration;
pub mod model;
pub mod providers;
pub mod vram;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::cli::{FinetuneMethod, OptimizerLib, QuantLevel};
use crate::plan::model::ModelSpec;

/// Version of the `plan --output json` schema. Bump on breaking changes.
pub const PLAN_SCHEMA_VERSION: u32 = 1;

/// Per-GPU VRAM estimate. All values in GiB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramBreakdown {
    /// GPUs the workload is spread over (sharding only with DeepSpeed ZeRO-3).
    pub gpu_count: u32,
    pub weights_gib: f64,
    pub gradients_gib: f64,
    pub optimizer_gib: f64,
    /// Activations with gradient checkpointing (per-layer checkpoints plus
    /// one layer's recompute working set).
    pub activations_gib: f64,
    /// Logits and the fp32 copy used by the cross-entropy loss.
    pub logits_gib: f64,
    /// CUDA context, cuBLAS workspace and allocator fragmentation.
    pub overhead_gib: f64,
    pub total_gib: f64,
    /// Assumptions behind the numbers, shown to the user.
    pub notes: Vec<String>,
}

/// Training configuration as the planner understands it.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TrainingSetup {
    pub method: FinetuneMethod,
    pub optimizer_lib: OptimizerLib,
    pub quantization: QuantLevel,
    pub micro_batch: u32,
    pub seq_len: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkloadSummary {
    pub model: ModelSpec,
    pub setup: TrainingSetup,
    /// Single-GPU VRAM estimate.
    pub vram: VramBreakdown,
    /// VRAM a single GPU must have.
    pub required_vram_gib: f64,
    pub fitting_tiers: Vec<String>,
    pub dataset_rows: Option<u64>,
    pub epochs: u32,
    pub total_tokens: Option<f64>,
    /// Hardware FLOPs including activation recomputation.
    pub total_hardware_flops: Option<f64>,
    /// Hardware utilization assumed for durations (0–1).
    pub assumed_utilization: f64,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityStatus {
    Available,
    Unavailable,
    Queue { estimated_wait_minutes: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListingFlag {
    /// Interruptible / preemptible instance.
    Spot,
    /// RunPod Community Cloud: third-party hosts, fewer guarantees.
    CommunityCloud,
    /// Marketplace price that may change before launch (Vast.ai).
    PriceVolatile,
    /// Host reliability score below 0.95.
    LowReliability,
    /// Listed but no capacity right now.
    CurrentlyUnavailable,
    /// GPU not in the spec database — no duration or cost estimate.
    UnknownGpu,
}

/// A raw offer from a provider (or a `--prices-file`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuListing {
    pub provider: String,
    pub gpu_model: String,
    #[serde(default = "one")]
    pub gpu_count: u32,
    pub vram_per_gpu_gib: f64,
    /// Price for the whole instance (all GPUs), USD per hour.
    pub hourly_usd: f64,
    #[serde(default = "available")]
    pub availability: AvailabilityStatus,
    #[serde(default)]
    pub flags: Vec<ListingFlag>,
}

fn one() -> u32 {
    1
}
fn available() -> AvailabilityStatus {
    AvailabilityStatus::Available
}

/// A listing enriched with fit, duration and cost.
#[derive(Debug, Clone, Serialize)]
pub struct RankedListing {
    #[serde(flatten)]
    pub listing: GpuListing,
    pub peak_tflops_per_gpu: Option<f64>,
    pub required_vram_per_gpu_gib: f64,
    pub duration_range: Option<DurationRange>,
    pub cost_range: Option<CostRange>,
    pub over_budget: bool,
}

impl RankedListing {
    pub fn display_gpu(&self) -> String {
        if self.listing.gpu_count > 1 {
            format!("{}× {}", self.listing.gpu_count, self.listing.gpu_model)
        } else {
            self.listing.gpu_model.clone()
        }
    }
    pub fn is_launchable(&self) -> bool {
        self.listing.availability == AvailabilityStatus::Available
            && !self
                .listing
                .flags
                .contains(&ListingFlag::CurrentlyUnavailable)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationRange {
    pub low_secs: f64,
    pub high_secs: f64,
}

impl DurationRange {
    pub fn display(&self) -> String {
        use crate::metrics::units::fmt_duration;
        let (a, b) = (fmt_duration(self.low_secs), fmt_duration(self.high_secs));
        if a == b {
            a
        } else {
            format!("{a}–{b}")
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRange {
    pub low_usd: f64,
    pub high_usd: f64,
}

impl CostRange {
    pub fn display(&self) -> String {
        use crate::metrics::units::fmt_usd;
        format!("{}–{}", fmt_usd(self.low_usd), fmt_usd(self.high_usd))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanRecommendation {
    pub listing: RankedListing,
    pub rationale: String,
    /// A stable-price alternative when the top pick is volatile.
    pub safe_alternative: Option<RankedListing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedProvider {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanReport {
    pub schema_version: u32,
    pub workload: WorkloadSummary,
    /// Listings that fit, sorted by estimated cost (ascending).
    pub listings: Vec<RankedListing>,
    /// Listings excluded because the model does not fit in their VRAM.
    pub excluded_too_small: usize,
    pub recommendation: Option<PlanRecommendation>,
    pub skipped_providers: Vec<SkippedProvider>,
    /// Where prices came from: "live", "cache", "file".
    pub price_source: String,
    pub generated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_formats() {
        assert_eq!(
            DurationRange {
                low_secs: 1200.0,
                high_secs: 2400.0
            }
            .display(),
            "20m–40m"
        );
        assert_eq!(
            DurationRange {
                low_secs: 3.0 * 3600.0,
                high_secs: 5.0 * 3600.0
            }
            .display(),
            "3.0h–5.0h"
        );
    }

    #[test]
    fn cost_formats() {
        let c = CostRange {
            low_usd: 0.46,
            high_usd: 1234.0,
        };
        assert_eq!(c.display(), "$0.46–$1,234");
    }

    #[test]
    fn listing_json_defaults() {
        let l: GpuListing = serde_json::from_str(
            r#"{"provider":"onprem","gpu_model":"H100 SXM","vram_per_gpu_gib":80,"hourly_usd":2.0}"#,
        )
        .unwrap();
        assert_eq!(l.gpu_count, 1);
        assert_eq!(l.availability, AvailabilityStatus::Available);
    }

    #[test]
    fn flags_round_trip() {
        let flags = vec![ListingFlag::Spot, ListingFlag::PriceVolatile];
        let json = serde_json::to_string(&flags).unwrap();
        let back: Vec<ListingFlag> = serde_json::from_str(&json).unwrap();
        assert_eq!(flags, back);
    }
}
