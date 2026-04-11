#![allow(dead_code)] // types consumed in Phase 2–5

pub mod duration;
pub mod model;
pub mod providers;
pub mod vram;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ── VRAM breakdown ─────────────────────────────────────────────────────────────

/// Per-component breakdown of estimated VRAM usage for a training run.
///
/// All values are in gibibytes (GiB). The sum of the components equals `total_gib`
/// after the optimizer efficiency factor is applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramBreakdown {
    /// Model weight tensors (parameter count × bytes per weight).
    pub weights_gib: f64,
    /// Gradient tensors. Zero for LoRA/QLoRA (only adapter gradients stored).
    pub gradients_gib: f64,
    /// Adam first and second moment optimizer states.
    pub optimizer_gib: f64,
    /// Intermediate activations stored for the backward pass.
    /// Assumes gradient checkpointing is enabled; without it this is ~8× larger.
    pub activations_gib: f64,
    /// Key-value attention cache proportional to sequence length and layers.
    pub kv_cache_gib: f64,
    /// Reduction applied by the optimizer library (e.g. Unsloth −40–60 %).
    /// Negative value represents the memory saved; zero when no library is used.
    pub library_savings_gib: f64,
    /// Total VRAM estimate (sum of components + library_savings_gib).
    pub total_gib: f64,
}

// ── Workload summary ───────────────────────────────────────────────────────────

/// Resolved model properties and the VRAM estimate derived from them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSummary {
    /// The model identifier or path as supplied by the user.
    pub model_id: String,
    /// Resolved parameter count in billions (e.g. 7.0 for a 7B model).
    pub param_count_b: f64,
    /// Full VRAM breakdown showing each component's contribution.
    pub vram_breakdown: VramBreakdown,
    /// Minimum VRAM (GiB) that a GPU must have to run this workload.
    /// Equal to `vram_breakdown.total_gib` rounded up to a safe margin.
    pub required_vram_gib: f64,
    /// Human-readable list of GPU memory tiers that can satisfy the requirement
    /// (e.g. ["16GB", "24GB", "40GB"]).
    pub fitting_tiers: Vec<String>,
}

// ── Provider listing ───────────────────────────────────────────────────────────

/// Availability of a GPU instance at query time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityStatus {
    /// Instance can be launched immediately.
    Available,
    /// Instance is listed but currently has no capacity.
    Unavailable,
    /// Instance is available but requires waiting in a queue.
    Queue { estimated_wait_minutes: u32 },
}

/// Flags that must be surfaced to the user alongside a listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListingFlag {
    /// Spot / preemptible instance — may be interrupted mid-job.
    Spot,
    /// Vast.ai or other auction-market listing — price may change before launch.
    PriceVolatile,
    /// Machine reliability score is below the 0.95 threshold.
    LowReliability,
    /// Instance is listed but currently unavailable to rent.
    CurrentlyUnavailable,
}

/// A single GPU offering from one provider, enriched with duration + cost estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedListing {
    /// Provider name (e.g. "RunPod", "Lambda", "Vast.ai").
    pub provider: String,
    /// GPU model name (e.g. "RTX 3090", "A100 SXM4 80GB").
    pub gpu_model: String,
    /// GPU VRAM in GiB.
    pub vram_gib: f64,
    /// Current hourly price in USD.
    pub hourly_usd: f64,
    /// Estimated training duration range.
    /// `None` when `--dataset-rows` was not provided.
    pub duration_range: Option<DurationRange>,
    /// Estimated total cost range.
    /// `None` when `--dataset-rows` was not provided.
    pub cost_range: Option<CostRange>,
    /// Availability at query time.
    pub availability: AvailabilityStatus,
    /// Flags that should be shown to the user.
    pub flags: Vec<ListingFlag>,
}

// ── Duration and cost ranges ───────────────────────────────────────────────────

/// Training duration range in seconds (low estimate to high estimate).
///
/// The range reflects ±MFU variability and real-world scheduling overhead
/// rather than false point-estimate precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationRange {
    pub low_secs: f64,
    pub high_secs: f64,
}

impl DurationRange {
    /// Format the range as a human-readable string (e.g. "1.5–2.5h").
    pub fn display(&self) -> String {
        let low_h = self.low_secs / 3600.0;
        let high_h = self.high_secs / 3600.0;
        if high_h < 1.0 {
            format!("{:.0}–{:.0}m", low_h * 60.0, high_h * 60.0)
        } else {
            format!("{:.1}–{:.1}h", low_h, high_h)
        }
    }
}

/// Estimated total cost range in USD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRange {
    pub low_usd: f64,
    pub high_usd: f64,
}

impl CostRange {
    /// Format as a human-readable string (e.g. "$0.46–$0.70").
    pub fn display(&self) -> String {
        format!("${:.2}–${:.2}", self.low_usd, self.high_usd)
    }
}

// ── Recommendation ─────────────────────────────────────────────────────────────

/// The tool's top pick, plus an optional conservative alternative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRecommendation {
    /// The recommended listing (lowest estimated cost among available GPUs).
    pub listing: RankedListing,
    /// One-sentence explanation of why this listing was chosen.
    pub rationale: String,
    /// A stable-price alternative when the top recommendation is volatile
    /// (e.g. the top pick is Vast.ai but a RunPod option is only marginally
    /// more expensive). `None` when the top pick is already stable.
    pub safe_alternative: Option<RankedListing>,
}

// ── Provider skip ──────────────────────────────────────────────────────────────

/// A provider whose listings could not be fetched, with the reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedProvider {
    pub name: String,
    pub reason: String,
}

// ── Top-level report ────────────────────────────────────────────────────────────

/// Complete output of a `calibrate plan` run, suitable for serialization to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanReport {
    /// Resolved model and VRAM analysis.
    pub workload: WorkloadSummary,
    /// All GPU listings that meet the VRAM requirement, sorted by estimated
    /// total cost (ascending). Listings that exceed the budget are included
    /// but flagged in the output.
    pub listings: Vec<RankedListing>,
    /// The recommended configuration, if at least one viable listing was found.
    pub recommendation: Option<PlanRecommendation>,
    /// Providers skipped due to API errors or timeouts.
    pub skipped_providers: Vec<SkippedProvider>,
    /// Timestamp at which provider pricing was fetched.
    pub generated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_range_formats_minutes_below_one_hour() {
        let dr = DurationRange {
            low_secs: 1200.0,
            high_secs: 2400.0,
        };
        let s = dr.display();
        assert!(s.contains('m'), "expected minutes format, got: {s}");
    }

    #[test]
    fn duration_range_formats_hours_above_one_hour() {
        let dr = DurationRange {
            low_secs: 5400.0,
            high_secs: 9000.0,
        };
        let s = dr.display();
        assert!(s.contains('h'), "expected hours format, got: {s}");
    }

    #[test]
    fn cost_range_display_includes_dollar_sign() {
        let cr = CostRange {
            low_usd: 0.46,
            high_usd: 0.70,
        };
        let s = cr.display();
        assert!(s.starts_with('$'), "got: {s}");
        assert!(s.contains("0.46"), "got: {s}");
        assert!(s.contains("0.70"), "got: {s}");
    }

    #[test]
    fn listing_flags_are_serializable() {
        let flags = vec![ListingFlag::Spot, ListingFlag::PriceVolatile];
        let json = serde_json::to_string(&flags).unwrap();
        let back: Vec<ListingFlag> = serde_json::from_str(&json).unwrap();
        assert_eq!(flags, back);
    }

    #[test]
    fn availability_status_queue_round_trips() {
        let status = AvailabilityStatus::Queue {
            estimated_wait_minutes: 15,
        };
        let json = serde_json::to_string(&status).unwrap();
        let back: AvailabilityStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, back);
    }
}
