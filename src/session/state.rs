use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;

use crate::analysis::bottleneck::Bottleneck;
use crate::analysis::recommendations::Recommendation;
use crate::metrics::breakdown::TimeBreakdown;
use crate::metrics::mfu::MfuEstimate;
use crate::metrics::units::{Celsius, Mib, Percent, Watts};

/// A complete point-in-time view of the monitoring session.
///
/// Cheaply cloneable via `Arc`; serializable to JSON for the JSON output mode.
/// Broadcast on a `tokio::sync::watch` channel each sampling interval.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSnapshot {
    /// Wall-clock elapsed time since the session started.
    pub elapsed: Duration,

    /// Name of the primary GPU being monitored.
    pub gpu_name: String,

    /// MFU estimate for the current window.
    pub mfu: MfuEstimate,

    /// Time breakdown across the five training-loop phases.
    pub breakdown: TimeBreakdown,

    /// Primary identified bottleneck.
    pub bottleneck: Bottleneck,

    /// Actionable recommendation for the current bottleneck.
    pub recommendation: Recommendation,

    // ── Thermal / Power ───────────────────────────────────────────────────
    pub temperature: Celsius,
    pub power_draw: Watts,
    pub power_limit: Watts,
    pub throttle_thermal: bool,

    // ── Memory ───────────────────────────────────────────────────────────
    pub vram_used_mib: Mib,
    pub vram_total_mib: Mib,
    pub vram_utilization: Percent,

    /// Cost impact, populated only when `--cost-per-hour` was supplied.
    pub cost_impact: Option<CostImpact>,

    /// Per-GPU data for multi-GPU training.
    pub per_gpu: Vec<GpuSnapshot>,

    /// Approximate number of training steps observed.
    pub steps_observed: u64,
}

/// Cost breakdown at the current MFU versus the 45% target.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CostImpact {
    /// Current cost per hour (user-supplied).
    pub cost_per_hour: f64,
    /// Estimated cost per epoch at current MFU (requires step-count heuristic).
    pub current_cost_usd: f64,
    /// What the same epoch would cost at 45% MFU.
    pub target_cost_usd: f64,
    /// Waste per hour (current - target).
    pub waste_per_hour: f64,
}

/// Snapshot for a single GPU device (for multi-GPU display).
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuSnapshot {
    pub gpu_index: u32,
    pub sm_utilization: Percent,
    pub vram_used_mib: Mib,
    pub temperature: Celsius,
}

/// Shared, mutably-updatable session state.
///
/// Wrapped in `Arc<RwLock<>>` so the sampler loop can write and the renderer
/// can read without blocking.
pub type SharedState = Arc<RwLock<Option<SessionSnapshot>>>;

pub fn new_shared_state() -> SharedState {
    Arc::new(RwLock::new(None))
}
