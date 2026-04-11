use std::time::Duration;

use crate::analysis::bottleneck::Bottleneck;
use crate::analysis::recommendations::Recommendation;
use crate::metrics::breakdown::TimeBreakdown;
use crate::metrics::mfu::MfuEstimate;
use crate::metrics::units::{Celsius, Mib, Percent, Watts};

// ── Snapshot──────

/// MFU distribution across the session window.
/// Present once ≥15 samples have been collected.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MfuPercentiles {
    pub p50: f32,
    pub p75: f32,
    pub p95: f32,
}

/// Point-in-time view of the monitoring session, broadcast on a watch channel
/// each sampling interval.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSnapshot {
    pub elapsed: Duration,
    pub gpu_name: String,
    pub mfu: MfuEstimate,
    /// Present once ≥15 samples are collected.
    pub mfu_percentiles: Option<MfuPercentiles>,
    pub peak_mfu_pct: f32,
    pub breakdown: TimeBreakdown,
    pub bottleneck: Bottleneck,
    pub recommendation: Recommendation,

    pub temperature: Celsius,
    pub power_draw: Watts,
    pub power_limit: Watts,
    pub throttle_thermal: bool,

    pub vram_used_mib: Mib,
    pub vram_total_mib: Mib,
    pub vram_utilization: Percent,
    pub peak_vram_mib: Mib,

    /// Populated only when `--cost-per-hour` was supplied.
    pub cost_impact: Option<CostImpact>,
    pub per_gpu: Vec<GpuSnapshot>,
    pub steps_observed: u64,

    /// True when any two GPUs' SM utilisation differs by >20 ppt.
    pub mfu_divergent: bool,
    /// Max SM utilisation spread across GPUs (ppt). 0.0 for single-GPU runs.
    pub gpu_mfu_divergence_ppt: f32,
    /// False on non-NVIDIA hardware or when the nvidia driver is absent.
    pub nvml_available: bool,

    /// True when VRAM has been monotonically increasing for 8+ consecutive
    /// samples — a potential memory leak in the training loop.
    pub vram_growing: bool,
    /// Mean inter-sample interval in ms. 0.0 until enough samples exist.
    pub step_time_ms_mean: f32,
    /// True when step-time coefficient of variation exceeds 0.3 (erratic).
    pub step_time_erratic: bool,
}

// ── Supporting types

/// Cost breakdown at the current MFU versus the 45% target.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CostImpact {
    pub cost_per_hour: f64,
    pub current_cost_usd: f64,
    pub target_cost_usd: f64,
    /// Waste per hour (current − target cost rate).
    pub waste_per_hour: f64,
}

/// Snapshot for a single GPU device (multi-GPU display).
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuSnapshot {
    pub gpu_index: u32,
    pub sm_utilization: Percent,
    pub vram_used_mib: Mib,
    pub temperature: Celsius,
}

// ── Watch channel──

/// Sender half of the snapshot broadcast channel.
///
/// Owned by `MonitoringSession` — dropped when the session loop exits, which
/// causes `SnapshotReceiver::changed()` to return `Err` and naturally
/// terminates any render loop.
pub type SnapshotSender = tokio::sync::watch::Sender<Option<SessionSnapshot>>;

/// Receiver half of the snapshot broadcast channel.
///
/// Cloneable — additional renderers can subscribe without extra coordination.
/// `changed().await` resolves immediately whenever a new snapshot is published.
pub type SnapshotReceiver = tokio::sync::watch::Receiver<Option<SessionSnapshot>>;

/// Create a new snapshot broadcast channel.
///
/// The initial value is `None` — renderers wait for the first sample before
/// drawing.
pub fn new_snapshot_channel() -> (SnapshotSender, SnapshotReceiver) {
    tokio::sync::watch::channel(None)
}
