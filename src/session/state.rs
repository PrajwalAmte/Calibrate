//! The `SessionSnapshot` published after every tick.
//!
//! This struct *is* the public JSON schema for `watch --output json`,
//! `--record` and the exporters. Bump [`SCHEMA_VERSION`] on any breaking
//! change (renamed/removed field, changed meaning); adding fields is
//! backwards compatible. See `docs/json-schema.md`.

use serde::Serialize;

use crate::analysis::bottleneck::Bottleneck;
use crate::analysis::recommendations::Recommendation;
use crate::collectors::{GpuScope, Throttle};
use crate::metrics::breakdown::TimeBreakdown;
use crate::metrics::mfu::MfuEstimate;
use crate::metrics::steps::StepStats;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct MfuPercentiles {
    pub p50: f32,
    pub p75: f32,
    pub p95: f32,
}

/// What is being watched.
#[derive(Debug, Clone, Serialize)]
pub struct TargetInfo {
    /// "process", "device" or "replay".
    pub mode: String,
    pub pid: Option<u32>,
    pub process_count: u32,
    pub scope: GpuScope,
    pub hostname: String,
}

/// Per-GPU state for the latest tick plus session peaks.
#[derive(Debug, Clone, Serialize)]
pub struct GpuStatus {
    pub index: u32,
    pub name: String,
    /// Peak dense BF16 TFLOPS from the spec DB (`None` if unknown).
    pub peak_tflops: Option<f64>,
    pub util_pct: Option<f32>,
    pub sm_active_pct: Option<f32>,
    pub tensor_active_pct: Option<f32>,
    pub dram_active_pct: Option<f32>,
    pub sm_clock_mhz: Option<u32>,
    pub clock_pct_of_boost: Option<f32>,
    pub vram_used_mib: Option<u64>,
    pub vram_total_mib: Option<u64>,
    pub peak_vram_mib: Option<u64>,
    pub temperature_c: Option<f32>,
    pub power_w: Option<f32>,
    pub power_limit_w: Option<f32>,
    pub throttle: Throttle,
    /// Sustained VRAM growth on this device — possible leak.
    pub vram_growing: bool,
}

/// Cost of running below target.
#[derive(Debug, Clone, Serialize)]
pub struct CostImpact {
    pub cost_per_hour: f64,
    /// Fraction of spend not converted into useful compute relative to target (0–1).
    pub waste_ratio: f64,
    pub waste_per_hour: f64,
    /// Spend so far in this session.
    pub spent_usd: f64,
    /// Waste so far in this session.
    pub wasted_usd: f64,
    /// True when MFU is only an upper bound — real waste is at least this.
    pub is_lower_bound: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSnapshot {
    pub schema_version: u32,
    pub calibrate_version: &'static str,
    pub timestamp_ms: u64,
    pub elapsed_secs: f64,
    pub target: TargetInfo,
    /// "nvml", "apple-iokit", "none" or "replay".
    pub backend: String,
    /// Utilization source in use ("nvml-gpm", "dcgm", "nvml", "iokit", ...).
    pub metrics_source: String,
    /// Display name for the GPU set, e.g. "8× NVIDIA H100 80GB HBM3".
    pub gpu_name: String,
    pub mfu: MfuEstimate,
    pub mfu_percentiles: Option<MfuPercentiles>,
    pub peak_mfu_pct: Option<f32>,
    pub breakdown: TimeBreakdown,
    pub bottleneck: Bottleneck,
    pub recommendation: Recommendation,
    pub gpus: Vec<GpuStatus>,
    /// Process-tree CPU, percent of one core.
    pub cpu_pct: Option<f32>,
    pub steps: Option<StepStats>,
    pub hook_peers: u32,
    pub cost: Option<CostImpact>,
    /// Max spread of mean GPU busy % across devices (multi-GPU only).
    pub gpu_divergence_ppt: Option<f32>,
    pub gpu_divergent: bool,
    pub vram_growing: bool,
    pub target_mfu_pct: f32,
    pub ticks_observed: u64,
    pub warnings: Vec<String>,
}

pub type SnapshotSender = tokio::sync::watch::Sender<Option<SessionSnapshot>>;
pub type SnapshotReceiver = tokio::sync::watch::Receiver<Option<SessionSnapshot>>;

/// Broadcast channel for snapshots; starts as `None` until the first tick.
pub fn new_snapshot_channel() -> (SnapshotSender, SnapshotReceiver) {
    tokio::sync::watch::channel(None)
}
