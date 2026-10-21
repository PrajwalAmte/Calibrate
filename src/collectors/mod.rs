//! Data collection: GPU backends, process-tree CPU accounting, the Python
//! training hook, and trace replay.
//!
//! A collector thread produces one [`Tick`] per sampling interval. A tick
//! carries every GPU that is relevant to the target plus process CPU usage
//! and any training-step events published by the Python hook. Metrics a
//! platform cannot measure are `None` — never a fabricated zero.

#[cfg(target_os = "macos")]
pub mod apple_gpu;
pub mod cpu;
// The NVIDIA backend compiles everywhere (so it is unit-tested on every
// platform) but is only selected at runtime on Linux.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub mod dcgm;
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub mod gpm;
pub mod hook;
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub mod nvml;
pub mod proc_tree;
pub mod replay;
pub mod runner;

use serde::{Deserialize, Serialize};

/// What the GPU numbers in a tick describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuScope {
    /// Only GPUs on which the target process tree has a context.
    Process,
    /// Explicitly selected devices (`--gpus`), regardless of who uses them.
    Device,
    /// Whole-system figures (macOS IOKit) — other apps contribute too.
    System,
    /// No GPU backend available — CPU metrics only.
    None,
}

/// Where profiling-grade activity counters came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfSource {
    /// NVML GPU Performance Monitoring (Hopper and newer).
    Gpm,
    /// DCGM profiling fields read through `dcgmi dmon`.
    Dcgm,
}

/// Profiling counters. Percentages of elapsed cycles, 0–100.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProfMetrics {
    pub source: ProfSource,
    /// Fraction of time at least one warp was resident on an SM, averaged over SMs.
    pub sm_active_pct: Option<f32>,
    /// Resident warps relative to the SM maximum.
    pub sm_occupancy_pct: Option<f32>,
    /// Fraction of cycles any tensor-core pipe was active. Closest hardware
    /// proxy for MFU on tensor-core workloads.
    pub tensor_active_pct: Option<f32>,
    /// Fraction of cycles the DRAM interface was busy.
    pub dram_active_pct: Option<f32>,
}

/// Active clock-throttle reasons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Throttle {
    pub hw_thermal: bool,
    pub sw_thermal: bool,
    pub power_cap: bool,
    pub hw_slowdown: bool,
}

impl Throttle {
    pub fn thermal(&self) -> bool {
        self.hw_thermal || self.sw_thermal
    }
    pub fn any(&self) -> bool {
        self.hw_thermal || self.sw_thermal || self.power_cap || self.hw_slowdown
    }
}

/// One GPU at one point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSample {
    pub index: u32,
    pub name: String,
    /// NVML "utilization.gpu" / IOKit device utilization: percent of time any
    /// kernel was running. Not a FLOP measure.
    pub util_pct: Option<f32>,
    pub prof: Option<ProfMetrics>,
    pub sm_clock_mhz: Option<u32>,
    pub sm_clock_max_mhz: Option<u32>,
    pub vram_used_mib: Option<u64>,
    pub vram_total_mib: Option<u64>,
    /// Memory-controller utilization (percent of time memory was read/written).
    pub mem_util_pct: Option<f32>,
    pub temperature_c: Option<f32>,
    pub power_w: Option<f32>,
    pub power_limit_w: Option<f32>,
    #[serde(default)]
    pub throttle: Throttle,
}

impl GpuSample {
    /// Best available "is the GPU busy" signal: SM activity when profiling
    /// counters exist, else kernel-busy utilization.
    pub fn busy_pct(&self) -> Option<f32> {
        self.prof.and_then(|p| p.sm_active_pct).or(self.util_pct)
    }
}

/// One training step reported by the Python hook (`calibrate_hook`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepEvent {
    pub pid: u32,
    #[serde(default)]
    pub rank: Option<u32>,
    pub step: u64,
    /// Wall-clock end of the step, Unix milliseconds.
    pub t_end_ms: u64,
    /// Total step wall time including data wait (seconds).
    pub step_s: f64,
    #[serde(default)]
    pub data_s: Option<f64>,
    #[serde(default)]
    pub compute_s: Option<f64>,
    #[serde(default)]
    pub optimizer_s: Option<f64>,
    #[serde(default)]
    pub tokens: Option<u64>,
    #[serde(default)]
    pub samples: Option<u64>,
    /// Model FLOPs performed in this step (as declared by the training code).
    #[serde(default)]
    pub flops: Option<f64>,
}

/// A process that has the hook installed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookPeer {
    pub pid: u32,
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    pub world_size: Option<u32>,
    #[serde(default)]
    pub framework: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
}

/// Everything collected in one sampling interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tick {
    pub timestamp_ms: u64,
    pub scope: GpuScope,
    pub gpus: Vec<GpuSample>,
    /// CPU use of the whole target process tree, percent of one core
    /// (400 = four cores fully busy). `None` in device mode.
    pub cpu_pct: Option<f32>,
    /// Number of live processes in the target tree.
    #[serde(default)]
    pub process_count: u32,
    #[serde(default)]
    pub steps: Vec<StepEvent>,
    #[serde(default)]
    pub hook_peers: Vec<HookPeer>,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
pub mod testing {
    //! Builders shared by unit tests across modules.
    use super::*;

    pub fn gpu(index: u32, util: f32) -> GpuSample {
        GpuSample {
            index,
            name: "Test GPU".to_string(),
            util_pct: Some(util),
            prof: None,
            sm_clock_mhz: Some(2000),
            sm_clock_max_mhz: Some(2000),
            vram_used_mib: Some(8192),
            vram_total_mib: Some(16384),
            mem_util_pct: Some(40.0),
            temperature_c: Some(65.0),
            power_w: Some(200.0),
            power_limit_w: Some(350.0),
            throttle: Throttle::default(),
        }
    }

    pub fn tick(ts_ms: u64, gpus: Vec<GpuSample>, cpu: Option<f32>) -> Tick {
        Tick {
            timestamp_ms: ts_ms,
            scope: GpuScope::Process,
            gpus,
            cpu_pct: cpu,
            process_count: 1,
            steps: vec![],
            hook_peers: vec![],
        }
    }
}
