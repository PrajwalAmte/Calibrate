#[cfg(target_os = "linux")]
pub mod cpu_only;
#[cfg(target_os = "linux")]
pub mod nvml;
#[cfg(target_os = "linux")]
pub mod proc;
#[cfg(target_os = "macos")]
pub mod apple_gpu;

use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

/// A single point-in-time snapshot of all observable hardware + process metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RawSample {
    pub timestamp_ms: u64,
    pub gpu_index: u32,

    pub sm_utilization: Percent,
    pub sm_clock_mhz: Mhz,
    /// Maximum (boost) SM clock for this GPU.
    pub sm_clock_max_mhz: Mhz,

    pub vram_used_mib: Mib,
    pub vram_total_mib: Mib,
    pub mem_utilization: Percent,

    pub temperature: Celsius,
    pub power_draw: Watts,
    pub power_limit: Watts,

    pub throttle_thermal: bool,
    pub throttle_power: bool,
    pub throttle_hw_slowdown: bool,

    /// CPU utilization of the training process (may exceed 100 on multi-core).
    pub cpu_utilization: Percent,
}

/// Anything that can produce [`RawSample`] values on a dedicated OS thread.
pub trait MetricsCollector: Send + 'static {
    fn run(self, tx: flume::Sender<RawSample>, stop: std::sync::Arc<std::sync::atomic::AtomicBool>);
}
