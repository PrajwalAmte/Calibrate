pub mod nvml;
pub mod proc;

use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

/// A single point-in-time snapshot of all observable hardware + process metrics.
///
/// Produced by [`MetricsCollector`] implementations and consumed by the
/// analytics pipeline in `metrics/`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RawSample {
    /// Wall-clock timestamp (Unix epoch millis) when this sample was taken.
    pub timestamp_ms: u64,

    /// Index of the GPU being sampled (0-based, matching NVML device index).
    pub gpu_index: u32,

    // ── Compute ──────────────────────────────────────────────────────────
    /// GPU streaming-multiprocessor utilization (0–100%).
    pub sm_utilization: Percent,

    /// Current SM clock frequency.
    pub sm_clock_mhz: Mhz,

    /// Maximum (boost) SM clock frequency for this GPU.
    pub sm_clock_max_mhz: Mhz,

    // ── Memory ───────────────────────────────────────────────────────────
    /// VRAM used by all processes combined.
    pub vram_used_mib: Mib,

    /// Total VRAM on the device.
    pub vram_total_mib: Mib,

    /// GPU memory-controller utilization (0–100%).
    pub mem_utilization: Percent,

    // ── Thermal / Power ───────────────────────────────────────────────────
    pub temperature: Celsius,
    pub power_draw: Watts,
    pub power_limit: Watts,

    // ── Throttle reasons (bitmask decoded to bools) ───────────────────────
    pub throttle_thermal: bool,
    pub throttle_power: bool,
    pub throttle_hw_slowdown: bool,

    // ── Process (from /proc) ─────────────────────────────────────────────
    /// CPU utilization of the training process (0–100%, may exceed 100 on multi-core).
    pub cpu_utilization: Percent,
}

/// Port: anything that can produce [`RawSample`] values.
///
/// Implementations: [`nvml::NvmlCollector`], and mock collectors in tests.
/// The trait is object-safe so it can be boxed for dependency injection.
pub trait MetricsCollector: Send + 'static {
    /// Start the collection loop, sending samples on `tx` until stop is signalled.
    ///
    /// This is intentionally a blocking call so it can be launched on a
    /// `std::thread` (NVML is not async-safe).
    fn run(self, tx: flume::Sender<RawSample>, stop: std::sync::Arc<std::sync::atomic::AtomicBool>);
}
