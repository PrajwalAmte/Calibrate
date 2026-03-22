use std::sync::{atomic::AtomicBool, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::{debug, error, warn};

use crate::collectors::{MetricsCollector, RawSample};
use crate::error::CalibrateError;
use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

/// Collects GPU metrics via NVML on a dedicated OS thread.
///
/// NVML's C library is synchronous and uses internal locking.  It must NOT
/// be called from within a tokio task — doing so would block the executor.
/// `NvmlCollector::run` is designed to be launched via `std::thread::spawn`.
pub struct NvmlCollector {
    /// GPU device indices that the target process is using.
    pub gpu_indices: Vec<u32>,
    /// Sampling interval.
    pub interval: Duration,
}

impl NvmlCollector {
    /// Create a new collector for the given GPU device indices.
    pub fn new(gpu_indices: Vec<u32>, interval: Duration) -> Self {
        Self {
            gpu_indices,
            interval,
        }
    }

    /// Attempt to initialize NVML once; returns an error with a helpful
    /// message if the driver is not present.
    pub fn probe() -> Result<(), CalibrateError> {
        nvml_wrapper::Nvml::init()
            .map(|_| ())
            .map_err(|e| CalibrateError::NvmlInit(e.to_string()))
    }
}

impl MetricsCollector for NvmlCollector {
    fn run(self, tx: flume::Sender<RawSample>, stop: Arc<AtomicBool>) {
        // NVML is initialized once per thread — re-using across calls on the
        // same thread is safe and avoids per-sample init overhead.
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(n) => n,
            Err(e) => {
                error!("NVML init failed in collector thread: {e}");
                return;
            }
        };

        loop {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            for &gpu_index in &self.gpu_indices {
                match sample_device(&nvml, gpu_index, timestamp_ms) {
                    Ok(sample) => {
                        if tx.send(sample).is_err() {
                            // Receiver dropped — normal shutdown path.
                            return;
                        }
                    }
                    Err(e) => {
                        warn!("GPU {gpu_index} sample failed: {e}");
                    }
                }
            }

            std::thread::sleep(self.interval);
            debug!("NvmlCollector tick (interval={:?})", self.interval);
        }
    }
}

/// Query a single device and return a [`RawSample`].
fn sample_device(
    nvml: &nvml_wrapper::Nvml,
    gpu_index: u32,
    timestamp_ms: u64,
) -> Result<RawSample, CalibrateError> {
    use nvml_wrapper::enum_wrappers::device::Clock;

    let device = nvml
        .device_by_index(gpu_index)
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let util = device
        .utilization_rates()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let sm_clock = device
        .clock_info(Clock::SM)
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let sm_clock_max = device
        .max_clock_info(Clock::SM)
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let mem_info = device
        .memory_info()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let temp = device
        .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let power_mw = device
        .power_usage()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let power_limit_mw = device
        .power_management_limit()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let throttle_reasons = device
        .current_throttle_reasons()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    use nvml_wrapper::bitmasks::device::ThrottleReasons;

    Ok(RawSample {
        timestamp_ms,
        gpu_index,
        sm_utilization: Percent::clamped(util.gpu as f32),
        sm_clock_mhz: Mhz(sm_clock),
        sm_clock_max_mhz: Mhz(sm_clock_max),
        vram_used_mib: Mib(mem_info.used / (1024 * 1024)),
        vram_total_mib: Mib(mem_info.total / (1024 * 1024)),
        mem_utilization: Percent::clamped(util.memory as f32),
        temperature: Celsius(temp as f32),
        power_draw: Watts(power_mw as f32 / 1000.0),
        power_limit: Watts(power_limit_mw as f32 / 1000.0),
        throttle_thermal: throttle_reasons.contains(ThrottleReasons::HW_THERMAL_SLOWDOWN),
        throttle_power: throttle_reasons.contains(ThrottleReasons::SW_POWER_CAP),
        throttle_hw_slowdown: throttle_reasons.contains(ThrottleReasons::HW_SLOWDOWN),
        // cpu_utilization is patched in by the orchestrator after /proc sampling.
        cpu_utilization: Percent(0.0),
    })
}
