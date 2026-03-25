use std::sync::{atomic::AtomicBool, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::{debug, error, info, warn};

use crate::collectors::{MetricsCollector, RawSample};
use crate::error::CalibrateError;
use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

/// Collects GPU metrics via NVML on a dedicated OS thread.
///
/// NVML is synchronous and must not be called from a tokio task.
/// GPU discovery is performed on every tick so multi-GPU jobs that acquire
/// devices after startup are automatically tracked.
pub struct NvmlCollector {
    /// PID of the training process being monitored.
    pid: u32,
    interval: Duration,
    shared_cpu: Arc<parking_lot::Mutex<Percent>>,
}

impl NvmlCollector {
    pub fn new(pid: u32, interval: Duration, shared_cpu: Arc<parking_lot::Mutex<Percent>>) -> Self {
        Self {
            pid,
            interval,
            shared_cpu,
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
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(n) => n,
            Err(e) => {
                error!("NVML init failed in collector thread: {e}");
                return;
            }
        };

        let mut consecutive_no_gpu: u32 = 0;
        const NO_GPU_EXIT_THRESHOLD: u32 = 5;

        loop {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let gpu_indices = match discover_gpu_indices(&nvml, self.pid) {
                Ok(v) => v,
                Err(e) => {
                    warn!("GPU discovery failed: {e}");
                    std::thread::sleep(self.interval);
                    continue;
                }
            };

            if gpu_indices.is_empty() {
                consecutive_no_gpu += 1;
                if consecutive_no_gpu >= NO_GPU_EXIT_THRESHOLD {
                    info!(
                        pid = self.pid,
                        "No GPU activity for {} consecutive ticks — process likely exited",
                        consecutive_no_gpu
                    );
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(self.interval);
                continue;
            }
            consecutive_no_gpu = 0;

            let cpu_pct = *self.shared_cpu.lock();

            for gpu_index in gpu_indices {
                match sample_device(&nvml, gpu_index, timestamp_ms, cpu_pct) {
                    Ok(sample) => {
                        if tx.send(sample).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        warn!("GPU {gpu_index} sample failed: {e}");
                    }
                }
            }

            std::thread::sleep(self.interval);
            debug!(
                "NvmlCollector tick (pid={}, interval={:?})",
                self.pid, self.interval
            );
        }
    }
}

/// Return the NVML device indices that have `pid` in their compute-processes list.
fn discover_gpu_indices(nvml: &nvml_wrapper::Nvml, pid: u32) -> Result<Vec<u32>, CalibrateError> {
    let device_count = nvml
        .device_count()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let mut indices = Vec::new();
    for i in 0..device_count {
        let device = nvml
            .device_by_index(i)
            .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

        let processes = device
            .running_compute_processes()
            .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

        if processes.iter().any(|p| p.pid == pid) {
            indices.push(i);
        }
    }
    Ok(indices)
}

/// Query a single NVML device and return a [`RawSample`].
fn sample_device(
    nvml: &nvml_wrapper::Nvml,
    gpu_index: u32,
    timestamp_ms: u64,
    cpu_pct: Percent,
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
        // cpu_utilization is provided by the companion ProcCollector thread.
        cpu_utilization: cpu_pct,
    })
}
