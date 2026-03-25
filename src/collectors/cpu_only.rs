use std::sync::{atomic::AtomicBool, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::{error, info, warn};

use crate::collectors::{MetricsCollector, RawSample};
use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

/// Fallback collector used when NVML is unavailable (non-NVIDIA GPU or missing
/// nvidia driver).
///
/// Produces `RawSample` values from `/proc/{pid}/stat` only.  All GPU-specific
/// fields (SM utilisation, VRAM, temperature, power) are zero so the analytics
/// pipeline can still run and display "N/A" rather than crashing.
///
/// Run on a dedicated `std::thread` just like `NvmlCollector`.
pub struct CpuOnlyCollector {
    pid: u32,
    interval: Duration,
}

impl CpuOnlyCollector {
    pub fn new(pid: u32, interval: Duration) -> Self {
        Self { pid, interval }
    }
}

impl MetricsCollector for CpuOnlyCollector {
    fn run(self, tx: flume::Sender<RawSample>, stop: Arc<AtomicBool>) {
        use crate::collectors::proc::ProcCollector;

        let mut proc = match ProcCollector::new(self.pid) {
            Ok(c) => c,
            Err(e) => {
                error!("CpuOnlyCollector: ProcCollector init failed: {e}");
                return;
            }
        };

        info!(pid = self.pid, "CpuOnlyCollector started (NVML unavailable)");

        loop {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            if !proc.is_alive() {
                info!(pid = self.pid, "Process exited — CpuOnlyCollector stopping");
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                break;
            }

            let cpu_pct = match proc.sample() {
                Ok(p) => p,
                Err(e) => {
                    warn!("CpuOnlyCollector: /proc read error: {e}");
                    Percent(0.0)
                }
            };

            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let sample = RawSample {
                timestamp_ms,
                gpu_index: 0,
                sm_utilization: Percent(0.0),
                sm_clock_mhz: Mhz(0),
                sm_clock_max_mhz: Mhz(0),
                vram_used_mib: Mib(0),
                vram_total_mib: Mib(0),
                mem_utilization: Percent(0.0),
                temperature: Celsius(0.0),
                power_draw: Watts(0.0),
                power_limit: Watts(0.0),
                throttle_thermal: false,
                throttle_power: false,
                throttle_hw_slowdown: false,
                cpu_utilization: cpu_pct,
            };

            if tx.send(sample).is_err() {
                break;
            }

            std::thread::sleep(self.interval);
        }
    }
}
