//! NVIDIA backend: NVML device queries plus optional GPM / DCGM profiling.

use std::collections::BTreeSet;
use std::time::Duration;

use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;
use tracing::{debug, info, warn};

use crate::cli::MetricsSource;
use crate::collectors::dcgm::DcgmReader;
use crate::collectors::gpm::Gpm;
use crate::collectors::{GpuSample, Throttle};
use crate::error::CalibrateError;

/// Initialise NVML with an actionable error.
pub fn init() -> Result<Nvml, CalibrateError> {
    Nvml::init().map_err(|e| CalibrateError::NvmlInit(e.to_string()))
}

/// Which profiling source is in effect (reported in snapshots).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveSource {
    Gpm,
    Dcgm,
    /// GPM where supported, kernel-busy elsewhere.
    Mixed,
    Nvml,
}

impl ActiveSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Gpm => "nvml-gpm",
            Self::Dcgm => "dcgm",
            Self::Mixed => "nvml-gpm+nvml",
            Self::Nvml => "nvml",
        }
    }
}

pub struct NvmlBackend {
    nvml: Nvml,
    gpm: Option<Gpm>,
    dcgm: Option<DcgmReader>,
    source: ActiveSource,
}

impl NvmlBackend {
    pub fn new(requested: MetricsSource, interval: Duration) -> Result<Self, CalibrateError> {
        let nvml = init()?;
        let count = nvml
            .device_count()
            .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

        let gpm_support = |gpm: &Gpm| -> (usize, usize) {
            let mut ok = 0;
            for i in 0..count {
                if let Ok(d) = nvml.device_by_index(i) {
                    if gpm.is_supported(&d) {
                        ok += 1;
                    }
                }
            }
            (ok, count as usize)
        };

        let (gpm, dcgm, source) = match requested {
            MetricsSource::Nvml => (None, None, ActiveSource::Nvml),
            MetricsSource::Gpm => {
                let gpm = Gpm::load().ok_or_else(|| {
                    CalibrateError::ProfilingUnavailable(
                        "NVML GPM symbols not found — requires driver R520+ and a Hopper or newer GPU"
                            .into(),
                    )
                })?;
                let (ok, total) = gpm_support(&gpm);
                if ok == 0 {
                    return Err(CalibrateError::ProfilingUnavailable(
                        "no GPU on this system supports NVML GPM (Hopper or newer required); \
                         try --metrics-source dcgm"
                            .into(),
                    ));
                }
                let src = if ok == total {
                    ActiveSource::Gpm
                } else {
                    ActiveSource::Mixed
                };
                (Some(gpm), None, src)
            }
            MetricsSource::Dcgm => {
                if !DcgmReader::available() {
                    return Err(CalibrateError::ProfilingUnavailable(
                        "`dcgmi` not found or the DCGM host engine is not running \
                         (install datacenter-gpu-manager and start nv-hostengine)"
                            .into(),
                    ));
                }
                let r = DcgmReader::start(&[], interval)
                    .map_err(|e| CalibrateError::ProfilingUnavailable(e.to_string()))?;
                (None, Some(r), ActiveSource::Dcgm)
            }
            MetricsSource::Auto => {
                let gpm = Gpm::load();
                let (ok, total) = gpm.as_ref().map(gpm_support).unwrap_or((0, count as usize));
                if total > 0 && ok == total {
                    (gpm, None, ActiveSource::Gpm)
                } else if DcgmReader::available() {
                    match DcgmReader::start(&[], interval) {
                        Ok(r) => (None, Some(r), ActiveSource::Dcgm),
                        Err(e) => {
                            warn!("dcgmi dmon failed to start: {e}");
                            mixed_or_nvml(gpm, ok)
                        }
                    }
                } else {
                    mixed_or_nvml(gpm, ok)
                }
            }
        };
        info!(source = source.label(), "NVML backend ready");
        Ok(Self {
            nvml,
            gpm,
            dcgm,
            source,
        })
    }

    pub fn source(&self) -> ActiveSource {
        self.source
    }

    pub fn device_count(&self) -> u32 {
        self.nvml.device_count().unwrap_or(0)
    }

    /// GPUs on which any of `pids` has a compute or graphics context.
    pub fn devices_for_pids(&self, pids: &[u32]) -> Result<Vec<u32>, CalibrateError> {
        let wanted: BTreeSet<u32> = pids.iter().copied().collect();
        let mut out = Vec::new();
        for i in 0..self.device_count() {
            let device = self
                .nvml
                .device_by_index(i)
                .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;
            let mut hit = false;
            if let Ok(procs) = device.running_compute_processes() {
                hit |= procs.iter().any(|p| wanted.contains(&p.pid));
            }
            if !hit {
                if let Ok(procs) = device.running_graphics_processes() {
                    hit |= procs.iter().any(|p| wanted.contains(&p.pid));
                }
            }
            if hit {
                out.push(i);
            }
        }
        Ok(out)
    }

    /// PIDs NVML reports on any device (used to detect PID-namespace mismatch).
    pub fn all_gpu_pids(&self) -> Vec<u32> {
        let mut out = BTreeSet::new();
        for i in 0..self.device_count() {
            if let Ok(d) = self.nvml.device_by_index(i) {
                for p in d.running_compute_processes().unwrap_or_default() {
                    out.insert(p.pid);
                }
            }
        }
        out.into_iter().collect()
    }

    /// Sample the given devices. Devices that fail entirely are skipped.
    pub fn sample(&mut self, indices: &[u32]) -> Vec<GpuSample> {
        let mut out = Vec::with_capacity(indices.len());
        if let Some(d) = self.dcgm.as_mut() {
            if !d.is_running() {
                warn!("dcgmi dmon exited — falling back to NVML utilization");
                self.dcgm = None;
                self.source = ActiveSource::Nvml;
            }
        }
        for &index in indices {
            let device = match self.nvml.device_by_index(index) {
                Ok(d) => d,
                Err(e) => {
                    warn!("GPU {index}: {e}");
                    continue;
                }
            };
            let prof = if let Some(g) = self.gpm.as_mut() {
                g.sample(index, &device)
            } else {
                self.dcgm.as_ref().and_then(|d| d.get(index))
            };
            let util = device.utilization_rates().ok();
            let mem = device.memory_info().ok();
            let throttle = device
                .current_throttle_reasons()
                .map(|r| Throttle {
                    hw_thermal: r.contains(ThrottleReasons::HW_THERMAL_SLOWDOWN),
                    sw_thermal: r.contains(ThrottleReasons::SW_THERMAL_SLOWDOWN),
                    power_cap: r.contains(ThrottleReasons::SW_POWER_CAP),
                    hw_slowdown: r.contains(ThrottleReasons::HW_SLOWDOWN)
                        || r.contains(ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN),
                })
                .unwrap_or_default();
            out.push(GpuSample {
                index,
                name: device.name().unwrap_or_else(|_| format!("GPU {index}")),
                util_pct: util.as_ref().map(|u| (u.gpu as f32).clamp(0.0, 100.0)),
                prof,
                sm_clock_mhz: device.clock_info(Clock::SM).ok(),
                sm_clock_max_mhz: device.max_clock_info(Clock::SM).ok(),
                vram_used_mib: mem.as_ref().map(|m| m.used / (1024 * 1024)),
                vram_total_mib: mem.as_ref().map(|m| m.total / (1024 * 1024)),
                mem_util_pct: util.as_ref().map(|u| (u.memory as f32).clamp(0.0, 100.0)),
                temperature_c: device
                    .temperature(TemperatureSensor::Gpu)
                    .ok()
                    .map(|t| t as f32),
                power_w: device.power_usage().ok().map(|mw| mw as f32 / 1000.0),
                power_limit_w: device
                    .enforced_power_limit()
                    .or_else(|_| device.power_management_limit())
                    .ok()
                    .map(|mw| mw as f32 / 1000.0),
                throttle,
            });
        }
        debug!(n = out.len(), "NVML sampled");
        out
    }
}

fn mixed_or_nvml(
    gpm: Option<Gpm>,
    supported: usize,
) -> (Option<Gpm>, Option<DcgmReader>, ActiveSource) {
    if supported > 0 {
        (gpm, None, ActiveSource::Mixed)
    } else {
        (None, None, ActiveSource::Nvml)
    }
}
