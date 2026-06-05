use anyhow::Result;

/// A point-in-time memory snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemorySnapshot {
    /// GPU VRAM used (MiB). 0.0 if NVML is unavailable.
    pub vram_used_mib: f64,
    /// Process resident set size (MiB).
    pub rss_mib: f64,
}

impl MemorySnapshot {
    /// Capture the current memory state.
    ///
    /// `device_index` is the NVML device index to query for VRAM.
    /// Pass `None` on CPU-only machines.
    pub fn take(device_index: Option<u32>) -> Self {
        let vram = device_index
            .and_then(|idx| read_vram_mib(idx).ok())
            .unwrap_or(0.0);
        let rss = read_rss_mib().unwrap_or(0.0);
        Self {
            vram_used_mib: vram,
            rss_mib: rss,
        }
    }

    /// Compute how much more memory is used relative to a baseline snapshot.
    pub fn delta_from(&self, baseline: &MemorySnapshot) -> MemoryDelta {
        MemoryDelta {
            vram_delta_mib: (self.vram_used_mib - baseline.vram_used_mib).max(0.0),
            rss_delta_mib: (self.rss_mib - baseline.rss_mib).max(0.0),
        }
    }
}

/// Difference in memory usage between two snapshots.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryDelta {
    pub vram_delta_mib: f64,
    pub rss_delta_mib: f64,
}

/// Tracks the highest memory watermark observed across multiple readings.
#[derive(Debug, Default)]
pub struct PeakMemoryTracker {
    peak: MemorySnapshot,
}

impl PeakMemoryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sample current memory and update the peak if this reading is higher.
    pub fn update(&mut self, device_index: Option<u32>) {
        let current = MemorySnapshot::take(device_index);
        if current.vram_used_mib > self.peak.vram_used_mib {
            self.peak.vram_used_mib = current.vram_used_mib;
        }
        if current.rss_mib > self.peak.rss_mib {
            self.peak.rss_mib = current.rss_mib;
        }
    }

    pub fn peak(&self) -> MemorySnapshot {
        self.peak
    }
}

/// Read GPU VRAM usage for the given device index (MiB).
///
/// On Linux: queries NVML for the specific device.
/// On macOS: queries IOKit for system-wide GPU memory; `device_index` is ignored
/// because Apple Silicon has a single unified GPU visible via IOKit.
#[cfg(target_os = "linux")]
fn read_vram_mib(device_index: u32) -> Result<f64> {
    let nvml = nvml_wrapper::Nvml::init()?;
    let device = nvml.device_by_index(device_index)?;
    let info = device.memory_info()?;
    Ok(info.used as f64 / (1024.0 * 1024.0))
}

#[cfg(target_os = "macos")]
fn read_vram_mib(_device_index: u32) -> Result<f64> {
    crate::collectors::apple_gpu::gpu_memory_used_mib()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn read_vram_mib(_device_index: u32) -> Result<f64> {
    anyhow::bail!("GPU memory reading not supported on this platform")
}

/// Read the current process resident set size (MiB).
///
/// Uses `getrusage(RUSAGE_SELF)` which is available on Linux and macOS.
/// Returns 0.0 on unsupported platforms.
fn read_rss_mib() -> Result<f64> {
    #[cfg(unix)]
    {
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if ret != 0 {
            anyhow::bail!("getrusage failed with errno {}", ret);
        }
        // Linux: ru_maxrss is in kilobytes.
        // macOS: ru_maxrss is in bytes.
        #[cfg(target_os = "macos")]
        let rss = usage.ru_maxrss as f64 / (1024.0 * 1024.0);
        #[cfg(not(target_os = "macos"))]
        let rss = usage.ru_maxrss as f64 / 1024.0;
        Ok(rss)
    }
    #[cfg(not(unix))]
    {
        Ok(0.0)
    }
}
