use std::os::raw::{c_char, c_int, c_void};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::{debug, info, warn};

use crate::collectors::{MetricsCollector, RawSample};
use crate::error::CalibrateError;
use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

// ── IOKit / CoreFoundation type aliases ───────────────────────────────────────

type IOObject = u32;
type KernReturn = c_int;

const KERN_SUCCESS: KernReturn = 0;
const CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
const CF_NUMBER_SINT64_TYPE: c_int = 4;

// ── IOKit FFI ─────────────────────────────────────────────────────────────────

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingServices(
        master_port: IOObject,
        matching: *mut c_void,
        existing: *mut IOObject,
    ) -> KernReturn;
    fn IOIteratorNext(iterator: IOObject) -> IOObject;
    fn IOObjectRelease(object: IOObject) -> KernReturn;
    fn IORegistryEntryCreateCFProperties(
        entry: IOObject,
        properties: *mut *mut c_void,
        allocator: *mut c_void,
        options: u32,
    ) -> KernReturn;
}

// ── CoreFoundation FFI ────────────────────────────────────────────────────────

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
    fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const c_char,
        encoding: u32,
    ) -> *mut c_void;
    fn CFNumberGetValue(
        number: *const c_void,
        the_type: c_int,
        value_ptr: *mut c_void,
    ) -> bool;
}

// ── CF helpers ────────────────────────────────────────────────────────────────

/// Look up a value in a `CFDictionary` by a NUL-terminated UTF-8 key.
///
/// Returns `None` if the key does not exist or the CF string could not be
/// created.  The returned pointer is owned by the dictionary; do not release
/// it.
///
/// # Safety
/// `dict` must be a valid, live `CFDictionaryRef`.
/// `key` must be a NUL-terminated byte slice.
unsafe fn cf_dict_value(dict: *const c_void, key: &[u8]) -> Option<*const c_void> {
    debug_assert!(*key.last().unwrap() == 0, "key must be NUL-terminated");
    let cf_key = CFStringCreateWithCString(
        std::ptr::null(),
        key.as_ptr() as *const c_char,
        CF_STRING_ENCODING_UTF8,
    );
    if cf_key.is_null() {
        return None;
    }
    let value = CFDictionaryGetValue(dict, cf_key);
    CFRelease(cf_key);
    if value.is_null() { None } else { Some(value) }
}

/// Extract an `i64` from a `CFDictionary` entry that holds a `CFNumber`.
///
/// # Safety
/// `dict` must be a valid, live `CFDictionaryRef`.
/// `key` must be a NUL-terminated byte slice.
unsafe fn cf_dict_i64(dict: *const c_void, key: &[u8]) -> Option<i64> {
    let value = cf_dict_value(dict, key)?;
    let mut result: i64 = 0;
    let ok = CFNumberGetValue(
        value,
        CF_NUMBER_SINT64_TYPE,
        &mut result as *mut i64 as *mut c_void,
    );
    if ok { Some(result) } else { None }
}

// ── GPU snapshot ──────────────────────────────────────────────────────────────

struct GpuSnapshot {
    utilization_pct: f32,
    memory_used_bytes: u64,
}

/// Query all IOAccelerator services and return one `GpuSnapshot` per device.
///
/// On Apple Silicon the underlying service is `AGXAccelerator`, which
/// conforms to the `IOAccelerator` protocol and is therefore discovered by
/// `IOServiceMatching("IOAccelerator")`.
///
/// `IOServiceGetMatchingServices` consumes (releases) the `matching`
/// dictionary, so we must not call `CFRelease` on it afterward.
fn query_gpu_stats() -> Result<Vec<GpuSnapshot>, CalibrateError> {
    let mut snapshots = Vec::new();

    unsafe {
        let matching = IOServiceMatching(b"IOAccelerator\0".as_ptr() as *const c_char);
        if matching.is_null() {
            return Err(CalibrateError::AppleGpuInit(
                "IOServiceMatching returned null".into(),
            ));
        }

        let mut iterator: IOObject = 0;
        // kIOMasterPortDefault == MACH_PORT_NULL == 0 on macOS 12+.
        let kr = IOServiceGetMatchingServices(0, matching, &mut iterator);
        if kr != KERN_SUCCESS || iterator == 0 {
            return Ok(snapshots);
        }

        loop {
            let service = IOIteratorNext(iterator);
            if service == 0 {
                break;
            }

            let mut props: *mut c_void = std::ptr::null_mut();
            let kr = IORegistryEntryCreateCFProperties(
                service,
                &mut props,
                std::ptr::null_mut(),
                0,
            );
            IOObjectRelease(service);

            if kr != KERN_SUCCESS || props.is_null() {
                continue;
            }

            if let Some(snap) = extract_snapshot(props) {
                snapshots.push(snap);
            }

            CFRelease(props);
        }

        IOObjectRelease(iterator);
    }

    Ok(snapshots)
}

/// Extract a `GpuSnapshot` from the property dictionary returned by
/// `IORegistryEntryCreateCFProperties`.
///
/// The `PerformanceStatistics` sub-dictionary holds the keys we care about.
/// Key names are consistent across Apple Silicon and AMD Intel GPUs for the
/// utilization field; the memory field is Apple Silicon-specific.
///
/// # Safety
/// `props` must be a valid, live `CFDictionaryRef`.
unsafe fn extract_snapshot(props: *const c_void) -> Option<GpuSnapshot> {
    let perf = cf_dict_value(props, b"PerformanceStatistics\0")?;

    // Try both key variants: Apple Silicon uses "Device Utilization %";
    // some AMD/Intel models expose "GPU Core Utilization" instead.
    let util_pct = cf_dict_i64(perf, b"Device Utilization %\0")
        .or_else(|| cf_dict_i64(perf, b"GPU Core Utilization\0"))
        .unwrap_or(0)
        .clamp(0, 100) as f32;

    let mem_used = cf_dict_i64(perf, b"In use system memory\0")
        .unwrap_or(0)
        .max(0) as u64;

    Some(GpuSnapshot {
        utilization_pct: util_pct,
        memory_used_bytes: mem_used,
    })
}

// ── System memory ─────────────────────────────────────────────────────────────

/// Total physical RAM in bytes via `sysctl hw.memsize`.
///
/// On Apple Silicon this equals the GPU's VRAM budget because of the unified
/// memory architecture.  Returns 0 on `sysctl` failure (should never happen).
fn total_memory_bytes() -> u64 {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            b"hw.memsize\0".as_ptr() as *const libc::c_char,
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { size } else { 0 }
}

// ── Process liveness ──────────────────────────────────────────────────────────

/// Returns `true` if `pid` is still alive.
///
/// Uses `kill(pid, 0)` (POSIX), which succeeds without sending a signal when
/// the process exists and is visible to the current user.
fn process_is_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

// ── Collector ─────────────────────────────────────────────────────────────────

/// Collects Apple Silicon (and AMD/Intel Mac) GPU metrics via IOKit on a
/// dedicated OS thread.
///
/// Provides **system-wide** GPU utilization and unified-memory usage.
/// macOS does not expose per-process GPU isolation to userspace without Metal
/// instrumentation, so utilization reflects the whole GPU — consistent with
/// Activity Monitor's "GPU" column.
///
/// `vram_total_mib` is populated with total physical RAM, which is the correct
/// VRAM budget on Apple Silicon unified-memory systems.
///
/// CPU utilization (`cpu_utilization`) is left at `0.0` in this initial
/// implementation; per-process CPU tracking via `proc_pidinfo` will be added
/// in the Phase 2 `watch` command integration.
pub struct AppleGpuCollector {
    pid: u32,
    interval: Duration,
}

impl AppleGpuCollector {
    pub fn new(pid: u32, interval: Duration) -> Self {
        Self { pid, interval }
    }

    /// Attempt a single IOKit GPU query to verify the framework is accessible.
    ///
    /// Called before spawning the collector thread so startup fails fast with
    /// a human-readable error if IOKit is unavailable.
    pub fn probe() -> Result<(), CalibrateError> {
        query_gpu_stats().map(|_| ())
    }
}

impl MetricsCollector for AppleGpuCollector {
    fn run(self, tx: flume::Sender<RawSample>, stop: Arc<AtomicBool>) {
        let total_bytes = total_memory_bytes();
        let total_mib = Mib((total_bytes / (1024 * 1024)) as u64);

        info!(pid = self.pid, "AppleGpuCollector started");

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }

            if !process_is_alive(self.pid) {
                info!(
                    pid = self.pid,
                    "Process exited — AppleGpuCollector stopping"
                );
                stop.store(true, Ordering::Relaxed);
                break;
            }

            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let stats = match query_gpu_stats() {
                Ok(s) => s,
                Err(e) => {
                    warn!("AppleGpuCollector: IOKit query failed: {e}");
                    std::thread::sleep(self.interval);
                    continue;
                }
            };

            for (idx, snap) in stats.into_iter().enumerate() {
                let mem_pct = if total_bytes > 0 {
                    (snap.memory_used_bytes as f32 / total_bytes as f32) * 100.0
                } else {
                    0.0
                };

                let sample = RawSample {
                    timestamp_ms,
                    gpu_index: idx as u32,
                    sm_utilization: Percent::clamped(snap.utilization_pct),
                    sm_clock_mhz: Mhz(0),
                    sm_clock_max_mhz: Mhz(0),
                    vram_used_mib: Mib(snap.memory_used_bytes / (1024 * 1024)),
                    vram_total_mib: total_mib,
                    mem_utilization: Percent::clamped(mem_pct),
                    temperature: Celsius(0.0),
                    power_draw: Watts(0.0),
                    power_limit: Watts(0.0),
                    throttle_thermal: false,
                    throttle_power: false,
                    throttle_hw_slowdown: false,
                    cpu_utilization: Percent(0.0),
                };

                if tx.send(sample).is_err() {
                    return;
                }
            }

            debug!(
                "AppleGpuCollector tick (pid={}, interval={:?})",
                self.pid, self.interval
            );
            std::thread::sleep(self.interval);
        }
    }
}

/// Return the total system-wide GPU memory in use (MiB) across all IOAccelerator
/// devices.
///
/// Used by `bench::memory` on macOS as the closest available equivalent to
/// NVML's per-device VRAM usage query.  On Apple Silicon unified memory, this
/// reflects the GPU memory pressure of all running processes.
pub fn gpu_memory_used_mib() -> anyhow::Result<f64> {
    let snapshots = query_gpu_stats()
        .map_err(|e| anyhow::anyhow!("IOKit GPU memory query failed: {e}"))?;
    let total_bytes: u64 = snapshots.iter().map(|s| s.memory_used_bytes).sum();
    Ok(total_bytes as f64 / (1024.0 * 1024.0))
}
