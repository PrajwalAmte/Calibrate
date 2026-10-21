use std::os::raw::{c_char, c_int, c_void};

use tracing::warn;

use crate::collectors::{GpuSample, Throttle};
use crate::error::CalibrateError;

// IOKit / CoreFoundation type aliases

type IOObject = u32;
type KernReturn = c_int;

const KERN_SUCCESS: KernReturn = 0;
const CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
const CF_NUMBER_SINT64_TYPE: c_int = 4;

// IOKit FFI

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

// CoreFoundation FFI

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
    fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const c_char,
        encoding: u32,
    ) -> *mut c_void;
    fn CFNumberGetValue(number: *const c_void, the_type: c_int, value_ptr: *mut c_void) -> bool;
}

// CF helpers

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
    if value.is_null() {
        None
    } else {
        Some(value)
    }
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
    if ok {
        Some(result)
    } else {
        None
    }
}

// GPU snapshot

struct GpuSnapshot {
    utilization_pct: Option<f32>,
    memory_used_bytes: Option<u64>,
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
        let matching = IOServiceMatching(c"IOAccelerator".as_ptr());
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
            let kr =
                IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null_mut(), 0);
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
        .map(|v| v.clamp(0, 100) as f32);

    let mem_used = cf_dict_i64(perf, b"In use system memory\0").map(|v| v.max(0) as u64);

    Some(GpuSnapshot {
        utilization_pct: util_pct,
        memory_used_bytes: mem_used,
    })
}

// System memory

/// Total physical RAM in bytes via `sysctl hw.memsize` (0 on failure).
fn total_memory_bytes() -> u64 {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        size
    } else {
        0
    }
}

/// GPU memory budget on unified-memory Macs.
///
/// Metal's `recommendedMaxWorkingSetSize` is what PyTorch MPS treats as the
/// usable limit (typically ~65–75 % of RAM); total RAM overstates it.
fn gpu_budget_bytes() -> u64 {
    let metal = metal::Device::system_default()
        .map(|d| d.recommended_max_working_set_size())
        .unwrap_or(0);
    if metal > 0 {
        metal
    } else {
        total_memory_bytes()
    }
}

/// Apple GPU backend (IOKit). Figures are **system-wide**: macOS exposes no
/// per-process GPU counters without Metal instrumentation inside the target.
pub struct AppleBackend {
    name: String,
    budget_mib: u64,
}

impl AppleBackend {
    pub fn new(name: String) -> Result<Self, CalibrateError> {
        query_gpu_stats()?;
        Ok(Self {
            name,
            budget_mib: gpu_budget_bytes() / (1024 * 1024),
        })
    }

    /// Verify IOKit is reachable.
    pub fn probe() -> Result<(), CalibrateError> {
        query_gpu_stats().map(|_| ())
    }

    pub fn sample(&mut self) -> Vec<GpuSample> {
        let stats = match query_gpu_stats() {
            Ok(s) => s,
            Err(e) => {
                warn!("IOKit query failed: {e}");
                return vec![];
            }
        };
        stats
            .into_iter()
            .enumerate()
            .map(|(idx, snap)| GpuSample {
                index: idx as u32,
                name: self.name.clone(),
                util_pct: snap.utilization_pct,
                prof: None,
                sm_clock_mhz: None,
                sm_clock_max_mhz: None,
                vram_used_mib: snap.memory_used_bytes.map(|b| b / (1024 * 1024)),
                vram_total_mib: (self.budget_mib > 0).then_some(self.budget_mib),
                mem_util_pct: None,
                temperature_c: None,
                power_w: None,
                power_limit_w: None,
                throttle: Throttle::default(),
            })
            .collect()
    }
}

/// Total system-wide GPU memory in use (MiB) across all IOAccelerator devices.
///
/// Used by `bench::memory` on macOS as the closest equivalent to NVML's
/// per-device VRAM usage.
pub fn gpu_memory_used_mib() -> anyhow::Result<f64> {
    let snapshots =
        query_gpu_stats().map_err(|e| anyhow::anyhow!("IOKit GPU memory query failed: {e}"))?;
    let total_bytes: u64 = snapshots.iter().filter_map(|s| s.memory_used_bytes).sum();
    Ok(total_bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iokit_reports_at_least_one_gpu_with_utilization() {
        let stats = query_gpu_stats().expect("IOKit must be reachable on macOS");
        assert!(!stats.is_empty(), "no IOAccelerator found");
        assert!(stats.iter().any(|s| s.utilization_pct.is_some()));
    }

    #[test]
    fn backend_reports_na_instead_of_zero() {
        let mut b = AppleBackend::new("Apple GPU".into()).unwrap();
        let s = b.sample();
        assert!(!s.is_empty());
        assert_eq!(s[0].temperature_c, None);
        assert_eq!(s[0].power_w, None);
        assert!(s[0].vram_total_mib.unwrap_or(0) > 0);
    }
}
