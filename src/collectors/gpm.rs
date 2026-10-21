//! NVML GPU Performance Monitoring (GPM).
//!
//! GPM exposes profiling-grade counters — SM activity, occupancy, tensor-core
//! pipe activity, DRAM bandwidth — through plain NVML on Hopper and newer
//! GPUs, without DCGM. The symbols are resolved at runtime so a driver that
//! predates GPM (or a GPU that does not support it) simply falls back to
//! kernel-busy utilization instead of failing.

use std::collections::HashMap;
use std::ffi::{c_char, c_uint, c_void};

use crate::collectors::{ProfMetrics, ProfSource};

const NVML_SUCCESS: c_uint = 0;
const GPM_METRICS_GET_VERSION: c_uint = 1;
const GPM_SUPPORT_VERSION: c_uint = 1;
const GPM_METRIC_MAX: usize = 98;

const METRIC_SM_UTIL: c_uint = 2;
const METRIC_SM_OCCUPANCY: c_uint = 3;
const METRIC_ANY_TENSOR_UTIL: c_uint = 5;
const METRIC_DRAM_BW_UTIL: c_uint = 10;

type Sample = *mut c_void;
type DeviceHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct GpmMetricInfo {
    short_name: *mut c_char,
    long_name: *mut c_char,
    unit: *mut c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpmMetric {
    metric_id: c_uint,
    nvml_return: c_uint,
    value: f64,
    info: GpmMetricInfo,
}

#[repr(C)]
struct GpmMetricsGet {
    version: c_uint,
    num_metrics: c_uint,
    sample1: Sample,
    sample2: Sample,
    metrics: [GpmMetric; GPM_METRIC_MAX],
}

#[repr(C)]
struct GpmSupport {
    version: c_uint,
    is_supported_device: c_uint,
}

type QuerySupportFn = unsafe extern "C" fn(DeviceHandle, *mut GpmSupport) -> c_uint;
type SampleAllocFn = unsafe extern "C" fn(*mut Sample) -> c_uint;
type SampleFreeFn = unsafe extern "C" fn(Sample) -> c_uint;
type SampleGetFn = unsafe extern "C" fn(DeviceHandle, Sample) -> c_uint;
type MetricsGetFn = unsafe extern "C" fn(*mut GpmMetricsGet) -> c_uint;

struct DeviceState {
    prev: Sample,
    cur: Sample,
    has_prev: bool,
}

/// Loaded GPM entry points plus per-device sample buffers.
pub struct Gpm {
    _lib: libloading::Library,
    query_support: QuerySupportFn,
    sample_alloc: SampleAllocFn,
    sample_free: SampleFreeFn,
    sample_get: SampleGetFn,
    metrics_get: MetricsGetFn,
    devices: HashMap<u32, DeviceState>,
    unsupported: std::collections::HashSet<u32>,
}

// The raw sample pointers are only touched from the owning collector thread.
unsafe impl Send for Gpm {}

impl Gpm {
    /// Resolve the GPM symbols from the NVML library. `None` when NVML is
    /// missing or too old.
    pub fn load() -> Option<Self> {
        let names: &[&str] = if cfg!(target_os = "windows") {
            &["nvml.dll"]
        } else {
            &["libnvidia-ml.so.1", "libnvidia-ml.so"]
        };
        // SAFETY: loading the vendor NVML library; its initializers are the
        // same ones nvml-wrapper already ran.
        let lib = names
            .iter()
            .find_map(|n| unsafe { libloading::Library::new(n).ok() })?;
        // SAFETY: the signatures match nvml.h for these symbols.
        unsafe {
            let query_support = *lib
                .get::<QuerySupportFn>(b"nvmlGpmQueryDeviceSupport\0")
                .ok()?;
            let sample_alloc = *lib.get::<SampleAllocFn>(b"nvmlGpmSampleAlloc\0").ok()?;
            let sample_free = *lib.get::<SampleFreeFn>(b"nvmlGpmSampleFree\0").ok()?;
            let sample_get = *lib.get::<SampleGetFn>(b"nvmlGpmSampleGet\0").ok()?;
            let metrics_get = *lib.get::<MetricsGetFn>(b"nvmlGpmMetricsGet\0").ok()?;
            Some(Self {
                _lib: lib,
                query_support,
                sample_alloc,
                sample_free,
                sample_get,
                metrics_get,
                devices: HashMap::new(),
                unsupported: Default::default(),
            })
        }
    }

    /// Whether the device supports GPM.
    pub fn is_supported(&self, device: &nvml_wrapper::Device) -> bool {
        // SAFETY: valid device handle from nvml-wrapper; struct is POD.
        unsafe {
            let mut s = GpmSupport {
                version: GPM_SUPPORT_VERSION,
                is_supported_device: 0,
            };
            let r = (self.query_support)(device.handle() as DeviceHandle, &mut s);
            r == NVML_SUCCESS && s.is_supported_device != 0
        }
    }

    /// Take a GPM sample and return metrics for the interval since the
    /// previous call for this device. The first call per device returns `None`.
    pub fn sample(&mut self, index: u32, device: &nvml_wrapper::Device) -> Option<ProfMetrics> {
        if self.unsupported.contains(&index) {
            return None;
        }
        if !self.devices.contains_key(&index) {
            if !self.is_supported(device) {
                self.unsupported.insert(index);
                return None;
            }
            let mut a: Sample = std::ptr::null_mut();
            let mut b: Sample = std::ptr::null_mut();
            // SAFETY: allocation out-params.
            unsafe {
                if (self.sample_alloc)(&mut a) != NVML_SUCCESS
                    || (self.sample_alloc)(&mut b) != NVML_SUCCESS
                {
                    self.unsupported.insert(index);
                    return None;
                }
            }
            self.devices.insert(
                index,
                DeviceState {
                    prev: a,
                    cur: b,
                    has_prev: false,
                },
            );
        }
        let handle = unsafe { device.handle() } as DeviceHandle;
        let st = self.devices.get_mut(&index)?;
        // SAFETY: handle and sample buffers are valid for this device.
        if unsafe { (self.sample_get)(handle, st.cur) } != NVML_SUCCESS {
            return None;
        }
        let result = if st.has_prev {
            // SAFETY: zeroed is a valid bit pattern for this C struct.
            let mut req: GpmMetricsGet = unsafe { std::mem::zeroed() };
            req.version = GPM_METRICS_GET_VERSION;
            req.sample1 = st.prev;
            req.sample2 = st.cur;
            let ids = [
                METRIC_SM_UTIL,
                METRIC_SM_OCCUPANCY,
                METRIC_ANY_TENSOR_UTIL,
                METRIC_DRAM_BW_UTIL,
            ];
            req.num_metrics = ids.len() as c_uint;
            for (slot, id) in req.metrics.iter_mut().zip(ids) {
                slot.metric_id = id;
            }
            // SAFETY: request is fully initialised.
            if unsafe { (self.metrics_get)(&mut req) } == NVML_SUCCESS {
                let value = |id: c_uint| {
                    req.metrics[..ids.len()]
                        .iter()
                        .find(|m| m.metric_id == id && m.nvml_return == NVML_SUCCESS)
                        .map(|m| m.value.clamp(0.0, 100.0) as f32)
                };
                Some(ProfMetrics {
                    source: ProfSource::Gpm,
                    sm_active_pct: value(METRIC_SM_UTIL),
                    sm_occupancy_pct: value(METRIC_SM_OCCUPANCY),
                    tensor_active_pct: value(METRIC_ANY_TENSOR_UTIL),
                    dram_active_pct: value(METRIC_DRAM_BW_UTIL),
                })
            } else {
                None
            }
        } else {
            None
        };
        std::mem::swap(&mut st.prev, &mut st.cur);
        st.has_prev = true;
        result
    }
}

impl Drop for Gpm {
    fn drop(&mut self) {
        for st in self.devices.values() {
            // SAFETY: buffers were allocated by sample_alloc.
            unsafe {
                (self.sample_free)(st.prev);
                (self.sample_free)(st.cur);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layout_matches_nvml_h() {
        // nvmlGpmMetric_t: uint, enum, double, 3 pointers.
        assert_eq!(
            std::mem::size_of::<GpmMetric>(),
            16 + 3 * std::mem::size_of::<usize>()
        );
        // nvmlGpmMetricsGet_t: 2 uints, 2 pointers, 98 metrics.
        assert_eq!(
            std::mem::size_of::<GpmMetricsGet>(),
            8 + 2 * std::mem::size_of::<usize>() + 98 * std::mem::size_of::<GpmMetric>()
        );
    }

    #[test]
    fn load_is_none_without_nvml() {
        // On machines without an NVIDIA driver this must not panic.
        if std::path::Path::new("/dev/nvidiactl").exists() {
            return;
        }
        assert!(Gpm::load().is_none());
    }
}
