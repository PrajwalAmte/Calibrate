use std::path::Path;

use crate::error::CalibrateError;

/// Details about a successfully attached training process.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub pid: u32,
    /// NVML device indices the process is actively using.
    pub gpu_indices: Vec<u32>,
    /// Human-readable name of the primary GPU (e.g. "NVIDIA GeForce RTX 3090").
    pub primary_gpu_name: String,
    /// Whether the tool itself is running inside a container.
    pub container_context: ContainerContext,
}

/// Result of container-environment detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerContext {
    /// Running on the host — full /proc access expected.
    Host,
    /// Running inside Docker.
    Docker,
    /// Running inside Kubernetes (Docker or containerd runtime).
    Kubernetes,
    /// Container runtime detected but type unknown.
    Unknown,
}

/// Validate that `pid` exists, is a GPU process, and return structured info.
///
/// This is the entry point used by `WatchCommand` before the sampling loop
/// starts.  It fails fast and clearly rather than silently producing empty
/// metrics.
pub fn attach(pid: u32) -> Result<ProcessInfo, CalibrateError> {
    // 1. Verify /proc entry exists.
    let proc_path = format!("/proc/{pid}");
    if !Path::new(&proc_path).exists() {
        return Err(CalibrateError::ProcessNotFound { pid });
    }

    // 2. Check read permission on /proc/{pid}/stat.
    let stat_path = format!("/proc/{pid}/stat");
    std::fs::metadata(&stat_path).map_err(|_| CalibrateError::PermissionDenied { pid })?;

    // 3. Find GPU indices for this PID via NVML.
    let (gpu_indices, primary_gpu_name) = find_gpu_indices(pid)?;
    if gpu_indices.is_empty() {
        return Err(CalibrateError::NoGpuProcess { pid });
    }

    // 4. Detect container environment.
    let container_context = crate::process::container::detect();

    Ok(ProcessInfo {
        pid,
        gpu_indices,
        primary_gpu_name,
        container_context,
    })
}

/// Use NVML to enumerate all devices and return those that have the target
/// PID in their running-compute-processes list.
fn find_gpu_indices(pid: u32) -> Result<(Vec<u32>, String), CalibrateError> {
    let nvml = nvml_wrapper::Nvml::init()
        .map_err(|e| CalibrateError::NvmlInit(e.to_string()))?;

    let device_count = nvml
        .device_count()
        .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

    let mut indices = Vec::new();
    let mut primary_name = String::from("Unknown GPU");

    for i in 0..device_count {
        let device = nvml
            .device_by_index(i)
            .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

        let processes = device
            .running_compute_processes()
            .map_err(|e| CalibrateError::NvmlQuery(e.to_string()))?;

        if processes.iter().any(|p| p.pid == pid) {
            if indices.is_empty() {
                primary_name = device
                    .name()
                    .unwrap_or_else(|_| "Unknown GPU".to_string());
            }
            indices.push(i);
        }
    }

    Ok((indices, primary_name))
}
