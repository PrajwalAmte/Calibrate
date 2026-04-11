use std::path::Path;

use crate::error::CalibrateError;

/// Details about a successfully attached training process.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    #[allow(dead_code)]
    pub pid: u32,
    /// NVML device indices the process is actively using.
    pub gpu_indices: Vec<u32>,
    /// Human-readable name of the primary GPU (e.g. "NVIDIA GeForce RTX 3090").
    pub primary_gpu_name: String,
    /// Whether the tool itself is running inside a container.
    pub container_context: ContainerContext,
    /// Whether NVML was successfully initialised.  False on non-NVIDIA systems
    /// or when the nvidia driver is not loaded.  GPU metrics will be absent.
    pub nvml_available: bool,
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
/// Fails fast with actionable errors rather than silently producing empty metrics.
pub fn attach(pid: u32) -> Result<ProcessInfo, CalibrateError> {
    let proc_path = format!("/proc/{pid}");
    if !Path::new(&proc_path).exists() {
        let ctx = crate::process::container::detect();
        if ctx != ContainerContext::Host {
            return Err(CalibrateError::ContainerPidIsolation { pid });
        }
        return Err(CalibrateError::ProcessNotFound { pid });
    }

    let stat_path = format!("/proc/{pid}/stat");
    std::fs::metadata(&stat_path).map_err(|_| CalibrateError::PermissionDenied { pid })?;

    // NVML init failure is non-fatal: the tool degrades to CPU-only mode.
    let (gpu_indices, primary_gpu_name, nvml_available) = match find_gpu_indices(pid) {
        Ok((indices, name)) => (indices, name, true),
        Err(CalibrateError::NvmlInit(_) | CalibrateError::NvmlUnavailable) => (
            vec![],
            "Unknown (non-NVIDIA GPU or missing driver)".to_string(),
            false,
        ),
        Err(e) => return Err(e),
    };

    if nvml_available && gpu_indices.is_empty() {
        return Err(CalibrateError::NoGpuProcess { pid });
    }

    let container_context = crate::process::container::detect();

    Ok(ProcessInfo {
        pid,
        gpu_indices,
        primary_gpu_name,
        container_context,
        nvml_available,
    })
}

/// Use NVML to enumerate all devices and return those that have the target
/// PID in their running-compute-processes list.
fn find_gpu_indices(pid: u32) -> Result<(Vec<u32>, String), CalibrateError> {
    let nvml = nvml_wrapper::Nvml::init().map_err(|e| CalibrateError::NvmlInit(e.to_string()))?;

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
                primary_name = device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
            }
            indices.push(i);
        }
    }

    Ok((indices, primary_name))
}

// ── Tests───────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: a PID that can never exist on Linux (max is typically 4_194_304).
    const NONEXISTENT_PID: u32 = 4_000_001;

    #[test]
    fn nonexistent_pid_returns_process_not_found() {
        // We are running on the host (CI or developer machine), so
        // the container detection will return Host, giving ProcessNotFound.
        // If this test runs inside a container the error will be
        // ContainerPidIsolation instead; both are acceptable.
        let result = attach(NONEXISTENT_PID);
        assert!(result.is_err());
        match result.unwrap_err() {
            CalibrateError::ProcessNotFound { pid } => {
                assert_eq!(pid, NONEXISTENT_PID);
            }
            CalibrateError::ContainerPidIsolation { .. } => {
                // Also acceptable: test is running inside a container.
            }
            other => panic!("unexpected error for nonexistent PID: {other:?}"),
        }
    }

    #[test]
    fn process_not_found_error_message_is_actionable() {
        let err = CalibrateError::ProcessNotFound { pid: 99999 };
        let msg = err.to_string();
        assert!(
            msg.contains("99999"),
            "error message should contain the PID: {msg}"
        );
        assert!(
            msg.contains("running"),
            "error message should mention whether process is running: {msg}"
        );
    }

    #[test]
    fn container_isolation_error_includes_docker_exec_hint() {
        let err = CalibrateError::ContainerPidIsolation { pid: 12345 };
        let msg = err.to_string();
        assert!(msg.contains("docker"), "should mention docker: {msg}");
        assert!(msg.contains("kubectl"), "should mention kubectl: {msg}");
        assert!(msg.contains("12345"), "should include the PID: {msg}");
    }

    #[test]
    fn permission_denied_error_mentions_sudo() {
        let err = CalibrateError::PermissionDenied { pid: 7 };
        let msg = err.to_string();
        assert!(
            msg.contains("sudo") || msg.contains("permission") || msg.contains("insufficient"),
            "error should mention privilege escalation: {msg}"
        );
    }

    #[test]
    fn nvml_init_error_mentions_driver() {
        let err = CalibrateError::NvmlInit("library not found".to_string());
        let msg = err.to_string();
        assert!(
            msg.contains("nvidia") || msg.contains("driver") || msg.contains("nvidia-smi"),
            "NVML init error should mention the driver: {msg}"
        );
    }

    #[test]
    fn nvml_unavailable_error_is_descriptive() {
        let err = CalibrateError::NvmlUnavailable;
        let msg = err.to_string();
        assert!(
            msg.contains("NVML") || msg.contains("non-NVIDIA"),
            "error should mention NVML or non-NVIDIA: {msg}"
        );
    }

    #[test]
    fn container_context_detect_returns_a_value() {
        // Only verifies the function runs without panicking.
        let _ctx = crate::process::container::detect();
    }
}
