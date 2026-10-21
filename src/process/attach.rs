use crate::collectors::proc_tree;
use crate::error::CalibrateError;

/// Details about a successfully attached target process.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub container_context: ContainerContext,
    /// Non-fatal problems to show the user (e.g. no GPU context yet).
    pub warnings: Vec<String>,
}

/// Result of container-environment detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerContext {
    Host,
    Docker,
    Kubernetes,
    /// A container runtime was detected but its type is unknown.
    Unknown,
}

/// Validate that `pid` exists and is observable.
///
/// A process that has not created a GPU context yet is *not* an error — jobs
/// often spend their first minute loading data. That situation is reported
/// as a warning and shown live instead.
pub fn attach(pid: u32) -> Result<ProcessInfo, CalibrateError> {
    let container_context = crate::process::container::detect();
    if !proc_tree::is_alive(pid) {
        return Err(if container_context != ContainerContext::Host {
            CalibrateError::ContainerPidIsolation { pid }
        } else {
            CalibrateError::ProcessNotFound { pid }
        });
    }

    let mut warnings = Vec::new();
    if crate::collectors::cpu::process_cpu_ns(pid).is_none() {
        #[cfg(target_os = "linux")]
        return Err(CalibrateError::PermissionDenied { pid });
        #[cfg(not(target_os = "linux"))]
        warnings.push(format!(
            "cannot read CPU usage of PID {pid} (owned by another user?) — run with sudo for \
             CPU and input-pipeline analysis"
        ));
    }

    #[cfg(target_os = "linux")]
    gpu_presence_warnings(pid, container_context, &mut warnings);

    Ok(ProcessInfo {
        container_context,
        warnings,
    })
}

/// Warn when no process of the tree holds a GPU context, with a specific hint
/// when NVML's PIDs are not visible here (container PID namespace).
#[cfg(target_os = "linux")]
fn gpu_presence_warnings(pid: u32, ctx: ContainerContext, warnings: &mut Vec<String>) {
    let Ok(backend) = crate::collectors::nvml::NvmlBackend::new(
        crate::cli::MetricsSource::Nvml,
        std::time::Duration::from_secs(1),
    ) else {
        return;
    };
    let tree = proc_tree::descendants(pid);
    if !backend
        .devices_for_pids(&tree)
        .unwrap_or_default()
        .is_empty()
    {
        return;
    }
    let gpu_pids = backend.all_gpu_pids();
    let invisible = !gpu_pids.is_empty() && gpu_pids.iter().all(|p| !proc_tree::is_alive(*p));
    if invisible || ctx != ContainerContext::Host {
        warnings.push(format!(
            "PID {pid} is not listed on any GPU, and the GPU's process IDs are not visible in \
             this PID namespace (container). Use `--gpus all` to monitor the devices directly, \
             or run calibrate on the host / with hostPID."
        ));
    } else {
        warnings.push(format!(
            "PID {pid} and its children have no GPU context yet — waiting for the job to start \
             using a GPU"
        ));
    }
}

/// Apple chip brand (`machdep.cpu.brand_string`) + " GPU", e.g. "Apple M3 Max GPU".
#[cfg(target_os = "macos")]
pub fn apple_gpu_name() -> String {
    let mut buf = [0u8; 256];
    let mut len = buf.len();
    let ret = unsafe {
        libc::sysctlbyname(
            c"machdep.cpu.brand_string".as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 && len > 1 {
        let s = std::str::from_utf8(&buf[..len - 1])
            .unwrap_or("Apple")
            .trim()
            .to_string();
        format!("{s} GPU")
    } else {
        "Apple GPU".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A PID that can never exist (Linux max is 4_194_304; macOS 99_999).
    const NONEXISTENT_PID: u32 = 4_000_001;

    #[test]
    fn nonexistent_pid_is_an_actionable_error() {
        match attach(NONEXISTENT_PID).unwrap_err() {
            CalibrateError::ProcessNotFound { pid }
            | CalibrateError::ContainerPidIsolation { pid } => {
                assert_eq!(pid, NONEXISTENT_PID)
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn own_process_attaches() {
        attach(std::process::id()).expect("self attach");
    }

    #[test]
    fn error_messages_are_actionable() {
        let msg = CalibrateError::ProcessNotFound { pid: 99999 }.to_string();
        assert!(msg.contains("pgrep"), "{msg}");
        let msg = CalibrateError::PermissionDenied { pid: 1 }.to_string();
        assert!(msg.contains("sudo"), "{msg}");
        let msg = CalibrateError::ContainerPidIsolation { pid: 7 }.to_string();
        assert!(
            msg.contains("docker exec") && msg.contains("--gpus all"),
            "{msg}"
        );
        let msg = CalibrateError::NvmlInit("x".into()).to_string();
        assert!(
            msg.contains("modprobe") && msg.contains("calibrate diagnose"),
            "{msg}"
        );
    }
}
