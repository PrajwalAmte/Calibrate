use thiserror::Error;

/// Errors at subsystem boundaries, each with remediation text.
/// Converted into `anyhow::Error` at the command layer.
#[derive(Debug, Error)]
#[allow(dead_code)] // some variants are only constructed on one platform
pub enum CalibrateError {
    #[error(
        "process {pid} not found — is the training job still running?\n\
         \n\
         To find the correct PID:\n\
           pgrep -f train.py          # by script name\n\
           pgrep -a -f torch          # all torch processes\n\
           calibrate watch --auto     # auto-detect active training processes"
    )]
    ProcessNotFound { pid: u32 },

    #[error(
        "insufficient permissions to read /proc/{pid}\n\
         \n\
         To fix:\n\
           sudo calibrate watch --pid {pid}\n\
           (or check whether /proc is mounted with hidepid)"
    )]
    PermissionDenied { pid: u32 },

    #[error(
        "NVML initialization failed: {0}\n\
         \n\
         Possible causes and fixes:\n\
         \n\
         1. Driver not loaded:\n\
            sudo modprobe nvidia\n\
         \n\
         2. libnvidia-ml library missing:\n\
            Ubuntu/Debian: sudo apt-get install libnvidia-ml1  (or the nvidia-utils package)\n\
            RHEL/CentOS:   sudo dnf install nvidia-driver-cuda-libs\n\
            Then reload:   sudo ldconfig\n\
         \n\
         3. Permission denied on /dev/nvidiactl:\n\
            sudo usermod -aG video $USER && newgrp video\n\
         \n\
         4. Docker without GPU passthrough:\n\
            docker run --gpus all ... (requires nvidia-container-toolkit)\n\
         \n\
         Run `calibrate diagnose` for a detailed per-component report."
    )]
    NvmlInit(String),

    #[error("NVML query error: {0}\n  Run `calibrate diagnose` to verify the driver stack.")]
    NvmlQuery(String),

    #[error("requested GPU profiling source is unavailable: {0}")]
    ProfilingUnavailable(String),

    #[error(
        "Apple GPU initialization failed: {0}\n\
         \n\
         To resolve this:\n\
         • Ensure the process is running on macOS 12 or later\n\
         • Verify IOKit access is not restricted by a sandbox profile"
    )]
    AppleGpuInit(String),

    #[error(
        "process {pid} not found inside this container's PID namespace.\n\
         \n\
         If the training process runs on the HOST or in another container:\n\
         \n\
         • Same pod (Kubernetes): set `shareProcessNamespace: true`\n\
         • Docker   : docker exec -it <training-container> calibrate watch --pid {pid}\n\
         • Host     : sudo calibrate watch --pid {pid}\n\
         • Any      : calibrate watch --gpus all   (device-level, no PID needed)"
    )]
    ContainerPidIsolation { pid: u32 },
}
