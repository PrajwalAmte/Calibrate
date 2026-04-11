use thiserror::Error;

/// Top-level error type for calibrate.
/// Each variant captures failures at a specific subsystem boundary.
/// Downstream code converts these into `anyhow::Error` at the command layer.
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum CalibrateError {
    #[error("process {pid} not found — is the training job still running?")]
    ProcessNotFound { pid: u32 },

    #[error("process {pid} is not using any NVIDIA GPU")]
    NoGpuProcess { pid: u32 },

    #[error("insufficient permissions to read /proc/{pid}; try running with sudo")]
    PermissionDenied { pid: u32 },

    #[error(
        "NVML initialization failed: {0}\n\
         \n\
         To resolve this:\n\
         • Run `nvidia-smi` — if it fails, the NVIDIA driver is not loaded\n\
         • On Linux: check driver status with `sudo modprobe nvidia`\n\
         • Ensure you have read access to /dev/nvidiactl (add user to `video` group)"
    )]
    NvmlInit(String),

    #[error("NVML query error: {0}")]
    NvmlQuery(String),

    #[error("NVML is not available on this system (non-NVIDIA GPU detected)")]
    NvmlUnavailable,

    #[error("failed to read /proc/{pid}/stat: {source}")]
    ProcRead {
        pid: u32,
        #[source]
        source: std::io::Error,
    },

    #[error("unexpected /proc/{pid}/stat format")]
    ProcFormat { pid: u32 },

    #[error("GPU spec fetch failed: {0}")]
    SpecFetch(String),

    #[error("GPU model '{name}' not found in spec database — MFU will be estimated")]
    SpecNotFound { name: String },

    #[error(
        "process {pid} not found inside this container PID namespace.\n\
             \n\
             If the training process is on the HOST, re-run calibrate there:\n\
             \n\
             • Docker : docker exec -it <container> calibrate watch --pid {pid}\n\
             • Kubernetes: kubectl exec -it <pod> -- calibrate watch --pid {pid}\n\
             • Host   : sudo calibrate watch --pid {pid}"
    )]
    ContainerPidIsolation { pid: u32 },

    #[error("training process {pid} exited before enough samples were collected")]
    ProcessExited { pid: u32 },

    #[error("sampling channel closed unexpectedly")]
    ChannelClosed,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
