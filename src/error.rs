use thiserror::Error;

/// Top-level error type for calibrate.
///
/// Each variant captures failures at a specific subsystem boundary.
/// Downstream code converts these into `anyhow::Error` at the command layer.
#[derive(Debug, Error)]
pub enum CalibrateError {
    // ── Process attachment ────────────────────────────────────────────────
    #[error("process {pid} not found — is the training job still running?")]
    ProcessNotFound { pid: u32 },

    #[error("process {pid} is not using any NVIDIA GPU")]
    NoGpuProcess { pid: u32 },

    #[error("insufficient permissions to read /proc/{pid}; try running with sudo")]
    PermissionDenied { pid: u32 },

    // ── NVML ──────────────────────────────────────────────────────────────
    #[error("NVML initialization failed: {0}\nIs the nvidia-smi driver installed?")]
    NvmlInit(String),

    #[error("NVML query error: {0}")]
    NvmlQuery(String),

    #[error("NVML is not available on this system (non-NVIDIA GPU detected)")]
    NvmlUnavailable,

    // ── /proc sampling ────────────────────────────────────────────────────
    #[error("failed to read /proc/{pid}/stat: {source}")]
    ProcRead {
        pid: u32,
        #[source]
        source: std::io::Error,
    },

    #[error("unexpected /proc/{pid}/stat format")]
    ProcFormat { pid: u32 },

    // ── GPU spec database ─────────────────────────────────────────────────
    #[error("GPU spec fetch failed: {0}")]
    SpecFetch(String),

    #[error("GPU model '{name}' not found in spec database — MFU will be estimated")]
    SpecNotFound { name: String },

    // ── Container / environment ───────────────────────────────────────────
    #[error("running inside a container without host PID namespace access\n\
             Re-run with: docker run --pid=host ...")]
    ContainerPidIsolation,

    // ── Sampling ──────────────────────────────────────────────────────────
    #[error("training process {pid} exited before enough samples were collected")]
    ProcessExited { pid: u32 },

    #[error("sampling channel closed unexpectedly")]
    ChannelClosed,

    // ── I/O ───────────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
