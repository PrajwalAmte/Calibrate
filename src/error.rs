use thiserror::Error;

/// Top-level error type for calibrate.
/// Each variant captures failures at a specific subsystem boundary.
/// Downstream code converts these into `anyhow::Error` at the command layer.
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum CalibrateError {
    #[error(
        "process {pid} not found — is the training job still running?\n\
         \n\
         To find the correct PID:\n\
           pgrep -f train.py          # by script name\n\
           pgrep -a -f torch          # all torch processes\n\
           calibrate diagnose         # auto-detect active training processes"
    )]
    ProcessNotFound { pid: u32 },

    #[error(
        "process {pid} is not using any NVIDIA GPU.\n\
         \n\
         Possible causes:\n\
         • Training is running on CPU — check your script for device='cpu'\n\
         • GPU allocation happened after you ran calibrate watch — try attaching again\n\
         • The process uses a non-NVIDIA GPU (AMD/Intel) — NVML cannot see it"
    )]
    NoGpuProcess { pid: u32 },

    #[error(
        "insufficient permissions to read /proc/{pid}\n\
         \n\
         To fix:\n\
           sudo calibrate watch --pid {pid}\n\
           Or add yourself to the process owner's group if running across users"
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
            Ubuntu/Debian: sudo apt-get install libnvidia-ml1\n\
            RHEL/CentOS:   sudo dnf install nvidia-driver-cuda-libs\n\
            Then reload:   sudo ldconfig\n\
         \n\
         3. Permission denied on /dev/nvidiactl:\n\
            sudo usermod -aG video $USER && newgrp video\n\
         \n\
         4. Docker without GPU passthrough:\n\
            Use: docker run --gpus all ... (requires nvidia-container-toolkit)\n\
         \n\
         Run `calibrate diagnose` for a detailed per-component report."
    )]
    NvmlInit(String),

    #[error("NVML query error: {0}\n  Run `calibrate diagnose` to verify the driver stack.")]
    NvmlQuery(String),

    #[error(
        "NVML is not available on this system.\n\
         \n\
         If you have an NVIDIA GPU, run `calibrate diagnose` to find the missing component.\n\
         If this is a non-NVIDIA system, `calibrate watch` requires Apple Silicon (macOS) or NVIDIA (Linux)."
    )]
    NvmlUnavailable,

    #[error(
        "Apple GPU initialization failed: {0}\n\
         \n\
         To resolve this:\n\
         • Ensure the process is running on macOS 12 or later\n\
         • Verify IOKit access is not restricted by a sandbox profile"
    )]
    AppleGpuInit(String),

    #[error("Apple GPU IOKit query failed: {0}")]
    AppleGpuQuery(String),

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
