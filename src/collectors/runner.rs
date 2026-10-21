//! The collector thread: one [`Tick`] per interval from whichever backend the
//! platform offers, plus process-tree CPU and hook events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::cli::{GpuSelection, MetricsSource};
use crate::collectors::cpu::TreeCpuSampler;
use crate::collectors::hook::HookClient;
use crate::collectors::{now_ms, proc_tree, GpuSample, GpuScope, Tick};

/// What to collect.
#[derive(Debug, Clone)]
pub enum Target {
    /// A process (and, unless disabled, its descendants).
    Process { root: u32, include_children: bool },
    /// Fixed devices with no process attribution.
    Devices(GpuSelection),
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub target: Target,
    pub interval: Duration,
    pub metrics_source: MetricsSource,
    pub hook_dir: PathBuf,
}

/// Static facts about the running collector, shown to the user.
#[derive(Debug, Clone)]
pub struct RunnerInfo {
    pub scope: GpuScope,
    /// "nvml", "apple-iokit", "none" or "replay".
    pub backend: String,
    /// Utilization source actually in use ("nvml-gpm", "dcgm", "nvml", ...).
    pub metrics_source: String,
    pub warnings: Vec<String>,
}

enum Backend {
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    Nvml(Box<crate::collectors::nvml::NvmlBackend>),
    #[cfg(target_os = "macos")]
    Apple(crate::collectors::apple_gpu::AppleBackend),
    None,
}

/// Start collection on a dedicated OS thread.
///
/// Backend initialisation happens before this returns so configuration
/// errors (e.g. `--metrics-source gpm` on an A100) surface immediately.
pub fn spawn(
    cfg: RunnerConfig,
    stop: Arc<AtomicBool>,
) -> Result<(flume::Receiver<Tick>, RunnerInfo)> {
    let mut warnings = Vec::new();
    let (backend, scope, backend_name, source) = init_backend(&cfg, &mut warnings)?;
    let info = RunnerInfo {
        scope,
        backend: backend_name.to_string(),
        metrics_source: source,
        warnings,
    };

    let (tx, rx) = flume::bounded::<Tick>(256);
    std::thread::Builder::new()
        .name("calibrate-collector".into())
        .spawn(move || run_loop(cfg, backend, scope, tx, stop))
        .context("failed to spawn collector thread")?;
    Ok((rx, info))
}

type BackendInit = (Backend, GpuScope, &'static str, String);

#[cfg(target_os = "macos")]
fn init_backend(cfg: &RunnerConfig, warnings: &mut Vec<String>) -> Result<BackendInit> {
    if cfg.metrics_source != MetricsSource::Auto && cfg.metrics_source != MetricsSource::Nvml {
        warnings.push("--metrics-source is ignored on macOS (IOKit only)".into());
    }
    let name = crate::process::attach::apple_gpu_name();
    Ok(
        match crate::collectors::apple_gpu::AppleBackend::new(name) {
            Ok(b) => {
                warnings.push(
                    "macOS reports system-wide GPU utilization and memory — other apps using the \
                 GPU are included"
                        .into(),
                );
                (
                    Backend::Apple(b),
                    GpuScope::System,
                    "apple-iokit",
                    "iokit".into(),
                )
            }
            Err(e) => {
                warnings.push(format!("GPU metrics unavailable: {e}"));
                (Backend::None, GpuScope::None, "none", "none".into())
            }
        },
    )
}

#[cfg(not(target_os = "macos"))]
fn init_backend(cfg: &RunnerConfig, warnings: &mut Vec<String>) -> Result<BackendInit> {
    let device_mode = matches!(cfg.target, Target::Devices(_));
    match crate::collectors::nvml::NvmlBackend::new(cfg.metrics_source, cfg.interval) {
        Ok(b) => {
            let source = b.source().label().to_string();
            if b.source() == crate::collectors::nvml::ActiveSource::Nvml {
                warnings.push(
                    "Only NVML kernel-busy utilization is available, so MFU is shown as an \
                     upper bound. Hopper+ GPUs (GPM), DCGM, or the calibrate Python hook give \
                     a real measurement."
                        .into(),
                );
            }
            let scope = if device_mode {
                GpuScope::Device
            } else {
                GpuScope::Process
            };
            Ok((Backend::Nvml(Box::new(b)), scope, "nvml", source))
        }
        Err(e @ crate::error::CalibrateError::ProfilingUnavailable(_)) => Err(e.into()),
        Err(e) if device_mode => Err(anyhow::Error::from(e).context("--gpus requires NVML")),
        Err(e) => {
            warnings.push(format!(
                "NVML unavailable — showing CPU metrics only ({})",
                first_line(&e.to_string())
            ));
            Ok((Backend::None, GpuScope::None, "none", "none".into()))
        }
    }
}

#[allow(dead_code)]
fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

fn run_loop(
    cfg: RunnerConfig,
    mut backend: Backend,
    scope: GpuScope,
    tx: flume::Sender<Tick>,
    stop: Arc<AtomicBool>,
) {
    let mut cpu = TreeCpuSampler::new();
    let mut hook = HookClient::new(cfg.hook_dir.clone());

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let started = Instant::now();

        let pids = match &cfg.target {
            Target::Process {
                root,
                include_children,
            } => {
                if !proc_tree::is_alive(*root) {
                    info!(pid = root, "target process exited — stopping collection");
                    break;
                }
                if *include_children {
                    let t = proc_tree::descendants(*root);
                    if t.is_empty() {
                        vec![*root]
                    } else {
                        t
                    }
                } else {
                    vec![*root]
                }
            }
            Target::Devices(_) => vec![],
        };

        let gpus: Vec<GpuSample> = match &mut backend {
            Backend::Nvml(b) => {
                let indices = match &cfg.target {
                    Target::Process { .. } => b.devices_for_pids(&pids).unwrap_or_else(|e| {
                        warn!("GPU discovery failed: {e}");
                        vec![]
                    }),
                    Target::Devices(GpuSelection::All) => (0..b.device_count()).collect(),
                    Target::Devices(GpuSelection::Indices(v)) => v.clone(),
                };
                b.sample(&indices)
            }
            #[cfg(target_os = "macos")]
            Backend::Apple(b) => b.sample(),
            Backend::None => vec![],
        };

        let (cpu_pct, steps) = if pids.is_empty() {
            (None, vec![])
        } else {
            (cpu.sample(&pids), hook.poll(&pids))
        };

        let tick = Tick {
            timestamp_ms: now_ms(),
            scope,
            gpus,
            cpu_pct,
            process_count: pids.len() as u32,
            steps,
            hook_peers: hook.peers(),
        };
        if tx.send(tick).is_err() {
            break;
        }

        // Sleep in short slices so a stop request is honoured quickly.
        while started.elapsed() < cfg.interval {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let left = cfg.interval.saturating_sub(started.elapsed());
            std::thread::sleep(left.min(Duration::from_millis(50)));
        }
    }
}
