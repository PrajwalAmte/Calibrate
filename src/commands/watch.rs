use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tracing::info;

use crate::cli::{OutputFormat, WatchArgs};
use crate::collectors::cpu_only::CpuOnlyCollector;
use crate::collectors::nvml::NvmlCollector;
use crate::collectors::proc::ProcCollector;
use crate::collectors::MetricsCollector;
use crate::gpu_specs;
use crate::metrics::units::Percent;
use crate::output::json::JsonRenderer;
use crate::output::terminal::TerminalRenderer;
use crate::output::OutputRenderer;
use crate::process::attach::{self, ContainerContext};
use crate::session::lifecycle::{state, MonitoringSession};
use crate::session::state::new_snapshot_channel;

/// Entry point for `calibrate watch`.
pub async fn run(args: WatchArgs) -> anyhow::Result<()> {
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!(
        "calibrate watch requires Linux with NVIDIA drivers installed.\n\
         On macOS, `calibrate bench` and `calibrate plan` are available."
    );

    let process_info = attach::attach(args.pid).context("Failed to attach to training process")?;

    // ── Container advisory — printed before the TUI takes over the screen ──
    match &process_info.container_context {
        ContainerContext::Docker => {
            eprintln!(
                "[calibrate] Note: running inside a Docker container. \
                 If the training process is on the host, run calibrate there instead:\n  \
                 docker exec -it <container> calibrate watch --pid {}",
                args.pid
            );
        }
        ContainerContext::Kubernetes => {
            eprintln!(
                "[calibrate] Note: running inside a Kubernetes pod. \
                 If the training process is on the host node, use:\n  \
                 kubectl exec -it <pod> -- calibrate watch --pid {}",
                args.pid
            );
        }
        _ => {}
    }

    // ── NVML advisory ─────────────────────────────────────────────────────
    if !process_info.nvml_available {
        eprintln!(
            "[calibrate] Warning: NVML unavailable — GPU metrics disabled.\n\
             Showing CPU metrics only.  To enable full monitoring:\n\
             • Confirm nvidia-smi is installed and the driver is loaded\n\
             • On non-NVIDIA hardware MFU cannot be computed"
        );
    };

    info!(
        pid = args.pid,
        gpu = %process_info.primary_gpu_name,
        "Attached to training process"
    );

    // ── 2. Load GPU spec ─────────────────────────────────────────────────
    let gpu_name = process_info.primary_gpu_name.clone();
    let gpu_spec = tokio::task::spawn_blocking(move || gpu_specs::resolve(&gpu_name))
        .await
        .context("GPU spec resolution panicked")?;

    info!(gpu_spec = ?gpu_spec, "GPU spec loaded");

    // ── 3. Set up shared stop flag ────────────────────────────────────────
    let stop = Arc::new(AtomicBool::new(false));

    // ── 4. Shared cpu_utilization written by ProcCollector, read by NvmlCollector ─
    //
    // Both threads share ownership of this `Mutex<Percent>`.  ProcCollector
    // writes to it; NvmlCollector reads from it once per tick and stamps the
    // value directly into each RawSample, eliminating the Percent(0.0) placeholder.
    let shared_cpu: Arc<parking_lot::Mutex<Percent>> =
        Arc::new(parking_lot::Mutex::new(Percent(0.0)));

    // ── 5a. Spawn ProcCollector on its own OS thread ──────────────────────
    let interval = Duration::from_secs_f64(args.interval);
    let pid = args.pid;

    std::thread::Builder::new()
        .name("proc-collector".to_string())
        .spawn({
            let shared_cpu = shared_cpu.clone();
            let stop = stop.clone();
            move || ProcCollector::run_background(pid, shared_cpu, stop, interval)
        })
        .context("Failed to spawn ProcCollector thread")?;

    let (tx, rx) = flume::bounded::<crate::collectors::RawSample>(64);

    if process_info.nvml_available {
        let nvml_collector = NvmlCollector::new(pid, interval, shared_cpu);
        std::thread::Builder::new()
            .name("nvml-collector".to_string())
            .spawn({
                let stop = stop.clone();
                move || nvml_collector.run(tx, stop)
            })
            .context("Failed to spawn NVML collector thread")?;
    } else {
        let cpu_collector = CpuOnlyCollector::new(pid, interval);
        std::thread::Builder::new()
            .name("cpu-only-collector".to_string())
            .spawn({
                let stop = stop.clone();
                move || cpu_collector.run(tx, stop)
            })
            .context("Failed to spawn CPU-only collector thread")?;
    };

    let (snap_tx, mut snap_rx) = new_snapshot_channel();
    let session = MonitoringSession::<state::Initializing>::new(
        pid,
        gpu_spec,
        args.cost_per_hour,
        snap_tx,
        stop.clone(),
        process_info.nvml_available,
    )
    .start();

    let session_handle = tokio::spawn(session.run(rx));

    let mut renderer: Box<dyn OutputRenderer> = match args.output {
        OutputFormat::Terminal => {
            Box::new(TerminalRenderer::new().context("Failed to initialize terminal renderer")?)
        }
        OutputFormat::Json => Box::new(JsonRenderer),
    };

    let stop_ctrlc = stop.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Ctrl+C received — stopping");
        stop_ctrlc.store(true, Ordering::Relaxed);
    });

    loop {
        tokio::select! {
            result = snap_rx.changed() => {
                if result.is_err() {
                    break;
                }
            }
            _ = tokio::time::sleep(interval) => {}
        }

        if stop.load(Ordering::Relaxed) {
            break;
        }

        let snapshot = snap_rx.borrow().clone();
        if let Some(ref snap) = snapshot {
            renderer.render(snap);
        }
    }

    let done_session = session_handle.await.context("Session task panicked")?;

    let final_snapshot = done_session.final_snapshot();
    renderer.finish(final_snapshot.as_ref());

    Ok(())
}
