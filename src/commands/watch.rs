use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tracing::info;

use crate::cli::{OutputFormat, WatchArgs};
use crate::collectors::nvml::NvmlCollector;
use crate::collectors::proc::ProcCollector;
use crate::collectors::MetricsCollector;
use crate::gpu_specs;
use crate::output::json::JsonRenderer;
use crate::output::terminal::TerminalRenderer;
use crate::output::OutputRenderer;
use crate::process::attach;
use crate::session::lifecycle::{state, MonitoringSession};
use crate::session::state::new_shared_state;

/// Entry point for `calibrate watch`.
///
/// Orchestration order:
/// 1. Validate PID + detect GPU indices
/// 2. Load GPU spec (remote / cache / fallback)
/// 3. Start NVML collector on a dedicated os thread
/// 4. Start ProcCollector polling loop (tokio task)
/// 5. Start MonitoringSession aggregation loop (tokio task)
/// 6. Start renderer refresh loop (tokio task)
/// 7. Await Ctrl+C or process exit
/// 8. Signal stop, await cleanup, print summary
pub async fn run(args: WatchArgs) -> anyhow::Result<()> {
    // ── 1. Attach to process ─────────────────────────────────────────────
    let process_info = attach::attach(args.pid)
        .context("Failed to attach to training process")?;

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
    let stop_nvml = stop.clone();
    let stop_proc = stop.clone();

    // ── 4. Start NVML collector on a dedicated OS thread ─────────────────
    let interval = Duration::from_secs_f64(args.interval);
    let (tx, rx) = flume::bounded::<crate::collectors::RawSample>(64);

    let nvml_collector = NvmlCollector::new(process_info.gpu_indices.clone(), interval);
    std::thread::Builder::new()
        .name("nvml-collector".to_string())
        .spawn(move || nvml_collector.run(tx, stop_nvml))
        .context("Failed to spawn NVML collector thread")?;

    // ── 5. Start ProcCollector as a blocking tokio task ───────────────────
    let pid = args.pid;
    let state = new_shared_state();
    let state_writer = state.clone();

    // ProcCollector runs in spawn_blocking because it does synchronous file I/O.
    // It patches cpu_utilization into the latest sample via shared state.
    // (For this phase the cpu_utilization in RawSample stays at 0.0 from NVML —
    // it will be wired through the session snapshot in a later phase.)
    tokio::task::spawn_blocking(move || {
        let mut proc = match ProcCollector::new(pid) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("ProcCollector setup failed: {e}");
                return;
            }
        };
        loop {
            if stop_proc.load(Ordering::Relaxed) {
                break;
            }
            if !proc.is_alive() {
                tracing::info!("Training process {pid} exited");
                stop_proc.store(true, Ordering::Relaxed);
                break;
            }
            match proc.sample() {
                Ok(cpu_pct) => {
                    // Patch the latest snapshot's cpu field in shared state.
                    if let Some(snap) = state_writer.write().as_mut() {
                        // We update the per_gpu vec's summary — full wiring
                        // happens in the session lifecycle.
                        let _ = cpu_pct; // used in later phase
                    }
                }
                Err(e) => tracing::warn!("ProcCollector sample failed: {e}"),
            }
            std::thread::sleep(interval);
        }
    });

    // ── 6. Start MonitoringSession ────────────────────────────────────────
    let session = MonitoringSession::<state::Initializing>::new(
        pid,
        gpu_spec,
        args.cost_per_hour,
        state.clone(),
        stop.clone(),
    )
    .start();

    // Run the aggregation loop in a separate task so the renderer can run
    // concurrently.
    let session_handle = tokio::spawn(session.run(rx));

    // ── 7. Start renderer ─────────────────────────────────────────────────
    let mut renderer: Box<dyn OutputRenderer> = match args.output {
        OutputFormat::Terminal => Box::new(
            TerminalRenderer::new().context("Failed to initialize terminal renderer")?,
        ),
        OutputFormat::Json => Box::new(JsonRenderer),
    };

    // Ctrl+C handler signals stop.
    let stop_ctrlc = stop.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Ctrl+C received — stopping");
        stop_ctrlc.store(true, Ordering::Relaxed);
    });

    // Render loop — polls shared state at the sampling interval.
    loop {
        tokio::time::sleep(interval).await;

        if stop.load(Ordering::Relaxed) {
            break;
        }

        let snapshot = state.read().clone();
        if let Some(ref snap) = snapshot {
            renderer.render(snap);
        }
    }

    // ── 8. Cleanup + summary ──────────────────────────────────────────────
    let done_session = session_handle
        .await
        .context("Session task panicked")?;

    let final_snapshot = done_session.final_snapshot();
    renderer.finish(final_snapshot.as_ref());

    Ok(())
}
