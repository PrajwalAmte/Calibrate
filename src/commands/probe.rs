use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;

use crate::cli::ProbeArgs;
use crate::collectors::nvml::NvmlCollector;
use crate::collectors::proc::ProcCollector;
use crate::collectors::MetricsCollector;
use crate::metrics::units::Percent;
use crate::process::attach;

/// Entry point for `calibrate probe`.
///
/// Validates that the collector pipeline is functioning end-to-end by:
///   1. Attaching to the given process and printing what we found
///   2. Streaming `--count` raw `RawSample` JSON objects to stdout, one per line
///   3. Exiting cleanly
///
/// This is the primary verification tool for Phase 2.  Run it against any
/// live GPU process to confirm that RawSamples arrive with all fields
/// (including `cpu_utilization`) populated.
pub async fn run(args: ProbeArgs) -> anyhow::Result<()> {
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!(
        "calibrate probe requires Linux with NVIDIA drivers installed.\n\
         On macOS, `calibrate bench` and `calibrate plan` are available."
    );

    // ── Probe NVML availability first ────────────────────────────────────
    NvmlCollector::probe().context(
        "NVML unavailable — is the NVIDIA driver installed and are you running as a user \
         with GPU access?",
    )?;

    // ── Attach to the process ─────────────────────────────────────────────
    let process_info = attach::attach(args.pid).context("Failed to attach to training process")?;

    eprintln!("Attached to PID {}", args.pid);
    eprintln!("  Primary GPU : {}", process_info.primary_gpu_name);
    eprintln!("  GPU indices : {:?}", process_info.gpu_indices);
    eprintln!("  Container   : {:?}", process_info.container_context);
    eprintln!();

    let n = args.count;
    let interval = Duration::from_secs_f64(args.interval);
    eprintln!(
        "Streaming {} RawSample(s) at {:.1}s intervals (JSON, one per line)...",
        n, args.interval
    );
    eprintln!();

    // ── Set up shared state ───────────────────────────────────────────────
    let stop = Arc::new(AtomicBool::new(false));
    let shared_cpu: Arc<parking_lot::Mutex<Percent>> =
        Arc::new(parking_lot::Mutex::new(Percent(0.0)));

    // ProcCollector thread
    std::thread::Builder::new()
        .name("proc-collector".to_string())
        .spawn({
            let shared_cpu = shared_cpu.clone();
            let stop = stop.clone();
            let pid = args.pid;
            move || ProcCollector::run_background(pid, shared_cpu, stop, interval)
        })
        .context("Failed to spawn ProcCollector thread")?;

    // NvmlCollector thread
    let (tx, rx) = flume::bounded::<crate::collectors::RawSample>(64);
    let nvml_collector = NvmlCollector::new(args.pid, interval, shared_cpu);

    std::thread::Builder::new()
        .name("nvml-collector".to_string())
        .spawn({
            let stop = stop.clone();
            move || nvml_collector.run(tx, stop)
        })
        .context("Failed to spawn NVML collector thread")?;

    // ── Collect N samples ─────────────────────────────────────────────────
    let mut received: u32 = 0;
    while received < n {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(sample) => {
                let json =
                    serde_json::to_string(&sample).context("Failed to serialize RawSample")?;
                println!("{json}");
                received += 1;
            }
            Err(flume::RecvTimeoutError::Timeout) => {
                eprintln!(
                    "Timeout waiting for sample {}/{} — is the process actively using the GPU?",
                    received + 1,
                    n
                );
            }
            Err(flume::RecvTimeoutError::Disconnected) => {
                eprintln!("Collector thread exited unexpectedly");
                break;
            }
        }
    }

    // ── Clean up ──────────────────────────────────────────────────────────
    stop.store(true, Ordering::Relaxed);

    if received == n {
        eprintln!();
        eprintln!("Done — {received} sample(s) received successfully.");
    } else {
        eprintln!();
        eprintln!("Received {received}/{n} sample(s) before stopping.");
    }

    Ok(())
}
