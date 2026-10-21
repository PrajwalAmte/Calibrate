use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::cli::ProbeArgs;
use crate::collectors::hook;
use crate::collectors::runner::{self, RunnerConfig, Target};
use crate::process::attach;

/// Entry point for `calibrate probe`: stream raw ticks as JSON so the
/// collector pipeline can be verified on a new machine.
pub async fn run(args: ProbeArgs) -> Result<()> {
    let target = match (args.pid, args.gpus.clone()) {
        (Some(pid), _) => {
            let info = attach::attach(pid).context("Failed to attach to process")?;
            for w in &info.warnings {
                eprintln!("[calibrate] {w}");
            }
            Target::Process {
                root: pid,
                include_children: true,
            }
        }
        (None, Some(sel)) => Target::Devices(sel),
        (None, None) => unreachable!("clap requires --pid or --gpus"),
    };

    let stop = Arc::new(AtomicBool::new(false));
    let (rx, info) = runner::spawn(
        RunnerConfig {
            target,
            interval: Duration::from_secs_f64(args.interval),
            metrics_source: args.metrics_source,
            hook_dir: hook::default_dir(),
        },
        stop.clone(),
    )?;
    eprintln!(
        "backend: {}  •  metrics source: {}  •  scope: {:?}",
        info.backend, info.metrics_source, info.scope
    );
    for w in &info.warnings {
        eprintln!("[calibrate] {w}");
    }
    eprintln!(
        "Streaming {} tick(s) at {:.1}s intervals (JSON, one per line)...",
        args.count, args.interval
    );

    let mut received = 0u32;
    let timeout = Duration::from_secs_f64(args.interval * 3.0 + 5.0);
    while received < args.count {
        match tokio::time::timeout(timeout, rx.recv_async()).await {
            Ok(Ok(tick)) => {
                println!("{}", serde_json::to_string(&tick)?);
                received += 1;
            }
            Ok(Err(_)) => {
                eprintln!("Collector stopped (target exited?)");
                break;
            }
            Err(_) => eprintln!("No sample within {timeout:?} — still waiting…"),
        }
    }
    stop.store(true, Ordering::Relaxed);
    eprintln!("Received {received}/{} tick(s).", args.count);
    if received == 0 {
        anyhow::bail!("no samples were collected");
    }
    Ok(())
}
