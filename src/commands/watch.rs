use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

use crate::cli::{MetricsSource, OutputFormat, WatchArgs};
use crate::collectors::replay::TraceWriter;
use crate::collectors::runner::{self, RunnerConfig, Target};
use crate::collectors::{hook, replay};
use crate::export::{otlp, prometheus};
use crate::gpu_specs::SpecDb;
use crate::output::json::JsonRenderer;
use crate::output::terminal::TerminalRenderer;
use crate::output::{NullRenderer, OutputRenderer};
use crate::process::attach::{self, ContainerContext};
use crate::process::discover;
use crate::session::lifecycle::{MonitoringSession, SessionConfig};
use crate::session::state::{new_snapshot_channel, SessionSnapshot};

/// Settings after merging flags, environment and the config file.
#[derive(Debug, Clone)]
pub struct WatchSettings {
    pub interval: Duration,
    pub target_mfu: f32,
    pub cost_per_hour: Option<f64>,
    pub metrics_source: MetricsSource,
    pub prometheus_listen: Option<String>,
    pub otlp_endpoint: Option<String>,
    pub hook_dir: std::path::PathBuf,
}

pub fn resolve_settings(args: &WatchArgs) -> Result<WatchSettings> {
    let file = &crate::config::get().file.watch;
    let interval = args.interval.or(file.interval).unwrap_or(2.0);
    if !interval.is_finite() || !(0.1..=3600.0).contains(&interval) {
        bail!("watch interval must be between 0.1 and 3600 seconds (got {interval})");
    }
    let target_mfu = args.target_mfu.or(file.target_mfu).unwrap_or(45.0);
    if !(target_mfu > 0.0 && target_mfu <= 100.0) {
        bail!("target_mfu must be in (0, 100] (got {target_mfu})");
    }
    let cost_per_hour = args.cost_per_hour.or(file.cost_per_hour);
    if cost_per_hour.is_some_and(|c| !c.is_finite() || c < 0.0) {
        bail!("cost_per_hour must be zero or positive");
    }
    Ok(WatchSettings {
        interval: Duration::from_secs_f64(interval),
        target_mfu,
        cost_per_hour,
        metrics_source: args
            .metrics_source
            .or(file.metrics_source)
            .unwrap_or(MetricsSource::Auto),
        prometheus_listen: args
            .prometheus_listen
            .clone()
            .or_else(|| file.prometheus_listen.clone()),
        otlp_endpoint: args
            .otlp_endpoint
            .clone()
            .or_else(|| file.otlp_endpoint.clone()),
        hook_dir: args
            .hook_dir
            .clone()
            .or_else(|| file.hook_dir.clone())
            .unwrap_or_else(hook::default_dir),
    })
}

/// Entry point for `calibrate watch`.
pub async fn run(args: WatchArgs) -> Result<()> {
    let s = resolve_settings(&args)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut warnings: Vec<String> = Vec::new();

    // 1. Start collection (live process, devices, or replay).
    let (ticks, info, mode, pid) = if let Some(path) = &args.replay {
        let (rx, info) = replay::spawn(path, args.replay_speed.unwrap_or(0.0), stop.clone())?;
        (rx, info, "replay", None)
    } else if let Some(sel) = args.gpus.clone() {
        let (rx, info) = runner::spawn(
            RunnerConfig {
                target: Target::Devices(sel),
                interval: s.interval,
                metrics_source: s.metrics_source,
                hook_dir: s.hook_dir.clone(),
            },
            stop.clone(),
        )?;
        (rx, info, "device", None)
    } else {
        let (pid_arg, name, auto) = (args.pid, args.process_name.clone(), args.auto);
        let pid = tokio::task::spawn_blocking(move || {
            discover::resolve_pid(pid_arg, name.as_deref(), auto)
        })
        .await
        .context("process discovery panicked")?
        .context("Failed to resolve target process")?;
        let pinfo = attach::attach(pid).context("Failed to attach to training process")?;
        match pinfo.container_context {
            ContainerContext::Docker | ContainerContext::Kubernetes | ContainerContext::Unknown => {
                warnings.push(
                    "calibrate is running inside a container: only processes in this PID \
                     namespace are visible (use `--gpus all` for device-level monitoring)"
                        .into(),
                )
            }
            ContainerContext::Host => {}
        }
        warnings.extend(pinfo.warnings);
        let (rx, info) = runner::spawn(
            RunnerConfig {
                target: Target::Process {
                    root: pid,
                    include_children: !args.no_children,
                },
                interval: s.interval,
                metrics_source: s.metrics_source,
                hook_dir: s.hook_dir.clone(),
            },
            stop.clone(),
        )?;
        (rx, info, "process", Some(pid))
    };
    warnings.extend(info.warnings.iter().cloned());
    info!(mode, backend = %info.backend, source = %info.metrics_source, "collection started");

    // 2. GPU spec DB (may touch the network, bounded by a short timeout).
    let specs = tokio::task::spawn_blocking(SpecDb::load)
        .await
        .context("GPU spec loading panicked")?;

    // 3. Optional recorders.
    let mut raw_writer = match &args.record_raw {
        Some(p) => Some(TraceWriter::create(p, &info, s.interval)?),
        None => None,
    };
    let mut snap_writer = match &args.record {
        Some(p) => Some(BufWriter::new(
            File::create(p).with_context(|| format!("creating {}", p.display()))?,
        )),
        None => None,
    };

    // 4. Session + exporters.
    let (snap_tx, snap_rx) = new_snapshot_channel();
    let session = MonitoringSession::new(
        SessionConfig {
            mode: mode.to_string(),
            pid,
            backend: info.backend.clone(),
            metrics_source: info.metrics_source.clone(),
            scope: info.scope,
            cost_per_hour: s.cost_per_hour,
            target_mfu: s.target_mfu,
            warnings,
        },
        specs,
        snap_tx,
        stop.clone(),
    )
    .start();

    if let Some(addr) = &s.prometheus_listen {
        let bound = prometheus::serve(addr, snap_rx.clone()).await?;
        eprintln!("[calibrate] Prometheus metrics on http://{bound}/metrics");
    }
    let otlp_task = match &s.otlp_endpoint {
        Some(ep) => {
            let t = otlp::spawn(
                otlp::OtlpConfig {
                    endpoint: ep.clone(),
                    headers: args.otlp_headers.clone(),
                    min_interval: s.interval.max(Duration::from_secs(1)),
                },
                snap_rx.clone(),
            )?;
            eprintln!(
                "[calibrate] Pushing OTLP metrics to {}",
                otlp::metrics_url(ep)
            );
            Some(t)
        }
        None => None,
    };

    let (stream_tx, stream_rx) = flume::unbounded::<SessionSnapshot>();
    let session_handle = tokio::spawn(session.run(
        ticks,
        move |tick| {
            if let Some(w) = raw_writer.as_mut() {
                if let Err(e) = w.write(tick) {
                    warn!("--record-raw write failed: {e}");
                }
            }
        },
        stream_tx,
    ));

    // 5. Signals: SIGINT (non-TUI modes) and SIGTERM (Kubernetes, systemd).
    spawn_signal_handler(stop.clone());

    let mut renderer: Box<dyn OutputRenderer> = match args.output {
        OutputFormat::Terminal => {
            use std::io::IsTerminal;
            if !std::io::stdout().is_terminal() {
                bail!("--output terminal needs an interactive terminal; use --output json or none");
            }
            Box::new(TerminalRenderer::new().context("Failed to initialize terminal renderer")?)
        }
        OutputFormat::Json => Box::new(JsonRenderer),
        OutputFormat::None => Box::new(NullRenderer),
    };

    // 6. Render loop.
    let started = Instant::now();
    let deadline = args.duration.map(Duration::from_secs_f64);
    let mut last: Option<SessionSnapshot> = None;
    loop {
        tokio::select! {
            msg = stream_rx.recv_async() => match msg {
                Ok(snap) => {
                    if let Some(w) = snap_writer.as_mut() {
                        if let Ok(line) = crate::output::json::snapshot_line(&snap) {
                            let _ = writeln!(w, "{line}").and_then(|_| w.flush());
                        }
                    }
                    renderer.render(&snap);
                    last = Some(snap);
                }
                Err(_) => break,
            },
            _ = tokio::time::sleep(Duration::from_millis(200)) => {
                if let Some(snap) = &last {
                    if args.output == OutputFormat::Terminal {
                        renderer.render(snap);
                    }
                }
            }
        }
        if renderer.quit_requested() || deadline.is_some_and(|d| started.elapsed() >= d) {
            stop.store(true, Ordering::Relaxed);
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
    stop.store(true, Ordering::Relaxed);

    let done = session_handle.await.context("Session task panicked")?;
    // Snapshots produced between the last render and shutdown.
    while let Ok(snap) = stream_rx.try_recv() {
        last = Some(snap);
    }
    let final_snapshot = done.final_snapshot().or(last);
    drop(done); // closes the snapshot channel → final OTLP push
    renderer.finish(final_snapshot.as_ref());
    if let Some(w) = snap_writer.as_mut() {
        let _ = writeln!(
            w,
            "{}",
            crate::output::json::session_end_line(final_snapshot.as_ref())
        );
        let _ = w.flush();
    }
    if let Some(t) = otlp_task {
        let _ = tokio::time::timeout(Duration::from_secs(5), t).await;
    }
    Ok(())
}

fn spawn_signal_handler(stop: Arc<AtomicBool>) {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut term = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    warn!("cannot install SIGTERM handler: {e}");
                    let _ = tokio::signal::ctrl_c().await;
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => info!("SIGINT received — stopping"),
                _ = term.recv() => info!("SIGTERM received — stopping"),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        stop.store(true, Ordering::Relaxed);
    });
}
