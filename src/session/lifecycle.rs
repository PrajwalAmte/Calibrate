//! Aggregation loop: ticks in, snapshots out.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use hdrhistogram::Histogram;
use tracing::{info, warn};

use crate::analysis::bottleneck::{BottleneckDetector, DetectInput};
use crate::analysis::recommendations::{Accelerator, RecommendationEngine};
use crate::collectors::{GpuScope, Tick};
use crate::gpu_specs::{ResolvedSpec, SpecDb};
use crate::metrics::window::{TickWindow, MIN_RELIABLE_SAMPLES};
use crate::metrics::{breakdown, mfu, steps};
use crate::session::state::{
    CostImpact, GpuStatus, MfuPercentiles, SessionSnapshot, SnapshotSender, TargetInfo,
    SCHEMA_VERSION,
};

/// Typestate markers for `MonitoringSession`.
pub mod state {
    pub struct Initializing;
    pub struct Sampling;
    pub struct Done;
}

/// Static configuration of a session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub mode: String,
    pub pid: Option<u32>,
    pub backend: String,
    pub metrics_source: String,
    pub scope: GpuScope,
    pub cost_per_hour: Option<f64>,
    pub target_mfu: f32,
    pub warnings: Vec<String>,
}

impl SessionConfig {
    fn accelerator(&self) -> Accelerator {
        match self.backend.as_str() {
            "nvml" => Accelerator::Cuda,
            "apple-iokit" => Accelerator::Mps,
            _ => Accelerator::Unknown,
        }
    }
    fn has_gpu_backend(&self) -> bool {
        self.backend != "none"
    }
}

/// Ticks of sustained growth required before flagging a leak, and the minimum
/// growth as a fraction of device memory (allocator warm-up is excluded by
/// requiring the session to be past its first minute).
const LEAK_TICKS: usize = 30;
const LEAK_MIN_GROWTH_FRAC: f64 = 0.02;
const LEAK_MIN_ELAPSED_SECS: f64 = 60.0;

/// Mutable analysis state, separate from the async loop so it can be unit-tested.
pub struct Analyzer {
    cfg: SessionConfig,
    specs_db: SpecDb,
    specs: HashMap<String, ResolvedSpec>,
    window: TickWindow,
    first_ts: Option<u64>,
    last_ts: Option<u64>,
    ticks: u64,
    hist: Histogram<u64>,
    peak_mfu: Option<f32>,
    peak_vram: BTreeMap<u32, u64>,
    vram_hist: BTreeMap<u32, VecDeque<u64>>,
    wasted_usd: f64,
    spent_usd: f64,
    hostname: String,
}

impl Analyzer {
    pub fn new(cfg: SessionConfig, specs_db: SpecDb) -> Self {
        Self {
            cfg,
            specs_db,
            specs: HashMap::new(),
            window: TickWindow::new(),
            first_ts: None,
            last_ts: None,
            ticks: 0,
            hist: Histogram::new_with_max(1000, 3).expect("valid histogram bounds"),
            peak_mfu: None,
            peak_vram: BTreeMap::new(),
            vram_hist: BTreeMap::new(),
            wasted_usd: 0.0,
            spent_usd: 0.0,
            hostname: hostname(),
        }
    }

    /// Ingest one tick and return the resulting snapshot.
    pub fn ingest(&mut self, tick: Tick) -> SessionSnapshot {
        let dt_secs = self
            .last_ts
            .map(|p| tick.timestamp_ms.saturating_sub(p) as f64 / 1000.0)
            .unwrap_or(0.0)
            .min(3600.0);
        self.first_ts.get_or_insert(tick.timestamp_ms);
        self.last_ts = Some(tick.timestamp_ms);
        self.ticks += 1;

        for g in &tick.gpus {
            if !self.specs.contains_key(&g.name) {
                let r = self.specs_db.resolve(&g.name);
                if !r.known() {
                    warn!(gpu = %g.name, "GPU not in spec database — peak TFLOPS unknown");
                }
                self.specs.insert(g.name.clone(), r);
            }
            if let Some(v) = g.vram_used_mib {
                let p = self.peak_vram.entry(g.index).or_insert(0);
                *p = (*p).max(v);
                let h = self.vram_hist.entry(g.index).or_default();
                h.push_back(v);
                while h.len() > LEAK_TICKS {
                    h.pop_front();
                }
            }
        }
        self.window.push(tick);

        let elapsed = self.elapsed_secs();
        let est = mfu::compute(&self.window, &self.specs);
        if let Some(m) = est.mfu_pct {
            let _ = self
                .hist
                .record((m * 10.0).round().clamp(0.0, 1000.0) as u64);
            self.peak_mfu = Some(self.peak_mfu.map_or(m, |p| p.max(m)));
        }
        let bd = breakdown::infer(&self.window);
        let step_stats = steps::compute(&self.window);

        let latest = self.window.latest().expect("just pushed");
        let gpus = self.gpu_statuses(latest, elapsed);
        let busy_mean = mfu::per_gpu_mean(&self.window, |g| g.busy_pct().map(f64::from));
        let divergence = (busy_mean.len() > 1).then(|| {
            let max = busy_mean.values().cloned().fold(f64::MIN, f64::max);
            let min = busy_mean.values().cloned().fold(f64::MAX, f64::min);
            (max - min) as f32
        });
        let busy_pct = (!busy_mean.is_empty())
            .then(|| (busy_mean.values().sum::<f64>() / busy_mean.len() as f64) as f32);
        let clocks: Vec<f32> = gpus.iter().filter_map(|g| g.clock_pct_of_boost).collect();
        let clock_pct =
            (!clocks.is_empty()).then(|| clocks.iter().sum::<f32>() / clocks.len() as f32);

        let detect = DetectInput {
            mfu: &est,
            breakdown: &bd,
            has_gpu_backend: self.cfg.has_gpu_backend(),
            gpus_in_use: latest.gpus.len(),
            thermal_throttle: latest.gpus.iter().any(|g| g.throttle.thermal()),
            power_cap: latest.gpus.iter().any(|g| g.throttle.power_cap),
            hw_slowdown: latest.gpus.iter().any(|g| g.throttle.hw_slowdown),
            clock_pct_of_boost: clock_pct,
            busy_pct,
            steps_erratic: step_stats.as_ref().is_some_and(|s| s.erratic),
            gpu_divergence_ppt: divergence,
            target_mfu: self.cfg.target_mfu,
            samples: self.window.len(),
        };
        let bottleneck = BottleneckDetector::detect(&detect);
        let recommendation = RecommendationEngine::recommend(
            bottleneck,
            self.cfg.accelerator(),
            est.mfu_pct,
            bd.data_wait_pct,
            self.cfg.target_mfu,
            !latest.hook_peers.is_empty(),
        );

        let cost = self.cfg.cost_per_hour.map(|cph| {
            let target = self.cfg.target_mfu as f64;
            let waste_ratio = match est.mfu_pct {
                Some(m) => ((target - m as f64) / target).clamp(0.0, 1.0),
                // No GPU in use at all → everything is waste; unknown → assume none.
                None if latest.gpus.is_empty() && self.cfg.has_gpu_backend() => 1.0,
                None => 0.0,
            };
            self.spent_usd += cph * dt_secs / 3600.0;
            self.wasted_usd += cph * waste_ratio * dt_secs / 3600.0;
            CostImpact {
                cost_per_hour: cph,
                waste_ratio,
                waste_per_hour: cph * waste_ratio,
                spent_usd: self.spent_usd,
                wasted_usd: self.wasted_usd,
                is_lower_bound: est.upper_bound,
            }
        });

        let percentiles =
            (self.hist.len() as usize >= MIN_RELIABLE_SAMPLES).then(|| MfuPercentiles {
                p50: self.hist.value_at_quantile(0.50) as f32 / 10.0,
                p75: self.hist.value_at_quantile(0.75) as f32 / 10.0,
                p95: self.hist.value_at_quantile(0.95) as f32 / 10.0,
            });

        let mut warnings = self.cfg.warnings.clone();
        for g in &gpus {
            if g.peak_tflops.is_none() {
                warnings.push(format!(
                    "{} is not in the GPU spec database — TFLOPS and peak-relative figures \
                     are unavailable",
                    g.name
                ));
            }
        }
        warnings.dedup();

        SessionSnapshot {
            schema_version: SCHEMA_VERSION,
            calibrate_version: env!("CARGO_PKG_VERSION"),
            timestamp_ms: latest.timestamp_ms,
            elapsed_secs: elapsed,
            target: TargetInfo {
                mode: self.cfg.mode.clone(),
                pid: self.cfg.pid,
                process_count: latest.process_count,
                scope: self.cfg.scope,
                hostname: self.hostname.clone(),
            },
            backend: self.cfg.backend.clone(),
            metrics_source: self.cfg.metrics_source.clone(),
            gpu_name: gpu_set_name(&gpus),
            mfu: est,
            mfu_percentiles: percentiles,
            peak_mfu_pct: self.peak_mfu,
            breakdown: bd,
            bottleneck,
            recommendation,
            vram_growing: gpus.iter().any(|g| g.vram_growing),
            gpus,
            cpu_pct: latest.cpu_pct,
            steps: step_stats,
            hook_peers: latest.hook_peers.len() as u32,
            cost,
            gpu_divergent: divergence.is_some_and(|d| d > 20.0),
            gpu_divergence_ppt: divergence,
            target_mfu_pct: self.cfg.target_mfu,
            ticks_observed: self.ticks,
            warnings,
        }
    }

    fn elapsed_secs(&self) -> f64 {
        match (self.first_ts, self.last_ts) {
            (Some(a), Some(b)) => b.saturating_sub(a) as f64 / 1000.0,
            _ => 0.0,
        }
    }

    fn gpu_statuses(&self, latest: &Tick, elapsed: f64) -> Vec<GpuStatus> {
        let mut out: Vec<GpuStatus> = latest
            .gpus
            .iter()
            .map(|g| {
                let spec = self.specs.get(&g.name).filter(|r| r.known());
                let clock_pct = match (g.sm_clock_mhz, spec) {
                    (Some(c), Some(r)) if r.spec.boost_clock_mhz > 0 => {
                        Some(c as f32 / r.spec.boost_clock_mhz as f32 * 100.0)
                    }
                    (Some(c), None) => g
                        .sm_clock_max_mhz
                        .filter(|&m| m > 0)
                        .map(|m| c as f32 / m as f32 * 100.0),
                    _ => None,
                };
                let total = g.vram_total_mib.unwrap_or(0);
                let growing = elapsed >= LEAK_MIN_ELAPSED_SECS
                    && self
                        .vram_hist
                        .get(&g.index)
                        .is_some_and(|h| is_leaking(h, total));
                GpuStatus {
                    index: g.index,
                    name: g.name.clone(),
                    peak_tflops: spec.map(|r| r.spec.bf16_tflops),
                    util_pct: g.util_pct,
                    sm_active_pct: g.prof.and_then(|p| p.sm_active_pct),
                    tensor_active_pct: g.prof.and_then(|p| p.tensor_active_pct),
                    dram_active_pct: g.prof.and_then(|p| p.dram_active_pct),
                    sm_clock_mhz: g.sm_clock_mhz,
                    clock_pct_of_boost: clock_pct,
                    vram_used_mib: g.vram_used_mib,
                    vram_total_mib: g.vram_total_mib,
                    peak_vram_mib: self.peak_vram.get(&g.index).copied(),
                    temperature_c: g.temperature_c,
                    power_w: g.power_w,
                    power_limit_w: g.power_limit_w,
                    throttle: g.throttle,
                    vram_growing: growing,
                }
            })
            .collect();
        out.sort_by_key(|g| g.index);
        out
    }
}

/// Sustained growth: the last `LEAK_TICKS` readings never decrease, most of
/// them increase, and total growth is a meaningful share of device memory.
pub fn is_leaking(hist: &VecDeque<u64>, total_mib: u64) -> bool {
    if hist.len() < LEAK_TICKS {
        return false;
    }
    let v: Vec<u64> = hist.iter().copied().collect();
    let non_decreasing = v.windows(2).all(|w| w[1] >= w[0]);
    let increases = v.windows(2).filter(|w| w[1] > w[0]).count();
    let growth = v[v.len() - 1].saturating_sub(v[0]) as f64;
    let min_growth = (total_mib.max(1) as f64 * LEAK_MIN_GROWTH_FRAC).max(64.0);
    non_decreasing && increases * 2 >= v.len() && growth >= min_growth
}

fn gpu_set_name(gpus: &[GpuStatus]) -> String {
    if gpus.is_empty() {
        return "no GPU in use".into();
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for g in gpus {
        *counts.entry(g.name.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(n, c)| {
            if c > 1 {
                format!("{c}× {n}")
            } else {
                n.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: gethostname writes at most buf.len() bytes.
    let r = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if r == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    } else {
        "unknown".into()
    }
}

/// Drives the sampling loop using a typestate machine:
/// `Initializing → Sampling → Done`.
pub struct MonitoringSession<S> {
    analyzer: Analyzer,
    snap_tx: SnapshotSender,
    stop: Arc<AtomicBool>,
    _phase: std::marker::PhantomData<S>,
}

impl MonitoringSession<state::Initializing> {
    pub fn new(
        cfg: SessionConfig,
        specs_db: SpecDb,
        snap_tx: SnapshotSender,
        stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            analyzer: Analyzer::new(cfg, specs_db),
            snap_tx,
            stop,
            _phase: std::marker::PhantomData,
        }
    }

    pub fn start(self) -> MonitoringSession<state::Sampling> {
        info!(mode = %self.analyzer.cfg.mode, "monitoring session starting");
        MonitoringSession {
            analyzer: self.analyzer,
            snap_tx: self.snap_tx,
            stop: self.stop,
            _phase: std::marker::PhantomData,
        }
    }
}

impl MonitoringSession<state::Sampling> {
    /// Consume ticks until the collector stops or `stop` is set. Each tick
    /// passes through `on_tick` (recording) before analysis; every snapshot
    /// is sent to `stream` (lossless, for JSON output and recording) and
    /// published on the watch channel (latest-only, for exporters).
    pub async fn run(
        mut self,
        rx: flume::Receiver<Tick>,
        mut on_tick: impl FnMut(&Tick) + Send,
        stream: flume::Sender<SessionSnapshot>,
    ) -> MonitoringSession<state::Done> {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            match tokio::time::timeout(Duration::from_millis(200), rx.recv_async()).await {
                Ok(Ok(tick)) => {
                    on_tick(&tick);
                    let snap = self.analyzer.ingest(tick);
                    let _ = stream.send(snap.clone());
                    let _ = self.snap_tx.send(Some(snap));
                }
                Ok(Err(_)) => {
                    info!("collector finished");
                    break;
                }
                Err(_) => {}
            }
        }
        MonitoringSession {
            analyzer: self.analyzer,
            snap_tx: self.snap_tx,
            stop: self.stop,
            _phase: std::marker::PhantomData,
        }
    }
}

impl MonitoringSession<state::Done> {
    pub fn final_snapshot(&self) -> Option<SessionSnapshot> {
        self.snap_tx.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::bottleneck::Bottleneck;
    use crate::collectors::testing::{gpu, tick};
    use crate::metrics::mfu::MfuSource;
    use crate::session::state::new_snapshot_channel;

    fn cfg(cost: Option<f64>) -> SessionConfig {
        SessionConfig {
            mode: "process".into(),
            pid: Some(1),
            backend: "nvml".into(),
            metrics_source: "nvml".into(),
            scope: GpuScope::Process,
            cost_per_hour: cost,
            target_mfu: 45.0,
            warnings: vec![],
        }
    }

    fn named(i: u32, util: f32) -> crate::collectors::GpuSample {
        let mut g = gpu(i, util);
        g.name = "NVIDIA GeForce RTX 4090".into();
        g.sm_clock_mhz = Some(2520);
        g.sm_clock_max_mhz = Some(3105);
        g
    }

    #[tokio::test]
    async fn session_produces_snapshots_and_ends_when_collector_closes() {
        let (snap_tx, _rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = flume::bounded(64);
        let session = MonitoringSession::new(cfg(None), SpecDb::builtin(), snap_tx, stop).start();
        let (stream_tx, stream_rx) = flume::unbounded();
        let handle = tokio::spawn(async move {
            let mut recorded = 0;
            let done = session.run(rx, |_| recorded += 1, stream_tx).await;
            (done.final_snapshot(), recorded)
        });
        for i in 0..20 {
            tx.send(tick(i * 2000, vec![named(0, 80.0)], Some(120.0)))
                .unwrap();
        }
        drop(tx);
        let (snap, recorded) = handle.await.unwrap();
        let snap = snap.expect("snapshot");
        assert_eq!(recorded, 20);
        assert_eq!(stream_rx.len(), 20, "every snapshot reaches the stream");
        assert_eq!(snap.ticks_observed, 20);
        assert_eq!(snap.schema_version, SCHEMA_VERSION);
        assert_eq!(snap.mfu.source, MfuSource::KernelBusy);
        assert!((snap.mfu.mfu_pct.unwrap() - 80.0).abs() < 0.5);
        assert!((snap.elapsed_secs - 38.0).abs() < 1e-9);
        assert!(snap.mfu_percentiles.is_some());
    }

    #[test]
    fn idle_gpu_counts_as_full_waste() {
        let mut a = Analyzer::new(cfg(Some(3.6)), SpecDb::builtin());
        a.ingest(tick(0, vec![named(0, 0.0)], Some(0.0)));
        let s = a.ingest(tick(1_000_000, vec![named(0, 0.0)], Some(0.0)));
        let c = s.cost.unwrap();
        assert!((c.waste_ratio - 1.0).abs() < 1e-9, "0% MFU is 100% waste");
        assert!((c.spent_usd - 1.0).abs() < 1e-6, "1000 s at $3.6/h");
        assert!((c.wasted_usd - 1.0).abs() < 1e-6);
        assert!(c.is_lower_bound);
    }

    #[test]
    fn healthy_job_wastes_nothing() {
        let mut a = Analyzer::new(cfg(Some(2.0)), SpecDb::builtin());
        let s = a.ingest(tick(0, vec![named(0, 99.0)], Some(100.0)));
        assert_eq!(s.cost.unwrap().waste_ratio, 0.0);
    }

    #[test]
    fn per_gpu_peaks_and_divergence() {
        let mut a = Analyzer::new(cfg(None), SpecDb::builtin());
        let mut last = None;
        for (i, v) in [1000u64, 5000, 9000, 7000].into_iter().enumerate() {
            let mut g0 = named(0, 90.0);
            g0.vram_used_mib = Some(v);
            let mut g1 = named(1, 40.0);
            g1.vram_used_mib = Some(100);
            last = Some(a.ingest(tick(i as u64 * 1000, vec![g0, g1], Some(100.0))));
        }
        let s = last.unwrap();
        assert_eq!(s.gpus.len(), 2);
        assert_eq!(s.gpus[0].peak_vram_mib, Some(9000));
        assert_eq!(s.gpus[1].peak_vram_mib, Some(100));
        assert!(s.gpu_divergent);
        assert!((s.gpu_divergence_ppt.unwrap() - 50.0).abs() < 0.1);
        assert_eq!(s.gpu_name, "2× NVIDIA GeForce RTX 4090");
    }

    #[test]
    fn leak_detection_needs_sustained_meaningful_growth() {
        let total = 24_576;
        let grow: VecDeque<u64> = (0..30).map(|i| 10_000 + i * 50).collect();
        assert!(is_leaking(&grow, total));
        let flat: VecDeque<u64> = (0..30).map(|_| 10_000).collect();
        assert!(!is_leaking(&flat, total));
        let tiny: VecDeque<u64> = (0..30).map(|i| 10_000 + i).collect();
        assert!(!is_leaking(&tiny, total), "allocator jitter is not a leak");
        let mut dip = grow.clone();
        dip[15] = 0;
        assert!(!is_leaking(&dip, total));
        let short: VecDeque<u64> = (0..10).map(|i| 10_000 + i * 500).collect();
        assert!(!is_leaking(&short, total));
    }

    #[test]
    fn no_backend_reports_no_gpu_metrics() {
        let mut c = cfg(None);
        c.backend = "none".into();
        let mut a = Analyzer::new(c, SpecDb::builtin());
        let s = a.ingest(tick(0, vec![], Some(50.0)));
        assert_eq!(s.bottleneck, Bottleneck::NoGpuMetrics);
        assert_eq!(s.mfu.mfu_pct, None);
        assert_eq!(s.gpu_name, "no GPU in use");
    }
}
