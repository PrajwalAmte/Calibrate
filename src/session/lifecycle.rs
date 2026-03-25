use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use tracing::{info, warn};

use crate::analysis::bottleneck::BottleneckDetector;
use crate::analysis::recommendations::RecommendationEngine;
use crate::collectors::RawSample;
use crate::gpu_specs::GpuSpec;
use crate::metrics::breakdown::TimeBreakdownInferrer;
use crate::metrics::mfu::MfuCalculator;
use crate::metrics::units::{Mib, Percent};
use crate::metrics::window::{MetricsWindow, MAX_WINDOW_SIZE, MIN_RELIABLE_SAMPLES};
use crate::session::state::{
    CostImpact, GpuSnapshot, MfuPercentiles, SessionSnapshot, SnapshotSender,
};

/// Typestate markers for `MonitoringSession`.
pub mod state {
    pub struct Initializing;
    pub struct Sampling;
    pub struct Done;
}

/// Drives the sampling loop using a typestate machine.
///
/// Transitions: `Initializing → Sampling → Done`.
/// The watch channel sender is owned here; dropping it on loop exit
/// causes any render loop awaiting `changed()` to unblock automatically.
pub struct MonitoringSession<S> {
    pid: u32,
    gpu_spec: GpuSpec,
    cost_per_hour: Option<f64>,
    snap_tx: SnapshotSender,
    stop: Arc<AtomicBool>,
    /// Whether NVML was available at startup.  Threaded through every
    /// typestate so the final `SessionSnapshot` can report it accurately.
    nvml_available: bool,
    _phase: std::marker::PhantomData<S>,
}

impl MonitoringSession<state::Initializing> {
    pub fn new(
        pid: u32,
        gpu_spec: GpuSpec,
        cost_per_hour: Option<f64>,
        snap_tx: SnapshotSender,
        stop: Arc<AtomicBool>,
        nvml_available: bool,
    ) -> Self {
        Self {
            pid,
            gpu_spec,
            cost_per_hour,
            snap_tx,
            stop,
            nvml_available,
            _phase: std::marker::PhantomData,
        }
    }

    pub fn start(self) -> MonitoringSession<state::Sampling> {
        info!("MonitoringSession starting for PID {}", self.pid);
        MonitoringSession {
            pid: self.pid,
            gpu_spec: self.gpu_spec,
            cost_per_hour: self.cost_per_hour,
            snap_tx: self.snap_tx,
            stop: self.stop,
            nvml_available: self.nvml_available,
            _phase: std::marker::PhantomData,
        }
    }
}

impl MonitoringSession<state::Sampling> {
    /// Run the aggregation loop, consuming samples from `rx` and broadcasting
    /// a `SessionSnapshot` on the watch channel after each sample.
    pub async fn run(self, rx: flume::Receiver<RawSample>) -> MonitoringSession<state::Done> {
        let mut window = MetricsWindow::new(MAX_WINDOW_SIZE);
        let started_at = Instant::now();
        let mut steps_observed: u64 = 0;
        let gpu_name = self.gpu_spec.name.clone();

        // hdrhistogram tracks MFU * 10 (integer tenths of a percent, 0–1000)
        // so we get 0.1% precision without floating-point histogram keys.
        let mut mfu_hist: Histogram<u64> =
            Histogram::new_with_max(1000, 3).expect("valid hdrhistogram config");

        let mut peak_mfu: f32 = 0.0;
        let mut peak_vram = Mib(0);

        let mut gpu_snapshots: HashMap<u32, GpuSnapshot> = HashMap::new();

        // Rolling VRAM values for monotonic-growth detection (memory leak signal).
        let mut vram_history: std::collections::VecDeque<u64> =
            std::collections::VecDeque::with_capacity(12);

        // Inter-sample timestamps for step-timing variance.
        let mut step_timestamps: std::collections::VecDeque<u64> =
            std::collections::VecDeque::with_capacity(32);

        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(sample) => {
                    if sample.vram_used_mib > peak_vram {
                        peak_vram = sample.vram_used_mib;
                    }

                    gpu_snapshots.insert(
                        sample.gpu_index,
                        GpuSnapshot {
                            gpu_index: sample.gpu_index,
                            sm_utilization: sample.sm_utilization,
                            vram_used_mib: sample.vram_used_mib,
                            temperature: sample.temperature,
                        },
                    );

                    window.push(sample.clone());
                    steps_observed += 1;

                    let mfu_calc = MfuCalculator::new(&self.gpu_spec);
                    let mfu = mfu_calc.compute(&window);

                    // hdrhistogram stores MFU * 10 (integer tenths) for 0.1% precision.
                    let bucket = (mfu.mfu_pct.0 * 10.0).clamp(0.0, 1000.0) as u64;
                    let _ = mfu_hist.record(bucket);

                    if mfu.mfu_pct.0 > peak_mfu {
                        peak_mfu = mfu.mfu_pct.0;
                    }

                    let mfu_percentiles = if steps_observed >= MIN_RELIABLE_SAMPLES as u64 {
                        Some(MfuPercentiles {
                            p50: mfu_hist.value_at_quantile(0.50) as f32 / 10.0,
                            p75: mfu_hist.value_at_quantile(0.75) as f32 / 10.0,
                            p95: mfu_hist.value_at_quantile(0.95) as f32 / 10.0,
                        })
                    } else {
                        None
                    };

                    let breakdown = TimeBreakdownInferrer::infer(&window);

                    let avg_clock_pct = window
                        .iter()
                        .map(|s| {
                            if s.sm_clock_max_mhz.0 > 0 {
                                s.sm_clock_mhz.0 as f32 / s.sm_clock_max_mhz.0 as f32 * 100.0
                            } else {
                                100.0
                            }
                        })
                        .sum::<f32>()
                        / window.len() as f32;

                    let bottleneck = BottleneckDetector::detect(
                        &mfu,
                        &breakdown,
                        sample.throttle_thermal,
                        sample.throttle_power,
                        avg_clock_pct,
                    );
                    let recommendation =
                        RecommendationEngine::recommend(&bottleneck, breakdown.data_loader_pct);

                    let cost_impact = self.cost_per_hour.map(|cph| {
                        let waste_ratio = if mfu.mfu_pct.0 > 0.0 {
                            (45.0 - mfu.mfu_pct.0.min(45.0)) / 45.0
                        } else {
                            0.0
                        };
                        CostImpact {
                            cost_per_hour: cph,
                            current_cost_usd: cph,
                            target_cost_usd: cph * (1.0 - waste_ratio as f64),
                            waste_per_hour: cph * waste_ratio as f64,
                        }
                    });

                    let vram_util = Percent::clamped(
                        sample.vram_used_mib.0 as f32 / sample.vram_total_mib.0.max(1) as f32
                            * 100.0,
                    );

                    // VRAM growth detection: flag if VRAM has increased every
                    // tick for the last 8+ consecutive samples.
                    vram_history.push_back(sample.vram_used_mib.0);
                    if vram_history.len() > 10 {
                        vram_history.pop_front();
                    }
                    let vram_growing = vram_history.len() >= 8 && {
                        let v: Vec<u64> = vram_history.iter().copied().collect();
                        v.windows(2).all(|w| w[1] > w[0])
                    };

                    // Step timing variance: CV > 0.3 → erratic.
                    step_timestamps.push_back(sample.timestamp_ms);
                    if step_timestamps.len() > 30 {
                        step_timestamps.pop_front();
                    }
                    let (step_time_ms_mean, step_time_erratic) = if step_timestamps.len() >= 5 {
                        let ts: Vec<u64> = step_timestamps.iter().copied().collect();
                        let deltas: Vec<f64> = ts
                            .windows(2)
                            .map(|w| w[1].saturating_sub(w[0]) as f64)
                            .collect();
                        let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
                        let variance = deltas.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
                            / deltas.len() as f64;
                        let cv = variance.sqrt() / mean.max(1.0);
                        (mean as f32, cv > 0.3)
                    } else {
                        (0.0, false)
                    };

                    // Multi-GPU divergence: flag when SM util spread > 20 ppt.
                    let (mfu_divergent, gpu_mfu_divergence_ppt) = {
                        let utils: Vec<f32> =
                            gpu_snapshots.values().map(|g| g.sm_utilization.0).collect();
                        if utils.len() > 1 {
                            let max_u = utils.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let min_u = utils.iter().cloned().fold(f32::INFINITY, f32::min);
                            let spread = max_u - min_u;
                            (spread > 20.0, spread)
                        } else {
                            (false, 0.0)
                        }
                    };
                    let per_gpu: Vec<GpuSnapshot> = {
                        let mut v: Vec<GpuSnapshot> = gpu_snapshots.values().cloned().collect();
                        v.sort_by_key(|g| g.gpu_index);
                        v
                    };

                    let snapshot = SessionSnapshot {
                        elapsed: started_at.elapsed(),
                        gpu_name: gpu_name.clone(),
                        mfu,
                        mfu_percentiles,
                        peak_mfu_pct: peak_mfu,
                        breakdown,
                        bottleneck,
                        recommendation,
                        temperature: sample.temperature,
                        power_draw: sample.power_draw,
                        power_limit: sample.power_limit,
                        throttle_thermal: sample.throttle_thermal,
                        vram_used_mib: sample.vram_used_mib,
                        vram_total_mib: sample.vram_total_mib,
                        vram_utilization: vram_util,
                        peak_vram_mib: peak_vram,
                        cost_impact,
                        per_gpu,
                        steps_observed,
                        mfu_divergent,
                        gpu_mfu_divergence_ppt,
                        nvml_available: self.nvml_available,
                        vram_growing,
                        step_time_ms_mean,
                        step_time_erratic,
                    };

                    let _ = self.snap_tx.send(Some(snapshot));
                }
                Err(flume::RecvTimeoutError::Timeout) => {}
                Err(flume::RecvTimeoutError::Disconnected) => {
                    warn!("Sample channel disconnected — training process likely exited");
                    break;
                }
            }
        }

        MonitoringSession {
            pid: self.pid,
            gpu_spec: self.gpu_spec,
            cost_per_hour: self.cost_per_hour,
            snap_tx: self.snap_tx,
            stop: self.stop,
            nvml_available: self.nvml_available,
            _phase: std::marker::PhantomData,
        }
    }
}

impl MonitoringSession<state::Done> {
    /// Return the final snapshot, if any samples were processed.
    ///
    /// Reads from the watch sender's current value — always the last snapshot
    /// that was published by the loop.
    pub fn final_snapshot(&self) -> Option<SessionSnapshot> {
        self.snap_tx.borrow().clone()
    }
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::RawSample;
    use crate::gpu_specs::GpuSpec;
    use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};
    use crate::session::state::new_snapshot_channel;

    fn test_spec() -> GpuSpec {
        GpuSpec {
            name: "Test GPU".to_string(),
            bf16_tflops: 100.0,
            fp32_tflops: 50.0,
            vram_gib: 16,
            boost_clock_mhz: 2000,
        }
    }

    fn make_sample(sm: f32, cpu: f32, vram_mib: u64) -> RawSample {
        make_sample_gpu(sm, cpu, vram_mib, 0)
    }

    fn make_sample_gpu(sm: f32, cpu: f32, vram_mib: u64, gpu_index: u32) -> RawSample {
        RawSample {
            timestamp_ms: 0,
            gpu_index,
            sm_utilization: Percent(sm),
            sm_clock_mhz: Mhz(2000),
            sm_clock_max_mhz: Mhz(2000),
            vram_used_mib: Mib(vram_mib),
            vram_total_mib: Mib(16384),
            mem_utilization: Percent(40.0),
            temperature: Celsius(65.0),
            power_draw: Watts(200.0),
            power_limit: Watts(350.0),
            throttle_thermal: false,
            throttle_power: false,
            throttle_hw_slowdown: false,
            cpu_utilization: Percent(cpu),
        }
    }

    #[tokio::test]
    async fn session_produces_snapshots_from_mock_collector() {
        let (snap_tx, _snap_rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (flume_tx, flume_rx) = flume::bounded(64);

        let session =
            MonitoringSession::new(9999, test_spec(), None, snap_tx, stop.clone(), true).start();

        let session_handle = tokio::spawn(session.run(flume_rx));

        // Send 20 samples — all forward/backward pattern.
        for _ in 0..20 {
            flume_tx.send(make_sample(80.0, 20.0, 8192)).unwrap();
        }
        // Dropping the sender closes the channel → lifecycle exits cleanly.
        drop(flume_tx);

        let done = session_handle.await.expect("session task must not panic");
        let snapshot = done
            .final_snapshot()
            .expect("must have at least one snapshot");

        assert_eq!(snapshot.steps_observed, 20);
        assert!(
            snapshot.mfu.mfu_pct.0 > 0.0,
            "MFU should be positive for active GPU"
        );
        // 80% SM at full clock → MFU ≈ 80%.
        assert!(
            (snapshot.mfu.mfu_pct.0 - 80.0).abs() < 1.0,
            "expected ~80% MFU, got {:.1}%",
            snapshot.mfu.mfu_pct.0
        );
        // All 20 samples are ForwardBackward → 100% forward/backward.
        let bd = &snapshot.breakdown;
        assert!(
            bd.forward_backward_pct > 90.0,
            "expected mostly forward/backward, got {:.1}%",
            bd.forward_backward_pct
        );
        // Percentages sum to 100.
        let sum = bd.forward_backward_pct
            + bd.data_loader_pct
            + bd.cuda_sync_pct
            + bd.memory_alloc_pct
            + bd.optimizer_pct;
        assert!(
            (sum - 100.0).abs() < 0.1,
            "breakdown percents must sum to 100, got {sum:.2}"
        );
    }

    #[tokio::test]
    async fn session_tracks_peak_vram_across_samples() {
        let (snap_tx, _snap_rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (flume_tx, flume_rx) = flume::bounded(64);

        let session =
            MonitoringSession::new(9999, test_spec(), None, snap_tx, stop.clone(), true).start();
        let handle = tokio::spawn(session.run(flume_rx));

        // Send samples with increasing VRAM usage, then one lower.
        for vram in [1000u64, 5000, 9000, 7000, 3000] {
            flume_tx.send(make_sample(80.0, 20.0, vram)).unwrap();
        }
        drop(flume_tx);

        let done = handle.await.unwrap();
        let snapshot = done.final_snapshot().unwrap();

        assert_eq!(
            snapshot.peak_vram_mib,
            Mib(9000),
            "peak VRAM should be 9000 MiB"
        );
    }

    #[tokio::test]
    async fn session_computes_percentiles_after_15_samples() {
        let (snap_tx, _snap_rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (flume_tx, flume_rx) = flume::bounded(64);

        let session =
            MonitoringSession::new(9999, test_spec(), None, snap_tx, stop.clone(), true).start();
        let handle = tokio::spawn(session.run(flume_rx));

        for _ in 0..15 {
            flume_tx.send(make_sample(60.0, 20.0, 8192)).unwrap();
        }
        drop(flume_tx);

        let done = handle.await.unwrap();
        let snapshot = done.final_snapshot().unwrap();

        assert!(
            snapshot.mfu_percentiles.is_some(),
            "percentiles should be present after 15 samples"
        );
        let p = snapshot.mfu_percentiles.unwrap();
        assert!(p.p50 > 0.0, "p50 should be positive: {}", p.p50);
        assert!(p.p95 >= p.p50, "p95 must be >= p50");
    }

    #[tokio::test]
    async fn session_propagates_cost_impact() {
        let (snap_tx, _snap_rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (flume_tx, flume_rx) = flume::bounded(64);

        let session =
            MonitoringSession::new(9999, test_spec(), Some(2.50), snap_tx, stop.clone(), true)
                .start();
        let handle = tokio::spawn(session.run(flume_rx));

        flume_tx.send(make_sample(20.0, 20.0, 8192)).unwrap();
        drop(flume_tx);

        let done = handle.await.unwrap();
        let snapshot = done.final_snapshot().unwrap();

        let cost = snapshot.cost_impact.expect("cost_impact must be set");
        assert!((cost.cost_per_hour - 2.50).abs() < 0.001);
        assert!(cost.waste_per_hour >= 0.0);
    }

    #[tokio::test]
    async fn multi_gpu_divergence_flagged_above_threshold() {
        let (snap_tx, _snap_rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (flume_tx, flume_rx) = flume::bounded(64);

        let session =
            MonitoringSession::new(9999, test_spec(), None, snap_tx, stop.clone(), true).start();
        let handle = tokio::spawn(session.run(flume_rx));

        // GPU 0 at 80% SM, GPU 1 at 40% SM — spread is 40 ppt (> 20 threshold).
        for _ in 0..3 {
            flume_tx.send(make_sample_gpu(80.0, 20.0, 8192, 0)).unwrap();
            flume_tx.send(make_sample_gpu(40.0, 20.0, 8192, 1)).unwrap();
        }
        drop(flume_tx);

        let done = handle.await.unwrap();
        let snapshot = done.final_snapshot().unwrap();

        assert_eq!(
            snapshot.per_gpu.len(),
            2,
            "both GPUs must appear in per_gpu"
        );
        assert!(
            snapshot.mfu_divergent,
            "divergence flag should be set when spread > 20 ppt"
        );
        assert!(
            snapshot.gpu_mfu_divergence_ppt >= 20.0,
            "divergence_ppt should reflect the spread: {}",
            snapshot.gpu_mfu_divergence_ppt
        );
    }

    #[tokio::test]
    async fn single_gpu_no_divergence() {
        let (snap_tx, _snap_rx) = new_snapshot_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (flume_tx, flume_rx) = flume::bounded(64);

        let session =
            MonitoringSession::new(9999, test_spec(), None, snap_tx, stop.clone(), true).start();
        let handle = tokio::spawn(session.run(flume_rx));

        for _ in 0..5 {
            flume_tx.send(make_sample(70.0, 20.0, 8192)).unwrap();
        }
        drop(flume_tx);

        let done = handle.await.unwrap();
        let snapshot = done.final_snapshot().unwrap();

        assert!(
            !snapshot.mfu_divergent,
            "single GPU must never trigger divergence flag"
        );
        assert_eq!(snapshot.gpu_mfu_divergence_ppt, 0.0);
    }
}
