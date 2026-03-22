use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::analysis::bottleneck::BottleneckDetector;
use crate::analysis::recommendations::RecommendationEngine;
use crate::collectors::RawSample;
use crate::gpu_specs::GpuSpec;
use crate::metrics::breakdown::TimeBreakdownInferrer;
use crate::metrics::mfu::MfuCalculator;
use crate::metrics::window::{MetricsWindow, MAX_WINDOW_SIZE};
use crate::session::state::{CostImpact, GpuSnapshot, SessionSnapshot, SharedState};

/// Typestate markers for `MonitoringSession`.
pub mod state {
    pub struct Initializing;
    pub struct Sampling;
    pub struct Done;
}

/// Drives the sampling loop using a typestate machine.
///
/// ```text
/// MonitoringSession<Initializing>
///     .start()
/// → MonitoringSession<Sampling>   (runs the loop, updating SharedState each interval)
///     .finish()
/// → MonitoringSession<Done>       (cleanup, returns final summary stats)
/// ```
pub struct MonitoringSession<S> {
    pid: u32,
    gpu_spec: GpuSpec,
    cost_per_hour: Option<f64>,
    state: SharedState,
    stop: Arc<AtomicBool>,
    _phase: std::marker::PhantomData<S>,
}

impl MonitoringSession<state::Initializing> {
    pub fn new(
        pid: u32,
        gpu_spec: GpuSpec,
        cost_per_hour: Option<f64>,
        state: SharedState,
        stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            pid,
            gpu_spec,
            cost_per_hour,
            state,
            stop,
            _phase: std::marker::PhantomData,
        }
    }

    /// Transition into the sampling phase.
    pub fn start(self) -> MonitoringSession<state::Sampling> {
        info!("MonitoringSession starting for PID {}", self.pid);
        MonitoringSession {
            pid: self.pid,
            gpu_spec: self.gpu_spec,
            cost_per_hour: self.cost_per_hour,
            state: self.state,
            stop: self.stop,
            _phase: std::marker::PhantomData,
        }
    }
}

impl MonitoringSession<state::Sampling> {
    /// Run the aggregation loop, consuming samples from `rx`.
    ///
    /// This is `async` so it participates in the tokio runtime.  NVML sample
    /// production happens on a separate `std::thread`; this loop only does
    /// analytics and state updates.
    pub async fn run(self, rx: flume::Receiver<RawSample>) -> MonitoringSession<state::Done> {
        let mut window = MetricsWindow::new(MAX_WINDOW_SIZE);
        let started_at = Instant::now();
        let mut steps_observed: u64 = 0;
        let gpu_name = self.gpu_spec.name.clone();

        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            // Use a short timeout so we can check the stop flag regularly.
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(mut sample) => {
                    // `cpu_utilization` is patched in by the proc collector
                    // if it ran in the same tick; the nvml collector sets it
                    // to 0.0 as a placeholder.
                    window.push(sample.clone());
                    steps_observed += 1;

                    let mfu_calc = MfuCalculator::new(&self.gpu_spec);
                    let mfu = mfu_calc.compute(&window);
                    let breakdown = TimeBreakdownInferrer::infer(&window);

                    // Average clock ratio across the window for underspeed check.
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

                    let snapshot = SessionSnapshot {
                        elapsed: started_at.elapsed(),
                        gpu_name: gpu_name.clone(),
                        mfu,
                        breakdown,
                        bottleneck,
                        recommendation,
                        temperature: sample.temperature,
                        power_draw: sample.power_draw,
                        power_limit: sample.power_limit,
                        throttle_thermal: sample.throttle_thermal,
                        vram_used_mib: sample.vram_used_mib,
                        vram_total_mib: sample.vram_total_mib,
                        vram_utilization: crate::metrics::units::Percent::clamped(
                            sample.vram_used_mib.0 as f32 / sample.vram_total_mib.0.max(1) as f32
                                * 100.0,
                        ),
                        cost_impact,
                        per_gpu: vec![GpuSnapshot {
                            gpu_index: sample.gpu_index,
                            sm_utilization: sample.sm_utilization,
                            vram_used_mib: sample.vram_used_mib,
                            temperature: sample.temperature,
                        }],
                        steps_observed,
                    };

                    *self.state.write() = Some(snapshot);
                }
                Err(flume::RecvTimeoutError::Timeout) => {
                    // No sample yet — keep waiting.
                }
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
            state: self.state,
            stop: self.stop,
            _phase: std::marker::PhantomData,
        }
    }
}

impl MonitoringSession<state::Done> {
    /// Return the final snapshot for use in the summary report.
    pub fn final_snapshot(&self) -> Option<SessionSnapshot> {
        self.state.read().clone()
    }
}
