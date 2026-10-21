//! Picks the single most important problem from the measured data.

use serde::Serialize;

use crate::metrics::breakdown::{BreakdownSource, TimeBreakdown};
use crate::metrics::mfu::MfuEstimate;

/// The primary identified bottleneck, in detection-priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    /// Not enough data yet (first seconds of a session).
    WarmingUp,
    /// No GPU metrics are available (CPU-only fallback).
    NoGpuMetrics,
    /// The target process tree is not using any GPU.
    NoGpuActivity,
    /// Clocks reduced because the GPU is too hot.
    ThermalThrottle,
    /// Clocks reduced by the board power limit.
    PowerCapped,
    /// Clocks reduced by a hardware slowdown signal (power brake, PSU, ...).
    HardwareSlowdown,
    /// GPUs of one job are unevenly loaded (straggler rank / device).
    GpuImbalance,
    /// The GPU waits on the input pipeline.
    DataLoader,
    /// GPU and CPU both idle: host sync, logging, checkpointing, collectives.
    SyncStall,
    /// Memory bandwidth bound with little compute.
    MemoryBound,
    /// The optimizer step takes a large share of each step.
    OptimizerHeavy,
    /// Clock well below boost with no throttle reason reported.
    ClockUnderspeed,
    /// Step times vary widely.
    ErraticSteps,
    /// MFU below target without a single dominant cause.
    BelowTargetMfu,
    /// Only an MFU upper bound is available and it is above target, so real
    /// efficiency cannot be judged.
    Unverified,
    /// No significant bottleneck.
    None,
}

/// Everything the detector looks at.
#[derive(Debug, Clone)]
pub struct DetectInput<'a> {
    pub mfu: &'a MfuEstimate,
    pub breakdown: &'a TimeBreakdown,
    pub has_gpu_backend: bool,
    pub gpus_in_use: usize,
    pub thermal_throttle: bool,
    pub power_cap: bool,
    pub hw_slowdown: bool,
    /// Mean SM clock relative to the spec's boost clock (percent), if known.
    pub clock_pct_of_boost: Option<f32>,
    /// Mean GPU busy fraction (percent).
    pub busy_pct: Option<f32>,
    pub steps_erratic: bool,
    /// Spread of mean busy % across GPUs (multi-GPU jobs).
    pub gpu_divergence_ppt: Option<f32>,
    pub target_mfu: f32,
    pub samples: usize,
}

pub struct BottleneckDetector;

impl BottleneckDetector {
    pub fn detect(i: &DetectInput) -> Bottleneck {
        if !i.has_gpu_backend {
            return Bottleneck::NoGpuMetrics;
        }
        if i.samples < 3 {
            return Bottleneck::WarmingUp;
        }
        if i.gpus_in_use == 0 {
            return Bottleneck::NoGpuActivity;
        }
        if i.thermal_throttle {
            return Bottleneck::ThermalThrottle;
        }
        let clock_low = i.clock_pct_of_boost.is_some_and(|c| c < 90.0);
        if i.power_cap && clock_low {
            return Bottleneck::PowerCapped;
        }
        if i.hw_slowdown {
            return Bottleneck::HardwareSlowdown;
        }
        if i.gpu_divergence_ppt.is_some_and(|d| d > 20.0) {
            return Bottleneck::GpuImbalance;
        }

        let b = i.breakdown;
        if b.source != BreakdownSource::Unavailable {
            if b.data_wait_pct > 15.0 {
                return Bottleneck::DataLoader;
            }
            if b.sync_stall_pct > 10.0 {
                return Bottleneck::SyncStall;
            }
            if b.memory_bound_pct > 15.0 {
                return Bottleneck::MemoryBound;
            }
            if b.optimizer_pct > 20.0 {
                return Bottleneck::OptimizerHeavy;
            }
        }

        let busy = i.busy_pct.unwrap_or(0.0);
        if busy > 50.0
            && i.clock_pct_of_boost.is_some_and(|c| c < 85.0)
            && !i.power_cap
            && !i.hw_slowdown
        {
            return Bottleneck::ClockUnderspeed;
        }
        if i.steps_erratic {
            return Bottleneck::ErraticSteps;
        }

        match i.mfu.mfu_pct {
            Some(m) if m < i.target_mfu => Bottleneck::BelowTargetMfu,
            Some(_) if i.mfu.upper_bound => Bottleneck::Unverified,
            Some(_) => Bottleneck::None,
            None => Bottleneck::Unverified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::mfu::{Confidence, MfuSource};

    fn mfu(pct: f32, upper: bool) -> MfuEstimate {
        MfuEstimate {
            mfu_pct: Some(pct),
            source: if upper {
                MfuSource::KernelBusy
            } else {
                MfuSource::TensorActive
            },
            upper_bound: upper,
            achieved_tflops: None,
            peak_tflops: None,
            confidence: Confidence::High,
        }
    }

    fn bd(data: f32, sync: f32, mem: f32) -> TimeBreakdown {
        TimeBreakdown {
            source: BreakdownSource::Heuristic,
            compute_pct: 100.0 - data - sync - mem,
            data_wait_pct: data,
            sync_stall_pct: sync,
            memory_bound_pct: mem,
            optimizer_pct: 0.0,
            other_pct: 0.0,
        }
    }

    fn input<'a>(m: &'a MfuEstimate, b: &'a TimeBreakdown) -> DetectInput<'a> {
        DetectInput {
            mfu: m,
            breakdown: b,
            has_gpu_backend: true,
            gpus_in_use: 1,
            thermal_throttle: false,
            power_cap: false,
            hw_slowdown: false,
            clock_pct_of_boost: Some(100.0),
            busy_pct: Some(90.0),
            steps_erratic: false,
            gpu_divergence_ppt: None,
            target_mfu: 45.0,
            samples: 30,
        }
    }

    #[test]
    fn straggler_gpu_is_reported() {
        let (m, b) = (mfu(42.0, false), bd(0.0, 0.0, 0.0));
        let mut i = input(&m, &b);
        i.gpu_divergence_ppt = Some(35.0);
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::GpuImbalance);
        i.gpu_divergence_ppt = Some(5.0);
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::BelowTargetMfu);
    }

    #[test]
    fn missing_backend_is_reported_honestly() {
        let (m, b) = (MfuEstimate::unavailable(), TimeBreakdown::unavailable());
        let mut i = input(&m, &b);
        i.has_gpu_backend = false;
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::NoGpuMetrics);
    }

    #[test]
    fn idle_process_is_no_activity_not_cuda_sync() {
        let (m, b) = (MfuEstimate::unavailable(), TimeBreakdown::unavailable());
        let mut i = input(&m, &b);
        i.gpus_in_use = 0;
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::NoGpuActivity);
    }

    #[test]
    fn thermal_beats_everything() {
        let (m, b) = (mfu(20.0, false), bd(40.0, 0.0, 0.0));
        let mut i = input(&m, &b);
        i.thermal_throttle = true;
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::ThermalThrottle);
    }

    #[test]
    fn power_cap_needs_reduced_clock() {
        let (m, b) = (mfu(60.0, false), bd(0.0, 0.0, 0.0));
        let mut i = input(&m, &b);
        i.power_cap = true;
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::None);
        i.clock_pct_of_boost = Some(80.0);
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::PowerCapped);
    }

    #[test]
    fn breakdown_driven_bottlenecks() {
        let m = mfu(30.0, false);
        let b = bd(20.0, 0.0, 0.0);
        assert_eq!(
            BottleneckDetector::detect(&input(&m, &b)),
            Bottleneck::DataLoader
        );
        let b = bd(0.0, 12.0, 0.0);
        assert_eq!(
            BottleneckDetector::detect(&input(&m, &b)),
            Bottleneck::SyncStall
        );
        let b = bd(0.0, 0.0, 20.0);
        assert_eq!(
            BottleneckDetector::detect(&input(&m, &b)),
            Bottleneck::MemoryBound
        );
    }

    #[test]
    fn clock_underspeed_only_when_busy_and_unexplained() {
        let (m, b) = (mfu(60.0, false), bd(0.0, 0.0, 0.0));
        let mut i = input(&m, &b);
        i.clock_pct_of_boost = Some(70.0);
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::ClockUnderspeed);
        i.busy_pct = Some(20.0);
        assert_ne!(BottleneckDetector::detect(&i), Bottleneck::ClockUnderspeed);
    }

    #[test]
    fn upper_bound_above_target_is_unverified_not_well_optimized() {
        let (m, b) = (mfu(97.0, true), bd(0.0, 0.0, 0.0));
        assert_eq!(
            BottleneckDetector::detect(&input(&m, &b)),
            Bottleneck::Unverified
        );
        let m = mfu(30.0, true);
        assert_eq!(
            BottleneckDetector::detect(&input(&m, &b)),
            Bottleneck::BelowTargetMfu,
            "an upper bound below target proves the job is below target"
        );
    }

    #[test]
    fn healthy_measured_job_is_none() {
        let (m, b) = (mfu(55.0, false), bd(2.0, 1.0, 0.0));
        assert_eq!(BottleneckDetector::detect(&input(&m, &b)), Bottleneck::None);
    }

    #[test]
    fn custom_target_is_respected() {
        let (m, b) = (mfu(50.0, false), bd(0.0, 0.0, 0.0));
        let mut i = input(&m, &b);
        i.target_mfu = 60.0;
        assert_eq!(BottleneckDetector::detect(&i), Bottleneck::BelowTargetMfu);
    }
}
