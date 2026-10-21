//! MFU (Model FLOP Utilization) estimation with explicit provenance.
//!
//! Three sources, best first:
//!
//! * **Instrumented** — the training code reports FLOPs per step through the
//!   calibrate Python hook. `MFU = achieved FLOP/s ÷ peak FLOP/s`. This is MFU
//!   in the PaLM-paper sense, provided the declared FLOPs are model FLOPs.
//! * **Tensor-active** — GPM/DCGM tensor-pipe activity × clock ratio. A
//!   hardware-level estimate (closer to HFU: it counts recomputation and
//!   cannot see FLOPs wasted on padding), typically within a few points of
//!   MFU for dense BF16 training.
//! * **Kernel-busy** — NVML `utilization.gpu` × clock ratio. This only says
//!   *some* kernel was running, so it is an **upper bound**: a GPU can be
//!   100 % "busy" at 30 % MFU.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::collectors::GpuSample;
use crate::gpu_specs::ResolvedSpec;
use crate::metrics::window::{TickWindow, MIN_RELIABLE_SAMPLES};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MfuSource {
    Instrumented,
    TensorActive,
    KernelBusy,
    Unavailable,
}

impl MfuSource {
    /// Short label for displays.
    pub fn label(self) -> &'static str {
        match self {
            Self::Instrumented => "MFU",
            Self::TensorActive => "MFU (est., tensor-core activity)",
            Self::KernelBusy => "MFU upper bound (GPU busy time)",
            Self::Unavailable => "MFU",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,
    Low,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MfuEstimate {
    /// 0–100, `None` when it cannot be computed (no GPU data / unknown peak).
    pub mfu_pct: Option<f32>,
    pub source: MfuSource,
    /// True when the real MFU is at most this value (kernel-busy source).
    pub upper_bound: bool,
    pub achieved_tflops: Option<f64>,
    /// Combined peak of the GPUs in use; `None` if any GPU is not in the spec DB.
    pub peak_tflops: Option<f64>,
    pub confidence: Confidence,
}

impl MfuEstimate {
    pub fn unavailable() -> Self {
        Self {
            mfu_pct: None,
            source: MfuSource::Unavailable,
            upper_bound: false,
            achieved_tflops: None,
            peak_tflops: None,
            confidence: Confidence::Low,
        }
    }
}

/// Clock relative to the clock at which peak TFLOPS is specified.
///
/// GPUs routinely run above the advertised boost clock, so the ratio may
/// exceed 1; it is capped to keep a misreported clock from inflating results.
pub fn clock_ratio(s: &GpuSample, spec: Option<&ResolvedSpec>) -> f64 {
    let Some(clock) = s.sm_clock_mhz.filter(|&c| c > 0) else {
        return 1.0;
    };
    let reference = spec
        .filter(|r| r.known() && r.spec.boost_clock_mhz > 0)
        .map(|r| r.spec.boost_clock_mhz)
        .or(s.sm_clock_max_mhz.filter(|&m| m > 0));
    match reference {
        Some(r) => (clock as f64 / r as f64).clamp(0.0, 1.25),
        None => 1.0,
    }
}

/// Mean of a per-GPU signal over the window, keyed by GPU index.
pub fn per_gpu_mean(
    window: &TickWindow,
    signal: impl Fn(&GpuSample) -> Option<f64>,
) -> BTreeMap<u32, f64> {
    let mut acc: BTreeMap<u32, (f64, usize)> = BTreeMap::new();
    for t in window.iter() {
        for g in &t.gpus {
            if let Some(v) = signal(g) {
                let e = acc.entry(g.index).or_insert((0.0, 0));
                e.0 += v;
                e.1 += 1;
            }
        }
    }
    acc.into_iter()
        .map(|(k, (sum, n))| (k, sum / n as f64))
        .collect()
}

/// Peak TFLOPS per GPU index seen in the window (`None` = unknown GPU).
fn peaks(window: &TickWindow, specs: &HashMap<String, ResolvedSpec>) -> BTreeMap<u32, Option<f64>> {
    let mut out = BTreeMap::new();
    for t in window.iter() {
        for g in &t.gpus {
            let peak = specs
                .get(&g.name)
                .filter(|r| r.known())
                .map(|r| r.spec.bf16_tflops);
            out.insert(g.index, peak);
        }
    }
    out
}

pub fn compute(window: &TickWindow, specs: &HashMap<String, ResolvedSpec>) -> MfuEstimate {
    let peaks = peaks(window, specs);
    let total_peak: Option<f64> = if peaks.is_empty() {
        None
    } else {
        peaks.values().copied().sum::<Option<f64>>()
    };

    if let Some(est) = instrumented(window, total_peak) {
        return est;
    }

    let reliable = if window.len() >= MIN_RELIABLE_SAMPLES {
        Confidence::High
    } else {
        Confidence::Low
    };
    let spec_of = |g: &GpuSample| specs.get(&g.name);

    // Use tensor activity only if every GPU sample in the window has it, so
    // the figure never mixes two different kinds of measurement.
    let all_samples: Vec<&GpuSample> = window.iter().flat_map(|t| t.gpus.iter()).collect();
    if all_samples.is_empty() {
        return MfuEstimate::unavailable();
    }
    let has_tensor = all_samples
        .iter()
        .all(|g| g.prof.and_then(|p| p.tensor_active_pct).is_some());
    let has_util = all_samples.iter().all(|g| g.util_pct.is_some());

    let (source, per_gpu) = if has_tensor {
        (
            MfuSource::TensorActive,
            per_gpu_mean(window, |g| {
                g.prof
                    .and_then(|p| p.tensor_active_pct)
                    .map(|v| v as f64 * clock_ratio(g, spec_of(g)))
            }),
        )
    } else if has_util {
        (
            MfuSource::KernelBusy,
            per_gpu_mean(window, |g| {
                g.util_pct.map(|v| v as f64 * clock_ratio(g, spec_of(g)))
            }),
        )
    } else {
        return MfuEstimate::unavailable();
    };

    // Peak-weighted average across GPUs; plain mean if any peak is unknown.
    let (pct, achieved) = match total_peak {
        Some(tp) if tp > 0.0 => {
            let achieved: f64 = per_gpu
                .iter()
                .map(|(i, v)| v / 100.0 * peaks.get(i).copied().flatten().unwrap_or(0.0))
                .sum();
            (achieved / tp * 100.0, Some(achieved))
        }
        _ => (
            per_gpu.values().sum::<f64>() / per_gpu.len().max(1) as f64,
            None,
        ),
    };

    MfuEstimate {
        mfu_pct: Some((pct as f32).clamp(0.0, 100.0)),
        source,
        upper_bound: source == MfuSource::KernelBusy,
        achieved_tflops: achieved,
        peak_tflops: total_peak,
        confidence: reliable,
    }
}

/// MFU from hook-reported FLOPs: Σ over processes of (FLOPs ÷ step time).
fn instrumented(window: &TickWindow, total_peak: Option<f64>) -> Option<MfuEstimate> {
    let mut by_pid: HashMap<u32, (f64, f64, usize)> = HashMap::new();
    for s in window.steps() {
        if let Some(f) = s.flops.filter(|f| f.is_finite() && *f > 0.0) {
            let e = by_pid.entry(s.pid).or_insert((0.0, 0.0, 0));
            e.0 += f;
            e.1 += s.step_s;
            e.2 += 1;
        }
    }
    let steps: usize = by_pid.values().map(|v| v.2).sum();
    if steps < 3 {
        return None;
    }
    let achieved_tflops: f64 = by_pid
        .values()
        .filter(|v| v.1 > 0.0)
        .map(|(f, t, _)| f / t / 1e12)
        .sum();
    let mfu_pct = total_peak
        .filter(|p| *p > 0.0)
        .map(|p| ((achieved_tflops / p * 100.0) as f32).clamp(0.0, 100.0));
    Some(MfuEstimate {
        mfu_pct,
        source: MfuSource::Instrumented,
        upper_bound: false,
        achieved_tflops: Some(achieved_tflops),
        peak_tflops: total_peak,
        confidence: if steps >= MIN_RELIABLE_SAMPLES {
            Confidence::High
        } else {
            Confidence::Low
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::testing::{gpu, tick};
    use crate::collectors::{ProfMetrics, ProfSource, StepEvent};
    use crate::gpu_specs::SpecDb;

    fn specs_for(name: &str) -> HashMap<String, ResolvedSpec> {
        HashMap::from([(name.to_string(), SpecDb::builtin().resolve(name))])
    }

    fn named(index: u32, name: &str, util: f32, clock: u32) -> GpuSample {
        let mut g = gpu(index, util);
        g.name = name.into();
        g.sm_clock_mhz = Some(clock);
        g.sm_clock_max_mhz = Some(2100);
        g
    }

    fn window_of(samples: Vec<Vec<GpuSample>>) -> TickWindow {
        let mut w = TickWindow::new();
        for (i, gpus) in samples.into_iter().enumerate() {
            w.push(tick(i as u64 * 1000, gpus, Some(50.0)));
        }
        w
    }

    #[test]
    fn kernel_busy_is_flagged_as_upper_bound() {
        let name = "NVIDIA GeForce RTX 3090";
        let w = window_of((0..20).map(|_| vec![named(0, name, 95.0, 1695)]).collect());
        let est = compute(&w, &specs_for(name));
        assert_eq!(est.source, MfuSource::KernelBusy);
        assert!(est.upper_bound);
        assert!((est.mfu_pct.unwrap() - 95.0).abs() < 0.5);
        assert_eq!(est.confidence, Confidence::High);
        assert!((est.peak_tflops.unwrap() - 71.2).abs() < 0.1);
    }

    #[test]
    fn clock_ratio_uses_spec_boost_not_max_clock() {
        // At exactly the advertised boost clock, no derating should apply even
        // though max_clock (2100) is higher.
        let name = "NVIDIA GeForce RTX 3090";
        let s = named(0, name, 100.0, 1695);
        let spec = SpecDb::builtin().resolve(name);
        assert!((clock_ratio(&s, Some(&spec)) - 1.0).abs() < 1e-9);
        // Unknown spec → fall back to max clock.
        assert!((clock_ratio(&s, None) - 1695.0 / 2100.0).abs() < 1e-9);
    }

    #[test]
    fn tensor_activity_preferred_over_util() {
        let name = "NVIDIA H100 80GB HBM3";
        let mk = || {
            let mut g = named(0, name, 99.0, 1830);
            g.prof = Some(ProfMetrics {
                source: ProfSource::Gpm,
                sm_active_pct: Some(90.0),
                sm_occupancy_pct: Some(40.0),
                tensor_active_pct: Some(38.0),
                dram_active_pct: Some(20.0),
            });
            vec![g]
        };
        let w = window_of((0..5).map(|_| mk()).collect());
        let est = compute(&w, &specs_for(name));
        assert_eq!(est.source, MfuSource::TensorActive);
        assert!(!est.upper_bound);
        assert!((est.mfu_pct.unwrap() - 38.0).abs() < 0.5);
        assert!((est.achieved_tflops.unwrap() - 0.38 * 989.0).abs() < 2.0);
    }

    #[test]
    fn instrumented_beats_hardware_counters() {
        let name = "NVIDIA H100 80GB HBM3";
        let mut w = TickWindow::new();
        for i in 0..10u64 {
            let mut t = tick(i * 1000, vec![named(0, name, 99.0, 1830)], Some(80.0));
            t.steps.push(StepEvent {
                pid: 1,
                rank: Some(0),
                step: i,
                t_end_ms: i * 1000,
                step_s: 1.0,
                data_s: Some(0.1),
                compute_s: Some(0.8),
                optimizer_s: Some(0.1),
                tokens: Some(1000),
                samples: None,
                // 400 TFLOP per 1 s step on a 989 TFLOPS GPU → 40.4 %.
                flops: Some(400e12),
            });
            w.push(t);
        }
        let est = compute(&w, &specs_for(name));
        assert_eq!(est.source, MfuSource::Instrumented);
        assert!(
            (est.mfu_pct.unwrap() - 40.4).abs() < 0.2,
            "{:?}",
            est.mfu_pct
        );
    }

    #[test]
    fn unknown_gpu_has_no_peak_but_still_reports_busy_fraction() {
        let name = "Mystery Accelerator";
        let w = window_of(vec![vec![named(0, name, 60.0, 1000)]]);
        let est = compute(&w, &specs_for(name));
        assert!(est.peak_tflops.is_none());
        assert!(est.achieved_tflops.is_none());
        assert!(est.mfu_pct.is_some());
        assert_eq!(est.confidence, Confidence::Low);
    }

    #[test]
    fn no_gpu_data_is_unavailable_not_zero() {
        let w = window_of(vec![vec![], vec![]]);
        let est = compute(&w, &HashMap::new());
        assert_eq!(est.source, MfuSource::Unavailable);
        assert_eq!(est.mfu_pct, None);
    }

    #[test]
    fn multi_gpu_is_peak_weighted() {
        let name = "NVIDIA GeForce RTX 4090";
        let w = window_of(
            (0..4)
                .map(|_| vec![named(0, name, 80.0, 2520), named(1, name, 40.0, 2520)])
                .collect(),
        );
        let est = compute(&w, &specs_for(name));
        assert!((est.mfu_pct.unwrap() - 60.0).abs() < 0.5);
        assert!((est.peak_tflops.unwrap() - 2.0 * 165.2).abs() < 0.1);
    }
}
