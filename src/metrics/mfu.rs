use crate::gpu_specs::GpuSpec;
use crate::metrics::units::{Percent, Tflops};
use crate::metrics::window::MetricsWindow;

/// The computed MFU estimate for the current window.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MfuEstimate {
    /// Model FLOP Utilization as a percentage (0–100).
    pub mfu_pct: Percent,
    /// Estimated actual TFLOPS being used for training math.
    pub actual_tflops: Tflops,
    /// GPU's theoretical peak (from spec DB).
    pub peak_tflops: Tflops,
    /// Confidence level — `Low` when fewer than [`MIN_RELIABLE_SAMPLES`] are available.
    pub confidence: Confidence,
}

/// How reliable the MFU estimate is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Confidence {
    /// Enough samples for reliable statistics (≥15 samples).
    High,
    /// Fewer than 15 samples — value is approximate.
    Low,
}

/// Calculates MFU from the metrics window and GPU specification.
///
/// ## Formula
/// ```text
/// clock_ratio  = current_sm_clock / boost_sm_clock
/// actual_pct   = sm_utilization * clock_ratio          (0–100)
/// mfu_pct      = actual_pct                            (same as above, intuition below)
/// actual_tflops = (actual_pct / 100) * peak_tflops
/// ```
///
/// The approximation: SM utilization tells us "what fraction of SMs are active",
/// the clock ratio tells us "at what fraction of peak frequency".  Their product
/// is the fraction of peak compute being delivered to the kernel.
///
/// This is an *external* approximation — the ground-truth MFU would require
/// counting FLOPs from inside the model.  We document this limitation clearly.
pub struct MfuCalculator<'a> {
    spec: &'a GpuSpec,
}

impl<'a> MfuCalculator<'a> {
    pub fn new(spec: &'a GpuSpec) -> Self {
        Self { spec }
    }

    pub fn compute(&self, window: &MetricsWindow) -> MfuEstimate {
        let confidence = if window.is_reliable() {
            Confidence::High
        } else {
            Confidence::Low
        };

        if window.is_empty() {
            return MfuEstimate {
                mfu_pct: Percent(0.0),
                actual_tflops: Tflops(0.0),
                peak_tflops: Tflops(self.spec.bf16_tflops),
                confidence: Confidence::Low,
            };
        }

        // Weighted average over the window; each sample contributes equally.
        let (sm_sum, clock_ratio_sum, count) = window.iter().fold(
            (0.0_f64, 0.0_f64, 0_usize),
            |(sm, cr, n), s| {
                let clock_ratio = if s.sm_clock_max_mhz.0 > 0 {
                    s.sm_clock_mhz.0 as f64 / s.sm_clock_max_mhz.0 as f64
                } else {
                    1.0
                };
                (sm + s.sm_utilization.0 as f64, cr + clock_ratio, n + 1)
            },
        );

        let avg_sm = sm_sum / count as f64;
        let avg_clock_ratio = clock_ratio_sum / count as f64;

        let actual_pct = (avg_sm / 100.0) * avg_clock_ratio * 100.0;
        let actual_tflops = (actual_pct / 100.0) * self.spec.bf16_tflops;

        MfuEstimate {
            mfu_pct: Percent::clamped(actual_pct as f32),
            actual_tflops: Tflops(actual_tflops),
            peak_tflops: Tflops(self.spec.bf16_tflops),
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::RawSample;
    use crate::gpu_specs::GpuSpec;
    use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};
    use crate::metrics::window::MetricsWindow;

    fn spec() -> GpuSpec {
        GpuSpec {
            name: "RTX 3090".to_string(),
            bf16_tflops: 35.6,
            fp32_tflops: 35.6,
            vram_gib: 24,
            boost_clock_mhz: 1695,
        }
    }

    fn sample_with(sm: f32, clock: u32, max_clock: u32) -> RawSample {
        RawSample {
            timestamp_ms: 0,
            gpu_index: 0,
            sm_utilization: Percent(sm),
            sm_clock_mhz: Mhz(clock),
            sm_clock_max_mhz: Mhz(max_clock),
            vram_used_mib: Mib(8192),
            vram_total_mib: Mib(24576),
            mem_utilization: Percent(40.0),
            temperature: Celsius(65.0),
            power_draw: Watts(250.0),
            power_limit: Watts(350.0),
            throttle_thermal: false,
            throttle_power: false,
            throttle_hw_slowdown: false,
            cpu_utilization: Percent(30.0),
        }
    }

    #[test]
    fn mfu_at_full_clock_and_sm() {
        let mut window = MetricsWindow::new(150);
        for _ in 0..20 {
            window.push(sample_with(100.0, 2000, 2000));
        }
        let s = spec();
        let calc = MfuCalculator::new(&s);
        let est = calc.compute(&window);
        assert!((est.mfu_pct.0 - 100.0).abs() < 1.0);
    }

    #[test]
    fn mfu_at_20_pct_sm() {
        let mut window = MetricsWindow::new(150);
        for _ in 0..20 {
            window.push(sample_with(20.0, 2000, 2000));
        }
        let s = spec();
        let calc = MfuCalculator::new(&s);
        let est = calc.compute(&window);
        assert!((est.mfu_pct.0 - 20.0).abs() < 1.0);
    }

    #[test]
    fn low_confidence_below_threshold() {
        let mut window = MetricsWindow::new(150);
        for _ in 0..5 {
            window.push(sample_with(50.0, 1800, 2000));
        }
        let s = spec();
        let calc = MfuCalculator::new(&s);
        let est = calc.compute(&window);
        assert_eq!(est.confidence, Confidence::Low);
    }
}
