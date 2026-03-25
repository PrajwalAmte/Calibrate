use crate::metrics::breakdown::TimeBreakdown;
use crate::metrics::mfu::MfuEstimate;

/// The primary identified bottleneck for the current window.
///
/// Variants are ordered by detection priority.  Only the first matching
/// condition is returned so the user always gets one clear answer.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum Bottleneck {
    /// Data loader is producing batches slower than the GPU can consume them.
    DataLoader,
    /// Excessive CUDA synchronization (`.item()`, scalar logging, etc.).
    CudaSync,
    /// High memory allocation/deallocation inside the training loop.
    MemoryFragmentation,
    /// GPU is reducing clock speed due to thermal stress.
    ThermalThrottle,
    /// GPU clock is running significantly below boost for non-thermal reasons.
    ClockUnderspeed,
    /// MFU is in the acceptable range but not optimal (25–45%).
    BelowTargetMfu,
    /// No significant bottleneck identified; training is well-optimized.
    None,
}

/// Detects the primary bottleneck from measured data.
pub struct BottleneckDetector;

impl BottleneckDetector {
    pub fn detect(
        mfu: &MfuEstimate,
        breakdown: &TimeBreakdown,
        throttle_thermal: bool,
        _throttle_power: bool,
        sm_clock_pct: f32, // (current / max) * 100
    ) -> Bottleneck {
        // Thermal throttling takes priority — it's an immediate hardware concern.
        if throttle_thermal {
            return Bottleneck::ThermalThrottle;
        }

        // Data loader > 15% of step time.
        if breakdown.data_loader_pct > 15.0 {
            return Bottleneck::DataLoader;
        }

        // CUDA sync > 8% of step time.
        if breakdown.cuda_sync_pct > 8.0 {
            return Bottleneck::CudaSync;
        }

        // Memory allocation > 10% of step time.
        if breakdown.memory_alloc_pct > 10.0 {
            return Bottleneck::MemoryFragmentation;
        }

        // Clock running at < 85% of boost (power-limited or driver state).
        if sm_clock_pct < 85.0 && !throttle_thermal {
            return Bottleneck::ClockUnderspeed;
        }

        // MFU below target but no specific bottleneck dominant.
        if mfu.mfu_pct.0 < 45.0 {
            return Bottleneck::BelowTargetMfu;
        }

        Bottleneck::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::breakdown::TimeBreakdown;
    use crate::metrics::mfu::{Confidence, MfuEstimate};
    use crate::metrics::units::{Percent, Tflops};

    fn mfu(pct: f32) -> MfuEstimate {
        MfuEstimate {
            mfu_pct: Percent(pct),
            actual_tflops: Tflops(1.0),
            peak_tflops: Tflops(35.6),
            confidence: Confidence::High,
        }
    }

    fn breakdown(dl: f32, sync: f32, mem: f32) -> TimeBreakdown {
        let rest = (100.0 - dl - sync - mem).max(0.0);
        TimeBreakdown {
            forward_backward_pct: rest,
            data_loader_pct: dl,
            cuda_sync_pct: sync,
            memory_alloc_pct: mem,
            optimizer_pct: 0.0,
        }
    }

    #[test]
    fn data_loader_wins() {
        let b =
            BottleneckDetector::detect(&mfu(20.0), &breakdown(28.0, 3.0, 1.0), false, false, 95.0);
        assert_eq!(b, Bottleneck::DataLoader);
    }

    #[test]
    fn thermal_has_highest_priority() {
        let b =
            BottleneckDetector::detect(&mfu(20.0), &breakdown(28.0, 9.0, 12.0), true, false, 95.0);
        assert_eq!(b, Bottleneck::ThermalThrottle);
    }

    #[test]
    fn no_bottleneck_at_good_mfu() {
        let b =
            BottleneckDetector::detect(&mfu(55.0), &breakdown(5.0, 2.0, 1.0), false, false, 97.0);
        assert_eq!(b, Bottleneck::None);
    }

    #[test]
    fn cuda_sync_detected() {
        let b =
            BottleneckDetector::detect(&mfu(30.0), &breakdown(5.0, 12.0, 1.0), false, false, 95.0);
        assert_eq!(b, Bottleneck::CudaSync);
    }

    #[test]
    fn memory_fragmentation_detected() {
        let b =
            BottleneckDetector::detect(&mfu(30.0), &breakdown(5.0, 2.0, 15.0), false, false, 95.0);
        assert_eq!(b, Bottleneck::MemoryFragmentation);
    }

    #[test]
    fn clock_underspeed_detected() {
        // Low clock (80%), no other bottleneck triggers.
        let b =
            BottleneckDetector::detect(&mfu(30.0), &breakdown(5.0, 2.0, 1.0), false, false, 80.0);
        assert_eq!(b, Bottleneck::ClockUnderspeed);
    }

    #[test]
    fn below_target_mfu_when_no_specific_bottleneck() {
        let b =
            BottleneckDetector::detect(&mfu(30.0), &breakdown(5.0, 2.0, 1.0), false, false, 95.0);
        assert_eq!(b, Bottleneck::BelowTargetMfu);
    }

    #[test]
    fn thermal_overrides_data_loader() {
        // Even when data loader is high, thermal takes priority.
        let b =
            BottleneckDetector::detect(&mfu(10.0), &breakdown(50.0, 0.0, 0.0), true, false, 95.0);
        assert_eq!(b, Bottleneck::ThermalThrottle);
    }

    #[test]
    fn data_loader_overrides_cuda_sync() {
        // data_loader_pct=20 and cuda_sync_pct=12 — data loader should win.
        let b =
            BottleneckDetector::detect(&mfu(20.0), &breakdown(20.0, 12.0, 1.0), false, false, 95.0);
        assert_eq!(b, Bottleneck::DataLoader);
    }
}
