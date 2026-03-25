use std::collections::VecDeque;
use std::time::Duration;

use crate::collectors::RawSample;

/// Maximum number of samples retained in the window (5 minutes at 2-second intervals).
pub const MAX_WINDOW_SIZE: usize = 150;

/// Minimum number of samples needed for reliable statistics.
pub const MIN_RELIABLE_SAMPLES: usize = 15;

/// A sliding-window circular buffer of [`RawSample`] values.
///
/// The analytics pipeline (MFU, breakdown, bottleneck) operates over this
/// window rather than the raw stream, providing smoothing and a bounded
/// memory footprint.
#[derive(Debug, Default)]
pub struct MetricsWindow {
    samples: VecDeque<RawSample>,
    max_size: usize,
}

impl MetricsWindow {
    pub fn new(max_size: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a sample, evicting the oldest if the window is full.
    pub fn push(&mut self, sample: RawSample) {
        if self.samples.len() == self.max_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration spanned by samples in the window.
    #[allow(dead_code)]
    pub fn duration(&self) -> Duration {
        match (self.samples.front(), self.samples.back()) {
            (Some(first), Some(last)) => {
                let ms = last.timestamp_ms.saturating_sub(first.timestamp_ms);
                Duration::from_millis(ms)
            }
            _ => Duration::ZERO,
        }
    }

    /// Duration in seconds, as an `f64`.  Convenience wrapper over [`duration`].
    #[allow(dead_code)]
    pub fn duration_secs(&self) -> f64 {
        self.duration().as_secs_f64()
    }

    /// Returns `true` when there are enough samples for reliable statistics.
    pub fn is_reliable(&self) -> bool {
        self.samples.len() >= MIN_RELIABLE_SAMPLES
    }

    /// Iterate over all samples in chronological order.
    pub fn iter(&self) -> impl Iterator<Item = &RawSample> {
        self.samples.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};

    fn make_sample(timestamp_ms: u64) -> RawSample {
        RawSample {
            timestamp_ms,
            gpu_index: 0,
            sm_utilization: Percent(80.0),
            sm_clock_mhz: Mhz(1800),
            sm_clock_max_mhz: Mhz(2000),
            vram_used_mib: Mib(8192),
            vram_total_mib: Mib(24576),
            mem_utilization: Percent(50.0),
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
    fn window_evicts_oldest() {
        let mut w = MetricsWindow::new(3);
        w.push(make_sample(1000));
        w.push(make_sample(2000));
        w.push(make_sample(3000));
        w.push(make_sample(4000)); // should evict sample at 1000
        assert_eq!(w.len(), 3);
        assert_eq!(w.iter().next().unwrap().timestamp_ms, 2000);
    }

    #[test]
    fn duration_calculation() {
        let mut w = MetricsWindow::new(150);
        w.push(make_sample(1_000));
        w.push(make_sample(3_000));
        assert_eq!(w.duration().as_millis(), 2000);
    }

    #[test]
    fn duration_secs_matches_duration() {
        let mut w = MetricsWindow::new(150);
        w.push(make_sample(0));
        w.push(make_sample(5_000));
        assert!((w.duration_secs() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn duration_secs_empty_is_zero() {
        let w = MetricsWindow::new(10);
        assert_eq!(w.duration_secs(), 0.0);
    }

    #[test]
    fn is_reliable_at_threshold() {
        let mut w = MetricsWindow::new(150);
        for i in 0..14 {
            w.push(make_sample(i as u64 * 2000));
        }
        assert!(!w.is_reliable(), "14 samples should not be reliable");
        w.push(make_sample(14 * 2000));
        assert!(w.is_reliable(), "15 samples should be reliable");
    }

    #[test]
    fn single_sample_not_reliable() {
        let mut w = MetricsWindow::new(150);
        w.push(make_sample(0));
        assert!(!w.is_reliable());
    }
}
