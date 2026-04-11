use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};

/// Summary statistics for a single (runtime, batch_size) benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchStats {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Actual sustained throughput (requests per second) measured via a
    /// separate fixed-duration load window — not simply 1 / mean_latency.
    pub throughput_rps: f64,
    pub stddev_ms: f64,
    /// Coefficient of variation (stddev / mean). Values above 0.20 indicate
    /// the system was under variable load and results may be unreliable.
    pub cv: f64,
    pub sample_count: u64,
}

/// Accumulates per-iteration latency samples and computes summary statistics.
///
/// Uses an HDR histogram for accurate percentile queries and a parallel raw
/// sample list for stddev / CV computation (HDR histograms quantise values
/// and cannot produce an exact stddev).
pub struct StatAccumulator {
    /// HDR histogram storing values in microseconds.
    /// Tracks latencies up to 1 hour (3.6 × 10⁹ µs) with 3 significant figures.
    histogram: Histogram<u64>,
    /// Raw samples in microseconds for exact stddev computation.
    samples: Vec<u64>,
}

impl StatAccumulator {
    pub fn new() -> Self {
        let histogram = Histogram::<u64>::new_with_bounds(1, 3_600_000_000, 3)
            .expect("histogram bounds are valid");
        Self {
            histogram,
            samples: Vec::new(),
        }
    }

    /// Record a single latency measurement given in microseconds.
    pub fn record_micros(&mut self, micros: u64) {
        let v = micros.max(1); // histogram requires values ≥ 1
        if self.histogram.record(v).is_err() {
            // Value exceeds the 1-hour ceiling — clamp rather than panic.
            let _ = self.histogram.record(3_600_000_000);
        }
        self.samples.push(v);
    }

    /// Consume the accumulator and produce a `BenchStats`.
    ///
    /// `throughput_rps` must be supplied by the caller because it is measured
    /// via a separate sustained-load window, not derivable from latency alone.
    pub fn finalize(self, throughput_rps: f64) -> BenchStats {
        let h = &self.histogram;
        let p50_us = h.value_at_quantile(0.50) as f64;
        let p95_us = h.value_at_quantile(0.95) as f64;
        let p99_us = h.value_at_quantile(0.99) as f64;
        let min_us = h.min() as f64;
        let max_us = h.max() as f64;

        let n = self.samples.len() as f64;
        let mean_us = if n > 0.0 {
            self.samples.iter().sum::<u64>() as f64 / n
        } else {
            0.0
        };
        let variance = if n > 1.0 {
            self.samples
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean_us;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let stddev_us = variance.sqrt();
        let cv = if mean_us > 0.0 {
            stddev_us / mean_us
        } else {
            0.0
        };

        BenchStats {
            p50_ms: p50_us / 1_000.0,
            p95_ms: p95_us / 1_000.0,
            p99_ms: p99_us / 1_000.0,
            min_ms: min_us / 1_000.0,
            max_ms: max_us / 1_000.0,
            throughput_rps,
            stddev_ms: stddev_us / 1_000.0,
            cv,
            sample_count: self.samples.len() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms_to_us(ms: u64) -> u64 {
        ms * 1_000
    }

    #[test]
    fn percentiles_correct() {
        let mut acc = StatAccumulator::new();
        // 100 samples: 98 × 10 ms, 2 × 100 ms.
        // 98% of values are at 10ms, so p99 falls at 100ms.
        for _ in 0..98 {
            acc.record_micros(ms_to_us(10));
        }
        for _ in 0..2 {
            acc.record_micros(ms_to_us(100));
        }
        let stats = acc.finalize(0.0);
        assert!((stats.p50_ms - 10.0).abs() < 1.0, "p50 ≈ 10ms");
        assert!(stats.p99_ms >= 50.0, "p99 should reflect the 100ms tail");
        assert!((stats.max_ms - 100.0).abs() < 1.0, "max should be 100ms");
        assert_eq!(stats.sample_count, 100);
    }

    #[test]
    fn cv_zero_for_uniform_samples() {
        let mut acc = StatAccumulator::new();
        for _ in 0..50 {
            acc.record_micros(ms_to_us(20));
        }
        let stats = acc.finalize(100.0);
        assert!(
            stats.cv < 0.01,
            "CV should be near zero for uniform samples"
        );
        assert!((stats.throughput_rps - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_accumulator_produces_zeros() {
        let acc = StatAccumulator::new();
        let stats = acc.finalize(0.0);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.cv, 0.0);
    }
}
