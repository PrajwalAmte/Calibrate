pub mod compare;
pub mod harness;
pub mod input;
pub mod memory;
pub mod runtime;
pub mod runtimes;
pub mod stats;

use serde::{Deserialize, Serialize};

pub use stats::BenchStats;

/// A single benchmarked (runtime, batch_size) combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub runtime: String,
    pub batch_size: u32,
    pub stats: BenchStats,
    /// Peak memory consumed during the measurement phase (MiB).
    pub peak_memory_mib: f64,
    /// Memory delta above the pre-load baseline (MiB).
    pub memory_delta_mib: f64,
    /// Wall-clock time to load the model into this runtime (ms).
    pub load_time_ms: u64,
    /// First warm-up iteration at which std dev of the rolling 5-sample window
    /// dropped below 10% of the mean, indicating performance has stabilized.
    pub warmup_stable_at: u32,
    /// True when coefficient of variation of measurement samples exceeds 0.20,
    /// indicating high system noise during this benchmark.
    pub flagged_unreliable: bool,
    /// True if the runtime ran out of memory at this batch size.
    pub oom: bool,
    /// Kernel microbenchmark rather than model inference (never recommended).
    #[serde(default)]
    pub synthetic: bool,
    /// Where `peak_memory_mib` came from: "device" (GPU/MPS allocator in the
    /// runtime's process), "host_rss" (runtime process RSS), "nvml", "none".
    #[serde(default)]
    pub memory_source: String,
}

/// A runtime that was not benchmarked, with the reason why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedRuntime {
    pub name: String,
    pub reason: String,
}

/// The recommended (runtime, batch_size) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub runtime: String,
    pub batch_size: u32,
    /// Plain-English explanation of why this configuration was chosen.
    pub rationale: String,
}

/// The complete output of a `calibrate bench` run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub model_path: String,
    pub optimize_for: String,
    pub results: Vec<BenchResult>,
    pub skipped: Vec<SkippedRuntime>,
    pub recommendation: Option<Recommendation>,
    pub ran_at: jiff::Timestamp,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report_json(ran_at: &str) -> String {
        format!(
            r#"{{"model_path":"m","optimize_for":"latency","results":[],"skipped":[],
               "recommendation":null,"ran_at":"{ran_at}"}}"#
        )
    }

    #[test]
    fn baselines_saved_by_older_versions_still_load() {
        // `bench --save` files written while timestamps used chrono, in the
        // forms chrono emitted (Z suffix, microseconds or whole seconds).
        for ts in [
            "2026-10-21T04:42:36.182661Z",
            "2026-10-21T04:42:36Z",
            "2026-10-21T04:42:36.182661123Z",
            "2026-10-21T04:42:36+00:00",
        ] {
            let r: BenchReport =
                serde_json::from_str(&report_json(ts)).unwrap_or_else(|e| panic!("{ts}: {e}"));
            assert_eq!(
                r.ran_at.strftime("%Y-%m-%d %H:%M").to_string(),
                "2026-10-21 04:42"
            );
        }
    }

    #[test]
    fn timestamps_serialize_as_rfc3339_utc() {
        let r: BenchReport = serde_json::from_str(&report_json("2026-10-21T04:42:36.5Z")).unwrap();
        let v = serde_json::to_value(&r).unwrap();
        let s = v["ran_at"].as_str().unwrap();
        assert!(
            s.starts_with("2026-10-21T04:42:36.5") && s.ends_with('Z'),
            "{s}"
        );
        let now = serde_json::to_value(jiff::Timestamp::now()).unwrap();
        let now = now.as_str().unwrap();
        assert!(
            now.ends_with('Z') && now.contains('T') && now.len() >= 20,
            "{now}"
        );
    }
}
