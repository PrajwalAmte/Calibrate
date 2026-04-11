pub mod compare;
pub mod harness;
pub mod input;
pub mod memory;
pub mod runtime;
pub mod runtimes;
pub mod stats;

use chrono::{DateTime, Utc};
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
    pub ran_at: DateTime<Utc>,
}
