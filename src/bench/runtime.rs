use std::path::Path;
use std::time::Duration;

use anyhow::Result;

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::SkippedRuntime;

/// Object-safe trait implemented by each inference runtime backend.
///
/// The harness owns all timing and memory measurement; runtimes are
/// responsible only for loading a model, executing a single forward pass,
/// and releasing resources cleanly.
pub trait Runtime: Send {
    /// Human-readable runtime name used in output tables and JSON reports.
    fn name(&self) -> &str;

    /// Load the model from disk into the runtime.
    ///
    /// Called once per (runtime, batch_size) pair. Returns the wall-clock
    /// time spent loading, which is reported separately in the results.
    fn load(&mut self, model_path: &Path) -> Result<Duration>;

    /// Execute one forward pass with the provided input.
    ///
    /// Called in a tight loop by the harness. Any lazy compilation or
    /// memory allocation that would produce uneven latency per call must
    /// be completed during `load()`, not here.
    fn infer(&mut self, input: &BenchInput) -> Result<()>;

    /// Release the loaded model and free all runtime-held resources.
    ///
    /// Called by the harness after each (runtime, batch_size) pair finishes,
    /// before the system-state cooldown begins.
    fn unload(&mut self);

    /// Return pre-collected samples if this runtime measures internally.
    ///
    /// Subprocess-based runtimes (onnxruntime, torchscript, tensorrt, llamacpp)
    /// run warm-up + measurement + throughput window entirely inside the child
    /// process on the first `infer()` call. They override this method to hand
    /// the recorded samples back to the harness so it can feed them directly
    /// into the `StatAccumulator` instead of timing a no-op replay loop.
    ///
    /// Returns `Some((latency_samples_us, throughput_rps, load_ms))` after the
    /// subprocess has completed. Returns `None` for in-process runtimes (e.g.
    /// candle) where `infer()` represents a single real forward pass.
    ///
    /// Must be called **after** the first `infer()` call.
    /// Implementations must be idempotent: a second call returns `None`.
    fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> {
        None
    }
}

/// Static descriptor for a runtime.
///
/// Holds availability and compatibility checks as plain function pointers
/// so the registry can be built without instantiating any runtime. Only
/// runtimes that pass both checks are instantiated via `create`.
pub struct RuntimeDescriptor {
    pub name: &'static str,
    /// Return `true` if the runtime binary or Python package is installed
    /// on the current system.
    pub is_available: fn() -> bool,
    /// Return `true` if the runtime can load a model of the given format.
    pub is_compatible: fn(ModelFormat) -> bool,
    /// Construct a boxed runtime instance ready to receive `load()` calls.
    pub create: fn() -> Box<dyn Runtime>,
}

/// Build the list of runtime instances to benchmark.
///
/// Checks availability and format compatibility for every registered
/// descriptor, instantiating only those that pass. Returns
/// `(ready_runtimes, skipped_entries)`.
pub fn build_runtime_list(
    requested: Option<&[String]>,
    format: ModelFormat,
) -> (Vec<Box<dyn Runtime>>, Vec<SkippedRuntime>) {
    let registry = crate::bench::runtimes::registry();
    let mut runtimes: Vec<Box<dyn Runtime>> = Vec::new();
    let mut skipped: Vec<SkippedRuntime> = Vec::new();

    for descriptor in &registry {
        // If the user named specific runtimes, skip anything not in the list.
        if let Some(names) = requested {
            if !names
                .iter()
                .any(|n| n.eq_ignore_ascii_case(descriptor.name))
            {
                continue;
            }
        }

        if !(descriptor.is_available)() {
            skipped.push(SkippedRuntime {
                name: descriptor.name.to_string(),
                reason: "not installed or not found in PATH".to_string(),
            });
            continue;
        }

        if !(descriptor.is_compatible)(format) {
            skipped.push(SkippedRuntime {
                name: descriptor.name.to_string(),
                reason: format!(
                    "incompatible with {} model format",
                    format.as_str()
                ),
            });
            continue;
        }

        runtimes.push((descriptor.create)());
    }

    (runtimes, skipped)
}
