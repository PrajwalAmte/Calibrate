use std::path::Path;
use std::time::{Duration, Instant};

use tracing::warn;

use crate::bench::input::BenchInput;
use crate::bench::memory::{MemorySnapshot, PeakMemoryTracker};
use crate::bench::runtime::Runtime;
use crate::bench::stats::StatAccumulator;
use crate::bench::{BenchResult, SkippedRuntime};

/// Configuration for a single benchmark run.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    /// Warm-up iterations whose timings are discarded.
    pub warmup: u32,
    /// Timed measurement iterations.
    pub iterations: u32,
    /// Duration of the sustained throughput measurement window.
    pub throughput_window: Duration,
    /// NVML device index for VRAM measurement. `None` on CPU-only machines.
    pub gpu_device_index: Option<u32>,
    /// Maximum allowed wall-clock time for one (runtime, batch_size) pair
    /// before the harness reduces the iteration count automatically.
    pub max_total_duration: Duration,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            warmup: 20,
            iterations: 100,
            throughput_window: Duration::from_secs(5),
            gpu_device_index: None,
            max_total_duration: Duration::from_secs(600), // 10 minutes
        }
    }
}

/// Run the full benchmark for one runtime across all requested batch sizes.
///
/// Returns `(results, additional_skipped)`. An entry is added to
/// `additional_skipped` for any batch size that cannot be completed (OOM,
/// load failure, etc.).
pub fn run_runtime_benchmarks(
    runtime: &mut dyn Runtime,
    model_path: &Path,
    batch_sizes: &[u32],
    inputs: &[(u32, BenchInput)],
    config: &HarnessConfig,
) -> (Vec<BenchResult>, Vec<SkippedRuntime>) {
    let mut results = Vec::new();
    let mut skipped = Vec::new();

    for &batch_size in batch_sizes {
        let input = match inputs.iter().find(|(b, _)| *b == batch_size) {
            Some((_, inp)) => inp,
            None => {
                warn!("no pre-generated input for batch_size={batch_size}, skipping");
                continue;
            }
        };

        // Snapshot memory before model load so the delta is meaningful.
        let baseline = MemorySnapshot::take(config.gpu_device_index);

        // Load the model and record how long it took.
        let load_start = Instant::now();
        match runtime.load(model_path) {
            Err(ref e) if is_oom_error(e) => {
                // OOM during load: record as an OOM result so it appears in
                // the table rather than being silently absent from the output.
                let load_time_ms = load_start.elapsed().as_millis() as u64;
                runtime.unload();
                results.push(BenchResult {
                    runtime: runtime.name().to_string(),
                    batch_size,
                    stats: StatAccumulator::new().finalize(0.0),
                    peak_memory_mib: 0.0,
                    memory_delta_mib: 0.0,
                    load_time_ms,
                    warmup_stable_at: 0,
                    flagged_unreliable: false,
                    oom: true,
                });
                continue;
            }
            Err(e) => {
                skipped.push(SkippedRuntime {
                    name: format!("{} (batch={})", runtime.name(), batch_size),
                    reason: format!("load failed: {e}"),
                });
                continue;
            }
            Ok(_) => {}
        }
        let load_time_ms = load_start.elapsed().as_millis() as u64;

        if load_time_ms > 60_000 {
            eprintln!(
                "Note: {} took {:.1}s to load the model.",
                runtime.name(),
                load_time_ms as f64 / 1_000.0
            );
        }

        // Warn about concurrent GPU workloads before wasting benchmark time.
        if let Some(idx) = config.gpu_device_index {
            check_concurrent_gpu_processes(idx);
        }

        // ── Delegated measurement (subprocess runtimes) ───────────────────
        //
        // Subprocess runtimes (onnxruntime, torchscript, tensorrt, llamacpp)
        // run warm-up + measurement + throughput window entirely inside the
        // child process on the first `infer()` call. Timing that call from
        // Rust would record subprocess launch overhead, not inference latency.
        //
        // After the first `infer()`, `pre_collected_samples()` returns the
        // raw latency samples and throughput measured inside the subprocess.
        // We inject them directly into the StatAccumulator rather than running
        // the normal timing loop.

        // Trigger the subprocess by calling infer() once.
        let first_infer = runtime.infer(input);

        if let Some((samples_us, throughput_rps, delegated_load_ms)) =
            runtime.pre_collected_samples()
        {
            // --- Delegated path (subprocess runtimes) --------------------
            // The runtime measured everything internally; use those numbers.
            if let Err(e) = first_infer {
                if is_oom_error(&e) {
                    runtime.unload();
                    results.push(BenchResult {
                        runtime: runtime.name().to_string(),
                        batch_size,
                        stats: crate::bench::stats::StatAccumulator::new()
                            .finalize(0.0),
                        peak_memory_mib: 0.0,
                        memory_delta_mib: 0.0,
                        load_time_ms,
                        warmup_stable_at: 0,
                        flagged_unreliable: false,
                        oom: true,
                    });
                    continue;
                }
                skipped.push(SkippedRuntime {
                    name: format!("{} (batch={})", runtime.name(), batch_size),
                    reason: format!("subprocess error: {e}"),
                });
                runtime.unload();
                continue;
            }

            let mut accumulator = StatAccumulator::new();
            for &us in &samples_us {
                accumulator.record_micros(us);
            }
            let stats = accumulator.finalize(throughput_rps);
            let effective_load_ms = if delegated_load_ms > 0 {
                delegated_load_ms
            } else {
                load_time_ms
            };

            let flagged_unreliable = stats.cv > 0.20;
            if flagged_unreliable {
                warn!(
                    runtime = runtime.name(),
                    batch_size,
                    cv = %format!("{:.2}", stats.cv),
                    "High variance in subprocess results (CV={:.2}). \
                     Results may be unreliable.",
                    stats.cv,
                );
            }

            // Memory: take a snapshot immediately after subprocess returns.
            let post_snapshot = MemorySnapshot::take(config.gpu_device_index);
            let delta = post_snapshot.delta_from(&baseline);
            let (peak_memory_mib, memory_delta_mib) = if config.gpu_device_index.is_some() {
                (post_snapshot.vram_used_mib, delta.vram_delta_mib)
            } else {
                (post_snapshot.rss_mib, delta.rss_delta_mib)
            };

            runtime.unload();
            wait_for_gpu_idle(config.gpu_device_index, Duration::from_secs(30));

            results.push(BenchResult {
                runtime: runtime.name().to_string(),
                batch_size,
                stats,
                peak_memory_mib,
                memory_delta_mib,
                load_time_ms: effective_load_ms,
                warmup_stable_at: 0, // managed inside subprocess
                flagged_unreliable,
                oom: false,
            });
            continue;
        }

        // ── In-process measurement path (e.g. candle) ─────────────────────
        //
        // The first infer() call has already executed one real forward pass;
        // check it succeeded before entering the measurement loop.
        if let Err(ref e) = first_infer {
            if is_oom_error(e) {
                runtime.unload();
                results.push(BenchResult {
                    runtime: runtime.name().to_string(),
                    batch_size,
                    stats: crate::bench::stats::StatAccumulator::new().finalize(0.0),
                    peak_memory_mib: 0.0,
                    memory_delta_mib: 0.0,
                    load_time_ms,
                    warmup_stable_at: 0,
                    flagged_unreliable: false,
                    oom: true,
                });
                continue;
            }
            skipped.push(SkippedRuntime {
                name: format!("{} (batch={})", runtime.name(), batch_size),
                reason: format!("first infer error: {}", first_infer.unwrap_err()),
            });
            runtime.unload();
            continue;
        }

        // Pilot iteration already ran (the first_infer call above). Use it to
        // estimate budget and guard against exceeding the time limit.
        let effective_iterations = budget_guard(
            config.iterations,
            config.warmup,
            runtime,
            input,
            config.max_total_duration,
        );

        // --- Warm-up phase -----------------------------------------------
        let warmup_stable_at = run_warmup(runtime, input, config.warmup);

        // --- Measurement phase -------------------------------------------
        let mut accumulator = StatAccumulator::new();
        let mut peak_tracker = PeakMemoryTracker::new();
        let mut oom = false;

        for i in 0..effective_iterations {
            // Sample memory every 10 iterations to limit NVML call overhead.
            if i % 10 == 0 {
                peak_tracker.update(config.gpu_device_index);
            }

            let start = Instant::now();
            match runtime.infer(input) {
                Ok(()) => {
                    let elapsed_us = start.elapsed().as_micros() as u64;
                    accumulator.record_micros(elapsed_us);
                }
                Err(ref e) if is_oom_error(e) => {
                    warn!(
                        runtime = runtime.name(),
                        batch_size,
                        "OOM at measurement iteration {i}: {e}"
                    );
                    oom = true;
                    break;
                }
                Err(e) => {
                    warn!(runtime = runtime.name(), batch_size, "infer error: {e}");
                    break;
                }
            }
        }

        // --- Sustained throughput window ---------------------------------
        // Only measured when there was no OOM; uses the already-warm runtime.
        let throughput_rps = if !oom {
            measure_throughput(runtime, input, config.throughput_window)
        } else {
            0.0
        };

        // --- Finalise statistics -----------------------------------------
        let stats = accumulator.finalize(throughput_rps);

        let flagged_unreliable = stats.cv > 0.20;
        if flagged_unreliable {
            warn!(
                runtime = runtime.name(),
                batch_size,
                cv = %format!("{:.2}", stats.cv),
                "High variance detected (CV={:.2}). Results may be unreliable. \
                 Stop other processes and re-run for accurate measurements.",
                stats.cv
            );
        }

        // --- Memory delta ------------------------------------------------
        let peak_snapshot = peak_tracker.peak();
        let delta = peak_snapshot.delta_from(&baseline);
        let (peak_memory_mib, memory_delta_mib) = if config.gpu_device_index.is_some() {
            (peak_snapshot.vram_used_mib, delta.vram_delta_mib)
        } else {
            (peak_snapshot.rss_mib, delta.rss_delta_mib)
        };

        // --- Unload and cool down ----------------------------------------
        runtime.unload();
        wait_for_gpu_idle(config.gpu_device_index, Duration::from_secs(30));

        results.push(BenchResult {
            runtime: runtime.name().to_string(),
            batch_size,
            stats,
            peak_memory_mib,
            memory_delta_mib,
            load_time_ms,
            warmup_stable_at,
            flagged_unreliable,
            oom,
        });
    }

    (results, skipped)
}

// ── Warm-up ──────────────────────────────────────────────────────────────────

/// Run warm-up iterations and return the first iteration index at which
/// performance stabilised (5-sample rolling std dev < 10% of the mean).
///
/// Returns `warmup_count` if stability was never detected within the budget.
fn run_warmup(runtime: &mut dyn Runtime, input: &BenchInput, warmup_count: u32) -> u32 {
    let mut window: Vec<f64> = Vec::with_capacity(6);
    let mut stable_at = warmup_count;

    for i in 0..warmup_count {
        let start = Instant::now();
        let _ = runtime.infer(input);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;

        window.push(elapsed_ms);
        if window.len() > 5 {
            window.remove(0);
        }

        if window.len() == 5 {
            let mean = window.iter().sum::<f64>() / 5.0;
            let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 4.0;
            let stddev = variance.sqrt();
            if mean > 0.0 && (stddev / mean) < 0.10 {
                stable_at = i;
                break;
            }
        }
    }

    stable_at
}

// ── Sustained throughput ──────────────────────────────────────────────────────

/// Measure sustained throughput by running `infer` as fast as possible for a
/// fixed time window and dividing completed calls by elapsed wall-clock time.
///
/// This differs from `1 / mean_latency` because it captures scheduling
/// overhead between requests under continuous load.
fn measure_throughput(
    runtime: &mut dyn Runtime,
    input: &BenchInput,
    window: Duration,
) -> f64 {
    let start = Instant::now();
    let mut count: u64 = 0;
    while start.elapsed() < window {
        if runtime.infer(input).is_ok() {
            count += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        count as f64 / elapsed
    } else {
        0.0
    }
}

// ── Iteration budget guard ────────────────────────────────────────────────────

/// Run one pilot iteration to estimate per-iteration cost.
/// Reduce the iteration count if the full benchmark would exceed `max_total`.
fn budget_guard(
    requested: u32,
    warmup: u32,
    runtime: &mut dyn Runtime,
    input: &BenchInput,
    max_total: Duration,
) -> u32 {
    let start = Instant::now();
    let _ = runtime.infer(input);
    let per_iter = start.elapsed();

    if per_iter.is_zero() {
        return requested;
    }

    // Estimate total time for warmup + measurement (pilot counts as first warmup step).
    let total_estimate = per_iter.saturating_mul(warmup + requested);
    if total_estimate <= max_total {
        return requested;
    }

    let warmup_cost = per_iter.saturating_mul(warmup);
    if warmup_cost >= max_total {
        let budget_mins = max_total.as_secs() / 60;
        eprintln!(
            "Warning: warm-up alone may exceed the {budget_mins}-minute time budget. \
             Proceeding with 10 measurement iterations."
        );
        return 10;
    }

    let remaining = max_total - warmup_cost;
    let per_ns = per_iter.as_nanos().max(1);
    let reduced = ((remaining.as_nanos() / per_ns) as u32).max(10);

    if reduced < requested {
        let budget_mins = max_total.as_secs() / 60;
        eprintln!(
            "Warning: reducing measurement iterations from {requested} to {reduced} \
             to stay within the {budget_mins}-minute time budget."
        );
    }

    reduced
}

// ── System-state utilities ────────────────────────────────────────────────────

/// Return `true` when the error message suggests an out-of-memory condition.
fn is_oom_error(e: &anyhow::Error) -> bool {
    let msg = format!("{e:?}").to_lowercase();
    msg.contains("out of memory")
        || msg.contains("cuda error")
        || msg.contains("oom")
        || msg.contains("cudaerrormemoryal")
}

/// Log a warning if processes other than the current one are using the GPU.
fn check_concurrent_gpu_processes(device_index: u32) {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return;
    };
    let Ok(device) = nvml.device_by_index(device_index) else {
        return;
    };
    let Ok(procs) = device.running_compute_processes() else {
        return;
    };
    let own_pid = std::process::id();
    let other_count = procs.iter().filter(|p| p.pid != own_pid).count();
    if other_count > 0 {
        eprintln!(
            "Warning: {other_count} other GPU compute process(es) detected. \
             Concurrent GPU usage will contaminate benchmark measurements. \
             Stop other GPU processes and re-run for reliable results."
        );
    }
}

/// Wait until GPU SM utilisation drops to ≤ 5% or `timeout` elapses.
///
/// Called between runtimes to ensure the GPU returns to a consistent idle
/// state before the next benchmark begins.
fn wait_for_gpu_idle(device_index: Option<u32>, timeout: Duration) {
    let Some(idx) = device_index else {
        return;
    };
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return;
    };
    let Ok(device) = nvml.device_by_index(idx) else {
        return;
    };

    let start = Instant::now();
    loop {
        if start.elapsed() >= timeout {
            eprintln!(
                "Warning: GPU did not return to idle within {}s. \
                 Proceeding with the next benchmark anyway.",
                timeout.as_secs()
            );
            break;
        }
        if let Ok(util) = device.utilization_rates() {
            if util.gpu <= 5 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom_error_detects_oom_string() {
        let e = anyhow::anyhow!("CUDA error: out of memory");
        assert!(is_oom_error(&e));
    }

    #[test]
    fn oom_error_detects_oom_abbreviation() {
        let e = anyhow::anyhow!("RuntimeError: CUDA out of memory (OOM)");
        assert!(is_oom_error(&e));
    }

    #[test]
    fn oom_error_does_not_match_unrelated_error() {
        let e = anyhow::anyhow!("model file not found");
        assert!(!is_oom_error(&e));
    }

    #[test]
    fn harness_config_default_values() {
        let cfg = HarnessConfig::default();
        assert_eq!(cfg.warmup, 20);
        assert_eq!(cfg.iterations, 100);
        assert_eq!(cfg.gpu_device_index, None);
        // 10-minute budget
        assert_eq!(cfg.max_total_duration, Duration::from_secs(600));
    }
}
