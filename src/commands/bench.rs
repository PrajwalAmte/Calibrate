use anyhow::Result;
use chrono::Utc;
use tracing::info;

use crate::bench::harness::{run_runtime_benchmarks, HarnessConfig};
use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::build_runtime_list;
use crate::bench::{BenchReport, Recommendation};
use crate::cli::{BenchArgs, BenchOutputFormat, OptimizeFor};

pub async fn run(args: BenchArgs) -> Result<()> {
    if !args.model.exists() {
        anyhow::bail!("model file not found: {}", args.model.display());
    }

    let model_format = ModelFormat::from_path(&args.model);
    info!(format = model_format.as_str(), "detected model format");

    let gpu_device_index = detect_gpu_device();
    if gpu_device_index.is_none() {
        eprintln!("No NVIDIA GPU detected — benchmarking CPU-capable runtimes only.");
    }

    // Build the list of runtimes to run.
    let requested = args.runtimes.as_deref();
    let (mut runtimes, mut all_skipped) = build_runtime_list(requested, model_format);

    if runtimes.is_empty() {
        eprintln!(
            "No benchmarkable runtimes found for the {} format.",
            model_format.as_str()
        );
        eprintln!("Install onnxruntime, llama.cpp, or PyTorch and try again.");
        if !all_skipped.is_empty() {
            eprintln!("\nSkipped:");
            for s in &all_skipped {
                eprintln!("  {}: {}", s.name, s.reason);
            }
        }
        return Ok(());
    }

    let config = HarnessConfig {
        warmup: args.warmup,
        iterations: args.iterations,
        gpu_device_index,
        ..HarnessConfig::default()
    };

    // Pre-generate one fixed input per batch size.
    // The same input is reused across all runtimes to eliminate input variability.
    let inputs: Vec<(u32, BenchInput)> = args
        .batch_sizes
        .iter()
        .map(|&b| {
            let shape = BenchInput::default_shape_for_format(model_format, b);
            (b, BenchInput::generate(&shape))
        })
        .collect();

    // Run benchmarks for each runtime in sequence.
    // Sequential execution is required — concurrent GPU usage contaminates measurements.
    let mut all_results = Vec::new();
    for runtime in &mut runtimes {
        eprintln!("Benchmarking {}...", runtime.name());
        let (results, extra_skipped) = run_runtime_benchmarks(
            runtime.as_mut(),
            &args.model,
            &args.batch_sizes,
            &inputs,
            &config,
        );
        all_results.extend(results);
        all_skipped.extend(extra_skipped);
    }

    let recommendation = build_recommendation(&all_results, args.optimize_for);

    let report = BenchReport {
        model_path: args.model.display().to_string(),
        optimize_for: format!("{:?}", args.optimize_for).to_lowercase(),
        results: all_results,
        skipped: all_skipped,
        recommendation,
        ran_at: Utc::now(),
    };

    // Optionally persist the report.
    if let Some(save_path) = &args.save {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(save_path, &json)?;
        eprintln!("Results saved to {}", save_path.display());
    }

    // Comparison mode: diff against a saved baseline and exit.
    if let Some(compare_path) = &args.compare {
        let baseline_json = std::fs::read_to_string(compare_path)?;
        let baseline: BenchReport = serde_json::from_str(&baseline_json)?;
        crate::bench::compare::print_comparison(&baseline, &report);
        return Ok(());
    }

    // Render final output.
    match args.output {
        BenchOutputFormat::Terminal => {
            crate::output::bench_terminal::render(&report)?;
        }
        BenchOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        BenchOutputFormat::Markdown => {
            print!("{}", crate::output::bench_markdown::render(&report));
        }
    }

    Ok(())
}

// ── GPU detection

fn detect_gpu_device() -> Option<u32> {
    nvml_wrapper::Nvml::init()
        .ok()
        .and_then(|nvml| nvml.device_by_index(0).ok().map(|_| 0u32))
}

// ── Recommendation logic

pub(crate) fn build_recommendation(
    results: &[crate::bench::BenchResult],
    optimize_for: OptimizeFor,
) -> Option<Recommendation> {
    // Only consider results that completed without OOM.
    let valid: Vec<_> = results.iter().filter(|r| !r.oom).collect();
    if valid.is_empty() {
        return None;
    }

    let best = match optimize_for {
        OptimizeFor::Latency => valid
            .iter()
            .min_by(|a, b| a.stats.p99_ms.partial_cmp(&b.stats.p99_ms).unwrap()),
        OptimizeFor::Throughput => valid.iter().max_by(|a, b| {
            a.stats
                .throughput_rps
                .partial_cmp(&b.stats.throughput_rps)
                .unwrap()
        }),
        OptimizeFor::Memory => valid
            .iter()
            .min_by(|a, b| a.peak_memory_mib.partial_cmp(&b.peak_memory_mib).unwrap()),
    }?;

    let rationale = match optimize_for {
        OptimizeFor::Latency => format!(
            "{} at batch={} achieves the lowest p99 latency ({:.1}ms).",
            best.runtime, best.batch_size, best.stats.p99_ms
        ),
        OptimizeFor::Throughput => format!(
            "{} at batch={} achieves the highest throughput ({:.0} req/s).",
            best.runtime, best.batch_size, best.stats.throughput_rps
        ),
        OptimizeFor::Memory => format!(
            "{} at batch={} uses the least memory ({:.0} MiB peak).",
            best.runtime, best.batch_size, best.peak_memory_mib
        ),
    };

    Some(Recommendation {
        runtime: best.runtime.clone(),
        batch_size: best.batch_size,
        rationale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{BenchResult, BenchStats};

    fn make_result(
        runtime: &str,
        batch: u32,
        p99_ms: f64,
        tput: f64,
        mem_mib: f64,
        oom: bool,
    ) -> BenchResult {
        BenchResult {
            runtime: runtime.to_string(),
            batch_size: batch,
            stats: BenchStats {
                p50_ms: p99_ms * 0.7,
                p95_ms: p99_ms * 0.9,
                p99_ms,
                min_ms: p99_ms * 0.5,
                max_ms: p99_ms * 1.1,
                throughput_rps: tput,
                stddev_ms: 1.0,
                cv: 0.05,
                sample_count: 100,
            },
            peak_memory_mib: mem_mib,
            memory_delta_mib: mem_mib / 2.0,
            load_time_ms: 200,
            warmup_stable_at: 5,
            flagged_unreliable: false,
            oom,
        }
    }

    #[test]
    fn recommendation_latency_picks_lowest_p99() {
        let results = vec![
            make_result("onnx", 1, 25.0, 40.0, 512.0, false),
            make_result("candle", 1, 10.0, 90.0, 380.0, false), // lowest p99
            make_result("llama", 1, 50.0, 20.0, 800.0, false),
        ];
        let rec = build_recommendation(&results, OptimizeFor::Latency).unwrap();
        assert_eq!(rec.runtime, "candle");
        assert_eq!(rec.batch_size, 1);
    }

    #[test]
    fn recommendation_throughput_picks_highest_rps() {
        let results = vec![
            make_result("onnx", 1, 20.0, 60.0, 512.0, false),
            make_result("candle", 1, 15.0, 120.0, 380.0, false), // highest tput
            make_result("llama", 1, 30.0, 30.0, 800.0, false),
        ];
        let rec = build_recommendation(&results, OptimizeFor::Throughput).unwrap();
        assert_eq!(rec.runtime, "candle");
    }

    #[test]
    fn recommendation_memory_picks_lowest_memory() {
        let results = vec![
            make_result("onnx", 1, 20.0, 60.0, 512.0, false),
            make_result("candle", 1, 15.0, 90.0, 200.0, false), // lowest memory
            make_result("llama", 1, 30.0, 30.0, 900.0, false),
        ];
        let rec = build_recommendation(&results, OptimizeFor::Memory).unwrap();
        assert_eq!(rec.runtime, "candle");
    }

    #[test]
    fn recommendation_excludes_oom_results() {
        let results = vec![
            make_result("onnx", 1, 20.0, 60.0, 512.0, false),
            make_result("candle", 1, 5.0, 999.0, 100.0, true), // oom — must be ignored
        ];
        let rec = build_recommendation(&results, OptimizeFor::Latency).unwrap();
        assert_eq!(rec.runtime, "onnx", "OOM result should not win");
    }

    #[test]
    fn recommendation_all_oom_returns_none() {
        let results = vec![
            make_result("onnx", 1, 0.0, 0.0, 0.0, true),
            make_result("candle", 1, 0.0, 0.0, 0.0, true),
        ];
        assert!(build_recommendation(&results, OptimizeFor::Latency).is_none());
    }
}
