use crate::bench::{BenchReport, BenchResult};

/// Print a side-by-side diff of two benchmark reports to stdout.
///
/// Highlights improvements (↓ latency, ↑ throughput, ↓ memory) and
/// regressions (↑ latency, ↓ throughput, ↑ memory) relative to `baseline`.
pub fn print_comparison(baseline: &BenchReport, current: &BenchReport) {
    println!("Benchmark comparison");
    println!(
        "  Baseline : {} ({})",
        baseline.model_path,
        baseline.ran_at.format("%Y-%m-%d")
    );
    println!(
        "  Current  : {} ({})",
        current.model_path,
        current.ran_at.format("%Y-%m-%d")
    );
    println!();

    let header = format!(
        "{:<20} {:>6}  {:>9} {:>9} {:>9}  {:>12}  {:>11}",
        "Runtime (batch)", "Change", "Δ p50", "Δ p99", "Δ tput", "Δ memory", "Status"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for cur in &current.results {
        let key = (&cur.runtime, cur.batch_size);
        let baseline_result = baseline
            .results
            .iter()
            .find(|b| b.runtime == cur.runtime && b.batch_size == cur.batch_size);

        let Some(base) = baseline_result else {
            println!(
                "{:<20} {:>6}",
                format!("{} ({})", cur.runtime, cur.batch_size),
                "NEW"
            );
            continue;
        };

        let d_p50 = cur.stats.p50_ms - base.stats.p50_ms;
        let d_p99 = cur.stats.p99_ms - base.stats.p99_ms;
        let d_tput = cur.stats.throughput_rps - base.stats.throughput_rps;
        let d_mem = cur.peak_memory_mib - base.peak_memory_mib;

        // A result is a regression if p99 increased by more than 5%.
        let p99_pct_change = if base.stats.p99_ms > 0.0 {
            (d_p99 / base.stats.p99_ms) * 100.0
        } else {
            0.0
        };

        let status = if p99_pct_change > 5.0 {
            "REGRESSION"
        } else if p99_pct_change < -5.0 {
            "improvement"
        } else {
            "unchanged"
        };

        println!(
            "{:<20} {:>6}  {:>+8.1}ms {:>+8.1}ms {:>+8.1}/s  {:>+10.0}MiB  {:>11}",
            format!("{} ({})", cur.runtime, cur.batch_size),
            "",
            d_p50,
            d_p99,
            d_tput,
            d_mem,
            status,
        );

        let _ = key; // suppress unused warning
    }

    // List baseline entries that are missing from current.
    for base in &baseline.results {
        let exists = current
            .results
            .iter()
            .any(|c| c.runtime == base.runtime && c.batch_size == base.batch_size);
        if !exists {
            println!(
                "{:<20} {:>6}",
                format!("{} ({})", base.runtime, base.batch_size),
                "REMOVED"
            );
        }
    }
}

/// Compute a percentage change, returning `None` when `base` is zero.
#[allow(dead_code)]
fn pct_change(base: f64, current: f64) -> Option<f64> {
    if base == 0.0 {
        None
    } else {
        Some(((current - base) / base) * 100.0)
    }
}

/// Return a formatted delta string like "+12.3% (improved)" or "-5.1% (regression)".
#[allow(dead_code)]
pub fn format_latency_delta(base: &BenchResult, current: &BenchResult) -> String {
    match pct_change(base.stats.p99_ms, current.stats.p99_ms) {
        None => "n/a".to_string(),
        Some(pct) if pct > 5.0 => format!("{pct:+.1}% (regression)"),
        Some(pct) if pct < -5.0 => format!("{pct:+.1}% (improved)"),
        Some(pct) => format!("{pct:+.1}% (unchanged)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{BenchResult, BenchStats};
    use chrono::Utc;

    fn make_result(runtime: &str, p99_ms: f64) -> BenchResult {
        BenchResult {
            runtime: runtime.to_string(),
            batch_size: 1,
            stats: BenchStats {
                p50_ms: p99_ms * 0.7,
                p95_ms: p99_ms * 0.9,
                p99_ms,
                min_ms: p99_ms * 0.5,
                max_ms: p99_ms * 1.1,
                throughput_rps: 50.0,
                stddev_ms: 1.0,
                cv: 0.05,
                sample_count: 100,
            },
            peak_memory_mib: 512.0,
            memory_delta_mib: 100.0,
            load_time_ms: 300,
            warmup_stable_at: 5,
            flagged_unreliable: false,
            oom: false,
        }
    }

    fn make_report(results: Vec<BenchResult>) -> BenchReport {
        BenchReport {
            model_path: "/tmp/model.onnx".to_string(),
            optimize_for: "latency".to_string(),
            results,
            skipped: vec![],
            recommendation: None,
            ran_at: Utc::now(),
        }
    }

    #[test]
    fn pct_change_zero_base_is_none() {
        assert!(pct_change(0.0, 10.0).is_none());
    }

    #[test]
    fn pct_change_computes_correctly() {
        // +10 %
        let v = pct_change(10.0, 11.0).unwrap();
        assert!((v - 10.0).abs() < 0.001, "expected 10.0, got {v}");
        // -20 %
        let v = pct_change(10.0, 8.0).unwrap();
        assert!((v - (-20.0)).abs() < 0.001, "expected -20.0, got {v}");
    }

    #[test]
    fn format_latency_delta_regression_label() {
        // 10 % increase → regression (threshold is > 5 %)
        let base = make_result("onnx", 10.0);
        let cur = make_result("onnx", 11.0);
        let s = format_latency_delta(&base, &cur);
        assert!(s.contains("regression"), "got: {s}");
    }

    #[test]
    fn format_latency_delta_improvement_label() {
        // 20 % decrease → improved
        let base = make_result("onnx", 10.0);
        let cur = make_result("onnx", 8.0);
        let s = format_latency_delta(&base, &cur);
        assert!(s.contains("improved"), "got: {s}");
    }

    #[test]
    fn format_latency_delta_unchanged_within_threshold() {
        // 3 % increase — within the ±5 % unchanged band
        let base = make_result("onnx", 10.0);
        let cur = make_result("onnx", 10.3);
        let s = format_latency_delta(&base, &cur);
        assert!(s.contains("unchanged"), "got: {s}");
    }

    #[test]
    fn print_comparison_does_not_panic_on_new_and_removed_entries() {
        let base = make_report(vec![make_result("onnx", 20.0)]);
        let cur = make_report(vec![make_result("candle", 15.0)]);
        // "onnx" is removed in current, "candle" is new — both code paths exercised.
        print_comparison(&base, &cur);
    }
}
