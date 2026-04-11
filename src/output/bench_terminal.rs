use anyhow::Result;

use crate::bench::BenchReport;

/// Print the benchmark results as a formatted table to stdout.
///
/// The recommended row is marked with an arrow. Unreliable rows are marked
/// with an asterisk and a footnote appears below the table.
pub fn render(report: &BenchReport) -> Result<()> {
    let results = &report.results;

    if results.is_empty() {
        println!("No benchmark results to display.");
        print_skipped(report);
        return Ok(());
    }

    // Column widths.
    let runtime_w = results
        .iter()
        .map(|r| r.runtime.len())
        .max()
        .unwrap_or(7)
        .max(7);

    println!();
    println!(
        "  {:<rw$}  {:>5}  {:>9}  {:>9}  {:>12}  {:>10}",
        "Runtime",
        "Batch",
        "p50",
        "p99",
        "Throughput",
        "Memory",
        rw = runtime_w,
    );
    println!(
        "  {:-<rw$}  {:-<5}  {:-<9}  {:-<9}  {:-<12}  {:-<10}",
        "",
        "",
        "",
        "",
        "",
        "",
        rw = runtime_w,
    );

    let recommended_key = report
        .recommendation
        .as_ref()
        .map(|r| (r.runtime.as_str(), r.batch_size));

    let mut any_unreliable = false;
    for result in results {
        let is_recommended = recommended_key
            .map(|(name, batch)| name == result.runtime && batch == result.batch_size)
            .unwrap_or(false);

        let flag = if result.flagged_unreliable {
            any_unreliable = true;
            "*"
        } else if result.oom {
            "!"
        } else {
            " "
        };

        let marker = if is_recommended { ">" } else { " " };

        let (p50, p99, tput, mem) = if result.oom {
            ("OOM".to_string(), "OOM".to_string(), "OOM".to_string(), "OOM".to_string())
        } else {
            (
                format!("{:.1}ms", result.stats.p50_ms),
                format!("{:.1}ms", result.stats.p99_ms),
                format!("{:.0} req/s", result.stats.throughput_rps),
                format!("{:.0} MiB", result.peak_memory_mib),
            )
        };

        println!(
            "{}{} {:<rw$}  {:>5}  {:>9}  {:>9}  {:>12}  {:>10}",
            marker,
            flag,
            result.runtime,
            result.batch_size,
            p50,
            p99,
            tput,
            mem,
            rw = runtime_w,
        );
    }

    println!();

    if any_unreliable {
        println!("  * High variance detected — results may be unreliable.");
        println!("    Stop other processes and re-run for accurate measurements.");
        println!();
    }

    print_recommendation(report);
    print_skipped(report);

    Ok(())
}

fn print_recommendation(report: &BenchReport) {
    if let Some(rec) = &report.recommendation {
        println!(
            "Recommendation (optimising for {}): {} at batch={}",
            report.optimize_for, rec.runtime, rec.batch_size
        );
        println!("  {}", rec.rationale);
        println!();
    }
}

fn print_skipped(report: &BenchReport) {
    if report.skipped.is_empty() {
        return;
    }
    println!("Skipped runtimes:");
    for s in &report.skipped {
        println!("  {}: {}", s.name, s.reason);
    }
    println!();
}
