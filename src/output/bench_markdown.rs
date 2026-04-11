use crate::bench::BenchReport;

/// Render the benchmark report as a Markdown document.
///
/// Returns the full document as a `String` suitable for writing to a file
/// or printing to stdout.
pub fn render(report: &BenchReport) -> String {
    let mut out = String::new();

    out.push_str("# calibrate bench report\n\n");
    out.push_str(&format!("**Model:** `{}`  \n", report.model_path));
    out.push_str(&format!("**Optimising for:** {}  \n", report.optimize_for));
    out.push_str(&format!(
        "**Ran at:** {}  \n",
        report.ran_at.format("%Y-%m-%d %H:%M UTC")
    ));
    out.push('\n');

    if !report.results.is_empty() {
        out.push_str("## Results\n\n");
        out.push_str("| Runtime | Batch | p50 | p99 | Throughput | Memory |\n");
        out.push_str("|---------|------:|----:|----:|-----------:|-------:|\n");

        let recommended_key = report
            .recommendation
            .as_ref()
            .map(|r| (r.runtime.as_str(), r.batch_size));

        for r in &report.results {
            let is_rec = recommended_key
                .map(|(name, batch)| name == r.runtime && batch == r.batch_size)
                .unwrap_or(false);

            let name = if is_rec {
                format!("**{}** ✓", r.runtime)
            } else {
                r.runtime.clone()
            };

            let (p50, p99, tput, mem) = if r.oom {
                (
                    "OOM".to_string(),
                    "OOM".to_string(),
                    "OOM".to_string(),
                    "OOM".to_string(),
                )
            } else {
                (
                    format!("{:.1}ms", r.stats.p50_ms),
                    format!("{:.1}ms", r.stats.p99_ms),
                    format!("{:.0} req/s", r.stats.throughput_rps),
                    format!("{:.0} MiB", r.peak_memory_mib),
                )
            };

            let flag = if r.flagged_unreliable { " *" } else { "" };

            out.push_str(&format!(
                "| {}{} | {} | {} | {} | {} | {} |\n",
                name, flag, r.batch_size, p50, p99, tput, mem
            ));
        }

        out.push('\n');

        if report.results.iter().any(|r| r.flagged_unreliable) {
            out.push_str(
                "> \\* High variance detected. \
                 Stop other processes and re-run for reliable results.\n\n",
            );
        }
    }

    if let Some(rec) = &report.recommendation {
        out.push_str("## Recommendation\n\n");
        out.push_str(&format!(
            "**{} at batch={}** — {}\n\n",
            rec.runtime, rec.batch_size, rec.rationale
        ));
    }

    if !report.skipped.is_empty() {
        out.push_str("## Skipped runtimes\n\n");
        for s in &report.skipped {
            out.push_str(&format!("- **{}**: {}\n", s.name, s.reason));
        }
        out.push('\n');
    }

    out
}
