use crate::metrics::units::{fmt_hms, fmt_mib, fmt_opt, fmt_pct, fmt_temp, fmt_usd};
use crate::session::state::SessionSnapshot;

fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(100)
}

/// Wrap `text` on word boundaries; continuation lines are indented by `indent`.
pub fn wrap_words(text: &str, width: usize, indent: usize) -> String {
    let room = width.saturating_sub(indent).max(20);
    let mut out = String::new();
    let mut col = 0;
    for word in text.split_whitespace() {
        let len = word.chars().count();
        if col > 0 && col + 1 + len > room {
            out.push('\n');
            out.push_str(&" ".repeat(indent));
            col = 0;
        } else if col > 0 {
            out.push(' ');
            col += 1;
        }
        out.push_str(word);
        col += len;
    }
    out
}

/// End-of-session summary printed after the TUI exits.
pub struct SummaryReport;

impl SummaryReport {
    pub fn render(s: &SessionSnapshot) -> String {
        let mut out = String::new();
        let mut line = |l: String| {
            out.push_str(&l);
            out.push('\n');
        };
        line(String::new());
        line("calibrate watch — session summary".into());
        line(format!("  GPU              : {}", s.gpu_name));
        line(format!(
            "  Duration         : {}   ({} samples, GPU counters: {})",
            fmt_hms(s.elapsed_secs),
            s.ticks_observed,
            s.metrics_source
        ));
        let bound = if s.mfu.upper_bound { "≤ " } else { "" };
        line(format!(
            "  {:<17}: {}{}  (target ≥{:.0}%)",
            s.mfu.source.label(),
            bound,
            fmt_pct(s.mfu.mfu_pct),
            s.target_mfu_pct
        ));
        if let Some(p) = &s.mfu_percentiles {
            line(format!(
                "  p50 / p75 / p95  : {:.1}% / {:.1}% / {:.1}%   (peak {})",
                p.p50,
                p.p75,
                p.p95,
                fmt_pct(s.peak_mfu_pct)
            ));
        }
        if let Some(st) = &s.steps {
            line(format!(
                "  Step time        : {:.0} ms mean, {:.0} ms p95{}",
                st.mean_ms,
                st.p95_ms,
                if st.erratic { "  (erratic)" } else { "" }
            ));
            if let Some(t) = st.tokens_per_sec {
                line(format!("  Throughput       : {t:.0} tokens/s"));
            }
        }
        for g in &s.gpus {
            line(format!(
                "  GPU{:<2} peak VRAM  : {} / {}   temp {}",
                g.index,
                fmt_opt(g.peak_vram_mib, fmt_mib),
                fmt_opt(g.vram_total_mib, fmt_mib),
                fmt_temp(g.temperature_c)
            ));
        }
        if s.elapsed_secs < 30.0 {
            line("  ⚠  Session under 30 s — figures are approximate.".into());
        }
        if s.gpu_divergent {
            line(format!(
                "  ⚠  GPU utilisation diverged by {:.0} ppt across devices.",
                s.gpu_divergence_ppt.unwrap_or(0.0)
            ));
        }
        if s.vram_growing {
            line("  ⚠  GPU memory grew steadily — check for tensors kept across steps.".into());
        }
        for w in &s.warnings {
            line(format!("  ⚠  {w}"));
        }
        line(format!("  Primary finding  : {}", s.recommendation.title));
        line(format!(
            "  Recommendation   : {}",
            wrap_words(&s.recommendation.action, terminal_width(), 21)
        ));
        if let Some(c) = &s.cost {
            let lb = if c.is_lower_bound { "≥ " } else { "" };
            line(format!(
                "  Cost             : {} spent, {lb}{} below target ({lb}{}/hr at ${:.2}/hr)",
                fmt_usd(c.spent_usd),
                fmt_usd(c.wasted_usd),
                fmt_usd(c.waste_per_hour),
                c.cost_per_hour
            ));
        }
        out
    }

    pub fn print(s: &SessionSnapshot) {
        print!("{}", Self::render(s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_on_words_with_indent() {
        let w = wrap_words("periodic evaluation inside the loop", 30, 21);
        assert_eq!(w.lines().next().unwrap().trim_end(), "periodic evaluation");
        assert!(w.contains("\n                     inside"), "{w}");
        for l in w.lines() {
            assert!(!l.trim().is_empty());
        }
        assert_eq!(wrap_words("one two", 100, 21), "one two");
    }
}
