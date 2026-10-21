//! Training-step statistics from hook events.

use serde::Serialize;

use crate::metrics::window::TickWindow;

/// Coefficient of variation above which step times are called erratic.
pub const ERRATIC_CV: f64 = 0.3;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepStats {
    /// Steps in the analysis window.
    pub steps_in_window: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub cv: f64,
    pub erratic: bool,
    /// Optimizer steps per second summed over ranks.
    pub steps_per_sec: f64,
    pub tokens_per_sec: Option<f64>,
    pub samples_per_sec: Option<f64>,
    /// Distinct processes (ranks) reporting.
    pub ranks: usize,
}

pub fn compute(window: &TickWindow) -> Option<StepStats> {
    let steps: Vec<_> = window.steps().collect();
    if steps.is_empty() {
        return None;
    }
    let mut ms: Vec<f64> = steps.iter().map(|s| s.step_s * 1000.0).collect();
    ms.sort_by(|a, b| a.total_cmp(b));
    let n = ms.len();
    let mean = ms.iter().sum::<f64>() / n as f64;
    let var = ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let cv = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
    let q = |p: f64| ms[((n - 1) as f64 * p).round() as usize];

    let mut pids: Vec<u32> = steps.iter().map(|s| s.pid).collect();
    pids.sort_unstable();
    pids.dedup();

    // Per-rank throughput, summed: Σ_rank (count ÷ busy time).
    let rate = |f: &dyn Fn(&crate::collectors::StepEvent) -> Option<f64>| -> Option<f64> {
        let mut total = 0.0;
        let mut any = false;
        for pid in &pids {
            let mine: Vec<_> = steps.iter().filter(|s| s.pid == *pid).collect();
            let time: f64 = mine.iter().map(|s| s.step_s).sum();
            let vals: Vec<f64> = mine.iter().filter_map(|s| f(s)).collect();
            if time > 0.0 && !vals.is_empty() {
                total += vals.iter().sum::<f64>() / time;
                any = true;
            }
        }
        any.then_some(total)
    };

    Some(StepStats {
        steps_in_window: n,
        mean_ms: mean,
        p50_ms: q(0.5),
        p95_ms: q(0.95),
        cv,
        erratic: n >= 10 && cv > ERRATIC_CV,
        steps_per_sec: rate(&|_| Some(1.0)).unwrap_or(0.0),
        tokens_per_sec: rate(&|s| s.tokens.map(|t| t as f64)),
        samples_per_sec: rate(&|s| s.samples.map(|t| t as f64)),
        ranks: pids.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::testing::tick;
    use crate::collectors::StepEvent;

    fn ev(pid: u32, step_s: f64, tokens: Option<u64>) -> StepEvent {
        StepEvent {
            pid,
            rank: None,
            step: 0,
            t_end_ms: 0,
            step_s,
            data_s: None,
            compute_s: None,
            optimizer_s: None,
            tokens,
            samples: None,
            flops: None,
        }
    }

    #[test]
    fn none_without_steps() {
        let mut w = TickWindow::new();
        w.push(tick(0, vec![], None));
        assert!(compute(&w).is_none());
    }

    #[test]
    fn stats_and_throughput() {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![], None);
        for _ in 0..10 {
            t.steps.push(ev(1, 0.5, Some(1000)));
            t.steps.push(ev(2, 0.5, Some(1000)));
        }
        w.push(t);
        let s = compute(&w).unwrap();
        assert_eq!(s.ranks, 2);
        assert!((s.mean_ms - 500.0).abs() < 1e-9);
        assert!(!s.erratic);
        assert!((s.steps_per_sec - 4.0).abs() < 1e-9);
        assert!((s.tokens_per_sec.unwrap() - 4000.0).abs() < 1e-6);
    }

    #[test]
    fn erratic_steps_are_flagged() {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![], None);
        for i in 0..12 {
            t.steps
                .push(ev(1, if i % 2 == 0 { 0.1 } else { 1.0 }, None));
        }
        w.push(t);
        let s = compute(&w).unwrap();
        assert!(s.erratic, "cv={}", s.cv);
        assert!(s.p95_ms >= s.p50_ms);
    }
}
