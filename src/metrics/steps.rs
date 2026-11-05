//! Training-step statistics from hook events.
//!
//! The first steps of a run are not representative: model load, shader or
//! kernel compilation and allocator growth can make step 1 take 50× longer
//! than step 100. They are excluded, and "erratic" is judged with robust
//! statistics, so one slow step cannot label a steady job erratic.

use serde::Serialize;

use crate::collectors::StepEvent;
use crate::metrics::window::TickWindow;

/// Steps at the start of a run excluded as warm-up (by the hook's step number,
/// so a watcher attaching mid-run keeps everything).
pub const WARMUP_STEPS: u64 = 10;
/// Robust coefficient of variation above which step times are called erratic.
pub const ERRATIC_ROBUST_CV: f64 = 0.3;
/// Share of step time lost to occasional long steps that also counts as erratic.
pub const ERRATIC_STALL_PCT: f64 = 20.0;
const MIN_STEPS_TO_JUDGE: usize = 10;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepStats {
    /// Steps in the analysis window, including warm-up.
    pub steps_in_window: usize,
    /// Timing figures below exclude warm-up steps when others are available.
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    /// Plain coefficient of variation (sensitive to single outliers).
    pub cv: f64,
    /// 1.4826 · MAD / median: the typical spread, ignoring rare outliers.
    pub robust_cv: f64,
    /// Share of step time spent beyond the normal range in occasional long steps.
    pub stall_time_pct: f64,
    pub erratic: bool,
    /// Coefficient of variation of tokens per step, if the hook reports tokens.
    pub tokens_cv: Option<f64>,
    /// Optimizer steps per second summed over ranks.
    pub steps_per_sec: f64,
    pub tokens_per_sec: Option<f64>,
    pub samples_per_sec: Option<f64>,
    /// Distinct processes (ranks) reporting.
    pub ranks: usize,
}

fn median(sorted: &[f64]) -> f64 {
    sorted[(sorted.len() - 1) / 2]
}

pub fn compute(window: &TickWindow) -> Option<StepStats> {
    let all: Vec<&StepEvent> = window.steps().collect();
    if all.is_empty() {
        return None;
    }
    let settled: Vec<&StepEvent> = all
        .iter()
        .copied()
        .filter(|s| s.step > WARMUP_STEPS)
        .collect();
    let judged = !settled.is_empty();
    let steps = if judged { settled } else { all.clone() };

    let mut ms: Vec<f64> = steps.iter().map(|s| s.step_s * 1000.0).collect();
    ms.sort_by(|a, b| a.total_cmp(b));
    let n = ms.len();
    let total: f64 = ms.iter().sum();
    let mean = total / n as f64;
    let var = ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let cv = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
    let q = |p: f64| ms[((n - 1) as f64 * p).round() as usize];

    let med = median(&ms);
    let mut dev: Vec<f64> = ms.iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(|a, b| a.total_cmp(b));
    let sigma = 1.4826 * median(&dev);
    let robust_cv = if med > 0.0 { sigma / med } else { 0.0 };
    // Time beyond a generous normal range (at least 50 % above the median).
    let limit = med + (5.0 * sigma).max(0.5 * med);
    let excess: f64 = ms.iter().map(|v| (v - limit).max(0.0)).sum();
    let stall_time_pct = if total > 0.0 {
        excess / total * 100.0
    } else {
        0.0
    };

    let tokens: Vec<f64> = steps
        .iter()
        .filter_map(|s| s.tokens.map(|t| t as f64))
        .collect();
    let tokens_cv = (!tokens.is_empty()).then(|| {
        let m = tokens.iter().sum::<f64>() / tokens.len() as f64;
        let v = tokens.iter().map(|t| (t - m).powi(2)).sum::<f64>() / tokens.len() as f64;
        if m > 0.0 {
            v.sqrt() / m
        } else {
            0.0
        }
    });

    let mut pids: Vec<u32> = steps.iter().map(|s| s.pid).collect();
    pids.sort_unstable();
    pids.dedup();

    // Per-rank throughput, summed: Σ_rank (count ÷ busy time).
    let rate = |f: &dyn Fn(&StepEvent) -> Option<f64>| -> Option<f64> {
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
        steps_in_window: all.len(),
        mean_ms: mean,
        p50_ms: q(0.5),
        p95_ms: q(0.95),
        cv,
        robust_cv,
        stall_time_pct,
        erratic: judged
            && n >= MIN_STEPS_TO_JUDGE
            && (robust_cv > ERRATIC_ROBUST_CV || stall_time_pct > ERRATIC_STALL_PCT),
        tokens_cv,
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

    fn ev(pid: u32, step: u64, step_s: f64, tokens: Option<u64>) -> StepEvent {
        StepEvent {
            pid,
            rank: None,
            step,
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

    fn stats(durations: impl Iterator<Item = (u64, f64)>) -> StepStats {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![], None);
        for (i, d) in durations {
            t.steps.push(ev(1, i, d, Some(100)));
        }
        w.push(t);
        compute(&w).unwrap()
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
        for i in 0..10 {
            t.steps.push(ev(1, 11 + i, 0.5, Some(1000)));
            t.steps.push(ev(2, 11 + i, 0.5, Some(1000)));
        }
        w.push(t);
        let s = compute(&w).unwrap();
        assert_eq!(s.ranks, 2);
        assert!((s.mean_ms - 500.0).abs() < 1e-9);
        assert!(!s.erratic);
        assert!((s.steps_per_sec - 4.0).abs() < 1e-9);
        assert!((s.tokens_per_sec.unwrap() - 4000.0).abs() < 1e-6);
        assert_eq!(s.tokens_cv, Some(0.0));
    }

    #[test]
    fn slow_warm_up_is_not_erratic() {
        // Regression (real trace): a 2.2 s first step before steady 45 ms steps
        // gave CV 1.28 and an "erratic" finding with fixed-length input.
        let s = stats((1..=300).map(|i| (i, if i == 1 { 2.2 } else { 0.045 })));
        assert!(!s.erratic, "{s:?}");
        assert!(s.cv < 0.01, "warm-up is excluded: {}", s.cv);
        assert!((s.mean_ms - 45.0).abs() < 1e-6);
    }

    #[test]
    fn one_slow_step_after_warm_up_is_not_erratic() {
        let s = stats((11..=300).map(|i| (i, if i == 150 { 3.0 } else { 0.3 })));
        assert!(!s.erratic, "{s:?}");
        assert!(s.cv > 0.3, "the plain CV still sees it");
    }

    #[test]
    fn genuinely_variable_steps_are_erratic() {
        let s = stats((11..=40).map(|i| (i, if i % 2 == 0 { 0.1 } else { 1.0 })));
        assert!(s.erratic, "{s:?}");
        assert!(s.p95_ms >= s.p50_ms);
    }

    #[test]
    fn periodic_stalls_are_erratic() {
        // A 6 s checkpoint every 10th step is most of the wall time.
        let s = stats((11..=110).map(|i| (i, if i % 10 == 0 { 6.0 } else { 0.3 })));
        assert!(s.robust_cv < 0.1, "typical steps are steady");
        assert!(s.stall_time_pct > 20.0, "{}", s.stall_time_pct);
        assert!(s.erratic);
    }

    #[test]
    fn warm_up_only_cannot_be_judged() {
        let s = stats((1..=10).map(|i| (i, 0.1 * i as f64)));
        assert!(!s.erratic);
        assert_eq!(s.steps_in_window, 10);
    }

    #[test]
    fn varying_tokens_are_measured() {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![], None);
        for i in 0..20u64 {
            t.steps.push(ev(
                1,
                11 + i,
                0.5,
                Some(if i % 2 == 0 { 500 } else { 1500 }),
            ));
        }
        w.push(t);
        assert!(compute(&w).unwrap().tokens_cv.unwrap() > 0.4);
    }
}
