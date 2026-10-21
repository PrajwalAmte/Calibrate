//! Where training time goes.
//!
//! * **Instrumented** (Python hook present): exact shares of step wall time
//!   spent waiting for data, in forward/backward, and in the optimizer.
//! * **Heuristic** (no hook): each GPU sample is classified from coarse
//!   signals — GPU busy, process-tree CPU, memory-controller load. Sampling is
//!   much slower than a training step, so this shows *tendencies* (e.g. "the
//!   GPU is idle while the CPU is busy"), not per-phase timings.
//! * **Unavailable**: required signals are missing (no GPU data, or no CPU
//!   data in device mode).

use serde::Serialize;

use crate::metrics::window::TickWindow;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakdownSource {
    Instrumented,
    Heuristic,
    Unavailable,
}

/// Percentages summing to 100 (or all zero when unavailable).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimeBreakdown {
    pub source: BreakdownSource,
    /// Forward/backward (instrumented) or GPU busy > 60 % (heuristic).
    pub compute_pct: f32,
    /// Waiting for the next batch (instrumented) or GPU idle while the CPU
    /// is busy (heuristic: likely input pipeline).
    pub data_wait_pct: f32,
    /// GPU idle while the CPU is also idle (heuristic only): host-device
    /// synchronisation, logging, checkpointing, collective waits.
    pub sync_stall_pct: f32,
    /// Memory controller saturated while compute is low (heuristic only).
    pub memory_bound_pct: f32,
    /// Optimizer step (instrumented only).
    pub optimizer_pct: f32,
    /// Everything else: partial GPU use (heuristic) or unaccounted step time.
    pub other_pct: f32,
}

impl TimeBreakdown {
    pub fn unavailable() -> Self {
        Self {
            source: BreakdownSource::Unavailable,
            compute_pct: 0.0,
            data_wait_pct: 0.0,
            sync_stall_pct: 0.0,
            memory_bound_pct: 0.0,
            optimizer_pct: 0.0,
            other_pct: 0.0,
        }
    }
}

pub fn infer(window: &TickWindow) -> TimeBreakdown {
    instrumented(window).unwrap_or_else(|| heuristic(window))
}

fn instrumented(window: &TickWindow) -> Option<TimeBreakdown> {
    let steps: Vec<_> = window.steps().collect();
    if steps.len() < 3 || steps.iter().all(|s| s.data_s.is_none()) {
        return None;
    }
    let (mut total, mut data, mut compute, mut opt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for s in &steps {
        let d = s.data_s.unwrap_or(0.0).clamp(0.0, s.step_s);
        total += s.step_s;
        data += d;
        match (s.compute_s, s.optimizer_s) {
            (Some(c), o) => {
                compute += c.max(0.0);
                opt += o.unwrap_or(0.0).max(0.0);
            }
            (None, Some(o)) => opt += o.max(0.0),
            (None, None) => compute += (s.step_s - d).max(0.0),
        }
    }
    if total <= 0.0 {
        return None;
    }
    let pct = |v: f64| ((v / total) * 100.0).clamp(0.0, 100.0) as f32;
    let data_pct = pct(data);
    let compute_pct = pct(compute);
    let opt_pct = pct(opt);
    let other = (100.0 - data_pct - compute_pct - opt_pct).max(0.0);
    Some(normalize(TimeBreakdown {
        source: BreakdownSource::Instrumented,
        compute_pct,
        data_wait_pct: data_pct,
        sync_stall_pct: 0.0,
        memory_bound_pct: 0.0,
        optimizer_pct: opt_pct,
        other_pct: other,
    }))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Compute,
    DataWait,
    SyncStall,
    MemoryBound,
    Other,
}

/// Classify one GPU observation. Thresholds are deliberately coarse.
fn classify(busy: f32, cpu: f32, mem: Option<f32>) -> Slot {
    if busy > 60.0 {
        return Slot::Compute;
    }
    if mem.is_some_and(|m| m > 80.0) {
        return Slot::MemoryBound;
    }
    if busy < 15.0 && cpu > 30.0 {
        return Slot::DataWait;
    }
    if busy < 15.0 && cpu < 10.0 {
        return Slot::SyncStall;
    }
    Slot::Other
}

fn heuristic(window: &TickWindow) -> TimeBreakdown {
    let mut counts = [0usize; 5];
    let mut n = 0usize;
    for t in window.iter() {
        let Some(cpu) = t.cpu_pct else { continue };
        for g in &t.gpus {
            let Some(busy) = g.busy_pct() else { continue };
            let mem = g.prof.and_then(|p| p.dram_active_pct).or(g.mem_util_pct);
            counts[classify(busy, cpu, mem) as usize] += 1;
            n += 1;
        }
    }
    if n == 0 {
        return TimeBreakdown::unavailable();
    }
    let pct = |s: Slot| counts[s as usize] as f32 / n as f32 * 100.0;
    TimeBreakdown {
        source: BreakdownSource::Heuristic,
        compute_pct: pct(Slot::Compute),
        data_wait_pct: pct(Slot::DataWait),
        sync_stall_pct: pct(Slot::SyncStall),
        memory_bound_pct: pct(Slot::MemoryBound),
        optimizer_pct: 0.0,
        other_pct: pct(Slot::Other),
    }
}

/// Rescale so the parts sum to exactly 100 (phase timings can overlap).
fn normalize(mut b: TimeBreakdown) -> TimeBreakdown {
    let sum = b.compute_pct
        + b.data_wait_pct
        + b.sync_stall_pct
        + b.memory_bound_pct
        + b.optimizer_pct
        + b.other_pct;
    if sum > 0.0 && (sum - 100.0).abs() > 0.01 {
        let k = 100.0 / sum;
        b.compute_pct *= k;
        b.data_wait_pct *= k;
        b.sync_stall_pct *= k;
        b.memory_bound_pct *= k;
        b.optimizer_pct *= k;
        b.other_pct *= k;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::testing::{gpu, tick};
    use crate::collectors::StepEvent;

    fn sum(b: &TimeBreakdown) -> f32 {
        b.compute_pct
            + b.data_wait_pct
            + b.sync_stall_pct
            + b.memory_bound_pct
            + b.optimizer_pct
            + b.other_pct
    }

    fn sample(busy: f32, cpu: f32, mem: f32) -> crate::collectors::Tick {
        let mut g = gpu(0, busy);
        g.mem_util_pct = Some(mem);
        tick(0, vec![g], Some(cpu))
    }

    #[test]
    fn classification_table() {
        assert_eq!(classify(75.0, 20.0, Some(50.0)), Slot::Compute);
        assert_eq!(classify(5.0, 60.0, Some(10.0)), Slot::DataWait);
        assert_eq!(classify(5.0, 5.0, Some(10.0)), Slot::SyncStall);
        assert_eq!(classify(40.0, 20.0, Some(90.0)), Slot::MemoryBound);
        assert_eq!(classify(35.0, 15.0, Some(30.0)), Slot::Other);
        assert_eq!(classify(35.0, 15.0, None), Slot::Other);
    }

    #[test]
    fn heuristic_mostly_data_loader() {
        let mut w = TickWindow::new();
        for _ in 0..7 {
            w.push(sample(5.0, 250.0, 20.0));
        }
        for _ in 0..3 {
            w.push(sample(85.0, 120.0, 50.0));
        }
        let b = infer(&w);
        assert_eq!(b.source, BreakdownSource::Heuristic);
        assert!((b.data_wait_pct - 70.0).abs() < 0.1);
        assert!((b.compute_pct - 30.0).abs() < 0.1);
        assert!((sum(&b) - 100.0).abs() < 0.01);
    }

    #[test]
    fn no_cpu_data_means_unavailable_not_cuda_sync() {
        // Regression: macOS used to report CPU = 0 and therefore "100 % CUDA sync".
        let mut w = TickWindow::new();
        for _ in 0..10 {
            w.push(tick(0, vec![gpu(0, 2.0)], None));
        }
        assert_eq!(infer(&w).source, BreakdownSource::Unavailable);
    }

    #[test]
    fn no_gpu_data_is_unavailable() {
        let mut w = TickWindow::new();
        w.push(tick(0, vec![], Some(100.0)));
        assert_eq!(infer(&w).source, BreakdownSource::Unavailable);
    }

    fn step(i: u64, total: f64, data: f64, compute: Option<f64>, opt: Option<f64>) -> StepEvent {
        StepEvent {
            pid: 1,
            rank: None,
            step: i,
            t_end_ms: i,
            step_s: total,
            data_s: Some(data),
            compute_s: compute,
            optimizer_s: opt,
            tokens: None,
            samples: None,
            flops: None,
        }
    }

    #[test]
    fn instrumented_uses_real_step_timings() {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![gpu(0, 99.0)], Some(100.0));
        for i in 0..4 {
            t.steps.push(step(i, 1.0, 0.25, Some(0.6), Some(0.1)));
        }
        w.push(t);
        let b = infer(&w);
        assert_eq!(b.source, BreakdownSource::Instrumented);
        assert!((b.data_wait_pct - 25.0).abs() < 0.01);
        assert!((b.compute_pct - 60.0).abs() < 0.01);
        assert!((b.optimizer_pct - 10.0).abs() < 0.01);
        assert!((b.other_pct - 5.0).abs() < 0.01);
    }

    #[test]
    fn instrumented_without_phase_split_attributes_rest_to_compute() {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![], None);
        for i in 0..3 {
            t.steps.push(step(i, 2.0, 0.5, None, None));
        }
        w.push(t);
        let b = infer(&w);
        assert_eq!(b.source, BreakdownSource::Instrumented);
        assert!((b.compute_pct - 75.0).abs() < 0.01);
        assert!((sum(&b) - 100.0).abs() < 0.01);
    }

    #[test]
    fn overlapping_phase_timings_are_normalised() {
        let mut w = TickWindow::new();
        let mut t = tick(0, vec![], None);
        for i in 0..3 {
            t.steps.push(step(i, 1.0, 0.3, Some(0.9), Some(0.2)));
        }
        w.push(t);
        assert!((sum(&infer(&w)) - 100.0).abs() < 0.01);
    }
}
