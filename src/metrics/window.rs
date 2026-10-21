//! Sliding window of recent ticks that all analysis runs over.

use std::collections::VecDeque;

use crate::collectors::{StepEvent, Tick};

/// Analysis horizon: the last five minutes.
pub const WINDOW_SPAN_MS: u64 = 5 * 60 * 1000;
/// Hard cap on retained ticks (5 min at 0.1 s).
pub const MAX_WINDOW_TICKS: usize = 3000;
/// Ticks (or steps) needed before figures are reported as reliable.
pub const MIN_RELIABLE_SAMPLES: usize = 15;

#[derive(Debug, Default)]
pub struct TickWindow {
    ticks: VecDeque<Tick>,
    span_ms: u64,
    max_len: usize,
}

impl TickWindow {
    pub fn new() -> Self {
        Self::with_limits(WINDOW_SPAN_MS, MAX_WINDOW_TICKS)
    }

    pub fn with_limits(span_ms: u64, max_len: usize) -> Self {
        Self {
            ticks: VecDeque::new(),
            span_ms,
            max_len: max_len.max(1),
        }
    }

    pub fn push(&mut self, tick: Tick) {
        let newest = tick.timestamp_ms;
        self.ticks.push_back(tick);
        while self.ticks.len() > self.max_len {
            self.ticks.pop_front();
        }
        while let Some(front) = self.ticks.front() {
            if newest.saturating_sub(front.timestamp_ms) > self.span_ms && self.ticks.len() > 1 {
                self.ticks.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Tick> {
        self.ticks.iter()
    }

    pub fn latest(&self) -> Option<&Tick> {
        self.ticks.back()
    }

    pub fn len(&self) -> usize {
        self.ticks.len()
    }

    #[allow(dead_code)] // pairs with `len` (clippy::len_without_is_empty)
    pub fn is_empty(&self) -> bool {
        self.ticks.is_empty()
    }

    /// All hook step events currently in the window.
    pub fn steps(&self) -> impl Iterator<Item = &StepEvent> {
        self.ticks.iter().flat_map(|t| t.steps.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::testing::{gpu, tick};

    #[test]
    fn evicts_by_count() {
        let mut w = TickWindow::with_limits(u64::MAX, 3);
        for i in 0..5 {
            w.push(tick(i * 1000, vec![gpu(0, 50.0)], None));
        }
        assert_eq!(w.len(), 3);
        assert_eq!(w.iter().next().unwrap().timestamp_ms, 2000);
    }

    #[test]
    fn evicts_by_age() {
        let mut w = TickWindow::with_limits(10_000, 1000);
        for i in 0..20 {
            w.push(tick(i * 1000, vec![], None));
        }
        assert_eq!(w.iter().next().unwrap().timestamp_ms, 9000);
        assert_eq!(w.latest().unwrap().timestamp_ms, 19000);
    }

    #[test]
    fn reliability_threshold() {
        let mut w = TickWindow::new();
        for i in 0..14 {
            w.push(tick(i, vec![], None));
        }
        assert!(w.len() < MIN_RELIABLE_SAMPLES);
        w.push(tick(14, vec![], None));
        assert!(w.len() >= MIN_RELIABLE_SAMPLES);
    }
}
