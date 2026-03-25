use crate::metrics::window::MetricsWindow;

/// How time is distributed across the five observable training-loop phases.
///
/// All fields sum to 100%.  Values are inferred from the correlation between
/// GPU SM utilization and CPU utilization — see the architecture notes for
/// the reasoning behind each threshold.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimeBreakdown {
    /// Time the GPU was doing forward/backward pass math.
    pub forward_backward_pct: f32,
    /// Time the GPU was idle waiting for the data loader to deliver a batch.
    pub data_loader_pct: f32,
    /// Time lost to CUDA synchronization (`.item()` calls, scalar logging, etc.).
    pub cuda_sync_pct: f32,
    /// Time lost to memory allocation/deallocation inside the training loop.
    pub memory_alloc_pct: f32,
    /// Time in the optimizer step.
    pub optimizer_pct: f32,
}

/// Classification of a single raw sample into a training-loop phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeSlot {
    ForwardBackward,
    DataLoader,
    CudaSync,
    MemoryAlloc,
    Optimizer,
}

/// Classifies samples in the window and aggregates percentages.
///
/// The classification is *correlation-based*, not direct instrumentation.
/// See [`classify_sample`] for the threshold logic.
pub struct TimeBreakdownInferrer;

impl TimeBreakdownInferrer {
    pub fn infer(window: &MetricsWindow) -> TimeBreakdown {
        if window.is_empty() {
            return TimeBreakdown {
                forward_backward_pct: 0.0,
                data_loader_pct: 0.0,
                cuda_sync_pct: 0.0,
                memory_alloc_pct: 0.0,
                optimizer_pct: 0.0,
            };
        }

        let mut counts = [0usize; 5]; // indexed by TimeSlot discriminant
        let total = window.len();

        for sample in window.iter() {
            let slot = classify_sample(
                sample.sm_utilization.0,
                sample.cpu_utilization.0,
                sample.mem_utilization.0,
            );
            counts[slot as usize] += 1;
        }

        let pct = |n: usize| (n as f32 / total as f32) * 100.0;

        TimeBreakdown {
            forward_backward_pct: pct(counts[TimeSlot::ForwardBackward as usize]),
            data_loader_pct: pct(counts[TimeSlot::DataLoader as usize]),
            cuda_sync_pct: pct(counts[TimeSlot::CudaSync as usize]),
            memory_alloc_pct: pct(counts[TimeSlot::MemoryAlloc as usize]),
            optimizer_pct: pct(counts[TimeSlot::Optimizer as usize]),
        }
    }
}

/// Classify one sample into a time slot using observable proxy metrics.
///
/// Priority order matters — a sample can only be in one slot.
///
/// | SM util | CPU %  | Classification             |
/// |---------|--------|----------------------------|
/// | >60%    | any    | forward/backward           |
/// | <15%    | >30%   | data loader waiting        |
/// | <15%    | <10%   | CUDA sync overhead         |
/// | any     | any, high mem churn | memory alloc |
/// | 15-60%  | low    | optimizer step             |
fn classify_sample(sm_util: f32, cpu_pct: f32, mem_util: f32) -> TimeSlot {
    if sm_util > 60.0 {
        return TimeSlot::ForwardBackward;
    }
    // High memory controller utilization with mid-range SM → memory-bound op
    // (common during gradient checkpointing or large embedding lookups).
    if mem_util > 80.0 && sm_util < 60.0 {
        return TimeSlot::MemoryAlloc;
    }
    if sm_util < 15.0 && cpu_pct > 30.0 {
        return TimeSlot::DataLoader;
    }
    if sm_util < 15.0 && cpu_pct < 10.0 {
        return TimeSlot::CudaSync;
    }
    // Residual: mid-range SM + low CPU → optimizer step
    TimeSlot::Optimizer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::RawSample;
    use crate::metrics::units::{Celsius, Mhz, Mib, Percent, Watts};
    use crate::metrics::window::MetricsWindow;

    fn sample(sm: f32, cpu: f32, mem_util: f32) -> RawSample {
        RawSample {
            timestamp_ms: 0,
            gpu_index: 0,
            sm_utilization: Percent(sm),
            sm_clock_mhz: Mhz(1800),
            sm_clock_max_mhz: Mhz(2000),
            vram_used_mib: Mib(8192),
            vram_total_mib: Mib(24576),
            mem_utilization: Percent(mem_util),
            temperature: Celsius(65.0),
            power_draw: Watts(200.0),
            power_limit: Watts(350.0),
            throttle_thermal: false,
            throttle_power: false,
            throttle_hw_slowdown: false,
            cpu_utilization: Percent(cpu),
        }
    }

    #[test]
    fn mostly_data_loader() {
        let mut w = MetricsWindow::new(150);
        // 70% data loader, 30% forward pass
        for _ in 0..7 {
            w.push(sample(5.0, 60.0, 20.0));
        }
        for _ in 0..3 {
            w.push(sample(80.0, 20.0, 50.0));
        }
        let bd = TimeBreakdownInferrer::infer(&w);
        assert!(
            bd.data_loader_pct > 60.0,
            "expected >60% data loader, got {:.1}%",
            bd.data_loader_pct
        );
        assert!(bd.forward_backward_pct > 20.0);
    }

    #[test]
    fn classify_forward_backward() {
        assert_eq!(classify_sample(75.0, 20.0, 50.0), TimeSlot::ForwardBackward);
    }

    #[test]
    fn classify_cuda_sync() {
        assert_eq!(classify_sample(5.0, 5.0, 10.0), TimeSlot::CudaSync);
    }

    #[test]
    fn classify_optimizer_step() {
        // Mid-range SM with low CPU and normal mem → optimizer.
        assert_eq!(classify_sample(35.0, 15.0, 30.0), TimeSlot::Optimizer);
    }

    #[test]
    fn classify_memory_alloc() {
        // High mem controller utilization with lower SM → memory alloc.
        assert_eq!(classify_sample(40.0, 20.0, 90.0), TimeSlot::MemoryAlloc);
    }

    #[test]
    fn all_forward_backward_gives_100pct() {
        let mut w = MetricsWindow::new(150);
        for _ in 0..20 {
            w.push(sample(80.0, 20.0, 50.0)); // all ForwardBackward
        }
        let bd = TimeBreakdownInferrer::infer(&w);
        assert!(
            (bd.forward_backward_pct - 100.0).abs() < 0.1,
            "expected 100% forward/backward, got {:.1}%",
            bd.forward_backward_pct
        );
        assert_eq!(bd.data_loader_pct, 0.0);
        assert_eq!(bd.cuda_sync_pct, 0.0);
    }

    #[test]
    fn breakdown_percentages_sum_to_100() {
        let mut w = MetricsWindow::new(150);
        // Mixed workload touching every slot.
        for _ in 0..4 {
            w.push(sample(80.0, 20.0, 50.0));
        } // ForwardBackward
        for _ in 0..3 {
            w.push(sample(5.0, 70.0, 20.0));
        } // DataLoader
        for _ in 0..2 {
            w.push(sample(5.0, 5.0, 10.0));
        } // CudaSync
        for _ in 0..1 {
            w.push(sample(40.0, 10.0, 85.0));
        } // MemoryAlloc
        let bd = TimeBreakdownInferrer::infer(&w);
        let sum = bd.forward_backward_pct
            + bd.data_loader_pct
            + bd.cuda_sync_pct
            + bd.memory_alloc_pct
            + bd.optimizer_pct;
        assert!(
            (sum - 100.0).abs() < 0.01,
            "percentages should sum to 100, got {sum:.2}"
        );
    }

    #[test]
    fn empty_window_returns_all_zeros() {
        let w = MetricsWindow::new(150);
        let bd = TimeBreakdownInferrer::infer(&w);
        assert_eq!(bd.forward_backward_pct, 0.0);
        assert_eq!(bd.data_loader_pct, 0.0);
        assert_eq!(bd.cuda_sync_pct, 0.0);
        assert_eq!(bd.memory_alloc_pct, 0.0);
        assert_eq!(bd.optimizer_pct, 0.0);
    }
}
