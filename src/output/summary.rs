use crate::output::OutputRenderer;
use crate::session::state::SessionSnapshot;

/// Prints a final session summary to stdout after the run ends.
pub struct SummaryReport;

impl SummaryReport {
    pub fn print(snapshot: &SessionSnapshot) {
        let elapsed = snapshot.elapsed;
        let h = elapsed.as_secs() / 3600;
        let m = (elapsed.as_secs() % 3600) / 60;
        let s = elapsed.as_secs() % 60;

        println!();
        println!("═══════════════════════════════════════════════════");
        println!("  calibrate watch — Session Summary");
        println!("═══════════════════════════════════════════════════");

        if !snapshot.nvml_available {
            println!("  ⚠  NVML unavailable — GPU metrics were not collected.");
            println!("     CPU utilisation is shown; MFU is not available.");
            println!();
        }

        println!("  GPU              : {}", snapshot.gpu_name);
        println!("  Duration         : {:02}:{:02}:{:02}", h, m, s);
        println!("  Steps observed   : {}", snapshot.steps_observed);
        println!();

        if snapshot.nvml_available {
            println!(
                "  MFU              : {:.1}%  (target: >45%)",
                snapshot.mfu.mfu_pct.0
            );
            println!("  Peak MFU         : {:.1}%", snapshot.peak_mfu_pct);
            if let Some(ref p) = snapshot.mfu_percentiles {
                println!(
                    "  MFU p50/p75/p95  : {:.1}% / {:.1}% / {:.1}%",
                    p.p50, p.p75, p.p95
                );
            }
            println!(
                "  Peak VRAM        : {} / {}",
                snapshot.peak_vram_mib, snapshot.vram_total_mib
            );
        }

        println!("  Temperature      : {}", snapshot.temperature);
        println!();

        if elapsed.as_secs() < 30 {
            println!("  ⚠  Run was under 30 seconds — MFU estimate is approximate.");
            println!();
        }

        if snapshot.mfu_divergent {
            println!(
                "  ⚠  GPU MFU divergence: {:.0} ppt spread across devices.",
                snapshot.gpu_mfu_divergence_ppt
            );
            println!("     Check NCCL/DDP configuration and per-card thermal limits.");
            println!();
        }

        if snapshot.vram_growing {
            println!("  ⚠  VRAM was growing every tick — possible memory leak in training loop.");
            println!("     Check for tensors accumulating outside torch.no_grad() or in lists.");
            println!();
        }

        if snapshot.step_time_ms_mean > 0.0 {
            let erratic_note = if snapshot.step_time_erratic {
                "  ← ERRATIC (CV > 0.3 — check data loader or memory allocation)"
            } else {
                ""
            };
            println!(
                "  Step time (mean)   : {:.0} ms{}",
                snapshot.step_time_ms_mean, erratic_note
            );
            println!();
        }

        if snapshot.per_gpu.len() > 1 {
            println!("  Per-GPU breakdown:");
            for g in &snapshot.per_gpu {
                println!(
                    "    GPU{}: SM {:.1}%  VRAM {}  Temp {}",
                    g.gpu_index, g.sm_utilization.0, g.vram_used_mib, g.temperature
                );
            }
            println!();
        }

        println!("  Primary bottleneck : {:?}", snapshot.bottleneck);
        println!("  Recommendation     : {}", snapshot.recommendation.action);
        println!();

        if let Some(ref cost) = snapshot.cost_impact {
            println!(
                "  Waste per hour   : ${:.2}  (at ${:.2}/hr)",
                cost.waste_per_hour, cost.cost_per_hour
            );
            println!();
        }

        println!("═══════════════════════════════════════════════════");
    }
}

impl OutputRenderer for SummaryReport {
    fn render(&mut self, _snapshot: &SessionSnapshot) {
        // Summary only outputs at the end.
    }

    fn finish(&mut self, snapshot: Option<&SessionSnapshot>) {
        if let Some(s) = snapshot {
            Self::print(s);
        }
    }
}
