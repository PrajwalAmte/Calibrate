use crate::output::OutputRenderer;
use crate::session::state::SessionSnapshot;

/// Prints a final summary report to stdout after the session ends.
///
/// This is invoked by both renderers (terminal and JSON) at the end of a run.
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
        println!("  GPU              : {}", snapshot.gpu_name);
        println!("  Duration         : {:02}:{:02}:{:02}", h, m, s);
        println!("  Steps observed   : {}", snapshot.steps_observed);
        println!();
        println!("  MFU              : {:.1}%  (target: >45%)",
            snapshot.mfu.mfu_pct.0);
        println!("  Peak VRAM        : {} / {}",
            snapshot.vram_used_mib,
            snapshot.vram_total_mib);
        println!("  Temperature      : {}", snapshot.temperature);
        println!();

        if snapshot.elapsed.as_secs() < 30 {
            println!("  ⚠  Run was under 30 seconds — MFU estimate is approximate.");
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
