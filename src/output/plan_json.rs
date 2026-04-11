use anyhow::Result;

use crate::plan::PlanReport;

/// Serialize the plan report to stdout as pretty-printed JSON.
pub fn render(report: &PlanReport) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{
        AvailabilityStatus, CostRange, DurationRange, PlanReport, RankedListing, SkippedProvider,
        VramBreakdown, WorkloadSummary,
    };
    use chrono::Utc;

    fn make_minimal_report() -> PlanReport {
        PlanReport {
            workload: WorkloadSummary {
                model_id: "test/model".to_string(),
                param_count_b: 7.0,
                vram_breakdown: VramBreakdown {
                    weights_gib: 13.0,
                    gradients_gib: 0.13,
                    optimizer_gib: 2.6,
                    activations_gib: 0.5,
                    kv_cache_gib: 0.4,
                    library_savings_gib: 0.0,
                    total_gib: 16.63,
                },
                required_vram_gib: 17.46,
                fitting_tiers: vec!["24G".to_string()],
            },
            listings: vec![RankedListing {
                provider: "RunPod".to_string(),
                gpu_model: "RTX 4090".to_string(),
                vram_gib: 24.0,
                hourly_usd: 0.44,
                duration_range: Some(DurationRange { low_secs: 3600.0, high_secs: 5400.0 }),
                cost_range: Some(CostRange { low_usd: 0.44, high_usd: 0.66 }),
                availability: AvailabilityStatus::Available,
                flags: vec![],
            }],
            recommendation: None,
            skipped_providers: vec![SkippedProvider {
                name: "Lambda".to_string(),
                reason: "API timeout".to_string(),
            }],
            generated_at: Utc::now(),
        }
    }

    /// The output must be valid JSON that can be round-tripped back to a PlanReport.
    #[test]
    fn json_output_round_trips() {
        let report = make_minimal_report();
        let json = serde_json::to_string_pretty(&report).unwrap();

        // Deserialize back and verify key fields survive the round-trip.
        let back: PlanReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.workload.model_id, report.workload.model_id);
        assert_eq!(back.listings.len(), 1);
        assert_eq!(back.listings[0].provider, "RunPod");
        assert_eq!(back.skipped_providers[0].name, "Lambda");
    }

    /// Rendering should not panic with an empty listings list.
    #[test]
    fn render_empty_report_no_panic() {
        let mut report = make_minimal_report();
        report.listings.clear();
        report.skipped_providers.clear();
        // render writes to stdout; we just verify it doesn't error.
        render(&report).unwrap();
    }

    /// JSON output contains the provider name as a string.
    #[test]
    fn json_contains_provider_name() {
        let report = make_minimal_report();
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("RunPod"));
    }
}
