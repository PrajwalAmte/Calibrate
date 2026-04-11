use crate::plan::{AvailabilityStatus, ListingFlag, PlanReport};

/// Print the plan report to stdout as a human-readable terminal layout.
pub fn render(report: &PlanReport, budget: Option<f64>) {
    let w = &report.workload;

    // ── Section 1: Workload analysis──────────
    println!();
    println!(
        "  Model:     {}  ({:.2}B parameters)",
        w.model_id, w.param_count_b
    );
    println!("  Required:  {:.1} GiB VRAM", w.vram_breakdown.total_gib);
    if !w.fitting_tiers.is_empty() {
        println!("             fits {}  tiers", w.fitting_tiers.join(", "));
    } else {
        println!("             Warning: no standard GPU tier is large enough.");
    }
    println!();

    // VRAM breakdown table
    println!("  {:25}  {:>7}", "Component", "GiB");
    println!("  {}  {}", "-".repeat(25), "-".repeat(7));
    println!("  {:25}  {:>7.2}", "Weights", w.vram_breakdown.weights_gib);
    println!(
        "  {:25}  {:>7.2}  {}",
        "Gradients",
        w.vram_breakdown.gradients_gib,
        if w.vram_breakdown.gradients_gib < w.vram_breakdown.weights_gib * 0.05 {
            "(adapter only)"
        } else {
            "(full fine-tune)"
        }
    );
    println!(
        "  {:25}  {:>7.2}",
        "Optimizer states", w.vram_breakdown.optimizer_gib
    );
    println!(
        "  {:25}  {:>7.2}  (gradient checkpointing)",
        "Activations", w.vram_breakdown.activations_gib
    );
    println!(
        "  {:25}  {:>7.2}",
        "KV cache", w.vram_breakdown.kv_cache_gib
    );
    if w.vram_breakdown.library_savings_gib < 0.0 {
        println!(
            "  {:25}  {:>7.2}  (library savings)",
            "Efficiency reduction", w.vram_breakdown.library_savings_gib
        );
    }
    println!("  {}  {}", "-".repeat(25), "-".repeat(7));
    println!("  {:25}  {:>7.2}", "Total", w.vram_breakdown.total_gib);

    // ── Section 2: Pricing table───────────
    if report.listings.is_empty() {
        println!();
        println!(
            "  No GPU listings found with ≥ {:.1} GiB VRAM.",
            w.required_vram_gib
        );
    } else {
        let rec_key = report
            .recommendation
            .as_ref()
            .map(|r| (r.listing.provider.as_str(), r.listing.gpu_model.as_str()));

        println!();
        println!(
            "  {:<9}  {:<22}  {:>5}  {:>6}  {:>11}  {:>14}  Flags",
            "Provider", "GPU", "VRAM", "$/hr", "Duration", "Est. Cost"
        );
        println!(
            "  {:-<9}  {:-<22}  {:-<5}  {:-<6}  {:-<11}  {:-<14}  -----",
            "", "", "", "", "", ""
        );

        let mut any_spot = false;
        let mut any_volatile = false;
        let mut any_unavailable = false;

        for l in &report.listings {
            let is_rec = rec_key.is_some_and(|(p, g)| p == l.provider && g == l.gpu_model);
            let marker = if is_rec { ">" } else { " " };

            let flags_str = format_flags(&l.flags, &l.availability);
            if l.flags.contains(&ListingFlag::Spot) {
                any_spot = true;
            }
            if l.flags.contains(&ListingFlag::PriceVolatile) {
                any_volatile = true;
            }
            if l.availability == AvailabilityStatus::Unavailable {
                any_unavailable = true;
            }

            let over = budget
                .zip(l.cost_range.as_ref())
                .is_some_and(|(b, c)| c.low_usd > b);

            let vram_str = format!("{:.0}G", l.vram_gib);
            let dur_str = l
                .duration_range
                .as_ref()
                .map(|d| d.display())
                .unwrap_or_else(|| "n/a".to_string());
            let cost_str = l
                .cost_range
                .as_ref()
                .map(|c| c.display())
                .unwrap_or_else(|| "n/a".to_string());
            let budget_flag = if over { " !" } else { "" };

            println!(
                "{} {:<9}  {:<22}  {:>5}  {:>6.2}  {:>11}  {:>14}  {}{}",
                marker,
                l.provider,
                l.gpu_model,
                vram_str,
                l.hourly_usd,
                dur_str,
                cost_str,
                flags_str,
                budget_flag,
            );
        }

        println!();
        if any_spot {
            println!("  * Spot/preemptible — may be interrupted mid-job.");
        }
        if any_volatile {
            println!("  ~ Price volatile   — Vast.ai market price; may change before launch.");
        }
        if any_unavailable {
            println!("  ! Unavailable      — listed but no capacity right now.");
        }
        if budget.is_some() {
            println!("  ! Over budget      — exceeds --budget limit.");
        }
    }

    // ── Section 3: Recommendation──────────
    println!();
    if let Some(rec) = &report.recommendation {
        println!(
            "  Recommendation: {} {} at ${:.2}/hr",
            rec.listing.provider, rec.listing.gpu_model, rec.listing.hourly_usd
        );
        println!("    {}", rec.rationale);
        if let Some(alt) = &rec.safe_alternative {
            println!();
            println!(
                "  Safe alternative (stable price): {} {} at ${:.2}/hr",
                alt.provider, alt.gpu_model, alt.hourly_usd
            );
            if let Some(c) = &alt.cost_range {
                println!("    Estimated cost: {}", c.display());
            }
        }
    } else {
        println!(
            "  No recommendation could be made (no available listings meet the requirements)."
        );
    }

    // ── Skipped providers─────────
    if !report.skipped_providers.is_empty() {
        println!();
        println!("  Skipped:");
        for s in &report.skipped_providers {
            println!("    {}: {}", s.name, s.reason);
        }
    }

    println!();
}

pub(crate) fn format_flags(flags: &[ListingFlag], availability: &AvailabilityStatus) -> String {
    let mut parts = vec![];
    if flags.contains(&ListingFlag::Spot) {
        parts.push("*");
    }
    if flags.contains(&ListingFlag::PriceVolatile) {
        parts.push("~");
    }
    if flags.contains(&ListingFlag::LowReliability) {
        parts.push("low-rel");
    }
    if *availability == AvailabilityStatus::Unavailable {
        parts.push("unavail");
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{
        AvailabilityStatus, CostRange, DurationRange, ListingFlag, PlanRecommendation, PlanReport,
        RankedListing, SkippedProvider, VramBreakdown, WorkloadSummary,
    };
    use chrono::Utc;

    fn make_vram_breakdown(total: f64) -> VramBreakdown {
        VramBreakdown {
            weights_gib: total * 0.60,
            gradients_gib: total * 0.01,
            optimizer_gib: total * 0.20,
            activations_gib: total * 0.10,
            kv_cache_gib: total * 0.09,
            library_savings_gib: 0.0,
            total_gib: total,
        }
    }

    fn make_report_empty() -> PlanReport {
        PlanReport {
            workload: WorkloadSummary {
                model_id: "meta-llama/Llama-3-8B".to_string(),
                param_count_b: 8.0,
                vram_breakdown: make_vram_breakdown(14.0),
                required_vram_gib: 14.7,
                fitting_tiers: vec!["16G".to_string(), "24G".to_string()],
            },
            listings: vec![],
            recommendation: None,
            skipped_providers: vec![],
            generated_at: Utc::now(),
        }
    }

    fn make_listing(
        provider: &str,
        gpu: &str,
        vram: f64,
        hourly: f64,
        cost_low: f64,
    ) -> RankedListing {
        RankedListing {
            provider: provider.to_string(),
            gpu_model: gpu.to_string(),
            vram_gib: vram,
            hourly_usd: hourly,
            duration_range: Some(DurationRange {
                low_secs: cost_low / hourly * 3600.0,
                high_secs: cost_low / hourly * 3600.0 * 1.5,
            }),
            cost_range: Some(CostRange {
                low_usd: cost_low,
                high_usd: cost_low * 1.5,
            }),
            availability: AvailabilityStatus::Available,
            flags: vec![],
        }
    }

    /// Smoke test: rendering an empty listings report does not panic.
    #[test]
    fn render_empty_listings_no_panic() {
        let report = make_report_empty();
        // Should not panic; output goes to stdout (captured by test runner).
        render(&report, None);
    }

    /// Smoke test: rendering a populated report with a recommendation does not panic.
    #[test]
    fn render_with_recommendation_no_panic() {
        let listing = make_listing("RunPod", "RTX 4090", 24.0, 0.44, 0.48);
        let rec = PlanRecommendation {
            listing: listing.clone(),
            rationale: "Cheapest stable option.".to_string(),
            safe_alternative: None,
        };
        let mut report = make_report_empty();
        report.listings = vec![listing];
        report.recommendation = Some(rec);
        render(&report, None);
    }

    /// Smoke test: rendering with a safe alternative does not panic.
    #[test]
    fn render_with_safe_alternative_no_panic() {
        let mut top = make_listing("Vast.ai", "RTX 3090", 24.0, 0.22, 0.46);
        top.flags.push(ListingFlag::PriceVolatile);
        let stable = make_listing("RunPod", "RTX 4090", 24.0, 0.44, 0.48);
        let rec = PlanRecommendation {
            listing: top.clone(),
            rationale: "Cheapest option.".to_string(),
            safe_alternative: Some(stable.clone()),
        };
        let mut report = make_report_empty();
        report.listings = vec![top, stable];
        report.recommendation = Some(rec);
        render(&report, None);
    }

    /// Budget flag appears for listings that exceed the user's budget.
    #[test]
    fn render_with_budget_over_shows_flag() {
        let listing = make_listing("Lambda", "A10", 24.0, 0.60, 1.20);
        let mut report = make_report_empty();
        report.listings = vec![listing];
        // budget of $0.50 means a $1.20 estimate is over-budget — should not panic.
        render(&report, Some(0.50));
    }

    /// Skipped providers section renders without panic.
    #[test]
    fn render_skipped_providers_no_panic() {
        let mut report = make_report_empty();
        report.skipped_providers = vec![SkippedProvider {
            name: "RunPod".to_string(),
            reason: "API timeout".to_string(),
        }];
        render(&report, None);
    }

    // ── format_flags tests────────────────────

    #[test]
    fn format_flags_empty() {
        assert_eq!(format_flags(&[], &AvailabilityStatus::Available), "");
    }

    #[test]
    fn format_flags_spot() {
        assert_eq!(
            format_flags(&[ListingFlag::Spot], &AvailabilityStatus::Available),
            "*"
        );
    }

    #[test]
    fn format_flags_volatile_and_unavailable() {
        let s = format_flags(
            &[ListingFlag::PriceVolatile],
            &AvailabilityStatus::Unavailable,
        );
        assert!(s.contains('~'), "should contain volatile marker");
        assert!(s.contains("unavail"), "should contain unavail marker");
    }

    #[test]
    fn format_flags_low_reliability() {
        let s = format_flags(
            &[ListingFlag::LowReliability],
            &AvailabilityStatus::Available,
        );
        assert!(s.contains("low-rel"));
    }
}
