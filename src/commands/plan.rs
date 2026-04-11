use anyhow::{Context, Result};
use chrono::Utc;

use crate::cli::{Availability, PlanArgs, PlanOutputFormat};
use crate::output;
use crate::plan::duration::estimator as dur;
use crate::plan::providers::{self, GpuListing};
use crate::plan::vram::estimator as vram;
use crate::plan::{
    AvailabilityStatus, ListingFlag, PlanRecommendation, PlanReport, RankedListing,
    SkippedProvider, WorkloadSummary,
};

pub async fn run(args: PlanArgs) -> Result<()> {
    // ── Step 1: Resolve model ─────────────────────────────────────────────────
    eprintln!("Resolving model '{}'...", args.model);
    let spec = crate::plan::model::resolver::resolve(&args.model, args.params_b)
        .await
        .with_context(|| format!("could not resolve model '{}'", args.model))?;

    // ── Step 2: Estimate VRAM ─────────────────────────────────────────────────
    let breakdown = vram::estimate(
        &spec,
        args.method,
        args.optimizer,
        args.quantization,
        args.batch_size,
    );
    let required_vram_gib = breakdown.total_gib * 1.05; // 5% safety margin
    let fitting = vram::fitting_tiers(breakdown.total_gib);
    let workload = WorkloadSummary {
        model_id: spec.model_id.clone(),
        param_count_b: spec.param_count_b,
        vram_breakdown: breakdown,
        required_vram_gib,
        fitting_tiers: fitting,
    };

    // ── Step 3: Fetch provider listings concurrently ─────────────────────────
    eprintln!("Fetching live GPU pricing...");
    let (raw_listings, mut all_skipped) = providers::fetch_all(args.providers.as_deref()).await;

    if raw_listings.is_empty() && !all_skipped.is_empty() {
        eprintln!("Warning: all providers failed to respond. Check your network connection.");
    }

    // ── Step 4: Filter by availability preference ────────────────────────────
    let filtered: Vec<GpuListing> = raw_listings
        .into_iter()
        .filter(|l| {
            // When --availability now, drop listings that are not immediately launchable.
            if args.availability == Availability::Now {
                return l.availability == AvailabilityStatus::Available
                    && !l.flags.contains(&ListingFlag::CurrentlyUnavailable);
            }
            true
        })
        .collect();

    // ── Step 5: Filter by VRAM, add duration + cost estimates ────────────────
    let mfu = args.mfu.unwrap_or(0.30);

    let mut ranked: Vec<RankedListing> = filtered
        .into_iter()
        .filter(|l| l.vram_gib >= required_vram_gib)
        .map(|l| {
            let duration = dur::estimate_duration_range(
                &spec,
                &l.gpu_model,
                args.dataset_rows,
                args.batch_size,
                args.epochs,
                Some(mfu),
            );
            let cost = duration.as_ref().map(|d| dur::cost_range(d, l.hourly_usd));
            let mut r = l.into_ranked();
            r.duration_range = duration;
            r.cost_range = cost;
            r
        })
        .collect();

    // Sort: primarily by estimated low cost, secondarily by hourly rate.
    // Listings without a duration estimate use hourly_usd * 24 as a proxy.
    ranked.sort_by(|a, b| {
        let cost_a = sort_key(a);
        let cost_b = sort_key(b);
        cost_a
            .partial_cmp(&cost_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── Step 6: Budget filter note (include in skipped if nothing fits) ───────
    if let Some(budget) = args.budget {
        let affordable = ranked
            .iter()
            .any(|l| l.cost_range.as_ref().map_or(true, |c| c.low_usd <= budget));
        if !affordable && !ranked.is_empty() {
            let cheapest = ranked.first().unwrap();
            let over_by = cheapest
                .cost_range
                .as_ref()
                .map_or(0.0, |c| c.low_usd - budget);
            all_skipped.push(SkippedProvider {
                name: "budget-filter".to_string(),
                reason: format!(
                    "No option fits within your ${budget:.2} budget. \
                     Cheapest option ({} {}) is ~${over_by:.2} over budget.",
                    cheapest.provider, cheapest.gpu_model
                ),
            });
        }
    }

    // ── Step 7: Build recommendation ─────────────────────────────────────────
    let recommendation = build_recommendation(&ranked, args.budget);

    // ── Step 8: Render ────────────────────────────────────────────────────────
    let report = PlanReport {
        workload,
        listings: ranked,
        recommendation,
        skipped_providers: all_skipped,
        generated_at: Utc::now(),
    };

    match args.output {
        PlanOutputFormat::Terminal => output::plan_terminal::render(&report, args.budget),
        PlanOutputFormat::Json => output::plan_json::render(&report)?,
    }

    Ok(())
}

// ── Recommendation logic ───────────────────────────────────────────────────────

fn build_recommendation(
    ranked: &[RankedListing],
    budget: Option<f64>,
) -> Option<PlanRecommendation> {
    // Only consider listings that are immediately available for recommendation.
    let available: Vec<&RankedListing> = ranked
        .iter()
        .filter(|l| {
            l.availability == AvailabilityStatus::Available
                && !l.flags.contains(&ListingFlag::CurrentlyUnavailable)
        })
        .collect();

    if available.is_empty() {
        return None;
    }

    // The first entry is already the cheapest (list is pre-sorted).
    let best = *available.first()?;

    let is_volatile = best.flags.contains(&ListingFlag::PriceVolatile);
    let best_cost = sort_key(best);

    // Offer a stable-price alternative when the top pick is volatile (Vast.ai),
    // but only if a non-volatile option exists within 50% of the top pick's cost.
    let safe_alternative = if is_volatile {
        available
            .iter()
            .copied()
            .find(|l| {
                !l.flags.contains(&ListingFlag::PriceVolatile) && sort_key(l) <= best_cost * 1.5
            })
            .cloned()
    } else {
        None
    };

    let rationale = build_rationale(best, budget);

    Some(PlanRecommendation {
        listing: best.clone(),
        rationale,
        safe_alternative,
    })
}

fn build_rationale(listing: &RankedListing, budget: Option<f64>) -> String {
    let cost_str = listing
        .cost_range
        .as_ref()
        .map(|c| format!("an estimated total cost of {}", c.display()))
        .unwrap_or_else(|| format!("${:.2}/hr", listing.hourly_usd));

    let over_budget = budget
        .zip(listing.cost_range.as_ref())
        .is_some_and(|(b, c)| c.low_usd > b);

    let base = format!(
        "{} {} ({:.0} GiB VRAM) offers {} for this workload.",
        listing.provider, listing.gpu_model, listing.vram_gib, cost_str
    );

    if over_budget {
        format!("{base} Note: this exceeds your specified budget.")
    } else {
        base
    }
}

/// Sort key: estimated low cost when available, otherwise hourly_usd × 24
/// (one day of compute as a rough proxy for relative ordering).
fn sort_key(l: &RankedListing) -> f64 {
    l.cost_range
        .as_ref()
        .map_or(l.hourly_usd * 24.0, |c| c.low_usd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{CostRange, DurationRange};

    fn make_listing(
        provider: &str,
        gpu: &str,
        vram: f64,
        hourly: f64,
        cost_low: f64,
        volatile: bool,
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
            flags: if volatile {
                vec![ListingFlag::PriceVolatile]
            } else {
                vec![]
            },
        }
    }

    #[test]
    fn recommendation_picks_cheapest_available() {
        let ranked = vec![
            make_listing("Vast.ai", "RTX 3090", 24.0, 0.22, 0.46, true),
            make_listing("RunPod", "RTX 4090", 24.0, 0.44, 0.48, false),
            make_listing("Lambda", "A10", 24.0, 0.60, 0.90, false),
        ];
        let rec = build_recommendation(&ranked, None).unwrap();
        assert_eq!(rec.listing.gpu_model, "RTX 3090");
    }

    #[test]
    fn safe_alternative_provided_when_top_is_volatile() {
        let ranked = vec![
            make_listing("Vast.ai", "RTX 3090", 24.0, 0.22, 0.46, true),
            make_listing("RunPod", "RTX 4090", 24.0, 0.44, 0.48, false),
        ];
        let rec = build_recommendation(&ranked, None).unwrap();
        assert!(
            rec.safe_alternative.is_some(),
            "should offer a stable alternative"
        );
        assert_eq!(rec.safe_alternative.unwrap().provider, "RunPod");
    }

    #[test]
    fn no_safe_alternative_when_top_is_stable() {
        let ranked = vec![
            make_listing("RunPod", "RTX 4090", 24.0, 0.44, 0.48, false),
            make_listing("Lambda", "A10", 24.0, 0.60, 0.90, false),
        ];
        let rec = build_recommendation(&ranked, None).unwrap();
        assert!(rec.safe_alternative.is_none());
    }

    #[test]
    fn empty_ranked_returns_none() {
        assert!(build_recommendation(&[], None).is_none());
    }

    #[test]
    fn unavailable_listing_excluded_from_recommendation() {
        let mut unavail = make_listing("Lambda", "A100", 80.0, 0.50, 0.20, false);
        unavail.availability = AvailabilityStatus::Unavailable;
        unavail.flags.push(ListingFlag::CurrentlyUnavailable);
        let ranked = vec![
            unavail,
            make_listing("RunPod", "RTX 4090", 24.0, 0.44, 0.48, false),
        ];
        let rec = build_recommendation(&ranked, None).unwrap();
        assert_eq!(
            rec.listing.provider, "RunPod",
            "unavailable listing should not be recommended"
        );
    }
}
