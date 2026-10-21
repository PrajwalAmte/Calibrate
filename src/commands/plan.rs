use anyhow::{Context, Result};
use chrono::Utc;

use crate::cli::{Availability, PlanArgs, PlanOutputFormat};
use crate::gpu_specs::SpecDb;
use crate::output;
use crate::plan::duration::estimator as dur;
use crate::plan::model::ModelSpec;
use crate::plan::providers;
use crate::plan::vram::estimator as vram;
use crate::plan::{
    GpuListing, ListingFlag, PlanRecommendation, PlanReport, RankedListing, SkippedProvider,
    TrainingSetup, WorkloadSummary, PLAN_SCHEMA_VERSION,
};

pub async fn run(args: PlanArgs) -> Result<()> {
    let settings = crate::config::get();
    let seq_len = args
        .seq_len
        .or(settings.file.plan.seq_len)
        .unwrap_or(2048)
        .max(1);
    let setup = TrainingSetup {
        method: args.method,
        optimizer_lib: args.optimizer,
        quantization: args.quantization,
        micro_batch: args.batch_size,
        seq_len,
    };
    vram::validate(&setup)?;

    eprintln!("Resolving model '{}'...", args.model);
    let spec = crate::plan::model::resolver::resolve(&args.model, args.params_b)
        .await
        .with_context(|| format!("could not resolve model '{}'", args.model))?;

    let (listings, skipped, price_source) = match &args.prices_file {
        Some(p) => (providers::load_prices_file(p)?, vec![], "file".to_string()),
        None => {
            eprintln!("Fetching GPU pricing...");
            let filter = args
                .providers
                .clone()
                .or_else(|| settings.file.plan.providers.clone());
            let prices = providers::fetch_all(filter.as_deref(), args.refresh_prices).await?;
            if prices.listings.is_empty() {
                eprintln!(
                    "Warning: no provider returned prices. Check network/proxy settings or use \
                     --prices-file."
                );
            }
            (prices.listings, prices.skipped, prices.source)
        }
    };

    let specs = if settings.offline {
        SpecDb::builtin()
    } else {
        tokio::task::spawn_blocking(SpecDb::load).await?
    };
    let report = build_report(
        &spec,
        &setup,
        &PlanParams {
            dataset_rows: args.dataset_rows,
            epochs: args.epochs,
            utilization: args.mfu,
            budget: args.budget,
            availability: args.availability,
        },
        listings,
        skipped,
        price_source,
        &specs,
    );

    match args.output {
        PlanOutputFormat::Terminal => output::plan_terminal::render(&report, args.budget),
        PlanOutputFormat::Json => output::plan_json::render(&report)?,
    }
    Ok(())
}

pub struct PlanParams {
    pub dataset_rows: Option<u64>,
    pub epochs: u32,
    pub utilization: Option<f64>,
    pub budget: Option<f64>,
    pub availability: Availability,
}

/// Pure planning pipeline (no I/O) — everything downstream of data fetching.
pub fn build_report(
    spec: &ModelSpec,
    setup: &TrainingSetup,
    p: &PlanParams,
    listings: Vec<GpuListing>,
    mut skipped: Vec<SkippedProvider>,
    price_source: String,
    specs: &SpecDb,
) -> PlanReport {
    let single = vram::estimate(spec, setup, 1);
    let utilization = p.utilization.unwrap_or(dur::DEFAULT_UTILIZATION);
    let total_tokens = p
        .dataset_rows
        .map(|r| dur::total_tokens(r, setup.seq_len, p.epochs));
    let total_flops =
        total_tokens.map(|t| t * dur::flops_per_token(spec, setup.method, setup.seq_len));

    let mut assumptions = vec![
        format!(
            "utilization {:.0}% of peak{}",
            utilization * 100.0,
            if p.utilization.is_some() {
                " (from --mfu)"
            } else {
                " (default; pass --mfu from `calibrate watch` for a measured value)"
            }
        ),
        "multi-GPU instances: data parallel at 90% scaling efficiency".into(),
        format!("{} tokens per example", setup.seq_len),
    ];
    if spec.architecture_estimated {
        assumptions.push(
            "architecture estimated from the parameter count (set HF_TOKEN for gated models)"
                .into(),
        );
    }

    let mut ranked = Vec::new();
    let mut too_small = 0usize;
    for l in listings {
        let launchable = l.availability == crate::plan::AvailabilityStatus::Available
            && !l.flags.contains(&ListingFlag::CurrentlyUnavailable);
        if p.availability == Availability::Now && !launchable {
            continue;
        }
        let need = vram::estimate(spec, setup, l.gpu_count).total_gib;
        if l.vram_per_gpu_gib + 0.5 < need {
            too_small += 1;
            continue;
        }
        let gpu = specs
            .find(&l.gpu_model)
            .map(|(s, _)| s.bf16_tflops)
            .filter(|t| *t > 0.0);
        let duration = match (total_flops, gpu) {
            (Some(f), Some(peak)) => {
                dur::estimate_duration_range(f, peak, l.gpu_count, utilization)
            }
            _ => None,
        };
        let cost = duration.as_ref().map(|d| dur::cost_range(d, l.hourly_usd));
        let over_budget = p
            .budget
            .zip(cost.as_ref())
            .is_some_and(|(b, c)| c.low_usd > b);
        let mut listing = l;
        if gpu.is_none() && !listing.flags.contains(&ListingFlag::UnknownGpu) {
            listing.flags.push(ListingFlag::UnknownGpu);
        }
        ranked.push(RankedListing {
            listing,
            peak_tflops_per_gpu: gpu,
            required_vram_per_gpu_gib: need,
            duration_range: duration,
            cost_range: cost,
            over_budget,
        });
    }

    // Estimated listings first (by cost), then the rest by hourly price.
    ranked.sort_by(|a, b| {
        b.cost_range
            .is_some()
            .cmp(&a.cost_range.is_some())
            .then(sort_key(a).total_cmp(&sort_key(b)))
            .then(a.listing.hourly_usd.total_cmp(&b.listing.hourly_usd))
    });

    if let Some(budget) = p.budget {
        if !ranked.is_empty() && ranked.iter().all(|l| l.over_budget) {
            let c = &ranked[0];
            skipped.push(SkippedProvider {
                name: "budget".into(),
                reason: format!(
                    "no option fits within ${budget:.2}; the cheapest ({} {}) is ~${:.2} over",
                    c.listing.provider,
                    c.display_gpu(),
                    c.cost_range.as_ref().map_or(0.0, |x| x.low_usd - budget)
                ),
            });
        }
    }

    let recommendation = build_recommendation(&ranked);
    PlanReport {
        schema_version: PLAN_SCHEMA_VERSION,
        workload: WorkloadSummary {
            model: spec.clone(),
            setup: *setup,
            required_vram_gib: single.total_gib,
            fitting_tiers: vram::fitting_tiers(single.total_gib),
            vram: single,
            dataset_rows: p.dataset_rows,
            epochs: p.epochs,
            total_tokens,
            total_hardware_flops: total_flops,
            assumed_utilization: utilization,
            assumptions,
        },
        listings: ranked,
        excluded_too_small: too_small,
        recommendation,
        skipped_providers: skipped,
        price_source,
        generated_at: Utc::now(),
    }
}

/// Never recommend something over budget, unavailable, or without an estimate
/// when estimated options exist.
fn build_recommendation(ranked: &[RankedListing]) -> Option<PlanRecommendation> {
    let viable: Vec<&RankedListing> = ranked
        .iter()
        .filter(|l| l.is_launchable() && !l.over_budget)
        .collect();
    let has_estimates = viable.iter().any(|l| l.cost_range.is_some());
    let pool: Vec<&RankedListing> = viable
        .into_iter()
        .filter(|l| !has_estimates || l.cost_range.is_some())
        .collect();
    let best = *pool.first()?;
    let volatile = best.listing.flags.contains(&ListingFlag::PriceVolatile)
        || best.listing.flags.contains(&ListingFlag::CommunityCloud);
    let safe_alternative = if volatile {
        let k = sort_key(best);
        pool.iter()
            .copied()
            .find(|l| {
                !l.listing.flags.contains(&ListingFlag::PriceVolatile)
                    && !l.listing.flags.contains(&ListingFlag::CommunityCloud)
                    && sort_key(l) <= k * 1.5
            })
            .cloned()
    } else {
        None
    };
    let what = best
        .cost_range
        .as_ref()
        .map(|c| {
            format!(
                "an estimated {} total over {}",
                c.display(),
                best.duration_range
                    .as_ref()
                    .map(|d| d.display())
                    .unwrap_or_default()
            )
        })
        .unwrap_or_else(|| format!("${:.2}/hr (no duration estimate)", best.listing.hourly_usd));
    Some(PlanRecommendation {
        rationale: format!(
            "{} {} ({:.0} GiB per GPU, {:.1} GiB needed) costs {what}.",
            best.listing.provider,
            best.display_gpu(),
            best.listing.vram_per_gpu_gib,
            best.required_vram_per_gpu_gib
        ),
        listing: best.clone(),
        safe_alternative,
    })
}

/// Estimated cost when known; otherwise a day of compute as an ordering proxy.
fn sort_key(l: &RankedListing) -> f64 {
    l.cost_range
        .as_ref()
        .map_or(l.listing.hourly_usd * 24.0, |c| c.low_usd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{FinetuneMethod, OptimizerLib, QuantLevel};
    use crate::plan::model::ParamSource;
    use crate::plan::AvailabilityStatus;

    fn qwen7b() -> ModelSpec {
        ModelSpec {
            model_id: "Qwen/Qwen2.5-7B".into(),
            param_count_b: 7.62,
            active_param_count_b: 7.62,
            param_source: ParamSource::HubSafetensors,
            num_layers: 28,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 152_064,
            tie_word_embeddings: false,
            architecture_estimated: false,
        }
    }

    fn listing(provider: &str, gpu: &str, n: u32, vram: f64, hourly: f64) -> GpuListing {
        GpuListing {
            provider: provider.into(),
            gpu_model: gpu.into(),
            gpu_count: n,
            vram_per_gpu_gib: vram,
            hourly_usd: hourly,
            availability: AvailabilityStatus::Available,
            flags: vec![],
        }
    }

    fn setup(method: FinetuneMethod) -> TrainingSetup {
        TrainingSetup {
            method,
            optimizer_lib: OptimizerLib::None,
            quantization: QuantLevel::None,
            micro_batch: 1,
            seq_len: 2048,
        }
    }

    fn params(budget: Option<f64>) -> PlanParams {
        PlanParams {
            dataset_rows: Some(50_000),
            epochs: 1,
            utilization: None,
            budget,
            availability: Availability::Flexible,
        }
    }

    fn report(method: FinetuneMethod, budget: Option<f64>, ls: Vec<GpuListing>) -> PlanReport {
        build_report(
            &qwen7b(),
            &setup(method),
            &params(budget),
            ls,
            vec![],
            "file".into(),
            &SpecDb::builtin(),
        )
    }

    #[test]
    fn golden_lora_7b_50k_rows() {
        let r = report(
            FinetuneMethod::Lora,
            None,
            vec![
                listing("A", "RTX 4090", 1, 24.0, 0.44),
                listing("B", "H100 SXM", 1, 80.0, 2.99),
                listing("C", "RTX 3080", 1, 10.0, 0.11),
                listing("D", "Mystery GPU", 1, 48.0, 0.01),
            ],
        );
        assert_eq!(
            r.excluded_too_small, 1,
            "10 GB card cannot hold LoRA on 7.6B"
        );
        assert!(r.workload.required_vram_gib > 16.0 && r.workload.required_vram_gib < 24.0);
        let h100 = r
            .listings
            .iter()
            .find(|l| l.listing.gpu_model == "H100 SXM")
            .unwrap();
        let d = h100.duration_range.as_ref().unwrap();
        // ~4.8e18 FLOPs ÷ (989 TFLOPS × 0.3) ≈ 4.5 h point estimate.
        assert!(
            d.low_secs / 3600.0 > 3.0 && d.high_secs / 3600.0 < 8.0,
            "{d:?}"
        );
        let c = h100.cost_range.as_ref().unwrap();
        assert!(c.low_usd > 5.0 && c.high_usd < 30.0, "{c:?}");
        // Unknown GPU is listed but never recommended over estimated options.
        let rec = r.recommendation.unwrap();
        assert_ne!(rec.listing.listing.gpu_model, "Mystery GPU");
        assert!(
            r.listings.last().unwrap().cost_range.is_none(),
            "listings without an estimate sort last"
        );
        let mystery = r
            .listings
            .iter()
            .find(|l| l.listing.gpu_model == "Mystery GPU")
            .unwrap();
        assert!(mystery.listing.flags.contains(&ListingFlag::UnknownGpu));
        assert!(mystery.cost_range.is_none());
    }

    #[test]
    fn multi_gpu_listing_uses_per_gpu_vram_and_aggregate_speed() {
        let r = report(
            FinetuneMethod::Full,
            None,
            vec![
                listing("L", "H100 SXM5", 8, 80.0, 23.92),
                listing("L", "A100 SXM4", 1, 80.0, 1.29),
            ],
        );
        // Full FT of 7.6B (~125 GiB) fits on neither a single 80 GB GPU nor
        // (without sharding) each GPU of an 8× box.
        assert!(r.listings.is_empty(), "{:?}", r.listings.len());
        assert_eq!(r.excluded_too_small, 2);
    }

    #[test]
    fn deepspeed_makes_8x_box_viable() {
        let mut s = setup(FinetuneMethod::Full);
        s.optimizer_lib = OptimizerLib::DeepSpeed;
        let r = build_report(
            &qwen7b(),
            &s,
            &params(None),
            vec![listing("L", "H100 SXM5", 8, 80.0, 23.92)],
            vec![],
            "file".into(),
            &SpecDb::builtin(),
        );
        assert_eq!(r.listings.len(), 1);
        assert_eq!(r.listings[0].display_gpu(), "8× H100 SXM5");
        assert!(r.recommendation.is_some());
    }

    #[test]
    fn over_budget_is_never_recommended() {
        let r = report(
            FinetuneMethod::Lora,
            Some(1.0),
            vec![listing("A", "RTX 4090", 1, 24.0, 0.44)],
        );
        assert!(r.listings[0].over_budget);
        assert!(r.recommendation.is_none());
        assert!(r.skipped_providers.iter().any(|s| s.name == "budget"));
    }

    #[test]
    fn volatile_pick_gets_stable_alternative() {
        let mut cheap = listing("Vast.ai", "RTX 4090", 1, 24.0, 0.30);
        cheap.flags.push(ListingFlag::PriceVolatile);
        let r = report(
            FinetuneMethod::Lora,
            None,
            vec![cheap, listing("RunPod", "RTX 4090", 1, 24.0, 0.40)],
        );
        let rec = r.recommendation.unwrap();
        assert_eq!(rec.listing.listing.provider, "Vast.ai");
        assert_eq!(rec.safe_alternative.unwrap().listing.provider, "RunPod");
    }

    #[test]
    fn availability_now_filters_unavailable() {
        let mut busy = listing("L", "H100 SXM", 1, 80.0, 2.0);
        busy.availability = AvailabilityStatus::Unavailable;
        let mut p = params(None);
        p.availability = Availability::Now;
        let r = build_report(
            &qwen7b(),
            &setup(FinetuneMethod::Lora),
            &p,
            vec![busy],
            vec![],
            "file".into(),
            &SpecDb::builtin(),
        );
        assert!(r.listings.is_empty());
    }
}
