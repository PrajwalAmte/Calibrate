use crate::metrics::units::fmt_usd;
use crate::plan::{ListingFlag, PlanReport, RankedListing};

/// Render the plan report as text.
pub fn render_to_string(report: &PlanReport, budget: Option<f64>) -> String {
    let mut o = String::new();
    let mut ln = |s: String| {
        o.push_str(&s);
        o.push('\n');
    };
    let w = &report.workload;
    let m = &w.model;
    let v = &w.vram;

    ln(String::new());
    let moe = if (m.active_param_count_b - m.param_count_b).abs() > 0.01 {
        format!(", {:.2}B active", m.active_param_count_b)
    } else {
        String::new()
    };
    ln(format!(
        "  Model:     {}  ({:.2}B parameters{moe}; source: {})",
        m.model_id,
        m.param_count_b,
        serde_json::to_value(m.param_source)
            .ok()
            .and_then(|x| x.as_str().map(str::to_string))
            .unwrap_or_default()
    ));
    ln(format!(
        "  Setup:     {:?} • {:?} • micro-batch {} • {} tokens/example",
        w.setup.method, w.setup.optimizer_lib, w.setup.micro_batch, w.setup.seq_len
    ));
    ln(format!(
        "  Required:  {:.1} GiB VRAM on a single GPU{}",
        w.required_vram_gib,
        if w.fitting_tiers.is_empty() {
            " — larger than any single GPU; use DeepSpeed ZeRO-3 across several".to_string()
        } else {
            format!("  (fits {})", w.fitting_tiers.join(", "))
        }
    ));
    if let (Some(t), Some(f)) = (w.total_tokens, w.total_hardware_flops) {
        ln(format!(
            "  Work:      {:.2}B tokens × {} epoch(s) → {:.2e} FLOPs (incl. recomputation)",
            t / 1e9 / w.epochs.max(1) as f64,
            w.epochs,
            f
        ));
    } else {
        ln("  Work:      pass --dataset-rows for duration and cost estimates".into());
    }
    ln(String::new());

    ln(format!("  {:<28}  {:>8}", "Component (per GPU)", "GiB"));
    ln(format!("  {}  {}", "-".repeat(28), "-".repeat(8)));
    for (name, val) in [
        ("Weights", v.weights_gib),
        ("Gradients", v.gradients_gib),
        ("Optimizer states", v.optimizer_gib),
        ("Activations (checkpointed)", v.activations_gib),
        ("Logits + loss buffer", v.logits_gib),
        ("CUDA context + allocator", v.overhead_gib),
    ] {
        ln(format!("  {name:<28}  {val:>8.2}"));
    }
    ln(format!("  {}  {}", "-".repeat(28), "-".repeat(8)));
    ln(format!("  {:<28}  {:>8.2}", "Total", v.total_gib));
    for n in &v.notes {
        ln(format!("    · {n}"));
    }

    if report.listings.is_empty() {
        ln(String::new());
        ln(format!(
            "  No listing has enough VRAM per GPU ({} too small).",
            report.excluded_too_small
        ));
    } else {
        let rec = report.recommendation.as_ref().map(|r| &r.listing);
        ln(String::new());
        ln(format!(
            "  {:<9}  {:<28}  {:>6}  {:>7}  {:>11}  {:>16}  Flags",
            "Provider", "GPU", "VRAM", "$/hr", "Duration", "Est. Cost"
        ));
        ln(format!(
            "  {:-<9}  {:-<28}  {:-<6}  {:-<7}  {:-<11}  {:-<16}  -----",
            "", "", "", "", "", ""
        ));
        for l in &report.listings {
            let marker = if rec.is_some_and(|r| same(r, l)) {
                ">"
            } else {
                " "
            };
            let gpu = l.display_gpu();
            let gpu = if gpu.chars().count() > 28 {
                format!("{}…", gpu.chars().take(27).collect::<String>())
            } else {
                gpu
            };
            ln(format!(
                "{marker} {:<9}  {:<28}  {:>5.0}G  {:>7.2}  {:>11}  {:>16}  {}{}",
                l.listing.provider,
                gpu,
                l.listing.vram_per_gpu_gib,
                l.listing.hourly_usd,
                l.duration_range
                    .as_ref()
                    .map(|d| d.display())
                    .unwrap_or_else(|| "n/a".into()),
                l.cost_range
                    .as_ref()
                    .map(|c| c.display())
                    .unwrap_or_else(|| "n/a".into()),
                format_flags(l),
                if l.over_budget { " $!" } else { "" }
            ));
        }
        ln(String::new());
        let any = |f: ListingFlag| report.listings.iter().any(|l| l.listing.flags.contains(&f));
        if any(ListingFlag::CommunityCloud) {
            ln("  c  Community cloud — third-party hosts, fewer reliability guarantees.".into());
        }
        if any(ListingFlag::Spot) {
            ln("  *  Spot / preemptible — may be interrupted.".into());
        }
        if any(ListingFlag::PriceVolatile) {
            ln("  ~  Marketplace price — may change before launch.".into());
        }
        if any(ListingFlag::CurrentlyUnavailable) {
            ln("  !  No capacity right now.".into());
        }
        if any(ListingFlag::UnknownGpu) {
            ln("  ?  GPU not in the spec database — no estimate.".into());
        }
        if budget.is_some() {
            ln("  $! Over --budget (never recommended).".into());
        }
        if report.excluded_too_small > 0 {
            ln(format!(
                "  {} listing(s) hidden: not enough VRAM per GPU.",
                report.excluded_too_small
            ));
        }
    }

    ln(String::new());
    match &report.recommendation {
        Some(r) => {
            ln(format!(
                "  Recommendation: {} {} at ${:.2}/hr",
                r.listing.listing.provider,
                r.listing.display_gpu(),
                r.listing.listing.hourly_usd
            ));
            ln(format!("    {}", r.rationale));
            if let Some(alt) = &r.safe_alternative {
                ln(format!(
                    "  Stable-price alternative: {} {} at ${:.2}/hr{}",
                    alt.listing.provider,
                    alt.display_gpu(),
                    alt.listing.hourly_usd,
                    alt.cost_range
                        .as_ref()
                        .map(|c| format!(" (est. {})", c.display()))
                        .unwrap_or_default()
                ));
            }
        }
        None => ln("  No recommendation: no launchable listing fits the VRAM and budget.".into()),
    }

    ln(String::new());
    ln("  Assumptions:".into());
    for a in &w.assumptions {
        ln(format!("    · {a}"));
    }
    ln(format!(
        "    · prices: {} (fetched {})",
        report.price_source,
        report.generated_at.format("%Y-%m-%d %H:%M UTC")
    ));
    if !report.skipped_providers.is_empty() {
        ln(String::new());
        ln("  Notes:".into());
        for s in &report.skipped_providers {
            ln(format!("    {}: {}", s.name, s.reason));
        }
    }
    let _ = fmt_usd;
    ln(String::new());
    o
}

pub fn render(report: &PlanReport, budget: Option<f64>) {
    print!("{}", render_to_string(report, budget));
}

fn same(a: &RankedListing, b: &RankedListing) -> bool {
    a.listing == b.listing
}

pub(crate) fn format_flags(l: &RankedListing) -> String {
    let f = &l.listing.flags;
    let mut parts = vec![];
    if f.contains(&ListingFlag::CommunityCloud) {
        parts.push("c");
    }
    if f.contains(&ListingFlag::Spot) {
        parts.push("*");
    }
    if f.contains(&ListingFlag::PriceVolatile) {
        parts.push("~");
    }
    if f.contains(&ListingFlag::LowReliability) {
        parts.push("low-rel");
    }
    if f.contains(&ListingFlag::CurrentlyUnavailable) {
        parts.push("!");
    }
    if f.contains(&ListingFlag::UnknownGpu) {
        parts.push("?");
    }
    parts.join(" ")
}
