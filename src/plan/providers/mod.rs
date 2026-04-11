pub mod lambda;
pub mod runpod;
pub mod vastai;

use crate::plan::{AvailabilityStatus, ListingFlag, RankedListing, SkippedProvider};

/// Raw GPU listing returned by a provider before duration/cost enrichment.
#[derive(Debug, Clone)]
pub struct GpuListing {
    pub provider: String,
    pub gpu_model: String,
    pub vram_gib: f64,
    pub hourly_usd: f64,
    pub availability: AvailabilityStatus,
    pub flags: Vec<ListingFlag>,
}

impl GpuListing {
    /// Promote to a `RankedListing`. Duration and cost fields are left `None`
    /// and filled in by the duration estimator pass in `commands/plan.rs`.
    pub fn into_ranked(self) -> RankedListing {
        RankedListing {
            provider: self.provider,
            gpu_model: self.gpu_model,
            vram_gib: self.vram_gib,
            hourly_usd: self.hourly_usd,
            duration_range: None,
            cost_range: None,
            availability: self.availability,
            flags: self.flags,
        }
    }
}

/// Fetch GPU listings from all enabled providers concurrently.
///
/// Provider failures are captured as `SkippedProvider` entries; one provider
/// being down does not prevent results from the others from being returned.
pub async fn fetch_all(
    provider_filter: Option<&[String]>,
) -> (Vec<GpuListing>, Vec<SkippedProvider>) {
    let want = |name: &str| {
        provider_filter.map_or(true, |f| {
            f.iter().any(|p| p.eq_ignore_ascii_case(name))
        })
    };

    let run_runpod = want("runpod");
    let run_lambda = want("lambda");
    let run_vastai = want("vastai") || want("vast.ai") || want("vast");

    // Fire all three fetches concurrently; disabled providers return Ok([]).
    let (rp, lm, va) = tokio::join!(
        async {
            if run_runpod { runpod::fetch_listings().await } else { Ok(vec![]) }
        },
        async {
            if run_lambda { lambda::fetch_listings().await } else { Ok(vec![]) }
        },
        async {
            if run_vastai { vastai::fetch_listings().await } else { Ok(vec![]) }
        },
    );

    let mut listings = Vec::new();
    let mut skipped = Vec::new();

    for (name, result) in [("RunPod", rp), ("Lambda", lm), ("Vast.ai", va)] {
        match result {
            Ok(mut ls) => listings.append(&mut ls),
            Err(e) => skipped.push(SkippedProvider {
                name: name.to_string(),
                reason: e.to_string(),
            }),
        }
    }

    (listings, skipped)
}
