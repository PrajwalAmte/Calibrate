use std::collections::HashMap;

use anyhow::Result;
use serde::Deserialize;

use crate::plan::{AvailabilityStatus, ListingFlag};

use super::GpuListing;

const ENDPOINT: &str = "https://console.vast.ai/api/v0/bundles/";
const TIMEOUT_SECS: u64 = 10;

/// Minimum reliability score to include a listing.
/// Machines below this threshold are too unreliable for training runs.
const MIN_RELIABILITY: f64 = 0.90;

#[derive(Deserialize)]
struct VastResponse {
    offers: Vec<VastOffer>,
}

#[derive(Deserialize)]
struct VastOffer {
    gpu_name: Option<String>,
    /// Total GPU RAM in MB for the listing (already multiplied by num_gpus).
    gpu_ram: Option<f64>,
    /// Dollars per hour for the entire listing.
    dph_total: Option<f64>,
    /// Machine reliability score, 0.0–1.0.
    reliability2: Option<f64>,
    rentable: Option<bool>,
    num_gpus: Option<u32>,
}

pub async fn fetch_listings() -> Result<Vec<GpuListing>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
        .user_agent("calibrate/0.1")
        .build()?;

    // Vast.ai accepts a JSON-serialized query object in the `q` query parameter.
    let query = serde_json::json!({
        "rentable": { "eq": true },
        "order": [["dph_total", "asc"]],
        "type": "on-demand"
    });

    let resp: VastResponse = client
        .get(ENDPOINT)
        .query(&[("q", query.to_string())])
        .send()
        .await?
        .json()
        .await?;

    // Group by GPU name, keeping only the cheapest listing per GPU model.
    // This gives users a "current market floor" price for each GPU type.
    let mut best_by_gpu: HashMap<String, GpuListing> = HashMap::new();

    for offer in resp.offers {
        let gpu_name = match offer.gpu_name.filter(|s| !s.is_empty()) {
            Some(n) => n,
            None => continue,
        };
        let dph = match offer.dph_total.filter(|&p| p > 0.0) {
            Some(p) => p,
            None => continue,
        };
        if !offer.rentable.unwrap_or(false) {
            continue;
        }
        let reliability = offer.reliability2.unwrap_or(0.0);
        if reliability < MIN_RELIABILITY {
            continue;
        }

        // gpu_ram is per-GPU in MB; multiply by num_gpus for total listing VRAM.
        let per_gpu_mb = match offer.gpu_ram.filter(|&r| r > 0.0) {
            Some(r) => r,
            None => continue,
        };
        let num_gpus = offer.num_gpus.unwrap_or(1).max(1);
        let vram_gib = per_gpu_mb * num_gpus as f64 / 1024.0;

        let mut flags = vec![ListingFlag::PriceVolatile];
        if reliability < 0.95 {
            flags.push(ListingFlag::LowReliability);
        }

        let listing = GpuListing {
            provider: "Vast.ai".to_string(),
            gpu_model: gpu_name.clone(),
            vram_gib,
            hourly_usd: dph,
            availability: AvailabilityStatus::Available,
            flags,
        };

        best_by_gpu
            .entry(gpu_name)
            .and_modify(|existing| {
                if dph < existing.hourly_usd {
                    *existing = listing.clone();
                }
            })
            .or_insert(listing);
    }

    Ok(best_by_gpu.into_values().collect())
}
