use anyhow::Result;
use serde::Deserialize;

use crate::plan::{AvailabilityStatus, ListingFlag};

use super::GpuListing;

const ENDPOINT: &str = "https://api.runpod.io/graphql";
const TIMEOUT_SECS: u64 = 10;

#[derive(Deserialize)]
struct GraphQlResponse {
    data: Option<GpuTypesData>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuTypesData {
    gpu_types: Option<Vec<GpuType>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuType {
    display_name: Option<String>,
    id: String,
    memory_in_gb: Option<f64>,
    /// On-demand "secure pod" price (USD/hr). `null` when none are available.
    secure_price: Option<f64>,
    /// Community cloud (spot / interruptible) price. `null` when unavailable.
    community_price: Option<f64>,
}

pub async fn fetch_listings() -> Result<Vec<GpuListing>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
        .user_agent("calibrate/0.1")
        .build()?;

    let body = serde_json::json!({
        "query": "{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }"
    });

    let resp: serde_json::Value = client
        .post(ENDPOINT)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    // Deserialize the inner data structure; tolerate partial / malformed responses.
    let gpu_types: Vec<GpuType> = match serde_json::from_value::<GraphQlResponse>(resp) {
        Ok(r) => r.data.and_then(|d| d.gpu_types).unwrap_or_default(),
        Err(_) => vec![],
    };

    let mut listings = Vec::new();

    for g in gpu_types {
        let vram_gib = match g.memory_in_gb {
            Some(v) if v > 0.0 => v,
            _ => continue,
        };
        let gpu_name = g.display_name.filter(|s| !s.is_empty()).unwrap_or(g.id);

        // On-demand (secure pod) listing.
        if let Some(price) = g.secure_price.filter(|&p| p > 0.0) {
            listings.push(GpuListing {
                provider: "RunPod".to_string(),
                gpu_model: gpu_name.clone(),
                vram_gib,
                hourly_usd: price,
                availability: AvailabilityStatus::Available,
                flags: vec![],
            });
        }

        // Community cloud (spot / preemptible) listing.
        if let Some(price) = g.community_price.filter(|&p| p > 0.0) {
            listings.push(GpuListing {
                provider: "RunPod".to_string(),
                gpu_model: format!("{gpu_name} (spot)"),
                vram_gib,
                hourly_usd: price,
                availability: AvailabilityStatus::Available,
                flags: vec![ListingFlag::Spot],
            });
        }
    }

    Ok(listings)
}
