use std::collections::HashMap;

use anyhow::Result;
use serde::Deserialize;

use crate::plan::{AvailabilityStatus, ListingFlag};

use super::GpuListing;

const ENDPOINT: &str = "https://cloud.lambdalabs.com/api/v1/instance-types";
const TIMEOUT_SECS: u64 = 10;

#[derive(Deserialize)]
struct LambdaResponse {
    data: HashMap<String, InstanceEntry>,
}

#[derive(Deserialize)]
struct InstanceEntry {
    instance_type: InstanceType,
    regions_with_capacity_available: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct InstanceType {
    description: Option<String>,
    price_cents_per_hour: Option<u64>,
    specs: InstanceSpecs,
}

#[derive(Deserialize)]
struct InstanceSpecs {
    gpus: Vec<GpuInfo>,
    gpu_count: Option<u32>,
}

#[derive(Deserialize)]
struct GpuInfo {
    name: Option<String>,
    memory_gib: Option<u32>,
}

pub async fn fetch_listings() -> Result<Vec<GpuListing>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
        .user_agent("calibrate/0.1")
        .build()?;

    let resp: LambdaResponse = client.get(ENDPOINT).send().await?.json().await?;

    let mut listings = Vec::new();

    for (_key, entry) in resp.data {
        let price_cents = match entry.instance_type.price_cents_per_hour {
            Some(p) => p,
            None => continue,
        };
        let hourly_usd = price_cents as f64 / 100.0;

        let gpu_count = entry.instance_type.specs.gpu_count.unwrap_or(1).max(1);

        let first_gpu = match entry.instance_type.specs.gpus.first() {
            Some(g) => g,
            None => continue,
        };
        let per_gpu_vram = match first_gpu.memory_gib {
            Some(m) => m as f64,
            None => continue,
        };
        let vram_gib = per_gpu_vram * gpu_count as f64;

        let gpu_name = first_gpu
            .name
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| entry.instance_type.description.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        let has_capacity = !entry.regions_with_capacity_available.is_empty();
        let availability = if has_capacity {
            AvailabilityStatus::Available
        } else {
            AvailabilityStatus::Unavailable
        };

        let mut flags = vec![];
        if !has_capacity {
            flags.push(ListingFlag::CurrentlyUnavailable);
        }

        listings.push(GpuListing {
            provider: "Lambda".to_string(),
            gpu_model: gpu_name,
            vram_gib,
            hourly_usd,
            availability,
            flags,
        });
    }

    Ok(listings)
}
