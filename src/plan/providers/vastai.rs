use std::collections::HashMap;

use anyhow::Result;

use crate::plan::{AvailabilityStatus, GpuListing, ListingFlag};

const ENDPOINT: &str = "https://console.vast.ai/api/v0/bundles/";

/// Offers below this reliability score are excluded.
const MIN_RELIABILITY: f64 = 0.90;

pub async fn fetch_listings() -> Result<Vec<GpuListing>> {
    let client = crate::net::client(None)?;
    let query = serde_json::json!({
        "rentable": { "eq": true },
        "order": [["dph_total", "asc"]],
        "type": "on-demand"
    });
    let v: serde_json::Value = client
        .get(ENDPOINT)
        .query(&[("q", query.to_string())])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    parse(&v)
}

/// Cheapest reliable offer per (GPU model, GPU count).
pub fn parse(v: &serde_json::Value) -> Result<Vec<GpuListing>> {
    let offers = v["offers"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("unexpected Vast.ai response shape (no `offers`)"))?;
    let mut best: HashMap<(String, u32), GpuListing> = HashMap::new();
    for o in offers {
        let Some(name) = o["gpu_name"].as_str().filter(|s| !s.is_empty()) else {
            continue;
        };
        let Some(dph) = o["dph_total"].as_f64().filter(|p| *p > 0.0) else {
            continue;
        };
        if !o["rentable"].as_bool().unwrap_or(false) {
            continue;
        }
        let rel = o["reliability2"]
            .as_f64()
            .or_else(|| o["reliability"].as_f64())
            .unwrap_or(0.0);
        if rel < MIN_RELIABILITY {
            continue;
        }
        // gpu_ram is per GPU, in MB.
        let Some(per_gpu_mb) = o["gpu_ram"].as_f64().filter(|r| *r > 0.0) else {
            continue;
        };
        let count = o["num_gpus"].as_u64().unwrap_or(1).max(1) as u32;
        let mut flags = vec![ListingFlag::PriceVolatile];
        if rel < 0.95 {
            flags.push(ListingFlag::LowReliability);
        }
        let listing = GpuListing {
            provider: "Vast.ai".into(),
            gpu_model: name.to_string(),
            gpu_count: count,
            vram_per_gpu_gib: per_gpu_mb / 1024.0,
            hourly_usd: dph,
            availability: AvailabilityStatus::Available,
            flags,
        };
        best.entry((name.to_string(), count))
            .and_modify(|e| {
                if dph < e.hourly_usd {
                    *e = listing.clone();
                }
            })
            .or_insert(listing);
    }
    Ok(best.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_cheapest_per_model_and_count() {
        let v = serde_json::json!({"offers":[
            {"gpu_name":"RTX 4090","dph_total":0.40,"rentable":true,"reliability2":0.99,"gpu_ram":24564,"num_gpus":1},
            {"gpu_name":"RTX 4090","dph_total":0.35,"rentable":true,"reliability2":0.93,"gpu_ram":24564,"num_gpus":1},
            {"gpu_name":"RTX 4090","dph_total":2.80,"rentable":true,"reliability2":0.99,"gpu_ram":24564,"num_gpus":8},
            {"gpu_name":"RTX 4090","dph_total":0.10,"rentable":true,"reliability2":0.50,"gpu_ram":24564,"num_gpus":1},
            {"gpu_name":"RTX 4090","dph_total":0.05,"rentable":false,"reliability2":0.99,"gpu_ram":24564,"num_gpus":1}
        ]});
        let mut l = parse(&v).unwrap();
        l.sort_by_key(|x| x.gpu_count);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].hourly_usd, 0.35);
        assert!(l[0].flags.contains(&ListingFlag::LowReliability));
        assert_eq!(l[1].gpu_count, 8);
        assert!(
            (l[1].vram_per_gpu_gib - 24.0).abs() < 0.1,
            "per GPU, not summed"
        );
    }
}
