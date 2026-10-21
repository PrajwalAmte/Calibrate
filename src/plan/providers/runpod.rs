use anyhow::{bail, Result};
use serde::Deserialize;

use crate::plan::{AvailabilityStatus, GpuListing, ListingFlag};

const ENDPOINT: &str = "https://api.runpod.io/graphql";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuType {
    display_name: Option<String>,
    id: String,
    memory_in_gb: Option<f64>,
    /// Secure Cloud on-demand price per GPU-hour.
    secure_price: Option<f64>,
    /// Community Cloud on-demand price per GPU-hour (third-party hosts).
    community_price: Option<f64>,
}

pub async fn fetch_listings() -> Result<Vec<GpuListing>> {
    let settings = crate::config::get();
    let client = crate::net::client(None)?;
    let body = serde_json::json!({
        "query": "{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }"
    });
    let mut req = client.post(ENDPOINT).json(&body);
    if let Some(k) = &settings.runpod_api_key {
        req = req.bearer_auth(k);
    }
    let resp = req.send().await?.error_for_status()?;
    let v: serde_json::Value = resp.json().await?;
    parse(&v)
}

pub fn parse(v: &serde_json::Value) -> Result<Vec<GpuListing>> {
    if let Some(errs) = v.get("errors").and_then(|e| e.as_array()) {
        if v.get("data").is_none_or(|d| d.is_null()) {
            let msg = errs
                .first()
                .and_then(|e| e["message"].as_str())
                .unwrap_or("unknown error");
            bail!("RunPod API error: {msg}");
        }
    }
    let types: Vec<GpuType> = serde_json::from_value(v["data"]["gpuTypes"].clone())
        .map_err(|e| anyhow::anyhow!("unexpected RunPod response shape: {e}"))?;
    let mut out = Vec::new();
    for g in types {
        let Some(vram) = g.memory_in_gb.filter(|v| *v > 0.0) else {
            continue;
        };
        let name = g.display_name.filter(|s| !s.is_empty()).unwrap_or(g.id);
        if let Some(p) = g.secure_price.filter(|p| *p > 0.0) {
            out.push(GpuListing {
                provider: "RunPod".into(),
                gpu_model: name.clone(),
                gpu_count: 1,
                vram_per_gpu_gib: vram,
                hourly_usd: p,
                availability: AvailabilityStatus::Available,
                flags: vec![],
            });
        }
        if let Some(p) = g.community_price.filter(|p| *p > 0.0) {
            out.push(GpuListing {
                provider: "RunPod".into(),
                gpu_model: format!("{name} (community)"),
                gpu_count: 1,
                vram_per_gpu_gib: vram,
                hourly_usd: p,
                availability: AvailabilityStatus::Available,
                flags: vec![ListingFlag::CommunityCloud],
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_secure_and_community() {
        let v = serde_json::json!({"data":{"gpuTypes":[
            {"id":"NVIDIA H100 80GB HBM3","displayName":"H100 SXM","memoryInGb":80,"securePrice":2.99,"communityPrice":2.69},
            {"id":"x","displayName":"Ghost","memoryInGb":0,"securePrice":1.0},
            {"id":"y","displayName":"NoPrice","memoryInGb":24,"securePrice":null,"communityPrice":null}
        ]}});
        let l = parse(&v).unwrap();
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].gpu_model, "H100 SXM");
        assert!(l[1].flags.contains(&ListingFlag::CommunityCloud));
        assert!(
            !l[1].flags.contains(&ListingFlag::Spot),
            "community cloud is not spot"
        );
    }

    #[test]
    fn surfaces_api_errors() {
        let v = serde_json::json!({"errors":[{"message":"rate limited"}],"data":null});
        assert!(parse(&v).unwrap_err().to_string().contains("rate limited"));
    }
}
