use anyhow::{bail, Result};
use regex::Regex;

use crate::plan::{AvailabilityStatus, GpuListing, ListingFlag};

const ENDPOINT: &str = "https://cloud.lambdalabs.com/api/v1/instance-types";

/// Lambda's instance-types API requires an API key (`LAMBDA_API_KEY` or
/// `[credentials] lambda_api_key`).
pub async fn fetch_listings() -> Result<Vec<GpuListing>> {
    let Some(key) = crate::config::get().lambda_api_key.clone() else {
        bail!("Lambda's pricing API needs an API key — set LAMBDA_API_KEY to include Lambda");
    };
    let client = crate::net::client(None)?;
    let resp = client
        .get(ENDPOINT)
        .basic_auth(key, Some(""))
        .send()
        .await?;
    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("Lambda rejected the API key (HTTP 401)");
    }
    let v: serde_json::Value = resp.error_for_status()?.json().await?;
    parse(&v)
}

/// "A100 (40 GB SXM4)" → ("A100 SXM4", 40.0)
pub fn parse_gpu_description(desc: &str) -> Option<(String, f64)> {
    let re = Regex::new(
        r"^\s*(?P<model>.+?)\s*\((?P<mem>\d+(?:\.\d+)?)\s*GB(?:\s+(?P<form>[^)]+))?\)\s*$",
    )
    .expect("valid regex");
    let c = re.captures(desc)?;
    let mut model = c["model"].trim().to_string();
    if let Some(form) = c.name("form") {
        model = format!("{model} {}", form.as_str().trim());
    }
    Some((model, c["mem"].parse().ok()?))
}

fn count_from_name(name: &str) -> Option<u32> {
    let re = Regex::new(r"gpu_(\d+)x_").expect("valid regex");
    re.captures(name)?.get(1)?.as_str().parse().ok()
}

pub fn parse(v: &serde_json::Value) -> Result<Vec<GpuListing>> {
    let Some(data) = v["data"].as_object() else {
        bail!("unexpected Lambda response shape (no `data` object)");
    };
    let mut out = Vec::new();
    for (key, entry) in data {
        let it = &entry["instance_type"];
        let Some(cents) = it["price_cents_per_hour"].as_f64().filter(|c| *c > 0.0) else {
            continue;
        };
        let name = it["name"].as_str().unwrap_or(key);
        let count = it["specs"]["gpus"]
            .as_u64()
            .map(|n| n as u32)
            .or_else(|| count_from_name(name))
            .unwrap_or(1)
            .max(1);
        let desc = it["gpu_description"]
            .as_str()
            .or_else(|| it["description"].as_str())
            .unwrap_or("");
        // "8x H100 (80 GB SXM5)" → strip the count prefix if present.
        let desc = desc
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches('x')
            .trim();
        let Some((model, vram)) = parse_gpu_description(desc) else {
            continue;
        };
        let has_capacity = entry["regions_with_capacity_available"]
            .as_array()
            .is_some_and(|a| !a.is_empty());
        out.push(GpuListing {
            provider: "Lambda".into(),
            gpu_model: model,
            gpu_count: count,
            vram_per_gpu_gib: vram,
            hourly_usd: cents / 100.0,
            availability: if has_capacity {
                AvailabilityStatus::Available
            } else {
                AvailabilityStatus::Unavailable
            },
            flags: if has_capacity {
                vec![]
            } else {
                vec![ListingFlag::CurrentlyUnavailable]
            },
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn description_parsing() {
        assert_eq!(
            parse_gpu_description("A100 (40 GB SXM4)"),
            Some(("A100 SXM4".into(), 40.0))
        );
        assert_eq!(
            parse_gpu_description("A10 (24 GB PCIe)"),
            Some(("A10 PCIe".into(), 24.0))
        );
        assert_eq!(
            parse_gpu_description("GH200 (96 GB)"),
            Some(("GH200".into(), 96.0))
        );
        assert_eq!(parse_gpu_description("weird"), None);
    }

    #[test]
    fn parses_multi_gpu_instances_per_gpu() {
        let v = serde_json::json!({"data": {
            "gpu_8x_h100_sxm5": {"instance_type": {
                "name":"gpu_8x_h100_sxm5","description":"8x H100 (80 GB SXM5)",
                "gpu_description":"H100 (80 GB SXM5)","price_cents_per_hour":2392,
                "specs":{"vcpus":208,"memory_gib":1800,"storage_gib":24780,"gpus":8}},
                "regions_with_capacity_available":[{"name":"us-east-1"}]},
            "gpu_1x_a10": {"instance_type": {
                "name":"gpu_1x_a10","description":"1x A10 (24 GB PCIe)",
                "price_cents_per_hour":75,"specs":{"gpus":1}},
                "regions_with_capacity_available":[]}
        }});
        let mut l = parse(&v).unwrap();
        l.sort_by(|a, b| a.gpu_count.cmp(&b.gpu_count));
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].gpu_model, "A10 PCIe");
        assert!(l[0].flags.contains(&ListingFlag::CurrentlyUnavailable));
        assert_eq!(l[1].gpu_count, 8);
        assert_eq!(l[1].vram_per_gpu_gib, 80.0, "VRAM is per GPU, not summed");
        assert_eq!(l[1].hourly_usd, 23.92);
    }

    #[test]
    fn count_from_instance_name() {
        assert_eq!(count_from_name("gpu_4x_a100"), Some(4));
        assert_eq!(count_from_name("cpu_only"), None);
    }
}
