//! Cloud GPU pricing: live provider APIs, an on-disk cache, or a local file.

pub mod lambda;
pub mod runpod;
pub mod vastai;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::plan::{GpuListing, SkippedProvider};

pub const PROVIDERS: &[&str] = &["runpod", "lambda", "vastai"];
const DEFAULT_CACHE_TTL_SECS: u64 = 15 * 60;

pub fn canonical_provider(name: &str) -> Option<&'static str> {
    match name.trim().to_lowercase().as_str() {
        "runpod" => Some("runpod"),
        "lambda" | "lambdalabs" | "lambda-labs" => Some("lambda"),
        "vastai" | "vast.ai" | "vast" => Some("vastai"),
        _ => None,
    }
}

/// Result of a pricing lookup.
pub struct Prices {
    pub listings: Vec<GpuListing>,
    pub skipped: Vec<SkippedProvider>,
    /// "live", "cache", "live+cache" or "file".
    pub source: String,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    fetched_at_unix: u64,
    listings: Vec<GpuListing>,
}

fn cache_path(provider: &str) -> Option<PathBuf> {
    let d = directories::ProjectDirs::from("", "", "calibrate")?;
    let dir = d.cache_dir().to_path_buf();
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(format!("prices-{provider}.json")))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn read_cache(provider: &str, max_age: Option<u64>) -> Option<(Vec<GpuListing>, u64)> {
    let e: CacheEntry =
        serde_json::from_str(&std::fs::read_to_string(cache_path(provider)?).ok()?).ok()?;
    let age = now().saturating_sub(e.fetched_at_unix);
    if max_age.is_some_and(|m| age > m) {
        return None;
    }
    Some((e.listings, age))
}

fn write_cache(provider: &str, listings: &[GpuListing]) {
    if let Some(p) = cache_path(provider) {
        let e = CacheEntry {
            fetched_at_unix: now(),
            listings: listings.to_vec(),
        };
        if let Ok(json) = serde_json::to_string(&e) {
            let _ = std::fs::write(p, json);
        }
    }
}

/// Load listings from a JSON file: either an array of listings or
/// `{"listings": [...]}`. See `docs/configuration.md` for the format.
pub fn load_prices_file(path: &Path) -> Result<Vec<GpuListing>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum File {
        List(Vec<GpuListing>),
        Wrapped { listings: Vec<GpuListing> },
    }
    let listings = match serde_json::from_str::<File>(&text)
        .with_context(|| format!("{}: expected a list of GPU listings", path.display()))?
    {
        File::List(l) | File::Wrapped { listings: l } => l,
    };
    for (i, l) in listings.iter().enumerate() {
        if !(l.hourly_usd.is_finite() && l.hourly_usd >= 0.0 && l.vram_per_gpu_gib > 0.0) {
            anyhow::bail!(
                "{}: listing {i} ({}) has an invalid price or VRAM",
                path.display(),
                l.gpu_model
            );
        }
    }
    Ok(listings)
}

/// Fetch every requested provider concurrently, using fresh cache entries
/// when allowed. One failing provider never hides the others.
pub async fn fetch_all(filter: Option<&[String]>, refresh: bool) -> Result<Prices> {
    let settings = crate::config::get();
    let ttl = settings
        .file
        .plan
        .price_cache_ttl_secs
        .unwrap_or(DEFAULT_CACHE_TTL_SECS);

    let wanted: Vec<&'static str> = match filter {
        Some(list) => {
            let mut v = Vec::new();
            for p in list {
                let c = canonical_provider(p).with_context(|| {
                    format!("unknown provider '{p}' (valid: {})", PROVIDERS.join(", "))
                })?;
                if !v.contains(&c) {
                    v.push(c);
                }
            }
            v
        }
        None => PROVIDERS.to_vec(),
    };

    let mut listings = Vec::new();
    let mut skipped = Vec::new();
    let mut used_cache = false;
    let mut used_live = false;
    let mut to_fetch = Vec::new();

    for p in &wanted {
        if settings.offline {
            match read_cache(p, None) {
                Some((l, age)) => {
                    used_cache = true;
                    skipped_note(&mut skipped, p, age);
                    listings.extend(l);
                }
                None => skipped.push(SkippedProvider {
                    name: display_name(p).into(),
                    reason: "offline and no cached prices — use --prices-file".into(),
                }),
            }
        } else if !refresh {
            match read_cache(p, Some(ttl)) {
                Some((l, _)) => {
                    used_cache = true;
                    listings.extend(l);
                }
                None => to_fetch.push(*p),
            }
        } else {
            to_fetch.push(*p);
        }
    }

    let fetch = |p: &'static str| async move {
        let r = match p {
            "runpod" => runpod::fetch_listings().await,
            "lambda" => lambda::fetch_listings().await,
            _ => vastai::fetch_listings().await,
        };
        (p, r)
    };
    let results = futures_join(to_fetch.into_iter().map(fetch).collect()).await;
    for (p, r) in results {
        match r {
            Ok(l) => {
                used_live = true;
                write_cache(p, &l);
                listings.extend(l);
            }
            Err(e) => {
                // Fall back to a stale cache rather than showing nothing.
                if let Some((l, age)) = read_cache(p, None) {
                    used_cache = true;
                    skipped.push(SkippedProvider {
                        name: display_name(p).into(),
                        reason: format!("{e:#} — using cached prices from {} min ago", age / 60),
                    });
                    listings.extend(l);
                } else {
                    skipped.push(SkippedProvider {
                        name: display_name(p).into(),
                        reason: format!("{e:#}"),
                    });
                }
            }
        }
    }
    let source = match (used_live, used_cache) {
        (true, true) => "live+cache",
        (false, true) => "cache",
        _ => "live",
    };
    Ok(Prices {
        listings,
        skipped,
        source: source.into(),
    })
}

fn skipped_note(skipped: &mut Vec<SkippedProvider>, p: &str, age: u64) {
    if age > DEFAULT_CACHE_TTL_SECS {
        skipped.push(SkippedProvider {
            name: display_name(p).into(),
            reason: format!("offline — prices are {} h old", age / 3600),
        });
    }
}

pub fn display_name(p: &str) -> &'static str {
    match p {
        "runpod" => "RunPod",
        "lambda" => "Lambda",
        _ => "Vast.ai",
    }
}

/// Await futures concurrently without extra dependencies.
async fn futures_join<F, T>(futs: Vec<F>) -> Vec<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let handles: Vec<_> = futs.into_iter().map(tokio::spawn).collect();
    let mut out = Vec::new();
    for h in handles {
        if let Ok(v) = h.await {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_aliases() {
        assert_eq!(canonical_provider("Vast.ai"), Some("vastai"));
        assert_eq!(canonical_provider("LAMBDA"), Some("lambda"));
        assert_eq!(canonical_provider("aws"), None);
    }

    #[test]
    fn prices_file_formats_and_validation() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.json");
        std::fs::write(
            &a,
            r#"[{"provider":"onprem","gpu_model":"H100 SXM","gpu_count":8,"vram_per_gpu_gib":80,"hourly_usd":20.0}]"#,
        )
        .unwrap();
        let l = load_prices_file(&a).unwrap();
        assert_eq!(l[0].gpu_count, 8);
        let b = dir.path().join("b.json");
        std::fs::write(
            &b,
            r#"{"listings":[{"provider":"x","gpu_model":"L4","vram_per_gpu_gib":24,"hourly_usd":0.5}]}"#,
        )
        .unwrap();
        assert_eq!(load_prices_file(&b).unwrap().len(), 1);
        let c = dir.path().join("c.json");
        std::fs::write(
            &c,
            r#"[{"provider":"x","gpu_model":"L4","vram_per_gpu_gib":0,"hourly_usd":0.5}]"#,
        )
        .unwrap();
        assert!(load_prices_file(&c).is_err());
    }

    #[tokio::test]
    async fn unknown_provider_is_an_error() {
        let r = fetch_all(Some(&["aws".to_string()]), false).await;
        assert!(r.is_err());
    }
}
