use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::gpu_specs::{GpuSpec, SpecsRepository};

/// URL of the canonical GPU spec JSON file.
/// Point this at a raw GitHub URL in the calibrate project once published.
const SPECS_URL: &str =
    "https://raw.githubusercontent.com/your-org/calibrate/main/assets/gpu_specs.json";

/// Cache TTL: re-fetch at most once per 24 hours.
const CACHE_TTL_SECS: u64 = 86_400;

#[derive(Debug, Serialize, Deserialize)]
struct CacheFile {
    fetched_at_unix: u64,
    specs: Vec<GpuSpec>,
}

/// Loads GPU specs, using a local cache when fresh and fetching remotely when stale.
pub struct HttpSpecsClient {
    specs: Vec<GpuSpec>,
}

impl HttpSpecsClient {
    /// Load specs synchronously — intended to be called via `tokio::task::spawn_blocking`.
    pub fn load() -> Self {
        let specs = load_specs_with_cache().unwrap_or_default();
        Self { specs }
    }
}

impl SpecsRepository for HttpSpecsClient {
    fn get_by_name(&self, name: &str) -> Option<GpuSpec> {
        crate::gpu_specs::find_best_match(&self.specs, name).cloned()
    }
}

fn cache_path() -> Option<PathBuf> {
    let dirs = ProjectDirs::from("", "", "calibrate")?;
    let cache = dirs.cache_dir().to_path_buf();
    std::fs::create_dir_all(&cache).ok()?;
    Some(cache.join("gpu-specs.json"))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn load_specs_with_cache() -> Option<Vec<GpuSpec>> {
    let path = cache_path()?;

    // Try cache first.
    if path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Ok(cache) = serde_json::from_str::<CacheFile>(&contents) {
                let age = now_unix().saturating_sub(cache.fetched_at_unix);
                if age < CACHE_TTL_SECS {
                    tracing::debug!("GPU specs loaded from cache (age={age}s)");
                    return Some(cache.specs);
                }
            }
        }
    }

    // Fetch remotely.
    match fetch_remote() {
        Ok(specs) => {
            let cache = CacheFile {
                fetched_at_unix: now_unix(),
                specs: specs.clone(),
            };
            if let Ok(json) = serde_json::to_string_pretty(&cache) {
                let _ = std::fs::write(&path, json);
            }
            tracing::debug!("GPU specs fetched from remote and cached");
            Some(specs)
        }
        Err(e) => {
            tracing::warn!("Remote GPU spec fetch failed: {e}; using cache or fallback");
            // Return stale cache if available rather than nothing.
            if path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    if let Ok(cache) = serde_json::from_str::<CacheFile>(&contents) {
                        return Some(cache.specs);
                    }
                }
            }
            None
        }
    }
}

fn fetch_remote() -> Result<Vec<GpuSpec>, Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(SPECS_URL)?;
    let specs: Vec<GpuSpec> = response.json()?;
    Ok(specs)
}
