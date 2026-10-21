//! Remote GPU spec database with an on-disk cache.
//!
//! The remote file has the same format as `assets/fallback_specs.json`. It is
//! fetched at most once per day; after a failed attempt calibrate waits six
//! hours before retrying, so firewalled clusters do not pay a timeout on every
//! run. `--offline` (or `[general] offline = true`) disables fetching
//! entirely; `CALIBRATE_SPECS_URL` points at a mirror.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::gpu_specs::GpuSpec;

const CACHE_TTL_SECS: u64 = 86_400;
const RETRY_AFTER_FAILURE_SECS: u64 = 6 * 3600;
const FETCH_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    #[serde(default)]
    fetched_at_unix: u64,
    #[serde(default)]
    last_attempt_unix: u64,
    #[serde(default)]
    source_url: String,
    #[serde(default)]
    specs: Vec<GpuSpec>,
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

fn read_cache(path: &PathBuf) -> Option<CacheFile> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn write_cache(path: &PathBuf, cache: &CacheFile) {
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

/// Specs from cache or network; `None` when neither is usable.
pub fn load_remote_specs() -> Option<Vec<GpuSpec>> {
    let settings = crate::config::get();
    let path = cache_path();
    let mut cache = path.as_ref().and_then(read_cache).unwrap_or_default();
    let url = settings.specs_url.clone();
    let now = now_unix();
    let same_source = cache.source_url == url;

    let fresh = same_source && now.saturating_sub(cache.fetched_at_unix) < CACHE_TTL_SECS;
    let backing_off =
        same_source && now.saturating_sub(cache.last_attempt_unix) < RETRY_AFTER_FAILURE_SECS;
    if fresh || settings.offline || backing_off {
        return (same_source && !cache.specs.is_empty()).then_some(cache.specs);
    }

    cache.last_attempt_unix = now;
    match fetch(&url) {
        Ok(specs) if !specs.is_empty() => {
            tracing::debug!(n = specs.len(), "GPU specs fetched from {url}");
            cache.fetched_at_unix = now;
            cache.source_url = url;
            cache.specs = specs;
        }
        Ok(_) => tracing::warn!("GPU spec database at {url} is empty; using built-in table"),
        Err(e) => {
            tracing::warn!("GPU spec fetch from {url} failed ({e}); using cached/built-in table");
            if !same_source {
                cache.specs.clear();
                cache.source_url = url;
            }
        }
    }
    if let Some(p) = &path {
        write_cache(p, &cache);
    }
    (!cache.specs.is_empty()).then_some(cache.specs)
}

fn fetch(url: &str) -> anyhow::Result<Vec<GpuSpec>> {
    let client = crate::net::blocking_client(Some(FETCH_TIMEOUT))?;
    let resp = client.get(url).send()?.error_for_status()?;
    Ok(resp.json()?)
}
