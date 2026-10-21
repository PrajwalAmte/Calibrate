//! Configuration file and process-wide settings.
//!
//! Precedence for every value: command-line flag > environment variable >
//! config file > built-in default. The config file is TOML:
//!
//! ```toml
//! [general]
//! offline = false
//! http_timeout_secs = 10
//! specs_url = "https://raw.githubusercontent.com/PrajwalAmte/Calibrate/main/assets/fallback_specs.json"
//!
//! [watch]
//! interval = 2.0
//! cost_per_hour = 2.49
//! target_mfu = 45.0
//! metrics_source = "auto"
//! prometheus_listen = "0.0.0.0:9400"
//! otlp_endpoint = "http://otel-collector:4318"
//! hook_dir = "/tmp"
//!
//! [plan]
//! providers = ["runpod", "lambda", "vastai"]
//! seq_len = 2048
//! price_cache_ttl_secs = 900
//!
//! [credentials]
//! hf_token = "hf_..."
//! lambda_api_key = "..."
//! runpod_api_key = "..."
//! ```

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::cli::MetricsSource;

/// Default location of the canonical GPU spec database.
pub const DEFAULT_SPECS_URL: &str =
    "https://raw.githubusercontent.com/PrajwalAmte/Calibrate/main/assets/fallback_specs.json";

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigFile {
    pub general: GeneralSection,
    pub watch: WatchSection,
    pub plan: PlanSection,
    pub credentials: CredentialsSection,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralSection {
    pub offline: Option<bool>,
    pub http_timeout_secs: Option<u64>,
    pub specs_url: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WatchSection {
    pub interval: Option<f64>,
    pub cost_per_hour: Option<f64>,
    pub target_mfu: Option<f32>,
    pub metrics_source: Option<MetricsSource>,
    pub prometheus_listen: Option<String>,
    pub otlp_endpoint: Option<String>,
    pub hook_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlanSection {
    pub providers: Option<Vec<String>>,
    pub seq_len: Option<u32>,
    pub price_cache_ttl_secs: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CredentialsSection {
    pub hf_token: Option<String>,
    pub lambda_api_key: Option<String>,
    pub runpod_api_key: Option<String>,
}

impl ConfigFile {
    /// Load the config file.
    ///
    /// An explicitly requested file (`--config` / `CALIBRATE_CONFIG`) must
    /// exist; the default location is optional.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let (path, required) = match explicit {
            Some(p) => (p.to_path_buf(), true),
            None => match default_config_path() {
                Some(p) => (p, false),
                None => return Ok(Self::default()),
            },
        };
        if !path.exists() {
            if required {
                anyhow::bail!("config file not found: {}", path.display());
            }
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing config file {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        Ok(toml::from_str(text)?)
    }
}

/// `~/.config/calibrate/config.toml` on Linux, the platform equivalent elsewhere.
pub fn default_config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "calibrate").map(|d| d.config_dir().join("config.toml"))
}

/// Resolved process-wide settings shared by every subsystem.
#[derive(Debug, Clone)]
pub struct Settings {
    pub offline: bool,
    pub http_timeout: Duration,
    pub specs_url: String,
    pub hf_token: Option<String>,
    pub lambda_api_key: Option<String>,
    pub runpod_api_key: Option<String>,
    pub file: ConfigFile,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            offline: false,
            http_timeout: Duration::from_secs(10),
            specs_url: DEFAULT_SPECS_URL.to_string(),
            hf_token: None,
            lambda_api_key: None,
            runpod_api_key: None,
            file: ConfigFile::default(),
        }
    }
}

impl Settings {
    /// Merge CLI flags, environment and the config file.
    pub fn resolve(file: ConfigFile, cli_offline: bool) -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let g = &file.general;
        let c = &file.credentials;

        let offline = cli_offline || g.offline.unwrap_or(false);
        let http_timeout = Duration::from_secs(g.http_timeout_secs.unwrap_or(10).clamp(1, 300));
        let specs_url = env("CALIBRATE_SPECS_URL")
            .or_else(|| g.specs_url.clone())
            .unwrap_or_else(|| DEFAULT_SPECS_URL.to_string());
        let hf_token = env("HF_TOKEN")
            .or_else(|| env("HUGGING_FACE_HUB_TOKEN"))
            .or_else(|| c.hf_token.clone())
            .or_else(read_hf_token_file);
        let lambda_api_key = env("LAMBDA_API_KEY").or_else(|| c.lambda_api_key.clone());
        let runpod_api_key = env("RUNPOD_API_KEY").or_else(|| c.runpod_api_key.clone());

        Self {
            offline,
            http_timeout,
            specs_url,
            hf_token,
            lambda_api_key,
            runpod_api_key,
            file,
        }
    }
}

/// The token file written by `huggingface-cli login`.
fn read_hf_token_file() -> Option<String> {
    let path = if let Ok(home) = std::env::var("HF_HOME") {
        PathBuf::from(home).join("token")
    } else {
        directories::BaseDirs::new()?
            .home_dir()
            .join(".cache/huggingface/token")
    };
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Install the resolved settings. Called once from `main`.
pub fn init(settings: Settings) {
    let _ = SETTINGS.set(settings);
}

/// Process-wide settings (defaults when `init` was never called, e.g. in tests).
pub fn get() -> &'static Settings {
    SETTINGS.get_or_init(Settings::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_example() {
        let cfg = ConfigFile::parse(
            r#"
            [general]
            offline = true
            http_timeout_secs = 5
            [watch]
            interval = 1.5
            metrics_source = "dcgm"
            prometheus_listen = "0.0.0.0:9400"
            [plan]
            providers = ["runpod"]
            seq_len = 4096
            [credentials]
            hf_token = "hf_x"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.general.offline, Some(true));
        assert_eq!(cfg.watch.interval, Some(1.5));
        assert_eq!(cfg.watch.metrics_source, Some(MetricsSource::Dcgm));
        assert_eq!(cfg.plan.seq_len, Some(4096));
        assert_eq!(cfg.credentials.hf_token.as_deref(), Some("hf_x"));
    }

    #[test]
    fn empty_file_is_default() {
        let cfg = ConfigFile::parse("").unwrap();
        assert!(cfg.general.offline.is_none());
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(ConfigFile::parse("[watch]\nintervall = 2\n").is_err());
    }

    #[test]
    fn missing_explicit_file_is_an_error() {
        assert!(ConfigFile::load(Some(Path::new("/definitely/not/here.toml"))).is_err());
    }

    #[test]
    fn cli_offline_wins() {
        let s = Settings::resolve(ConfigFile::default(), true);
        assert!(s.offline);
        let mut f = ConfigFile::default();
        f.general.offline = Some(true);
        assert!(Settings::resolve(f, false).offline);
    }
}
