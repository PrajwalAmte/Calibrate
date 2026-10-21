//! Shared HTTP client construction.
//!
//! Every outbound request goes through these helpers so timeouts, the user
//! agent and `--offline` are enforced consistently. Proxies are honoured via
//! the standard `HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY` variables and TLS
//! uses the operating system trust store, so corporate CAs work.

use std::time::Duration;

use anyhow::{bail, Result};

pub const USER_AGENT: &str = concat!("calibrate/", env!("CARGO_PKG_VERSION"));

/// Fail fast when the user asked for no network access.
pub fn ensure_online(what: &str) -> Result<()> {
    if crate::config::get().offline {
        bail!("{what} requires network access, but --offline is set");
    }
    Ok(())
}

/// Async client with the configured timeout (or `timeout` when given).
pub fn client(timeout: Option<Duration>) -> Result<reqwest::Client> {
    let t = timeout.unwrap_or(crate::config::get().http_timeout);
    Ok(reqwest::Client::builder()
        .timeout(t)
        .connect_timeout(t.min(Duration::from_secs(5)))
        .user_agent(USER_AGENT)
        .build()?)
}

/// Blocking client for code that runs on plain threads.
pub fn blocking_client(timeout: Option<Duration>) -> Result<reqwest::blocking::Client> {
    let t = timeout.unwrap_or(crate::config::get().http_timeout);
    Ok(reqwest::blocking::Client::builder()
        .timeout(t)
        .connect_timeout(t.min(Duration::from_secs(5)))
        .user_agent(USER_AGENT)
        .build()?)
}
