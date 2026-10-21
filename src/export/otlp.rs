//! OTLP/HTTP (JSON encoding) metric push to an OpenTelemetry collector.
//!
//! Uses the protobuf-JSON mapping of `ExportMetricsServiceRequest`, which
//! every OpenTelemetry Collector accepts on `:4318/v1/metrics`. Gauges map to
//! OTLP gauges; the cumulative cost counters map to monotonic sums.

use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::export::{points, Kind, Point};
use crate::session::state::{SessionSnapshot, SnapshotReceiver};

pub struct OtlpConfig {
    pub endpoint: String,
    pub headers: Vec<(String, String)>,
    pub min_interval: Duration,
}

/// `http://host:4318` → `http://host:4318/v1/metrics`.
pub fn metrics_url(endpoint: &str) -> String {
    let e = endpoint.trim_end_matches('/');
    if e.ends_with("/v1/metrics") {
        e.to_string()
    } else {
        format!("{e}/v1/metrics")
    }
}

fn attr(k: &str, v: &str) -> Value {
    json!({"key": k, "value": {"stringValue": v}})
}

/// Build the OTLP JSON request body for one snapshot.
pub fn request_body(s: &SessionSnapshot, start_unix_nano: u128) -> Value {
    let now = (s.timestamp_ms as u128) * 1_000_000;
    let pts = points(s);
    let mut names: Vec<&str> = Vec::new();
    for p in &pts {
        if !names.contains(&p.name) {
            names.push(p.name);
        }
    }
    let metrics: Vec<Value> = names
        .iter()
        .map(|name| {
            let group: Vec<&Point> = pts.iter().filter(|p| p.name == *name).collect();
            let first = group[0];
            let data_points: Vec<Value> = group
                .iter()
                .map(|p| {
                    let attrs: Vec<Value> = p
                        .labels
                        .iter()
                        .filter(|(k, _)| *k != "host")
                        .map(|(k, v)| attr(k, v))
                        .collect();
                    json!({
                        "startTimeUnixNano": start_unix_nano.to_string(),
                        "timeUnixNano": now.to_string(),
                        "asDouble": p.value,
                        "attributes": attrs,
                    })
                })
                .collect();
            match first.kind {
                Kind::Gauge => json!({
                    "name": name, "description": first.help, "unit": "1",
                    "gauge": {"dataPoints": data_points}
                }),
                Kind::Counter => json!({
                    "name": name, "description": first.help, "unit": "1",
                    "sum": {"dataPoints": data_points, "aggregationTemporality": 2, "isMonotonic": true}
                }),
            }
        })
        .collect();

    json!({
        "resourceMetrics": [{
            "resource": {"attributes": [
                attr("service.name", "calibrate"),
                attr("service.version", env!("CARGO_PKG_VERSION")),
                attr("host.name", &s.target.hostname),
            ]},
            "scopeMetrics": [{
                "scope": {"name": "calibrate", "version": env!("CARGO_PKG_VERSION")},
                "metrics": metrics,
            }]
        }]
    })
}

/// Push every new snapshot (rate-limited) until the session ends, then once more.
pub fn spawn(cfg: OtlpConfig, mut rx: SnapshotReceiver) -> Result<tokio::task::JoinHandle<()>> {
    crate::net::ensure_online("--otlp-endpoint")?;
    let client = crate::net::client(Some(Duration::from_secs(5)))?;
    let url = metrics_url(&cfg.endpoint);
    let start = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(tokio::spawn(async move {
        let mut last_push: Option<Instant> = None;
        let mut last_warn: Option<Instant> = None;
        loop {
            let ended = rx.changed().await.is_err();
            if !ended && last_push.is_some_and(|t| t.elapsed() < cfg.min_interval) {
                continue;
            }
            let snap = rx.borrow().clone();
            if let Some(s) = snap {
                let mut req = client.post(&url).json(&request_body(&s, start));
                for (k, v) in &cfg.headers {
                    req = req.header(k, v);
                }
                match req.send().await.and_then(|r| r.error_for_status()) {
                    Ok(_) => debug!("OTLP push ok"),
                    Err(e) => {
                        if last_warn.is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
                            warn!("OTLP push to {url} failed: {e}");
                            last_warn = Some(Instant::now());
                        }
                    }
                }
                last_push = Some(Instant::now());
            }
            if ended {
                break;
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::tests::sample_snapshot;

    #[test]
    fn url_normalisation() {
        assert_eq!(metrics_url("http://c:4318"), "http://c:4318/v1/metrics");
        assert_eq!(metrics_url("http://c:4318/"), "http://c:4318/v1/metrics");
        assert_eq!(
            metrics_url("https://c/otlp/v1/metrics"),
            "https://c/otlp/v1/metrics"
        );
    }

    #[test]
    fn body_has_otlp_shape() {
        let b = request_body(&sample_snapshot(), 1);
        let rm = &b["resourceMetrics"][0];
        assert_eq!(
            rm["resource"]["attributes"][0]["value"]["stringValue"],
            "calibrate"
        );
        let metrics = rm["scopeMetrics"][0]["metrics"].as_array().unwrap();
        let mfu = metrics
            .iter()
            .find(|m| m["name"] == "calibrate_mfu_ratio")
            .unwrap();
        let dp = &mfu["gauge"]["dataPoints"][0];
        assert!(dp["asDouble"].as_f64().unwrap() > 0.0);
        assert!(dp["timeUnixNano"].as_str().unwrap().parse::<u128>().is_ok());
        let wasted = metrics
            .iter()
            .find(|m| m["name"] == "calibrate_cost_wasted_usd_total")
            .unwrap();
        assert_eq!(wasted["sum"]["isMonotonic"], true);
        assert_eq!(wasted["sum"]["aggregationTemporality"], 2);
    }

    #[tokio::test]
    async fn pushes_to_a_collector_with_headers_and_final_flush() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 65536];
            // Read headers + body (Content-Length bounded).
            loop {
                let n = sock.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(h) = text.find("\r\n\r\n") {
                    let len: usize = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= h + 4 + len || n == 0 {
                        break;
                    }
                }
            }
            sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .await
                .unwrap();
            String::from_utf8_lossy(&buf).to_string()
        });

        let (tx, rx) = crate::session::state::new_snapshot_channel();
        let handle = spawn(
            OtlpConfig {
                endpoint: format!("http://{addr}"),
                headers: vec![("Authorization".into(), "Bearer t0ken".into())],
                min_interval: Duration::from_secs(3600),
            },
            rx,
        )
        .unwrap();
        tx.send(Some(sample_snapshot())).unwrap();
        drop(tx); // session end → final flush
        let req = tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .unwrap()
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        assert!(
            req.starts_with("POST /v1/metrics "),
            "{}",
            &req[..40.min(req.len())]
        );
        assert!(req
            .to_ascii_lowercase()
            .contains("authorization: bearer t0ken"));
        assert!(req.contains("calibrate_mfu_ratio"));
        assert!(req.contains("\"service.name\""));
    }
}
