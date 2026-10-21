//! Minimal Prometheus scrape endpoint (`GET /metrics`, `GET /healthz`).
//!
//! A tiny hand-rolled HTTP/1.1 responder on tokio avoids pulling a web
//! framework into a CLI. It only serves GETs with small bounded reads.

use std::fmt::Write as _;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::export::{points, Kind};
use crate::session::state::{SessionSnapshot, SnapshotReceiver};

/// Render the text exposition format (version 0.0.4).
pub fn render(snapshot: Option<&SessionSnapshot>) -> String {
    let mut out = String::new();
    out.push_str("# HELP calibrate_up 1 while calibrate is running.\n# TYPE calibrate_up gauge\ncalibrate_up 1\n");
    let Some(s) = snapshot else {
        return out;
    };
    let pts = points(s);
    let mut seen: Vec<&str> = Vec::new();
    for p in &pts {
        if !seen.contains(&p.name) {
            seen.push(p.name);
            let kind = match p.kind {
                Kind::Gauge => "gauge",
                Kind::Counter => "counter",
            };
            let _ = writeln!(out, "# HELP {} {}", p.name, p.help);
            let _ = writeln!(out, "# TYPE {} {}", p.name, kind);
            for q in pts.iter().filter(|q| q.name == p.name) {
                let labels = q
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{k}=\"{}\"", escape(v)))
                    .collect::<Vec<_>>()
                    .join(",");
                let _ = writeln!(out, "{}{{{}}} {}", q.name, labels, fmt_value(q.value));
            }
        }
    }
    out
}

fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Bind and serve in the background. Returns the bound address (useful with port 0).
pub async fn serve(addr: &str, rx: SnapshotReceiver) -> Result<SocketAddr> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding Prometheus endpoint on {addr}"))?;
    let local = listener.local_addr()?;
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            let rx = rx.clone();
            tokio::spawn(async move {
                let _ = tokio::time::timeout(Duration::from_secs(10), handle(stream, rx)).await;
            });
        }
    });
    Ok(local)
}

async fn handle(mut stream: TcpStream, rx: SnapshotReceiver) -> std::io::Result<()> {
    let mut buf = vec![0u8; 8192];
    let mut len = 0;
    loop {
        let n = stream.read(&mut buf[len..]).await?;
        if n == 0 {
            break;
        }
        len += n;
        if buf[..len].windows(4).any(|w| w == b"\r\n\r\n") || len == buf.len() {
            break;
        }
    }
    let req = String::from_utf8_lossy(&buf[..len]);
    let mut parts = req.lines().next().unwrap_or("").split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/").split('?').next().unwrap_or("/");

    let (status, ctype, body) = match (method, path) {
        ("GET", "/metrics") => {
            let snap = rx.borrow().clone();
            (
                "200 OK",
                "text/plain; version=0.0.4; charset=utf-8",
                render(snap.as_ref()),
            )
        }
        ("GET", "/healthz") => ("200 OK", "text/plain", "ok\n".to_string()),
        ("GET", "/") => (
            "200 OK",
            "text/html",
            "<html><body><a href=\"/metrics\">/metrics</a></body></html>\n".to_string(),
        ),
        ("GET", _) => ("404 Not Found", "text/plain", "not found\n".to_string()),
        _ => (
            "405 Method Not Allowed",
            "text/plain",
            "method not allowed\n".to_string(),
        ),
    };
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::tests::sample_snapshot;

    #[test]
    fn exposition_format_is_well_formed() {
        let text = render(Some(&sample_snapshot()));
        assert!(text.contains("# TYPE calibrate_mfu_ratio gauge"));
        assert!(text.contains("# TYPE calibrate_cost_wasted_usd_total counter"));
        // Each metric family is declared exactly once.
        assert_eq!(text.matches("# TYPE calibrate_gpu_throttle ").count(), 1);
        assert_eq!(text.matches("calibrate_gpu_throttle{").count(), 4);
        // Quotes in label values are escaped.
        assert!(text.contains("\\\"special\\\""));
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let (lhs, value) = line.rsplit_once(' ').expect("name value");
            assert!(value.parse::<f64>().is_ok(), "bad value in {line}");
            assert!(lhs.starts_with("calibrate_"), "{line}");
        }
    }

    #[test]
    fn empty_before_first_snapshot() {
        assert_eq!(
            render(None).lines().filter(|l| !l.starts_with('#')).count(),
            1
        );
    }

    #[tokio::test]
    async fn serves_metrics_over_http() {
        let (tx, rx) = crate::session::state::new_snapshot_channel();
        tx.send(Some(sample_snapshot())).unwrap();
        let addr = serve("127.0.0.1:0", rx).await.unwrap();
        let client = reqwest::Client::new();
        let body = client
            .get(format!("http://{addr}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("calibrate_mfu_ratio{"));
        let health = client
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), 200);
        let missing = client
            .get(format!("http://{addr}/nope"))
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status(), 404);
    }
}
