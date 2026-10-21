//! Record and replay of raw sample ticks.
//!
//! `calibrate watch --record-raw trace.jsonl` writes a header line followed by
//! one [`Tick`] per line. `--replay trace.jsonl` feeds those ticks through the
//! exact same analysis pipeline, which makes field reports reproducible and
//! lets the analysis be regression-tested against real traces.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::collectors::runner::RunnerInfo;
use crate::collectors::{GpuScope, Tick};

pub const TRACE_FORMAT_VERSION: u32 = 1;

/// First line of a trace file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHeader {
    pub calibrate_trace: u32,
    pub calibrate_version: String,
    pub backend: String,
    pub metrics_source: String,
    pub scope: GpuScope,
    #[serde(default)]
    pub interval_secs: Option<f64>,
}

/// Appends ticks to a JSONL trace.
pub struct TraceWriter {
    out: BufWriter<File>,
}

impl TraceWriter {
    pub fn create(path: &Path, info: &RunnerInfo, interval: Duration) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut out = BufWriter::new(file);
        let header = TraceHeader {
            calibrate_trace: TRACE_FORMAT_VERSION,
            calibrate_version: env!("CARGO_PKG_VERSION").into(),
            backend: info.backend.clone(),
            metrics_source: info.metrics_source.clone(),
            scope: info.scope,
            interval_secs: Some(interval.as_secs_f64()),
        };
        serde_json::to_writer(&mut out, &header)?;
        out.write_all(b"\n")?;
        Ok(Self { out })
    }

    pub fn write(&mut self, tick: &Tick) -> Result<()> {
        serde_json::to_writer(&mut self.out, tick)?;
        self.out.write_all(b"\n")?;
        self.out.flush()?;
        Ok(())
    }
}

/// Load a whole trace (header optional, for hand-written fixtures).
pub fn load(path: &Path) -> Result<(Option<TraceHeader>, Vec<Tick>)> {
    let file = File::open(path).with_context(|| format!("opening trace {}", path.display()))?;
    let mut header = None;
    let mut ticks = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if n == 0 && line.contains("\"calibrate_trace\"") {
            let h: TraceHeader = serde_json::from_str(line)
                .with_context(|| format!("{}: invalid trace header", path.display()))?;
            if h.calibrate_trace > TRACE_FORMAT_VERSION {
                bail!(
                    "trace format v{} is newer than this calibrate supports (v{})",
                    h.calibrate_trace,
                    TRACE_FORMAT_VERSION
                );
            }
            header = Some(h);
            continue;
        }
        let tick: Tick = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid tick", path.display(), n + 1))?;
        ticks.push(tick);
    }
    if ticks.is_empty() {
        bail!("trace {} contains no samples", path.display());
    }
    Ok((header, ticks))
}

/// Stream a trace into a channel. `speed` 0 = as fast as possible,
/// 1 = original timing, 2 = twice as fast, ...
pub fn spawn(
    path: &Path,
    speed: f64,
    stop: Arc<AtomicBool>,
) -> Result<(flume::Receiver<Tick>, RunnerInfo)> {
    let (header, ticks) = load(path)?;
    let scope = header
        .as_ref()
        .map(|h| h.scope)
        .unwrap_or_else(|| ticks[0].scope);
    let info = RunnerInfo {
        scope,
        // Keep the recorded backend so the analysis behaves as it did live.
        backend: header
            .as_ref()
            .map(|h| h.backend.clone())
            .unwrap_or_else(|| "replay".into()),
        metrics_source: header
            .as_ref()
            .map(|h| h.metrics_source.clone())
            .unwrap_or_else(|| "replay".into()),
        warnings: vec![format!(
            "Replaying {} ({} ticks)",
            path.display(),
            ticks.len()
        )],
    };
    let (tx, rx) = flume::bounded(1024);
    std::thread::Builder::new()
        .name("calibrate-replay".into())
        .spawn(move || {
            let mut prev_ts: Option<u64> = None;
            for tick in ticks {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if speed > 0.0 {
                    if let Some(p) = prev_ts {
                        let gap = tick.timestamp_ms.saturating_sub(p) as f64 / speed;
                        std::thread::sleep(Duration::from_millis(gap.min(60_000.0) as u64));
                    }
                }
                prev_ts = Some(tick.timestamp_ms);
                if tx.send(tick).is_err() {
                    break;
                }
            }
        })?;
    Ok((rx, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::testing::{gpu, tick};

    #[test]
    fn round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let info = RunnerInfo {
            scope: GpuScope::Process,
            backend: "nvml".into(),
            metrics_source: "nvml-gpm".into(),
            warnings: vec![],
        };
        let mut w = TraceWriter::create(&path, &info, Duration::from_secs(2)).unwrap();
        let t1 = tick(1000, vec![gpu(0, 90.0)], Some(120.0));
        let t2 = tick(3000, vec![gpu(0, 10.0)], Some(300.0));
        w.write(&t1).unwrap();
        w.write(&t2).unwrap();
        drop(w);

        let (h, ticks) = load(&path).unwrap();
        let h = h.unwrap();
        assert_eq!(h.metrics_source, "nvml-gpm");
        assert_eq!(ticks, vec![t1, t2]);
    }

    #[test]
    fn rejects_empty_and_future_traces() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("e.jsonl");
        std::fs::write(&empty, "").unwrap();
        assert!(load(&empty).is_err());

        let future = dir.path().join("f.jsonl");
        std::fs::write(
            &future,
            r#"{"calibrate_trace":99,"calibrate_version":"9","backend":"x","metrics_source":"x","scope":"process"}"#,
        )
        .unwrap();
        assert!(load(&future).is_err());
    }

    #[test]
    fn replay_delivers_every_tick_then_closes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let lines: Vec<String> = (0..5)
            .map(|i| serde_json::to_string(&tick(i * 2000, vec![gpu(0, 50.0)], None)).unwrap())
            .collect();
        std::fs::write(&path, lines.join("\n")).unwrap();
        let (rx, info) = spawn(&path, 0.0, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(
            info.backend, "replay",
            "headerless traces are labelled replay"
        );
        let got: Vec<Tick> = rx.iter().collect();
        assert_eq!(got.len(), 5);
    }
}
