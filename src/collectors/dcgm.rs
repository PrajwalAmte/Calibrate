//! DCGM profiling metrics via `dcgmi dmon`.
//!
//! DCGM's DCP fields give SM activity, occupancy, tensor-pipe activity and
//! DRAM activity on data-centre GPUs from Volta onwards (A100, L4, L40S, …),
//! which is where GPM is unavailable. Rather than linking libdcgm we run
//! `dcgmi dmon` as a child process and parse its rolling output; this works
//! anywhere the DCGM host engine (`nv-hostengine`) is reachable.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::collectors::{ProfMetrics, ProfSource};

/// DCGM field IDs, in the column order we request them.
pub const FIELD_SM_ACTIVE: u32 = 1002;
pub const FIELD_SM_OCCUPANCY: u32 = 1003;
pub const FIELD_TENSOR_ACTIVE: u32 = 1004;
pub const FIELD_DRAM_ACTIVE: u32 = 1005;

type Latest = Arc<Mutex<HashMap<u32, (Instant, ProfMetrics)>>>;

/// A running `dcgmi dmon` child plus the latest values per GPU.
pub struct DcgmReader {
    child: Child,
    latest: Latest,
    max_age: Duration,
}

impl DcgmReader {
    /// `true` when `dcgmi` is on PATH and the host engine answers.
    pub fn available() -> bool {
        Command::new("dcgmi")
            .arg("discovery")
            .arg("-l")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Start streaming for `gpus` (all GPUs when empty) every `interval`.
    pub fn start(gpus: &[u32], interval: Duration) -> std::io::Result<Self> {
        let ms = interval.as_millis().clamp(100, 3_600_000).to_string();
        let mut cmd = Command::new("dcgmi");
        cmd.args(["dmon", "-e"])
            .arg(format!(
                "{FIELD_SM_ACTIVE},{FIELD_SM_OCCUPANCY},{FIELD_TENSOR_ACTIVE},{FIELD_DRAM_ACTIVE}"
            ))
            .args(["-d", &ms]);
        if !gpus.is_empty() {
            let list: Vec<String> = gpus.iter().map(|g| g.to_string()).collect();
            cmd.args(["-i", &list.join(",")]);
        }
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()?;
        let stdout = child.stdout.take().expect("piped stdout");
        let latest: Latest = Arc::default();
        let sink = latest.clone();
        std::thread::Builder::new()
            .name("dcgmi-reader".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if let Some((gpu, m)) = parse_dmon_line(&line) {
                        sink.lock().insert(gpu, (Instant::now(), m));
                    }
                }
            })?;
        Ok(Self {
            child,
            latest,
            max_age: interval * 3 + Duration::from_secs(1),
        })
    }

    /// Latest metrics for a GPU, if fresh.
    pub fn get(&self, gpu: u32) -> Option<ProfMetrics> {
        let map = self.latest.lock();
        let (at, m) = map.get(&gpu)?;
        (at.elapsed() <= self.max_age).then_some(*m)
    }

    /// `false` once the child process has exited.
    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for DcgmReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parse one data line of `dcgmi dmon -e 1002,1003,1004,1005`, e.g.
/// `GPU 0     0.512    0.250    0.311    0.120`. Values are ratios (0–1)
/// and may be `N/A`. Header and comment lines return `None`.
pub fn parse_dmon_line(line: &str) -> Option<(u32, ProfMetrics)> {
    let mut parts = line.split_whitespace();
    if parts.next()? != "GPU" {
        return None;
    }
    let gpu: u32 = parts.next()?.parse().ok()?;
    let vals: Vec<Option<f32>> = parts
        .map(|v| {
            v.parse::<f32>()
                .ok()
                .filter(|x| x.is_finite())
                .map(|x| (x * 100.0).clamp(0.0, 100.0))
        })
        .collect();
    if vals.len() < 4 || vals.iter().all(Option::is_none) {
        return None;
    }
    Some((
        gpu,
        ProfMetrics {
            source: ProfSource::Dcgm,
            sm_active_pct: vals[0],
            sm_occupancy_pct: vals[1],
            tensor_active_pct: vals[2],
            dram_active_pct: vals[3],
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_line() {
        let (gpu, m) = parse_dmon_line("GPU 3     0.512    0.250    0.311    0.120").unwrap();
        assert_eq!(gpu, 3);
        assert!((m.sm_active_pct.unwrap() - 51.2).abs() < 1e-3);
        assert!((m.tensor_active_pct.unwrap() - 31.1).abs() < 1e-3);
        assert!((m.dram_active_pct.unwrap() - 12.0).abs() < 1e-3);
        assert_eq!(m.source, ProfSource::Dcgm);
    }

    #[test]
    fn handles_na_values() {
        let (_, m) = parse_dmon_line("GPU 0  N/A  0.100  N/A  0.050").unwrap();
        assert_eq!(m.sm_active_pct, None);
        assert!(m.sm_occupancy_pct.is_some());
    }

    #[test]
    fn ignores_headers_and_all_na() {
        assert!(parse_dmon_line("#Entity   SMACT  SMOCC  TENSO  DRAMA").is_none());
        assert!(parse_dmon_line("ID").is_none());
        assert!(parse_dmon_line("").is_none());
        assert!(parse_dmon_line("GPU 0 N/A N/A N/A N/A").is_none());
        assert!(parse_dmon_line("GPU 0 0.1 0.2").is_none());
    }
}
