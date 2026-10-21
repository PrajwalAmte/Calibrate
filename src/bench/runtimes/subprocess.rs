//! Shared implementation for runtimes measured inside a Python subprocess.
//!
//! The driver script runs warm-up, timed iterations and a throughput window
//! itself (so interpreter start-up is never timed) and reports JSON lines:
//!
//! ```text
//! {"type":"load_ms","value":540}
//! {"type":"latency_us","value":12345}            (one per iteration)
//! {"type":"throughput_rps","value":83.2}
//! {"type":"peak_mem","rss_mib":812.0,"device_mib":2048.0}
//! {"type":"error","message":"..."}
//! ```
//!
//! Argument protocol: `script MODEL BATCH WARMUP ITERS TPUT_WINDOW_S SEQ_LEN SHAPE`
//! where SHAPE is `auto` or the non-batch dimensions joined by `x` (`3x224x224`).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::bench::input::BenchInput;
use crate::bench::runtime::{ReportedMemory, Runtime, SubprocessSettings};

/// Appended to every driver: reports the child's own peak memory, which the
/// parent cannot observe (its RSS is calibrate's, not the model's).
const PEAK_MEM_SNIPPET: &str = r#"

def _calibrate_peak_mem():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    dev = None
    t = sys.modules.get("torch")
    try:
        if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
            dev = t.cuda.max_memory_allocated() / 2**20
        elif t is not None and hasattr(t, "mps") and t.backends.mps.is_available():
            dev = t.mps.driver_allocated_memory() / 2**20
    except Exception:
        dev = None
    emit({"type": "peak_mem", "rss_mib": rss_mib, "device_mib": dev})

_calibrate_peak_mem()
"#;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Record {
    LoadMs {
        value: f64,
    },
    LatencyUs {
        value: u64,
    },
    ThroughputRps {
        value: f64,
    },
    PeakMem {
        rss_mib: f64,
        #[serde(default)]
        device_mib: Option<f64>,
    },
    Error {
        message: String,
    },
    #[serde(other)]
    Other,
}

/// Results of one driver run.
#[derive(Debug, Default, PartialEq)]
pub struct DriverOutput {
    pub load_ms: u64,
    pub latencies_us: Vec<u64>,
    pub throughput_rps: f64,
    pub memory: Option<ReportedMemory>,
}

pub fn parse_output(stdout: &[u8]) -> Result<DriverOutput> {
    let text = String::from_utf8_lossy(stdout);
    let mut out = DriverOutput::default();
    for line in text.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        // Libraries sometimes print JSON-looking noise; skip what we cannot parse.
        let Ok(r) = serde_json::from_str::<Record>(line) else {
            continue;
        };
        match r {
            Record::LoadMs { value } => out.load_ms = value.max(0.0) as u64,
            Record::LatencyUs { value } => out.latencies_us.push(value),
            Record::ThroughputRps { value } => out.throughput_rps = value.max(0.0),
            Record::PeakMem {
                rss_mib,
                device_mib,
            } => {
                out.memory = Some(ReportedMemory {
                    rss_mib,
                    device_mib,
                })
            }
            Record::Error { message } => bail!("{message}"),
            Record::Other => {}
        }
    }
    if out.latencies_us.is_empty() {
        bail!("driver produced no latency samples");
    }
    Ok(out)
}

/// `python3 -c "import <module>"` succeeds.
pub fn python_module_available(module: &str) -> bool {
    crate::util::output_with_timeout(
        Command::new(python()).args(["-c", &format!("import {module}")]),
        Duration::from_secs(60),
    )
    .map(|o| o.status.success())
    .unwrap_or(false)
}

/// Python interpreter: `$CALIBRATE_PYTHON` or `python3`.
pub fn python() -> String {
    std::env::var("CALIBRATE_PYTHON").unwrap_or_else(|_| "python3".into())
}

pub struct SubprocessRuntime {
    name: &'static str,
    script: &'static str,
    settings: SubprocessSettings,
    model_path: Option<PathBuf>,
    result: Option<DriverOutput>,
}

impl SubprocessRuntime {
    pub fn new(name: &'static str, script: &'static str) -> Self {
        Self {
            name,
            script,
            settings: SubprocessSettings::default(),
            model_path: None,
            result: None,
        }
    }

    fn run_driver(&self, model: &Path, batch: usize) -> Result<DriverOutput> {
        let mut f = tempfile::Builder::new()
            .prefix(&format!("calibrate_{}_", self.name))
            .suffix(".py")
            .tempfile()?;
        f.write_all(self.script.as_bytes())?;
        f.write_all(PEAK_MEM_SNIPPET.as_bytes())?;
        f.flush()?;
        let s = &self.settings;
        let mut cmd = Command::new(python());
        cmd.arg(f.path())
            .arg(model)
            .arg(batch.to_string())
            .arg(s.warmup.to_string())
            .arg(s.iterations.to_string())
            .arg(format!("{:.3}", s.throughput_window.as_secs_f64()))
            .arg(s.seq_len.to_string())
            .arg(match &s.input_shape {
                Some(d) if !d.is_empty() => d
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                _ => "auto".to_string(),
            });
        let output = crate::util::output_with_timeout(&mut cmd, s.timeout)
            .with_context(|| format!("{} driver", self.name))?;
        if !output.status.success() {
            // Prefer the structured error the driver emitted, then stderr.
            if let Err(e) = parse_output(&output.stdout) {
                if !e.to_string().contains("no latency samples") {
                    bail!("{e}");
                }
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.lines().rev().take(5).collect();
            bail!(
                "{} driver exited with {}: {}",
                self.name,
                output.status,
                tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
            );
        }
        parse_output(&output.stdout)
    }
}

impl Runtime for SubprocessRuntime {
    fn name(&self) -> &str {
        self.name
    }

    fn configure(&mut self, settings: &SubprocessSettings) {
        self.settings = settings.clone();
    }

    fn load(&mut self, model_path: &Path) -> Result<Duration> {
        if !model_path.exists() {
            bail!("model not found: {}", model_path.display());
        }
        self.model_path = Some(model_path.to_path_buf());
        self.result = None;
        Ok(Duration::ZERO)
    }

    fn infer(&mut self, input: &BenchInput) -> Result<()> {
        if self.result.is_some() {
            return Ok(());
        }
        let model = self
            .model_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("{}: load() not called", self.name))?;
        let batch = input.shape.first().copied().unwrap_or(1);
        self.result = Some(self.run_driver(&model, batch)?);
        Ok(())
    }

    fn unload(&mut self) {
        self.model_path = None;
        self.result = None;
    }

    fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> {
        let r = self.result.as_mut()?;
        if r.latencies_us.is_empty() {
            return None;
        }
        Some((
            std::mem::take(&mut r.latencies_us),
            r.throughput_rps,
            r.load_ms,
        ))
    }

    fn reported_memory(&mut self) -> Option<ReportedMemory> {
        self.result.as_ref().and_then(|r| r.memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_protocol() {
        let out = parse_output(
            b"noise\n{\"type\":\"load_ms\",\"value\":12.5}\n{\"type\":\"latency_us\",\"value\":100}\n\
              {\"type\":\"latency_us\",\"value\":120}\n{\"type\":\"throughput_rps\",\"value\":9.5}\n\
              {\"type\":\"peak_mem\",\"rss_mib\":800.0,\"device_mib\":null}\n{\"type\":\"future\"}\n",
        )
        .unwrap();
        assert_eq!(out.load_ms, 12);
        assert_eq!(out.latencies_us, vec![100, 120]);
        assert_eq!(out.throughput_rps, 9.5);
        assert_eq!(out.memory.unwrap().device_mib, None);
    }

    #[test]
    fn surfaces_driver_errors() {
        let e =
            parse_output(b"{\"type\":\"error\",\"message\":\"CUDA out of memory\"}\n").unwrap_err();
        assert!(e.to_string().contains("out of memory"));
        assert!(parse_output(b"").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn runs_a_driver_end_to_end_with_settings() {
        if !python_module_available("json") {
            return; // no python3 on this machine
        }
        const SCRIPT: &str = r#"
import sys, json
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
model, batch, warm, iters, win, seq = sys.argv[1:7]
emit({"type": "load_ms", "value": 1})
for i in range(int(iters)):
    emit({"type": "latency_us", "value": int(batch) * 10 + int(seq)})
emit({"type": "throughput_rps", "value": float(win)})
"#;
        let mut rt = SubprocessRuntime::new("fake", SCRIPT);
        rt.configure(&SubprocessSettings {
            warmup: 1,
            iterations: 7,
            throughput_window: Duration::from_millis(1500),
            timeout: Duration::from_secs(60),
            seq_len: 5,
            input_shape: None,
        });
        let model = tempfile::NamedTempFile::new().unwrap();
        rt.load(model.path()).unwrap();
        rt.infer(&BenchInput::generate(&[3, 4])).unwrap();
        let (lat, tput, _) = rt.pre_collected_samples().unwrap();
        assert_eq!(lat.len(), 7, "--iterations is honoured");
        assert_eq!(lat[0], 35);
        assert_eq!(tput, 1.5);
        assert!(rt.reported_memory().unwrap().rss_mib > 0.0);
    }

    #[cfg(unix)]
    #[test]
    fn hung_driver_is_killed() {
        if !python_module_available("time") {
            return;
        }
        let mut rt = SubprocessRuntime::new("hang", "import time\ntime.sleep(60)\n");
        rt.configure(&SubprocessSettings {
            timeout: Duration::from_millis(800),
            ..SubprocessSettings::default()
        });
        let model = tempfile::NamedTempFile::new().unwrap();
        rt.load(model.path()).unwrap();
        let e = rt.infer(&BenchInput::generate(&[1, 4])).unwrap_err();
        assert!(format!("{e:#}").contains("timed out"), "{e:#}");
    }
}
