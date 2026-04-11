use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::{Runtime, RuntimeDescriptor};

// ── Embedded Python driver script

const DRIVER_SCRIPT: &str = r#"
import sys, json, time
import numpy as np

model_path      = sys.argv[1]
batch_size      = int(sys.argv[2])
warmup_iters    = int(sys.argv[3])
measure_iters   = int(sys.argv[4])
tput_window_s   = float(sys.argv[5])

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

try:
    import torch
except ImportError as e:
    emit({"type": "error", "message": f"torch not installed: {e}"})
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
try:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    # Try GPU if available.
    if torch.cuda.is_available():
        try:
            model = model.cuda()
            device = "cuda"
        except Exception:
            device = "cpu"
    else:
        device = "cpu"
except Exception as e:
    emit({"type": "error", "message": f"load failed: {e}"})
    sys.exit(1)
load_ms = (time.perf_counter() - t0) * 1000
emit({"type": "load_ms", "value": load_ms})

# ── Build input ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=0xCAFEBABE)

# Attempt to infer input shape from the model's graph.
# Fall back to a [batch, 512] tensor if introspection fails.
try:
    graph = model.graph
    first_input = list(graph.inputs())[0]
    type_info = first_input.type()
    sizes = list(type_info.sizes())
    shape = [batch_size if (i == 0 or s <= 0) else s for i, s in enumerate(sizes)]
except Exception:
    shape = [batch_size, 512]

inp = torch.tensor(rng.random(shape, dtype=np.float32))
if device == "cuda":
    inp = inp.cuda()

def run_once():
    with torch.no_grad():
        model(inp)
    if device == "cuda":
        torch.cuda.synchronize()

# ── Warm-up ───────────────────────────────────────────────────────────────────
for _ in range(warmup_iters):
    run_once()

# ── Measurement ───────────────────────────────────────────────────────────────
for _ in range(measure_iters):
    t0 = time.perf_counter()
    run_once()
    elapsed_us = int((time.perf_counter() - t0) * 1_000_000)
    emit({"type": "latency_us", "value": elapsed_us})

# ── Throughput window ─────────────────────────────────────────────────────────
count = 0
deadline = time.perf_counter() + tput_window_s
while time.perf_counter() < deadline:
    run_once()
    count += 1
elapsed = time.perf_counter() - (deadline - tput_window_s)
emit({"type": "throughput_rps", "value": count / elapsed if elapsed > 0 else 0})
"#;

// ── Deserialization helpers

#[derive(Deserialize)]
#[serde(tag = "type")]
enum DriverRecord {
    #[serde(rename = "load_ms")]
    LoadMs { value: f64 },
    #[serde(rename = "latency_us")]
    LatencyUs { value: u64 },
    #[serde(rename = "throughput_rps")]
    ThroughputRps { value: f64 },
    #[serde(rename = "error")]
    Error { message: String },
}

fn parse_driver_output(stdout: &[u8]) -> Result<Vec<DriverRecord>> {
    let text = std::str::from_utf8(stdout)?;
    let mut records = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: DriverRecord = serde_json::from_str(line)
            .with_context(|| format!("failed to parse driver line: {line}"))?;
        records.push(record);
    }
    Ok(records)
}

// ── TorchScriptRuntime

pub struct TorchScriptRuntime {
    model_path: Option<std::path::PathBuf>,
    latency_samples_us: Vec<u64>,
    throughput_rps: f64,
    load_ms: u64,
}

impl TorchScriptRuntime {
    pub fn new() -> Self {
        Self {
            model_path: None,
            latency_samples_us: Vec::new(),
            throughput_rps: 0.0,
            load_ms: 0,
        }
    }

    pub fn descriptor() -> RuntimeDescriptor {
        RuntimeDescriptor {
            name: "torchscript",
            is_available: || {
                Command::new("python3")
                    .args(["-c", "import torch; print(torch.__version__)"])
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            },
            is_compatible: |fmt| matches!(fmt, ModelFormat::TorchScript),
            create: || Box::new(TorchScriptRuntime::new()),
        }
    }

    fn write_driver() -> Result<tempfile::NamedTempFile> {
        let mut f = tempfile::Builder::new()
            .prefix("calibrate_ts_")
            .suffix(".py")
            .tempfile()?;
        f.write_all(DRIVER_SCRIPT.as_bytes())?;
        f.flush()?;
        Ok(f)
    }
}

impl Runtime for TorchScriptRuntime {
    fn name(&self) -> &str {
        "torchscript"
    }

    fn load(&mut self, model_path: &Path) -> Result<Duration> {
        if !model_path.exists() {
            bail!("model file not found: {}", model_path.display());
        }
        self.model_path = Some(model_path.to_path_buf());
        self.latency_samples_us.clear();
        self.throughput_rps = 0.0;
        self.load_ms = 0;
        Ok(Duration::ZERO)
    }

    fn infer(&mut self, input: &BenchInput) -> Result<()> {
        if !self.latency_samples_us.is_empty() {
            return Ok(());
        }

        let model_path = self
            .model_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("torchscript: load() not called"))?;

        let batch_size = input.shape.first().copied().unwrap_or(1);
        let driver = Self::write_driver()?;

        let _t0 = Instant::now();
        let output = Command::new("python3")
            .args([
                driver.path().to_str().unwrap_or_default(),
                model_path.to_str().unwrap_or_default(),
                &batch_size.to_string(),
                "20",
                "100",
                "5.0",
            ])
            .output()
            .context("failed to launch torchscript driver subprocess")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "torchscript driver exited with {}: {}",
                output.status,
                stderr
            );
        }

        let records = parse_driver_output(&output.stdout)?;
        let mut has_error = false;
        for record in records {
            match record {
                DriverRecord::LoadMs { value } => self.load_ms = value as u64,
                DriverRecord::LatencyUs { value } => self.latency_samples_us.push(value),
                DriverRecord::ThroughputRps { value } => self.throughput_rps = value,
                DriverRecord::Error { message } => {
                    eprintln!("torchscript driver error: {message}");
                    has_error = true;
                }
            }
        }

        if has_error {
            bail!("torchscript driver reported errors (see stderr)");
        }

        Ok(())
    }

    fn unload(&mut self) {
        self.model_path = None;
        self.latency_samples_us.clear();
        self.throughput_rps = 0.0;
        self.load_ms = 0;
    }

    fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> {
        if self.latency_samples_us.is_empty() {
            return None;
        }
        Some((
            std::mem::take(&mut self.latency_samples_us),
            self.throughput_rps,
            self.load_ms,
        ))
    }
}
