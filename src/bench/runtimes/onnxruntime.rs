use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::{Runtime, RuntimeDescriptor};

// ── Embedded Python driver script ─────────────────────────────────────────────

/// A self-contained Python script written to a tempfile and executed as a
/// subprocess. Returns newline-delimited JSON timing records to stdout and
/// writes errors to stderr.
///
/// Protocol:
///   argv[1]  path to the .onnx model file
///   argv[2]  batch_size (integer)
///   argv[3]  warmup_iterations (integer)
///   argv[4]  measure_iterations (integer)
///   argv[5]  throughput_window_secs (float)
///
/// stdout: one JSON object per line:
///   {"type":"latency_us","value":12345}
///   {"type":"throughput_rps","value":83.2}
///   {"type":"load_ms","value":540}
///   {"type":"error","message":"..."}
const DRIVER_SCRIPT: &str = r#"
import sys, json, time, os
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
    import onnxruntime as ort
except ImportError as e:
    emit({"type": "error", "message": f"onnxruntime not installed: {e}"})
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
try:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress verbose ONNX logs
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
except Exception as e:
    emit({"type": "error", "message": f"load failed: {e}"})
    sys.exit(1)
load_ms = (time.perf_counter() - t0) * 1000
emit({"type": "load_ms", "value": load_ms})

# ── Build input ───────────────────────────────────────────────────────────────
inputs = {}
for inp in session.get_inputs():
    shape = [batch_size if (isinstance(d, str) or d is None or d == 0) else d
             for d in inp.shape]
    dtype_map = {
        "tensor(float)":   np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)":   np.int64,
        "tensor(int32)":   np.int32,
    }
    dtype = dtype_map.get(inp.type, np.float32)
    inputs[inp.name] = np.random.default_rng(seed=0xCAFEBABE).random(shape).astype(dtype)

output_names = [o.name for o in session.get_outputs()]

def run_once():
    session.run(output_names, inputs)

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

// ── OnnxRuntime struct ────────────────────────────────────────────────────────

pub struct OnnxRuntime {
    model_path: Option<std::path::PathBuf>,
    /// Collected latency samples in microseconds from the last subprocess run.
    latency_samples_us: Vec<u64>,
    throughput_rps: f64,
    load_ms: u64,
}

impl OnnxRuntime {
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
            name: "onnxruntime",
            is_available: || {
                Command::new("python3")
                    .args(["-c", "import onnxruntime as ort; print(ort.__version__)"])
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            },
            is_compatible: |fmt| matches!(fmt, ModelFormat::Onnx),
            create: || Box::new(OnnxRuntime::new()),
        }
    }

    /// Write the embedded driver script to a tempfile and return the path.
    fn write_driver() -> Result<tempfile::NamedTempFile> {
        let mut f = tempfile::Builder::new()
            .prefix("calibrate_ort_")
            .suffix(".py")
            .tempfile()?;
        f.write_all(DRIVER_SCRIPT.as_bytes())?;
        f.flush()?;
        Ok(f)
    }
}

// ── Deserialization helpers ───────────────────────────────────────────────────

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

// ── Runtime impl ─────────────────────────────────────────────────────────────

impl Runtime for OnnxRuntime {
    fn name(&self) -> &str {
        "onnxruntime"
    }

    /// `load()` for subprocess runtimes: validate the model file exists and
    /// store the path. The actual subprocess is launched in `infer()` because
    /// the harness calls `load()` once per (runtime, batch_size) pair and then
    /// calls `infer()` in a loop.
    ///
    /// For subprocess runtimes all measurement (warm-up, timing, throughput)
    /// is delegated to the driver script in a single subprocess invocation
    /// triggered by the first `infer()` call. Subsequent `infer()` calls are
    /// no-ops that return the pre-recorded samples.
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

    /// The first call to `infer()` launches the driver subprocess, which
    /// runs warm-up, full measurement, and the throughput window internally,
    /// then streams results back as JSON. The harness timing loop calls
    /// `infer()` `iterations` times; only the first call does real work —
    /// subsequent calls return immediately via the replay buffer.
    fn infer(&mut self, input: &BenchInput) -> Result<()> {
        // If we already have samples from a previous subprocess run, replay them.
        if !self.latency_samples_us.is_empty() {
            return Ok(());
        }

        let model_path = self
            .model_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("onnxruntime: load() not called"))?;

        let batch_size = input.shape.first().copied().unwrap_or(1);
        let driver = Self::write_driver()?;

        let t0 = Instant::now();
        let output = Command::new("python3")
            .args([
                driver.path().to_str().unwrap_or_default(),
                model_path.to_str().unwrap_or_default(),
                &batch_size.to_string(),
                "20",  // warmup iterations
                "100", // measurement iterations
                "5.0", // throughput window seconds
            ])
            .output()
            .context("failed to launch onnxruntime driver subprocess")?;

        let subprocess_ms = t0.elapsed().as_millis() as u64;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "onnxruntime driver exited with {}: {}",
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
                    eprintln!("onnxruntime driver error: {message}");
                    has_error = true;
                }
            }
        }

        if has_error {
            bail!("onnxruntime driver reported errors (see stderr)");
        }

        let _ = subprocess_ms; // total time tracked by harness wrapper
        Ok(())
    }

    fn unload(&mut self) {
        self.model_path = None;
        self.latency_samples_us.clear();
        self.throughput_rps = 0.0;
        self.load_ms = 0;
    }

    /// Returns the samples collected inside the subprocess.
    ///
    /// Returns `Some` exactly once — the samples are moved out and cleared so
    /// that a second call returns `None`, preventing double-injection.
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
