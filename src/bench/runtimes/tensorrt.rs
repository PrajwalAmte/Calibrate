use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::{Runtime, RuntimeDescriptor};

// ── Embedded Python driver script

/// Driver that converts an ONNX model to a TensorRT engine, runs inference,
/// and streams timing results back as JSONL.
///
/// Engine compilation is counted as part of `load_ms` because TensorRT
/// builds an optimised kernel for the target GPU at load time — this cost
/// is real for any deployment that does not cache the engine.
const DRIVER_SCRIPT: &str = r#"
import sys, json, time, tempfile, os
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
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as e:
    emit({"type": "error", "message": f"tensorrt/pycuda not installed: {e}"})
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# ── Build TensorRT engine from ONNX ──────────────────────────────────────────
t0 = time.perf_counter()
try:
    builder  = trt.Builder(TRT_LOGGER)
    network  = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser   = trt.OnnxParser(network, TRT_LOGGER)
    config   = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    with open(model_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            emit({"type": "error", "message": "ONNX parse failed: " + "; ".join(errors)})
            sys.exit(1)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        emit({"type": "error", "message": "TensorRT engine build failed"})
        sys.exit(1)

    runtime_trt = trt.Runtime(TRT_LOGGER)
    engine  = runtime_trt.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
except Exception as e:
    emit({"type": "error", "message": f"engine build failed: {e}"})
    sys.exit(1)

load_ms = (time.perf_counter() - t0) * 1000
emit({"type": "load_ms", "value": load_ms})

# ── Allocate I/O buffers ──────────────────────────────────────────────────────
rng = np.random.default_rng(seed=0xCAFEBABE)
h_inputs, d_inputs, d_outputs, h_outputs, bindings = [], [], [], [], []

for i in range(engine.num_io_tensors):
    name  = engine.get_tensor_name(i)
    shape = list(engine.get_tensor_shape(name))
    shape = [batch_size if (j == 0 or s <= 0) else s for j, s in enumerate(shape)]
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        h = np.ascontiguousarray(rng.random(shape).astype(dtype))
        d = cuda.mem_alloc(nbytes)
        cuda.memcpy_htod(d, h)
        context.set_tensor_address(name, int(d))
        h_inputs.append(h); d_inputs.append(d)
    else:
        h = np.empty(shape, dtype=dtype)
        d = cuda.mem_alloc(nbytes)
        context.set_tensor_address(name, int(d))
        h_outputs.append(h); d_outputs.append(d)

stream = cuda.Stream()

def run_once():
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

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

// ── TensorRtRuntime

pub struct TensorRtRuntime {
    model_path: Option<std::path::PathBuf>,
    latency_samples_us: Vec<u64>,
    throughput_rps: f64,
    load_ms: u64,
}

impl TensorRtRuntime {
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
            name: "tensorrt",
            is_available: || {
                // TensorRT requires both the tensorrt Python package and pycuda.
                Command::new("python3")
                    .args(["-c", "import tensorrt; import pycuda.driver"])
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            },
            // TensorRT only accepts ONNX as input (builds an engine from it).
            is_compatible: |fmt| matches!(fmt, ModelFormat::Onnx),
            create: || Box::new(TensorRtRuntime::new()),
        }
    }

    fn write_driver() -> Result<tempfile::NamedTempFile> {
        let mut f = tempfile::Builder::new()
            .prefix("calibrate_trt_")
            .suffix(".py")
            .tempfile()?;
        f.write_all(DRIVER_SCRIPT.as_bytes())?;
        f.flush()?;
        Ok(f)
    }
}

impl Runtime for TensorRtRuntime {
    fn name(&self) -> &str {
        "tensorrt"
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
            .ok_or_else(|| anyhow::anyhow!("tensorrt: load() not called"))?;

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
            .context("failed to launch tensorrt driver subprocess")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("tensorrt driver exited with {}: {}", output.status, stderr);
        }

        let records = parse_driver_output(&output.stdout)?;
        let mut has_error = false;
        for record in records {
            match record {
                DriverRecord::LoadMs { value } => self.load_ms = value as u64,
                DriverRecord::LatencyUs { value } => self.latency_samples_us.push(value),
                DriverRecord::ThroughputRps { value } => self.throughput_rps = value,
                DriverRecord::Error { message } => {
                    eprintln!("tensorrt driver error: {message}");
                    has_error = true;
                }
            }
        }

        if has_error {
            bail!("tensorrt driver reported errors (see stderr)");
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
