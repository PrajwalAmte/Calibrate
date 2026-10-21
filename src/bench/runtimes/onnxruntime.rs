//! ONNX Runtime (CUDA / CoreML / CPU execution providers) via a Python driver.

use crate::bench::input::ModelFormat;
use crate::bench::runtime::RuntimeDescriptor;
use crate::bench::runtimes::subprocess::{python_module_available, SubprocessRuntime};

pub const NAME: &str = "onnxruntime";

pub const SCRIPT: &str = r#"
import sys, json, time, os
import numpy as np

model_path      = sys.argv[1]
batch_size      = int(sys.argv[2])
warmup_iters    = int(sys.argv[3])
measure_iters   = int(sys.argv[4])
tput_window_s   = float(sys.argv[5])
seq_len         = int(sys.argv[6]) if len(sys.argv) > 6 else 512
shape_override  = sys.argv[7] if len(sys.argv) > 7 else "auto"
override_dims   = None if shape_override == "auto" else [int(d) for d in shape_override.split("x")]

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

try:
    import onnxruntime as ort
except ImportError as e:
    emit({"type": "error", "message": f"onnxruntime not installed: {e}"})
    sys.exit(1)

# Load 
t0 = time.perf_counter()
try:
    import platform
    if platform.system() == "Darwin":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress verbose ONNX logs
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
except Exception as e:
    emit({"type": "error", "message": f"load failed: {e}"})
    sys.exit(1)
load_ms = (time.perf_counter() - t0) * 1000
emit({"type": "load_ms", "value": load_ms})

# Build input 
inputs = {}
for inp in session.get_inputs():
    # Batch dimension → batch size; other dynamic dimensions → --input-shape
    # (if given) or the sequence length.
    shape = []
    for i, d in enumerate(inp.shape):
        dynamic = isinstance(d, str) or d is None or d == 0
        if i == 0:
            shape.append(batch_size if dynamic else d)
        elif override_dims is not None and i - 1 < len(override_dims):
            shape.append(override_dims[i - 1])
        else:
            shape.append(seq_len if dynamic else d)
    dtype_map = {
        "tensor(float)":   np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)":   np.int64,
        "tensor(int32)":   np.int32,
    }
    dtype = dtype_map.get(inp.type, np.float32)
    rng = np.random.default_rng(seed=0xCAFEBABE)
    if np.issubdtype(dtype, np.integer):
        inputs[inp.name] = rng.integers(0, 1000, size=shape).astype(dtype)
    else:
        inputs[inp.name] = rng.random(shape).astype(dtype)

output_names = [o.name for o in session.get_outputs()]

def run_once():
    session.run(output_names, inputs)

# Warm-up 
for _ in range(warmup_iters):
    run_once()

# Measurement 
for _ in range(measure_iters):
    t0 = time.perf_counter()
    run_once()
    elapsed_us = int((time.perf_counter() - t0) * 1_000_000)
    emit({"type": "latency_us", "value": elapsed_us})

# Throughput window
count = 0
deadline = time.perf_counter() + tput_window_s
while time.perf_counter() < deadline:
    run_once()
    count += 1
elapsed = time.perf_counter() - (deadline - tput_window_s)
emit({"type": "throughput_rps", "value": count / elapsed if elapsed > 0 else 0})
"#;

pub fn descriptor() -> RuntimeDescriptor {
    RuntimeDescriptor {
        name: NAME,
        is_available: || python_module_available("onnxruntime"),
        is_compatible: |fmt| matches!(fmt, ModelFormat::Onnx),
        create: || Box::new(SubprocessRuntime::new(NAME, SCRIPT)),
        opt_in: false,
    }
}
