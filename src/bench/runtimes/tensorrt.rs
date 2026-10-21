//! TensorRT: builds an engine from ONNX (build time counts as load time) and runs it.

use crate::bench::input::ModelFormat;
use crate::bench::runtime::RuntimeDescriptor;
use crate::bench::runtimes::subprocess::{python_module_available, SubprocessRuntime};

pub const NAME: &str = "tensorrt";

pub const SCRIPT: &str = r#"
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

# Build TensorRT engine from ONNX 
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

# Allocate I/O buffers 
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

# Warm-up ─
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
        is_available: || python_module_available("tensorrt") && python_module_available("pycuda"),
        is_compatible: |fmt| matches!(fmt, ModelFormat::Onnx),
        create: || Box::new(SubprocessRuntime::new(NAME, SCRIPT)),
        opt_in: false,
    }
}
