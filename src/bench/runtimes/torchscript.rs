//! TorchScript models via PyTorch (CUDA, MPS or CPU).

use crate::bench::input::ModelFormat;
use crate::bench::runtime::RuntimeDescriptor;
use crate::bench::runtimes::subprocess::{python_module_available, SubprocessRuntime};

pub const NAME: &str = "torchscript";

pub const SCRIPT: &str = r#"
import sys, json, time
import numpy as np

model_path      = sys.argv[1]
batch_size      = int(sys.argv[2])
warmup_iters    = int(sys.argv[3])
measure_iters   = int(sys.argv[4])
tput_window_s   = float(sys.argv[5])
seq_len         = int(sys.argv[6]) if len(sys.argv) > 6 else 512
shape_override  = sys.argv[7] if len(sys.argv) > 7 else "auto"

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

try:
    import torch
except ImportError as e:
    emit({"type": "error", "message": f"torch not installed: {e}"})
    sys.exit(1)

# Load 
t0 = time.perf_counter()
try:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    # Prefer MPS on macOS, CUDA on Linux/Windows, fall back to CPU.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            model = model.to("mps")
            device = "mps"
        except Exception:
            device = "cpu"
    elif torch.cuda.is_available():
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

# Build input 
rng = np.random.default_rng(seed=0xCAFEBABE)

# Input shape: --input-shape override, then the traced graph's sizes, then
# the first layer's weights (Linear → [b, in], Conv2d → [b, C, 224, 224],
# Conv1d → [b, C, seq], Embedding → integer ids [b, seq]).
integer_ids = False
shape = None
if shape_override != "auto":
    shape = [batch_size] + [int(d) for d in shape_override.split("x")]
if shape is None:
    try:
        sizes = list(list(model.graph.inputs())[1].type().sizes())
        if sizes:
            shape = [batch_size if (i == 0 or s <= 0) else s for i, s in enumerate(sizes)]
    except Exception:
        shape = None
if shape is None:
    try:
        for name, mod in model.named_modules():
            kind = getattr(mod, "original_name", "")
            w = getattr(mod, "weight", None) if kind else None
            if kind == "Embedding":
                integer_ids, shape, vocab = True, [batch_size, seq_len], int(w.shape[0])
                break
            if kind == "Linear":
                shape = [batch_size, int(w.shape[1])]; break
            if kind == "Conv2d":
                shape = [batch_size, int(w.shape[1]), 224, 224]; break
            if kind == "Conv1d":
                shape = [batch_size, int(w.shape[1]), seq_len]; break
    except Exception:
        shape = None
if shape is None:
    shape = [batch_size, 512]

if integer_ids:
    inp = torch.tensor(rng.integers(0, vocab, size=shape), dtype=torch.long)
else:
    inp = torch.tensor(rng.random(shape, dtype=np.float32))
if device == "cuda":
    inp = inp.cuda()
elif device == "mps":
    inp = inp.to("mps")

def run_once():
    with torch.no_grad():
        model(inp)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

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
        is_available: || python_module_available("torch"),
        is_compatible: |fmt| matches!(fmt, ModelFormat::TorchScript),
        create: || Box::new(SubprocessRuntime::new(NAME, SCRIPT)),
        opt_in: false,
    }
}
