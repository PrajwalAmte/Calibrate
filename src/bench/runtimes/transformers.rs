//! Hugging Face Transformers: real forward passes of a model directory
//! (`config.json` + `*.safetensors`) on CUDA, MPS or CPU.

use crate::bench::input::ModelFormat;
use crate::bench::runtime::RuntimeDescriptor;
use crate::bench::runtimes::subprocess::{python_module_available, SubprocessRuntime};

pub const NAME: &str = "transformers";

pub const SCRIPT: &str = r#"
import sys, json, time, os

model_path    = sys.argv[1]
batch_size    = int(sys.argv[2])
warmup_iters  = int(sys.argv[3])
measure_iters = int(sys.argv[4])
tput_window_s = float(sys.argv[5])
seq_len       = int(sys.argv[6])

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

try:
    import torch
    from transformers import AutoConfig, AutoModel
except ImportError as e:
    emit({"type": "error", "message": f"torch/transformers not installed: {e}"})
    sys.exit(1)

if os.path.isfile(model_path):
    model_path = os.path.dirname(model_path) or "."

if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float32

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

t0 = time.perf_counter()
try:
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    # PyTorch's MPS SDPA kernel aborts the process on grouped-query attention
    # in some releases; the eager path is slower but correct.
    extra = {"attn_implementation": "eager"} if device == "mps" else {}
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    # `dtype` replaced `torch_dtype` in transformers 4.56.
    extra["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
    model = AutoModel.from_pretrained(model_path, local_files_only=True, **extra)
    model.to(device).eval()
except Exception as e:
    msg = str(e)
    if "out of memory" in msg.lower():
        emit({"type": "error", "message": f"out of memory while loading: {msg}"})
    else:
        emit({"type": "error", "message": f"load failed: {msg}"})
    sys.exit(1)
sync()
emit({"type": "load_ms", "value": (time.perf_counter() - t0) * 1000})

vocab = getattr(cfg, "vocab_size", None) or getattr(getattr(cfg, "text_config", None), "vocab_size", 32000)
max_pos = getattr(cfg, "max_position_embeddings", None) or seq_len
seq = max(1, min(seq_len, max_pos))
g = torch.Generator().manual_seed(0xCAFE)
ids = torch.randint(0, vocab, (batch_size, seq), generator=g).to(device)
mask = torch.ones_like(ids)

def run_once():
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=mask)
    sync()

try:
    for _ in range(warmup_iters):
        run_once()
    for _ in range(measure_iters):
        t0 = time.perf_counter()
        run_once()
        emit({"type": "latency_us", "value": int((time.perf_counter() - t0) * 1_000_000)})
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < tput_window_s:
        run_once()
        count += 1
    elapsed = time.perf_counter() - start
    emit({"type": "throughput_rps", "value": count / elapsed if elapsed > 0 else 0})
except Exception as e:
    emit({"type": "error", "message": str(e)})
    sys.exit(1)
"#;

pub fn descriptor() -> RuntimeDescriptor {
    RuntimeDescriptor {
        name: NAME,
        is_available: || python_module_available("transformers"),
        is_compatible: |fmt| matches!(fmt, ModelFormat::HfDirectory),
        create: || Box::new(SubprocessRuntime::new(NAME, SCRIPT)),
        opt_in: false,
    }
}
