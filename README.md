# calibrate

GPU training efficiency analyzer for NVIDIA and Apple Silicon. Attach to a
running training job and see how much of the GPU's peak it really uses, where
the time goes, and the one change most likely to help. Benchmark inference
runtimes. Plan and cost a fine-tuning run against live cloud pricing.

```bash
calibrate watch --pid 38291 --cost-per-hour 2.49        # live TUI (q / Esc / Ctrl+C to quit)
calibrate watch --process-name "train\.py" --output json  # versioned JSON lines
calibrate watch --gpus all --output none --prometheus-listen 0.0.0.0:9400   # node agent
calibrate bench --model ./my-model-dir --batch-sizes 1,8
calibrate plan  --model meta-llama/Llama-3.1-8B --method qlora --dataset-rows 50000
```

---

## What the numbers mean

MFU (Model FLOP Utilization) is only as good as its source, so calibrate
always says which one it used:

| Source | How | When | Shown as |
|---|---|---|---|
| **Instrumented** | FLOPs and step times reported by the [Python hook](docs/hook.md) | any GPU, three lines of code | `MFU` |
| **Tensor-active** | tensor-core pipe activity × clock, from NVML GPM or DCGM | Hopper+ (GPM), A100/L4/L40S… (DCGM) | `MFU (est., tensor-core activity)` |
| **Kernel-busy** | NVML "GPU utilization" × clock | everywhere | `MFU upper bound (GPU busy time)` |

NVML's "GPU utilization" only says that *some* kernel was running; a GPU can be
100 % busy at 30 % MFU. calibrate therefore never calls a busy-time figure
"well optimized" — it reports it as an upper bound and tells you how to get a
real measurement. Metrics a platform cannot provide (e.g. temperature on
macOS) are shown as N/A, never as 0. Details: [docs/metrics.md](docs/metrics.md).

---

## Subcommands

| Command | Description |
|---|---|
| `diagnose` | Check the driver stack, profiling support (GPM/DCGM), process visibility, config and network |
| `watch` | Live efficiency analysis of a process tree (`--pid`, `--process-name`, `--auto`), of devices (`--gpus`), or of a recorded trace (`--replay`) |
| `probe` | Stream raw sample ticks as JSON to verify collection on a new machine |
| `bench` | Compare inference runtimes (Transformers, TorchScript, ONNX Runtime, TensorRT, llama.cpp) |
| `plan` | Estimate per-GPU VRAM, training time and cost across RunPod, Lambda and Vast.ai, or your own price list |

---

## Install

**Release binaries** (from v0.2.0) — Linux x86-64 / ARM64 (glibc ≥ 2.17) and macOS:

```bash
VERSION=v0.2.0 TARGET=x86_64-unknown-linux-gnu   # or aarch64-unknown-linux-gnu, aarch64-apple-darwin
curl -LO https://github.com/PrajwalAmte/Calibrate/releases/download/$VERSION/calibrate-$VERSION-$TARGET.tar.gz
curl -LO https://github.com/PrajwalAmte/Calibrate/releases/download/$VERSION/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
tar xzf calibrate-$VERSION-$TARGET.tar.gz && sudo mv calibrate-$VERSION-$TARGET/calibrate /usr/local/bin/
```

Every artifact has a signed build-provenance attestation
(`gh attestation verify <file> --repo PrajwalAmte/Calibrate`).

**Container image** (linux/amd64, linux/arm64):

```bash
docker run --rm --gpus all ghcr.io/prajwalamte/calibrate:0.2.0 diagnose
```

**From source** (Rust 1.88+):

```bash
cargo install --git https://github.com/PrajwalAmte/Calibrate --locked
```

**Python hook** (optional, for measured MFU): `pip install ./python` from a checkout.

On Linux, NVML (`libnvidia-ml.so.1`, shipped with the NVIDIA driver) is loaded
at runtime — the binary has no hard dependency on it, and no OpenSSL
dependency (TLS uses rustls with the system trust store).

---

## calibrate watch

```
Target (exactly one):
  -p, --pid <PID>              Attach to a process; children (ranks, DataLoader workers) included
  -n, --process-name <REGEX>   Match the command line; a launcher and its ranks resolve to the launcher
  -a, --auto                   Find training jobs (python + torch/jax/…, torchrun, accelerate, deepspeed)
      --gpus <all|0,1,…>       Device-level mode: no PID needed (node agents, containers)
      --replay <FILE>          Replay a trace recorded with --record-raw

Options (combine freely with any target):
  -c, --cost-per-hour <USD>    Dollar cost of running below target
  -i, --interval <SECS>        0.1 – 3600 [default: 2]
      --target-mfu <PCT>       [default: 45]
      --duration <SECS>        Stop automatically
  -o, --output <FORMAT>        terminal | json | none
      --metrics-source <SRC>   auto | gpm | dcgm | nvml
      --record <FILE>          Append snapshots (JSONL)
      --record-raw <FILE>      Append raw ticks (replayable)
      --prometheus-listen <ADDR>   Serve /metrics and /healthz
      --otlp-endpoint <URL>    Push to an OpenTelemetry collector (OTLP/HTTP)
      --no-children            Only the target PID itself
```

The TUI shows the MFU source, a time breakdown (measured with the hook,
otherwise labelled *inferred from sampling*), per-GPU status for multi-GPU
jobs, throttle reasons, memory growth, and one recommendation. SIGTERM and
SIGINT stop it cleanly, so it works under systemd and Kubernetes.

**Measured MFU in three lines** ([docs/hook.md](docs/hook.md)):

```python
import calibrate_hook
hook = calibrate_hook.start(model=model, seq_len=2048)
for batch in loader:
    with hook.step(tokens=batch["input_ids"].numel()):
        ...  # forward, backward, optimizer step
```

---

## calibrate bench

```
calibrate bench --model <FILE|DIR> [--runtimes transformers,torchscript,...]
                [--batch-sizes 1,8,32] [--iterations 100] [--warmup 20]
                [--seq-len 512] [--input-shape 3,224,224] [--timeout 600]
                [--optimize-for latency|throughput|memory] [--output terminal|json|markdown]
                [--save base.json] [--compare base.json]
```

| Format | Runtimes |
|---|---|
| Hugging Face directory (`config.json` + safetensors) | `transformers` |
| `.pt` / `.pth` TorchScript | `torchscript` (input shape inferred from the first layer) |
| `.onnx` | `onnxruntime`, `tensorrt` |
| `.gguf` | `llamacpp` (`llama-bench`) |

Each runtime runs in its own subprocess with a hard timeout; memory is the
peak measured *inside* that process (device allocator or RSS). `metal-gemv`
is an opt-in kernel microbenchmark on macOS, labelled synthetic and never
recommended.

---

## calibrate plan

```
calibrate plan --model <HF_ID|PATH> [--params-b 7.6] [--method full|lora|qlora]
               [--optimizer none|unsloth|deep-speed] [--quantization none|8bit|4bit]
               [--dataset-rows N] [--seq-len 2048] [--batch-size 1] [--epochs 1]
               [--mfu 0.38] [--budget 100] [--availability now|flexible]
               [--providers runpod,lambda,vastai] [--prices-file prices.json] [--output json]
```

* Parameter counts are exact when possible (Hub safetensors metadata, which is
  public even for gated models; local safetensors headers); `HF_TOKEN` unlocks
  the gated `config.json` for exact architecture figures.
* VRAM is estimated **per GPU**: weights, gradients, optimizer state,
  checkpointed activations, the logits/loss buffer and CUDA overhead.
* Training time uses tokens = rows × `--seq-len`, hardware FLOPs including
  recomputation (8N full, 6N LoRA per token, plus attention), the GPU's dense
  BF16 peak, the instance's GPU count and your utilization (`--mfu`, ideally
  from `calibrate watch`).
* Lambda needs `LAMBDA_API_KEY`. Prices are cached for 15 minutes;
  `--prices-file` supports on-prem or negotiated pricing and air-gapped use.

---

## Enterprise deployment

* **Kubernetes:** a DaemonSet node agent (device metrics), a per-job sidecar
  (hook metrics) and Prometheus Operator `PodMonitor`s — [deploy/kubernetes](deploy/kubernetes).
* **Prometheus / OpenTelemetry:** identical metric names on both paths —
  [docs/deployment.md](docs/deployment.md).
* **Configuration:** `~/.config/calibrate/config.toml`, environment variables,
  `--offline` for air-gapped clusters, proxy support — [docs/configuration.md](docs/configuration.md).
* **Stable JSON:** `schema_version` on every snapshot and plan report —
  [docs/json-schema.md](docs/json-schema.md).
* **Reproducible analysis:** `--record-raw` + `--replay` turn field problems
  into regression tests (`tests/fixtures/`).

## Requirements

| | Linux | macOS |
|---|---|---|
| `watch` / `probe` | NVIDIA driver (NVML); optional DCGM | Apple Silicon or AMD/Intel GPU via IOKit (system-wide figures) |
| `bench` | Python runtimes as needed | same |
| `plan` | network or `--prices-file` | same |

Windows is not supported.

## Development

```bash
cargo test                          # unit + binary-level tests (offline)
./scripts/test-all.sh               # strict release smoke test
(cd python && python3 -m unittest discover -s tests)
./scripts/gpu-validation.sh         # on an NVIDIA machine: validate estimators, record real traces
```

## License

MIT
