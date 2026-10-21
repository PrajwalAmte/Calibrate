# Changelog

All notable changes are documented here. The JSON output formats are
versioned separately (see `docs/json-schema.md`).

## [0.2.0]

### Fixed
- `watch --pid` could not be combined with any option (`--cost-per-hour`,
  `--interval`, `--output`, …); the README's own examples failed.
- The live TUI could not be quit: raw mode swallowed Ctrl+C and all keys were
  discarded. `q`, `Esc` and Ctrl+C now quit; the terminal is restored on panic.
- `plan` ignored sequence length (one token per row), reporting minutes and
  `$0.00` for multi-hour jobs. It now uses `--seq-len`, recomputation-aware
  FLOPs and multi-GPU scaling.
- Parameter counts used `12·L·h²`, ignoring GQA, gated MLPs and embeddings
  (Qwen2.5-7B came out as 4.3B). Exact counts now come from Hub/safetensors
  metadata, with an architecture-aware formula (including MoE) as fallback.
- Gated Hugging Face models failed (no token support); 401 was reported as
  "not found". `HF_TOKEN` is supported and gated repos still resolve.
- Multi-GPU listings summed VRAM across GPUs and priced 8-GPU instances
  against one GPU's speed. Fit is now per GPU (sharded only with DeepSpeed);
  speed scales with GPU count.
- macOS reported CPU as 0, which classified idle GPUs as "100 % CUDA sync" and
  recommended removing `.item()` calls. macOS now has real process CPU and
  missing metrics are N/A.
- The GPU spec database URL was a placeholder, and the fetch was blocking with
  no timeout. It now has a 3 s timeout, back-off, offline mode, validation and
  merges with the built-in table.
- A command line with a multi-byte character at byte 80 crashed discovery.
- Ordinary Linux login sessions were detected as containers.
- Idle GPUs counted as zero waste; they now count as full waste.
- Multi-GPU samples were mixed in one window (MFU, leak detection and "step
  time" were wrong); "step time" was actually the sampling interval.
- NVML's real H100 name ("H100 80GB HBM3") was not recognised, and an RTX 3090 Ti or an
  A100 could match the 3090 or A10 spec. RTX 30-series peaks were FP32 numbers
  (half the tensor rate).
- Clock ratio used max clock instead of the spec's boost clock.
- `bench` subprocesses ignored `--warmup`/`--iterations`, had no timeout,
  classified any CUDA error as OOM, reported calibrate's own RSS as model
  memory, and llama.cpp results were 100 copies of an average.
- Watching an exited (zombie) process continued indefinitely on macOS.

### Changed
- MFU is labelled by source: measured (hook), tensor-core activity
  (GPM/DCGM) or a busy-time upper bound. A busy GPU is never called "well
  optimized".
- The time breakdown is labelled measured or inferred; heuristic classes were
  renamed to what they actually detect.
- Recommendations quote expected gains only where they follow from data.
- The `candle` bench runtime was removed: it did not run the model. The Metal
  GEMV kernel test is opt-in (`--runtimes metal-gemv`) and labelled synthetic.
- RunPod Community Cloud listings are no longer mislabelled as spot.
- TLS via rustls (no OpenSSL); MSRV corrected to 1.88; `Cargo.lock` committed.

### Added
- Process-tree support: DataLoader workers and torchrun/DeepSpeed ranks.
- NVML GPM and DCGM profiling sources (`--metrics-source`).
- Python training hook (`python/`) with HF Trainer and Lightning callbacks.
- Device mode (`--gpus`), `--duration`, `--record`, `--record-raw`,
  `--replay`, `--target-mfu`, `--output none`, SIGTERM handling.
- Prometheus endpoint and OTLP/HTTP exporter.
- Versioned JSON schemas, structured logs (`--log-format json`).
- Config file, offline mode, proxy support, price cache, `--prices-file`.
- `transformers` bench runtime; `--input-shape`, `--seq-len`, `--timeout`.
- GPU imbalance (straggler) and optimizer-heavy bottlenecks; leak detection
  requires sustained, significant growth.
- New GPUs: L4, L40, L40S, H100 NVL, H200, GH200, B200, RTX 5090/5080,
  RTX 3090 Ti, 4080/4070 Super, RTX A4500, RTX 4000 Ada (+SFF).
- CI on Linux x86-64/ARM64 and macOS, MSRV check, cargo-deny, Python tests,
  strict smoke test; releases with glibc 2.17 builds, checksums, provenance
  attestations and a multi-arch container image.
- Kubernetes DaemonSet, sidecar and PodMonitor manifests; Dockerfile.
- Binary-level tests, replayable trace fixtures, and a real-GPU validation
  script.

## [0.1.0]
- Initial release: `watch`, `probe`, `bench`, `plan`, `diagnose`.
