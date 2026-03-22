# calibrate

GPU training efficiency analyzer. Attach to a running training job and immediately see your Model FLOP Utilization (MFU), where compute time is being lost, and the single change that would fix it.

```
calibrate watch --pid 38291 --cost-per-hour 0.34
```

## Subcommands

| Command | Status | Description |
|---|---|---|
| `watch` | ✅ Week 1 | Attach to a training process, measure MFU & bottlenecks in real time |
| `bench` | 🔜 Week 2 | Compare runtime latency across ONNX, llama.cpp, PyTorch |
| `plan`  | 🔜 Week 3 | Fetch live GPU cloud prices and recommend cheapest option for your workload |

## Install

```bash
cargo install calibrate
```

Or build from source:

```bash
git clone https://github.com/your-org/calibrate
cd calibrate
cargo build --release
./target/release/calibrate watch --pid <PID>
```

## Usage: calibrate watch

```
calibrate watch --pid <PID> [OPTIONS]

Options:
  -p, --pid <PID>               Process ID of the running training job
  -c, --cost-per-hour <USD/HR>  Hourly GPU cost (enables dollar waste display)
  -i, --interval <SECS>         Sampling interval [default: 2]
  -o, --output <FORMAT>         terminal | json  [default: terminal]
  -h, --help                    Print help
```

### Example output

```
GPU: NVIDIA GeForce RTX 3090  •  $0.34/hr  •  Elapsed: 00:04:22

MFU ████████░░░░░░░░░░░░░░░░░░░░░░  19.3%  •  6.9 / 35.6 TFLOPS  (target >45%)

Time Breakdown
  Forward/backward  ████████████████░░░░░░░░░░░░░░   61.0%
  Data loader wait  ████████░░░░░░░░░░░░░░░░░░░░░░   28.0%  <- primary bottleneck
  CUDA sync         ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░    7.0%
  Optimizer step    █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    4.0%

Hardware: Temp: 68°C  •  Power: 210W / 350W  •  VRAM: 8.0 GiB / 24.0 GiB

Recommendation: Data loader is the primary bottleneck
  [+17 ppt MFU expected]
  Add num_workers=4, pin_memory=True to your DataLoader.
  If the dataset fits in RAM, use an in-memory dataset.
```

## Requirements

- Linux (primary; non-NVIDIA support planned)
- NVIDIA GPU with drivers installed (`nvidia-smi` must work)
- Training job running in the same host PID namespace

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design: hexagonal architecture, typestate session lifecycle, NVML threading model, and MFU estimation methodology.

## License

MIT
