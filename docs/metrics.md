# What calibrate measures

This page explains every figure `calibrate watch` reports, where it comes
from, and how far to trust it.

## MFU (Model FLOP Utilization)

`MFU = achieved model FLOP/s ÷ peak FLOP/s of the GPUs in use`.

calibrate picks the best available source and labels it (`mfu.source` in
JSON, the gauge label in the TUI):

| `source` | Computation | Trust |
|---|---|---|
| `instrumented` | Σ over processes of (FLOPs ÷ step time) from the [Python hook](hook.md), ÷ Σ peak | Exact, provided the declared FLOPs are model FLOPs (`6N` + attention for training). |
| `tensor_active` | mean over GPUs of tensor-pipe activity × (SM clock ÷ boost clock), peak-weighted | Hardware estimate. Counts recomputation and padding, so it sits between MFU and HFU; typically within a few points of MFU for dense BF16 transformer training. Validate on your hardware with `scripts/gpu-validation.sh`. |
| `kernel_busy` | NVML `utilization.gpu` × clock ratio | **Upper bound** (`mfu.upper_bound = true`). "Busy" only means some kernel ran during the sample period. |
| `unavailable` | no GPU data | `mfu_pct` is `null`, never 0. |

Tensor-active is used only when *every* GPU sample in the window has it, so a
figure never mixes two kinds of measurement.

**Clock ratio.** Peak TFLOPS are specified at the boost clock, so activity at a
lower clock delivers proportionally less. calibrate divides by the spec
database's boost clock (falling back to NVML's max clock for unknown GPUs) and
caps the ratio at 1.25.

**Peak TFLOPS** come from `assets/fallback_specs.json`: dense (no sparsity)
BF16/FP16 tensor throughput **with FP32 accumulation** at boost clock. GeForce
cards run FP32-accumulate tensor math at half their FP16-accumulate rate; the
table uses the FP32-accumulate figure because that is what mixed-precision
training uses. Unknown GPUs get no peak (TFLOPS shown as unknown) rather than
an invented one; `calibrate diagnose` flags them.

## Time breakdown

| `breakdown.source` | Meaning |
|---|---|
| `instrumented` | Exact shares of step wall time from the hook: `data_wait` (gap before the step), `compute` (forward+backward), `optimizer`, `other`. Phase split requires `sync="phases"`. |
| `heuristic` | Each GPU sample is classified from GPU busy %, process-tree CPU % and memory-controller load. Sampling (seconds) is far slower than a training step (milliseconds), so this shows tendencies, not phase timings. |
| `unavailable` | Required inputs missing (no GPU data, or no CPU data in `--gpus` mode). |

Heuristic classes (thresholds are deliberately coarse):

| Class | Rule |
|---|---|
| `compute` | busy > 60 % |
| `memory_bound` | memory controller / DRAM activity > 80 % |
| `data_wait` | busy < 15 % and process-tree CPU > 30 % of a core |
| `sync_stall` | busy < 15 % and CPU < 10 % |
| `other` | everything else (partial GPU use) |

"Busy" is SM activity when profiling counters exist, else kernel-busy time.
CPU is summed over the **whole process tree**, so DataLoader workers count.

## Bottlenecks

Checked in this order; the first match is reported with one recommendation:

`warming_up` → `no_gpu_metrics` → `no_gpu_activity` → `thermal_throttle` →
`power_capped` (power-cap reason **and** clock < 90 % of boost) →
`hardware_slowdown` → `gpu_imbalance` (busy spread > 20 ppt) →
`data_loader` (wait > 15 %) → `sync_stall` (> 10 %) → `memory_bound` (> 15 %) →
`optimizer_heavy` (> 20 %, instrumented only) → `clock_underspeed` (busy > 50 %,
clock < 85 % of boost, no throttle reason) → `erratic_steps` (step-time CV > 0.3)
→ `below_target_mfu` → `unverified` (only a busy-time bound, and it is above
target) → `none`.

Expected MFU gains are only quoted where they follow from measurement: removing
an input-pipeline wait of fraction *w* raises throughput by 1/(1−*w*).

## Other signals

* **Memory growth** (`gpus[].vram_growing`): 30 consecutive samples that never
  decrease, at least half of them increasing, growing by ≥ 2 % of device memory
  (≥ 64 MiB), after the first minute (allocator warm-up is excluded).
* **GPU divergence** (`gpu_divergence_ppt`): spread of mean busy % across the
  job's GPUs over the window.
* **Cost** (`cost`): `waste_ratio = max(0, target − MFU) / target`; spend and
  waste are integrated over the session. With a busy-time upper bound the waste
  is a lower bound (`is_lower_bound`). A GPU the job holds but does not use
  counts as 100 % waste.
* **Steps** (`steps`, hook only): mean/p50/p95 step time, coefficient of
  variation, steps/s, tokens/s and samples/s summed over ranks.

## Platform notes

* **Linux / NVIDIA:** per-process GPU attribution from NVML's compute and
  graphics process lists across the whole process tree. Inside containers NVML
  reports host PIDs; use `--gpus` (device mode) or run with `hostPID`.
* **macOS:** IOKit exposes **system-wide** GPU utilization and memory only, so
  other apps are included (`target.scope = "system"`). Temperature and power
  need root (`powermetrics`) and are reported as N/A. The GPU memory budget is
  Metal's `recommendedMaxWorkingSetSize`.
