# Machine-readable output

All machine-readable output is versioned. **Adding** a field is
backwards-compatible; renaming, removing or changing the meaning of a field
bumps the version. Consumers should ignore unknown fields.

| Output | Version field | Current |
|---|---|---|
| `watch --output json`, `watch --record` | `schema_version` | 1 |
| `plan --output json` | `schema_version` | 1 |
| `watch --record-raw` traces | `calibrate_trace` (header line) | 1 |
| Python hook socket | `v` | 1 |

Durations are seconds (`*_secs`) or milliseconds (`*_ms`); percentages are
0–100 (`*_pct`); memory is MiB (`*_mib`). A metric that is not available is
`null`, never `0`.

## `watch` JSON lines

One `{"event":"snapshot", ...}` per sample tick, then one
`{"event":"session_end", ...}`.

```jsonc
{
  "event": "snapshot",
  "schema_version": 1,
  "calibrate_version": "0.2.0",
  "timestamp_ms": 1760000038000,
  "elapsed_secs": 38.0,
  "target": { "mode": "process", "pid": 4242, "process_count": 9,
              "scope": "process", "hostname": "gpu-node-17" },
  "backend": "nvml",                 // nvml | apple-iokit | none
  "metrics_source": "nvml-gpm",      // nvml-gpm | dcgm | nvml-gpm+nvml | nvml | iokit | none
  "gpu_name": "8× NVIDIA H100 80GB HBM3",
  "mfu": { "mfu_pct": 41.3, "source": "tensor_active", "upper_bound": false,
           "achieved_tflops": 3268.0, "peak_tflops": 7912.0, "confidence": "high" },
  "mfu_percentiles": { "p50": 41.0, "p75": 43.2, "p95": 45.9 },   // null until 15 samples
  "peak_mfu_pct": 47.1,
  "breakdown": { "source": "heuristic", "compute_pct": 88.0, "data_wait_pct": 4.0,
                 "sync_stall_pct": 2.0, "memory_bound_pct": 0.0,
                 "optimizer_pct": 0.0, "other_pct": 6.0 },
  "bottleneck": "below_target_mfu",
  "recommendation": { "kind": "below_target_mfu", "title": "...", "action": "...",
                      "expected_mfu_gain_ppt": null },
  "gpus": [ { "index": 0, "name": "NVIDIA H100 80GB HBM3", "peak_tflops": 989.0,
              "util_pct": 99.0, "sm_active_pct": 91.0, "tensor_active_pct": 42.0,
              "dram_active_pct": 31.0, "sm_clock_mhz": 1980, "clock_pct_of_boost": 108.2,
              "vram_used_mib": 61234, "vram_total_mib": 81559, "peak_vram_mib": 62001,
              "temperature_c": 61.0, "power_w": 612.0, "power_limit_w": 700.0,
              "throttle": { "hw_thermal": false, "sw_thermal": false,
                            "power_cap": false, "hw_slowdown": false },
              "vram_growing": false } ],
  "cpu_pct": 412.0,                  // process tree, % of one core
  "steps": null,                     // StepStats when the hook is active
  "hook_peers": 0,
  "cost": { "cost_per_hour": 23.92, "waste_ratio": 0.08, "waste_per_hour": 1.91,
            "spent_usd": 0.25, "wasted_usd": 0.02, "is_lower_bound": false },
  "gpu_divergence_ppt": 3.1,
  "gpu_divergent": false,
  "vram_growing": false,
  "target_mfu_pct": 45.0,
  "ticks_observed": 20,
  "warnings": []
}
```

`bottleneck` values are listed in [metrics.md](metrics.md#bottlenecks).
`steps` (hook only): `steps_in_window, mean_ms, p50_ms, p95_ms, cv, erratic,
steps_per_sec, tokens_per_sec, samples_per_sec, ranks`.

```jsonc
{ "event": "session_end", "schema_version": 1, "elapsed_secs": 600.0,
  "ticks_observed": 300, "mfu_pct": 41.0, "mfu_source": "tensor_active",
  "mfu_upper_bound": false, "peak_mfu_pct": 47.1, "mfu_percentiles": {...},
  "primary_bottleneck": "below_target_mfu", "cost": {...} }
```

## Raw traces (`--record-raw`)

First line is a header, then one tick per line:

```jsonc
{"calibrate_trace":1,"calibrate_version":"0.2.0","backend":"nvml",
 "metrics_source":"nvml-gpm","scope":"process","interval_secs":2.0}
{"timestamp_ms":...,"scope":"process","gpus":[{GpuSample}],"cpu_pct":412.0,
 "process_count":9,"steps":[{StepEvent}],"hook_peers":[...]}
```

`GpuSample` fields mirror `gpus[]` above without the derived ones (`util_pct`,
`prof{source,sm_active_pct,sm_occupancy_pct,tensor_active_pct,dram_active_pct}`,
`sm_clock_mhz`, `sm_clock_max_mhz`, `vram_used_mib`, `vram_total_mib`,
`mem_util_pct`, `temperature_c`, `power_w`, `power_limit_w`, `throttle`).
`StepEvent` is the hook's step message ([hook.md](hook.md#protocol-v1)).
Replay with `calibrate watch --replay trace.jsonl [--replay-speed 1]`.

## `plan --output json`

Top level: `schema_version`, `workload`, `listings[]`, `excluded_too_small`,
`recommendation`, `skipped_providers[]`, `price_source` (`live`, `cache`,
`live+cache`, `file`), `generated_at`.

* `workload`: `model` (`param_count_b`, `active_param_count_b`, `param_source`,
  architecture fields, `architecture_estimated`), `setup`, `vram` (per-GPU
  breakdown + `notes`), `required_vram_gib`, `fitting_tiers`, `dataset_rows`,
  `epochs`, `total_tokens`, `total_hardware_flops`, `assumed_utilization`,
  `assumptions[]`.
* `listings[]`: `provider`, `gpu_model`, `gpu_count`, `vram_per_gpu_gib`,
  `hourly_usd` (whole instance), `availability`, `flags[]`,
  `peak_tflops_per_gpu`, `required_vram_per_gpu_gib`, `duration_range`
  (`low_secs`, `high_secs`), `cost_range` (`low_usd`, `high_usd`),
  `over_budget`.
