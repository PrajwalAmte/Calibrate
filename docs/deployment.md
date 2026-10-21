# Deployment and integrations

## Topologies

| Topology | Command | What you get |
|---|---|---|
| Interactive | `calibrate watch --pid N` | TUI on a training node |
| Node agent | `calibrate watch --gpus all --output none --prometheus-listen :9400` | every GPU on the node, continuously |
| Job sidecar | `calibrate watch --process-name torchrun --output none --prometheus-listen :9401` | one job's process tree + hook metrics |
| Batch report | `calibrate watch --pid N --duration 600 --output none --record run.jsonl --record-raw run.trace` | artefacts for CI or post-mortems |

`SIGTERM`/`SIGINT` stop cleanly and flush recordings and a final OTLP push.

## Kubernetes

Manifests in [`deploy/kubernetes`](../deploy/kubernetes):

* `daemonset.yaml` — node agent with RuntimeClass `nvidia`, non-root,
  read-only root filesystem, `/healthz` probes, headless Service.
* `sidecar-example.yaml` — per-job sidecar sharing the pod's PID namespace and
  a hook-socket `emptyDir`.
* `podmonitor.yaml` — Prometheus Operator scrape configs adding
  `node`/`pod`/`namespace` labels.

NVML reports host PIDs, which a pod cannot see, so GPU device metrics come
from the DaemonSet (device mode) while the sidecar supplies job-level,
instrumented metrics. Join them in PromQL on `node` and `gpu`.

## systemd

```ini
[Unit]
Description=calibrate GPU efficiency exporter
After=nvidia-persistenced.service

[Service]
ExecStart=/usr/local/bin/calibrate watch --gpus all --output none \
          --prometheus-listen 0.0.0.0:9400 --log-format json
Restart=on-failure
DynamicUser=yes
SupplementaryGroups=video

[Install]
WantedBy=multi-user.target
```

## Prometheus metrics

All series carry `host`, `mode` and (process mode) `pid` labels. Absent
readings are omitted, never exported as 0.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `calibrate_up` | gauge | | always 1 |
| `calibrate_info` | gauge | `version, backend, metrics_source, gpu_name` | build/session info |
| `calibrate_session_elapsed_seconds` | gauge | | |
| `calibrate_mfu_ratio` | gauge | `source, upper_bound` | MFU 0–1 |
| `calibrate_achieved_tflops` | gauge | `source` | |
| `calibrate_peak_tflops` | gauge | | combined peak of GPUs in use |
| `calibrate_target_mfu_ratio` | gauge | | |
| `calibrate_process_cpu_cores` | gauge | | process-tree CPU in cores |
| `calibrate_gpu_utilization_ratio` | gauge | `gpu, gpu_name` | kernel-busy time |
| `calibrate_gpu_sm_active_ratio` | gauge | `gpu, gpu_name` | GPM/DCGM |
| `calibrate_gpu_tensor_active_ratio` | gauge | `gpu, gpu_name` | GPM/DCGM |
| `calibrate_gpu_dram_active_ratio` | gauge | `gpu, gpu_name` | GPM/DCGM |
| `calibrate_gpu_sm_clock_hertz` | gauge | `gpu, gpu_name` | |
| `calibrate_gpu_memory_used_bytes` / `_total_bytes` | gauge | `gpu, gpu_name` | |
| `calibrate_gpu_temperature_celsius` | gauge | `gpu, gpu_name` | |
| `calibrate_gpu_power_watts` / `_power_limit_watts` | gauge | `gpu, gpu_name` | |
| `calibrate_gpu_throttle` | gauge | `gpu, gpu_name, reason` | 1 when active |
| `calibrate_gpu_memory_growing` | gauge | `gpu, gpu_name` | leak signal |
| `calibrate_time_breakdown_ratio` | gauge | `phase, source` | |
| `calibrate_step_duration_seconds` | gauge | `quantile` | hook |
| `calibrate_steps_per_second`, `calibrate_tokens_per_second` | gauge | | hook |
| `calibrate_bottleneck` | gauge | `kind` | 1 for the current finding |
| `calibrate_gpu_divergence_ratio` | gauge | | multi-GPU spread |
| `calibrate_cost_waste_ratio` | gauge | `lower_bound` | |
| `calibrate_cost_spent_usd_total`, `calibrate_cost_wasted_usd_total` | counter | | |

Example alerts:

```yaml
- alert: GpuTrainingInputBound
  expr: calibrate_bottleneck{kind="data_loader"} == 1
  for: 15m
- alert: GpuThermalThrottle
  expr: max by (host, gpu) (calibrate_gpu_throttle{reason=~".*thermal"}) == 1
  for: 5m
- alert: LowMeasuredMfu
  expr: calibrate_mfu_ratio{upper_bound="false"} < 0.25
  for: 30m
```

## OpenTelemetry

`--otlp-endpoint http://collector:4318` pushes the same metrics with OTLP/HTTP
(JSON encoding) to `/v1/metrics`, at most once per sampling interval (minimum
1 s), plus a final push on exit. Add authentication with repeated
`--otlp-header Authorization=Bearer\ $TOKEN`. Gauges map to OTLP gauges; the
cost totals to cumulative monotonic sums. Resource attributes: `service.name`,
`service.version`, `host.name`.
