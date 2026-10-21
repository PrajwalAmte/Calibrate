#!/usr/bin/env python3
"""Generate the synthetic traces in tests/fixtures/.

Each trace is in the exact `calibrate watch --record-raw` format, so real
traces recorded on GPU machines can be added next to these and replayed by
the same tests. Deterministic (fixed seed) so fixtures are reproducible:

    python3 scripts/gen_fixtures.py && git diff --stat tests/fixtures
"""

import json
import os
import random

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
T0 = 1_760_000_000_000
rnd = random.Random(42)


def header(source="nvml-gpm", scope="process"):
    return {"calibrate_trace": 1, "calibrate_version": "0.2.0", "backend": "nvml",
            "metrics_source": source, "scope": scope, "interval_secs": 2.0}


def gpu(index=0, name="NVIDIA H100 80GB HBM3", util=95.0, tensor=None, sm=None,
        clock=1830, vram=60000, total=81559, temp=62.0, power=550.0, limit=700.0,
        mem=40.0, throttle=None):
    prof = None
    if tensor is not None or sm is not None:
        prof = {"source": "gpm", "sm_active_pct": sm, "sm_occupancy_pct": 35.0,
                "tensor_active_pct": tensor, "dram_active_pct": mem}
    return {"index": index, "name": name, "util_pct": util, "prof": prof,
            "sm_clock_mhz": clock, "sm_clock_max_mhz": 1980, "vram_used_mib": vram,
            "vram_total_mib": total, "mem_util_pct": mem, "temperature_c": temp,
            "power_w": power, "power_limit_w": limit,
            "throttle": throttle or {"hw_thermal": False, "sw_thermal": False,
                                     "power_cap": False, "hw_slowdown": False}}


def tick(i, gpus, cpu=150.0, steps=None, peers=None):
    return {"timestamp_ms": T0 + i * 2000, "scope": "process", "gpus": gpus,
            "cpu_pct": cpu, "process_count": 5, "steps": steps or [],
            "hook_peers": peers or []}


def jitter(v, pct=0.05):
    return max(0.0, v * (1 + rnd.uniform(-pct, pct)))


def write(name, hdr, ticks):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(json.dumps(hdr) + "\n")
        for t in ticks:
            f.write(json.dumps(t) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)

    # Healthy H100 job: 55 % tensor-core activity.
    write("healthy_h100_gpm.jsonl", header(), [
        tick(i, [gpu(util=jitter(98, .01), tensor=jitter(55), sm=jitter(92, .02))])
        for i in range(40)])

    # Input-pipeline bound: GPU idle while the DataLoader workers are busy.
    write("dataloader_bound.jsonl", header(), [
        tick(i, [gpu(util=4.0 if i % 2 else 85.0, tensor=1.0 if i % 2 else 40.0,
                     sm=5.0 if i % 2 else 85.0)], cpu=380.0)
        for i in range(40)])

    # Thermal throttling with depressed clocks.
    hot = {"hw_thermal": True, "sw_thermal": True, "power_cap": False, "hw_slowdown": False}
    write("thermal_throttle.jsonl", header(), [
        tick(i, [gpu(util=97, tensor=30, sm=90, clock=1300, temp=88, throttle=hot)])
        for i in range(30)])

    # 4-GPU DDP job with a straggler on GPU 2.
    write("multi_gpu_straggler.jsonl", header(), [
        tick(i, [gpu(index=g, util=95 if g != 2 else 45, tensor=50 if g != 2 else 20,
                     sm=90 if g != 2 else 45) for g in range(4)])
        for i in range(30)])

    # Only NVML available (A10G, no GPM/DCGM): busy-time upper bound only.
    write("kernel_busy_only_a10g.jsonl", header(source="nvml"), [
        tick(i, [gpu(name="NVIDIA A10G", util=jitter(98, .01), clock=1695, total=23028,
                     vram=20000, power=280, limit=300)])
        for i in range(40)])

    # Steady VRAM growth after warm-up (leak).
    write("vram_leak.jsonl", header(), [
        tick(i, [gpu(util=95, tensor=50, sm=90, vram=40000 + max(0, i - 10) * 300)])
        for i in range(80)])

    # Hooked job: measured FLOPs. 8 steps/tick, 0.25 s each, 20 % data wait,
    # 2.0e14 FLOPs/step → 800 TFLOP/s achieved of 989 peak... scaled to 40 %:
    steps_per_tick, step_s = 8, 0.25
    flops = 0.40 * 989e12 * step_s
    ticks = []
    for i in range(30):
        steps = [{"pid": 4242, "rank": 0, "step": i * steps_per_tick + k,
                  "t_end_ms": T0 + i * 2000 + k * 250, "step_s": step_s,
                  "data_s": 0.05, "compute_s": 0.17, "optimizer_s": 0.03,
                  "tokens": 65536, "flops": flops} for k in range(steps_per_tick)]
        ticks.append(tick(i, [gpu(util=80, tensor=38, sm=75)], steps=steps,
                          peers=[{"pid": 4242, "rank": 0, "world_size": 1,
                                  "framework": "pytorch", "device": "cuda:0"}]))
    write("hooked_instrumented.jsonl", header(), ticks)

    # CPU-only fallback (no NVML): no GPU data at all.
    no_gpu = header(source="none", scope="none")
    no_gpu["backend"] = "none"
    write("cpu_only.jsonl", no_gpu, [
        {**tick(i, [], cpu=120.0), "scope": "none"} for i in range(10)])

    print("wrote fixtures to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
