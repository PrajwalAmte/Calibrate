//! Metric export: Prometheus scrape endpoint and OTLP/HTTP push.
//!
//! Both exporters render the same [`Point`] list, so metric names, units and
//! labels are identical whichever path a deployment uses. See
//! `docs/deployment.md` for the full metric reference.

pub mod otlp;
pub mod prometheus;

use crate::session::state::SessionSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gauge,
    Counter,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub name: &'static str,
    pub help: &'static str,
    pub kind: Kind,
    pub value: f64,
    pub labels: Vec<(&'static str, String)>,
}

struct Builder {
    base: Vec<(&'static str, String)>,
    out: Vec<Point>,
}

impl Builder {
    fn push(
        &mut self,
        name: &'static str,
        help: &'static str,
        kind: Kind,
        value: Option<f64>,
        extra: &[(&'static str, String)],
    ) {
        let Some(value) = value.filter(|v| v.is_finite()) else {
            return;
        };
        let mut labels = self.base.clone();
        labels.extend_from_slice(extra);
        self.out.push(Point {
            name,
            help,
            kind,
            value,
            labels,
        });
    }
    fn gauge(
        &mut self,
        name: &'static str,
        help: &'static str,
        v: Option<f64>,
        extra: &[(&'static str, String)],
    ) {
        self.push(name, help, Kind::Gauge, v, extra);
    }
}

fn ratio(pct: Option<f32>) -> Option<f64> {
    pct.map(|p| p as f64 / 100.0)
}

fn enum_str<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Flatten a snapshot into metric points.
pub fn points(s: &SessionSnapshot) -> Vec<Point> {
    let mut base = vec![
        ("host", s.target.hostname.clone()),
        ("mode", s.target.mode.clone()),
    ];
    if let Some(pid) = s.target.pid {
        base.push(("pid", pid.to_string()));
    }
    let mut b = Builder { base, out: vec![] };

    b.gauge(
        "calibrate_info",
        "Build and session information (always 1).",
        Some(1.0),
        &[
            ("version", s.calibrate_version.to_string()),
            ("backend", s.backend.clone()),
            ("metrics_source", s.metrics_source.clone()),
            ("gpu_name", s.gpu_name.clone()),
        ],
    );
    b.gauge(
        "calibrate_session_elapsed_seconds",
        "Seconds since the first sample.",
        Some(s.elapsed_secs),
        &[],
    );
    b.gauge(
        "calibrate_mfu_ratio",
        "Model FLOP utilization (0-1). Check the source and upper_bound labels.",
        ratio(s.mfu.mfu_pct),
        &[
            ("source", enum_str(&s.mfu.source)),
            ("upper_bound", s.mfu.upper_bound.to_string()),
        ],
    );
    b.gauge(
        "calibrate_achieved_tflops",
        "Estimated achieved TFLOPS across GPUs in use.",
        s.mfu.achieved_tflops,
        &[("source", enum_str(&s.mfu.source))],
    );
    b.gauge(
        "calibrate_peak_tflops",
        "Combined dense BF16 peak TFLOPS of GPUs in use.",
        s.mfu.peak_tflops,
        &[],
    );
    b.gauge(
        "calibrate_target_mfu_ratio",
        "Configured MFU target (0-1).",
        Some(s.target_mfu_pct as f64 / 100.0),
        &[],
    );
    b.gauge(
        "calibrate_process_cpu_cores",
        "CPU used by the target process tree, in cores.",
        s.cpu_pct.map(|c| c as f64 / 100.0),
        &[],
    );

    for g in &s.gpus {
        let gl = [("gpu", g.index.to_string()), ("gpu_name", g.name.clone())];
        b.gauge(
            "calibrate_gpu_utilization_ratio",
            "Fraction of time any kernel was running (NVML/IOKit).",
            ratio(g.util_pct),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_sm_active_ratio",
            "SM activity from GPM/DCGM profiling counters.",
            ratio(g.sm_active_pct),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_tensor_active_ratio",
            "Tensor-core pipe activity from GPM/DCGM profiling counters.",
            ratio(g.tensor_active_pct),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_dram_active_ratio",
            "DRAM interface activity from GPM/DCGM profiling counters.",
            ratio(g.dram_active_pct),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_sm_clock_hertz",
            "Current SM clock.",
            g.sm_clock_mhz.map(|c| c as f64 * 1e6),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_memory_used_bytes",
            "Device memory in use.",
            g.vram_used_mib.map(|m| m as f64 * 1048576.0),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_memory_total_bytes",
            "Device memory capacity (GPU budget on unified-memory Macs).",
            g.vram_total_mib.map(|m| m as f64 * 1048576.0),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_temperature_celsius",
            "GPU core temperature.",
            g.temperature_c.map(f64::from),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_power_watts",
            "Board power draw.",
            g.power_w.map(f64::from),
            &gl,
        );
        b.gauge(
            "calibrate_gpu_power_limit_watts",
            "Enforced board power limit.",
            g.power_limit_w.map(f64::from),
            &gl,
        );
        for (reason, on) in [
            ("hw_thermal", g.throttle.hw_thermal),
            ("sw_thermal", g.throttle.sw_thermal),
            ("power_cap", g.throttle.power_cap),
            ("hw_slowdown", g.throttle.hw_slowdown),
        ] {
            let mut l = gl.to_vec();
            l.push(("reason", reason.to_string()));
            b.gauge(
                "calibrate_gpu_throttle",
                "1 when the throttle reason is active.",
                Some(on as u8 as f64),
                &l,
            );
        }
        b.gauge(
            "calibrate_gpu_memory_growing",
            "1 when device memory has grown steadily (possible leak).",
            Some(g.vram_growing as u8 as f64),
            &gl,
        );
    }

    let src = enum_str(&s.breakdown.source);
    if s.breakdown.source != crate::metrics::breakdown::BreakdownSource::Unavailable {
        let bd = &s.breakdown;
        for (phase, v) in [
            ("compute", bd.compute_pct),
            ("data_wait", bd.data_wait_pct),
            ("sync_stall", bd.sync_stall_pct),
            ("memory_bound", bd.memory_bound_pct),
            ("optimizer", bd.optimizer_pct),
            ("other", bd.other_pct),
        ] {
            b.gauge(
                "calibrate_time_breakdown_ratio",
                "Share of time per phase (instrumented or heuristic, see source).",
                Some(v as f64 / 100.0),
                &[("phase", phase.to_string()), ("source", src.clone())],
            );
        }
    }

    if let Some(st) = &s.steps {
        for (q, v) in [("0.5", st.p50_ms), ("0.95", st.p95_ms)] {
            b.gauge(
                "calibrate_step_duration_seconds",
                "Training step wall time from the Python hook.",
                Some(v / 1000.0),
                &[("quantile", q.to_string())],
            );
        }
        b.gauge(
            "calibrate_steps_per_second",
            "Training steps per second summed over ranks.",
            Some(st.steps_per_sec),
            &[],
        );
        b.gauge(
            "calibrate_tokens_per_second",
            "Training tokens per second summed over ranks.",
            st.tokens_per_sec,
            &[],
        );
    }

    b.gauge(
        "calibrate_bottleneck",
        "1 for the currently detected primary bottleneck.",
        Some(1.0),
        &[("kind", enum_str(&s.bottleneck))],
    );
    b.gauge(
        "calibrate_gpu_divergence_ratio",
        "Spread of mean GPU busy fraction across devices.",
        s.gpu_divergence_ppt.map(|d| d as f64 / 100.0),
        &[],
    );

    if let Some(c) = &s.cost {
        b.gauge(
            "calibrate_cost_waste_ratio",
            "Share of spend below the MFU target (lower bound if MFU is an upper bound).",
            Some(c.waste_ratio),
            &[("lower_bound", c.is_lower_bound.to_string())],
        );
        b.push(
            "calibrate_cost_spent_usd_total",
            "GPU spend during this session.",
            Kind::Counter,
            Some(c.spent_usd),
            &[],
        );
        b.push(
            "calibrate_cost_wasted_usd_total",
            "GPU spend below the MFU target during this session.",
            Kind::Counter,
            Some(c.wasted_usd),
            &[("lower_bound", c.is_lower_bound.to_string())],
        );
    }
    b.out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::collectors::testing::{gpu, tick};
    use crate::collectors::GpuScope;
    use crate::gpu_specs::SpecDb;
    use crate::session::lifecycle::{Analyzer, SessionConfig};

    pub fn sample_snapshot() -> SessionSnapshot {
        let cfg = SessionConfig {
            mode: "process".into(),
            pid: Some(42),
            backend: "nvml".into(),
            metrics_source: "nvml".into(),
            scope: GpuScope::Process,
            cost_per_hour: Some(2.0),
            target_mfu: 45.0,
            warnings: vec![],
        };
        let mut a = Analyzer::new(cfg, SpecDb::builtin());
        let mut g = gpu(0, 70.0);
        g.name = "NVIDIA GeForce RTX 4090 \"special\"".into();
        let mut last = None;
        for i in 0..5 {
            last = Some(a.ingest(tick(i * 1000, vec![g.clone()], Some(150.0))));
        }
        last.unwrap()
    }

    #[test]
    fn points_cover_core_metrics_and_skip_missing() {
        let snap = sample_snapshot();
        let pts = points(&snap);
        let names: Vec<&str> = pts.iter().map(|p| p.name).collect();
        for n in [
            "calibrate_info",
            "calibrate_mfu_ratio",
            "calibrate_gpu_utilization_ratio",
            "calibrate_gpu_memory_used_bytes",
            "calibrate_bottleneck",
            "calibrate_cost_wasted_usd_total",
            "calibrate_time_breakdown_ratio",
        ] {
            assert!(names.contains(&n), "missing {n}");
        }
        assert!(
            !names.contains(&"calibrate_gpu_tensor_active_ratio"),
            "absent profiling counters must not be exported as 0"
        );
        let mfu = pts
            .iter()
            .find(|p| p.name == "calibrate_mfu_ratio")
            .unwrap();
        assert!(mfu.labels.contains(&("upper_bound", "true".to_string())));
        assert!((mfu.value - snap.mfu.mfu_pct.unwrap() as f64 / 100.0).abs() < 1e-6);
    }
}
