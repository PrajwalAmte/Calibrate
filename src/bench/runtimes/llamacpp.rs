use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::{Runtime, RuntimeDescriptor};

/// llama.cpp runtime — invokes `llama-bench` as a subprocess and parses its
/// JSON output.
///
/// `llama-bench` handles its own warm-up, measurement, and reporting so we
/// delegate everything to it and parse the structured output rather than
/// running a generic Python driver.
///
/// Requires `llama-bench` (or `llama-cli`) to be on the PATH. These ship with
/// llama.cpp pre-built binaries and most package manager distributions.
pub struct LlamaCppRuntime {
    model_path: Option<std::path::PathBuf>,
    latency_samples_us: Vec<u64>,
    throughput_rps: f64,
    load_ms: u64,
    /// Which binary was found at availability-check time.
    binary: &'static str,
}

impl LlamaCppRuntime {
    pub fn new() -> Self {
        Self {
            model_path: None,
            latency_samples_us: Vec::new(),
            throughput_rps: 0.0,
            load_ms: 0,
            binary: detect_binary(),
        }
    }

    pub fn descriptor() -> RuntimeDescriptor {
        RuntimeDescriptor {
            name: "llamacpp",
            is_available: || !detect_binary().is_empty(),
            is_compatible: |fmt| matches!(fmt, ModelFormat::Gguf),
            create: || Box::new(LlamaCppRuntime::new()),
        }
    }
}

impl Runtime for LlamaCppRuntime {
    fn name(&self) -> &str {
        "llamacpp"
    }

    fn load(&mut self, model_path: &Path) -> Result<Duration> {
        if !model_path.exists() {
            bail!("model file not found: {}", model_path.display());
        }
        self.model_path = Some(model_path.to_path_buf());
        self.latency_samples_us.clear();
        self.throughput_rps = 0.0;
        self.load_ms = 0;
        Ok(Duration::ZERO)
    }

    fn infer(&mut self, input: &BenchInput) -> Result<()> {
        if !self.latency_samples_us.is_empty() {
            return Ok(());
        }

        let model_path = self
            .model_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("llamacpp: load() not called"))?;

        let binary = self.binary;
        if binary.is_empty() {
            bail!("llama-bench not found in PATH");
        }

        // n_prompt tokens = batch_size, n_gen = 1 (we measure prompt-processing speed)
        let batch_size = input.shape.first().copied().unwrap_or(1);

        let t0 = Instant::now();
        let output = Command::new(binary)
            .args([
                "--model",
                model_path.to_str().unwrap_or_default(),
                "--n-prompt",
                &batch_size.to_string(),
                "--n-gen",
                "0",
                "--output",
                "jsonl",
                "--warmup",
                "5",
            ])
            .output()
            .with_context(|| format!("failed to launch {binary}"))?;

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("{binary} exited with {}: {}", output.status, stderr);
        }

        parse_llama_bench_output(&output.stdout, elapsed_ms, self)?;

        Ok(())
    }

    fn unload(&mut self) {
        self.model_path = None;
        self.latency_samples_us.clear();
        self.throughput_rps = 0.0;
        self.load_ms = 0;
    }

    fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> {
        if self.latency_samples_us.is_empty() {
            return None;
        }
        Some((
            std::mem::take(&mut self.latency_samples_us),
            self.throughput_rps,
            self.load_ms,
        ))
    }
}

// ── Output parsing

/// Parse JSONL output from `llama-bench --output jsonl`.
///
/// Each line is a JSON object. We look for `t_pp_ms` (prompt processing time)
/// or `t_p_ms` to derive per-sample latency and throughput.
fn parse_llama_bench_output(
    stdout: &[u8],
    total_elapsed_ms: u64,
    runtime: &mut LlamaCppRuntime,
) -> Result<()> {
    let text = std::str::from_utf8(stdout).context("llama-bench output is not valid UTF-8")?;

    let mut found_any = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }

        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // llama-bench JSONL schema has fields like:
        //   "t_pp_ms"   — total time for prompt processing (ms)
        //   "n_pp"      — number of tokens processed
        //   "t_tg_ms"   — total time for token generation (ms)
        //   "t_load_ms" — model load time (ms)
        if let Some(load_ms) = v.get("t_load_ms").and_then(|v| v.as_f64()) {
            runtime.load_ms = load_ms as u64;
        }

        // Prefer prompt-processing time since that corresponds to batch inference.
        // t_pp_ms is the total across n_pp tokens; per-run latency = t_pp_ms / n_runs.
        if let (Some(t_pp_ms), Some(n_pp)) = (
            v.get("t_pp_ms").and_then(|v| v.as_f64()),
            v.get("n_pp").and_then(|v| v.as_u64()),
        ) {
            if n_pp > 0 {
                // Synthesise pseudo-samples from the aggregate time.
                let per_run_us = (t_pp_ms * 1000.0 / n_pp as f64) as u64;
                // Emit 100 synthetic samples at this average to populate the histogram.
                for _ in 0..100 {
                    runtime.latency_samples_us.push(per_run_us.max(1));
                }
                let tput = n_pp as f64 / (t_pp_ms / 1000.0).max(f64::EPSILON);
                runtime.throughput_rps = tput;
                found_any = true;
            }
        }
    }

    if !found_any {
        // llama-bench may use a different schema version; fall back to reporting
        // total elapsed time as the single latency sample.
        let elapsed_us = total_elapsed_ms.saturating_mul(1000);
        runtime.latency_samples_us.push(elapsed_us.max(1));
        runtime.throughput_rps = if total_elapsed_ms > 0 {
            1000.0 / total_elapsed_ms as f64
        } else {
            0.0
        };
        eprintln!(
            "Warning: could not parse llama-bench structured output; \
             using total elapsed time as a single-sample estimate."
        );
    }

    Ok(())
}

// ── Binary detection

fn detect_binary() -> &'static str {
    for candidate in &["llama-bench", "llama-cli"] {
        if which_exists(candidate) {
            // SAFETY: both string literals are &'static str.
            return candidate;
        }
    }
    ""
}

fn which_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
