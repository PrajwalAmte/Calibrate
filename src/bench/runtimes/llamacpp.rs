//! llama.cpp via `llama-bench`.
//!
//! Measures prompt processing of `--seq-len` tokens with the bench batch size
//! as llama.cpp's logical batch (`-b`). `llama-bench -o jsonl` reports one
//! object per test with `samples_ns` (one entry per repetition), `avg_ns` and
//! `avg_ts` (tokens/s); repetitions are taken from `--iterations`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::{Runtime, RuntimeDescriptor, SubprocessSettings};

pub struct LlamaCppRuntime {
    model_path: Option<PathBuf>,
    settings: SubprocessSettings,
    samples_us: Vec<u64>,
    tokens_per_sec: f64,
}

/// Parsed `llama-bench` result.
#[derive(Debug, PartialEq)]
pub struct LlamaBenchResult {
    pub samples_us: Vec<u64>,
    pub tokens_per_sec: f64,
}

impl LlamaCppRuntime {
    pub fn new() -> Self {
        Self {
            model_path: None,
            settings: SubprocessSettings::default(),
            samples_us: vec![],
            tokens_per_sec: 0.0,
        }
    }

    pub fn descriptor() -> RuntimeDescriptor {
        RuntimeDescriptor {
            name: "llamacpp",
            is_available: || which("llama-bench").is_some(),
            is_compatible: |fmt| matches!(fmt, ModelFormat::Gguf),
            create: || Box::new(LlamaCppRuntime::new()),
            opt_in: false,
        }
    }
}

impl Runtime for LlamaCppRuntime {
    fn name(&self) -> &str {
        "llamacpp"
    }

    fn configure(&mut self, settings: &SubprocessSettings) {
        self.settings = settings.clone();
    }

    fn load(&mut self, model_path: &Path) -> Result<Duration> {
        if !model_path.exists() {
            bail!("model file not found: {}", model_path.display());
        }
        self.model_path = Some(model_path.to_path_buf());
        self.samples_us.clear();
        Ok(Duration::ZERO)
    }

    fn infer(&mut self, input: &BenchInput) -> Result<()> {
        if !self.samples_us.is_empty() {
            return Ok(());
        }
        let model = self
            .model_path
            .clone()
            .context("llamacpp: load() not called")?;
        let batch = input.shape.first().copied().unwrap_or(1).max(1);
        let s = &self.settings;
        let mut cmd = Command::new("llama-bench");
        cmd.arg("-m")
            .arg(&model)
            .args(["-p", &s.seq_len.to_string()])
            .args(["-n", "0"])
            .args(["-b", &batch.max(1).to_string()])
            .args(["-r", &s.iterations.max(1).to_string()])
            .args(["-o", "jsonl"]);
        let out = crate::util::output_with_timeout(&mut cmd, s.timeout)?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            bail!(
                "llama-bench exited with {}: {}",
                out.status,
                err.lines().last().unwrap_or("")
            );
        }
        let r = parse_llama_bench(&out.stdout)?;
        self.samples_us = r.samples_us;
        self.tokens_per_sec = r.tokens_per_sec;
        Ok(())
    }

    fn unload(&mut self) {
        self.model_path = None;
        self.samples_us.clear();
        self.tokens_per_sec = 0.0;
    }

    fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> {
        if self.samples_us.is_empty() {
            return None;
        }
        // Throughput here is prompts per second (one prompt = seq_len tokens).
        let rps = self.tokens_per_sec / self.settings.seq_len.max(1) as f64;
        Some((std::mem::take(&mut self.samples_us), rps, 0))
    }
}

/// Parse `llama-bench -o jsonl` output into per-repetition latencies.
pub fn parse_llama_bench(stdout: &[u8]) -> Result<LlamaBenchResult> {
    let text = std::str::from_utf8(stdout).context("llama-bench output is not UTF-8")?;
    for line in text.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        // Prompt-processing test: n_prompt > 0, n_gen == 0.
        if v["n_prompt"].as_u64().unwrap_or(0) == 0 {
            continue;
        }
        let mut samples: Vec<u64> = v["samples_ns"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|ns| (ns / 1000.0) as u64)
                    .collect()
            })
            .unwrap_or_default();
        if samples.is_empty() {
            if let Some(avg) = v["avg_ns"].as_f64() {
                samples.push((avg / 1000.0) as u64);
            }
        }
        if samples.is_empty() {
            continue;
        }
        let tps = v["avg_ts"].as_f64().unwrap_or_else(|| {
            let mean_s = samples.iter().sum::<u64>() as f64 / samples.len() as f64 / 1e6;
            v["n_prompt"].as_f64().unwrap_or(0.0) / mean_s.max(f64::EPSILON)
        });
        return Ok(LlamaBenchResult {
            samples_us: samples.into_iter().map(|s| s.max(1)).collect(),
            tokens_per_sec: tps,
        });
    }
    bail!("could not find a prompt-processing result in llama-bench output")
}

fn which(program: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|d| d.join(program))
            .find(|c| c.is_file())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_samples_ns() {
        let out = br#"{"build_commit":"abc","n_prompt":512,"n_gen":0,"n_batch":8,"avg_ns":2000000,"avg_ts":256000.0,"samples_ns":[1900000,2000000,2100000]}"#;
        let r = parse_llama_bench(out).unwrap();
        assert_eq!(
            r.samples_us,
            vec![1900, 2000, 2100],
            "real per-repetition samples"
        );
        assert_eq!(r.tokens_per_sec, 256000.0);
    }

    #[test]
    fn falls_back_to_avg_and_skips_generation_tests() {
        let out = b"{\"n_prompt\":0,\"n_gen\":128,\"avg_ns\":5}\n{\"n_prompt\":64,\"n_gen\":0,\"avg_ns\":1000000}\n";
        let r = parse_llama_bench(out).unwrap();
        assert_eq!(r.samples_us, vec![1000]);
        assert!((r.tokens_per_sec - 64_000.0).abs() < 1.0);
    }

    #[test]
    fn rejects_unparseable_output() {
        assert!(parse_llama_bench(b"| model | size |\n").is_err());
    }
}
