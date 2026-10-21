use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// GPU training efficiency analyzer.
///
/// Attach to a running training job and see how efficiently it uses the GPU,
/// where time is being lost, and the single change most likely to help.
#[derive(Debug, Parser)]
#[command(
    name = "calibrate",
    version,
    about,
    long_about = None,
    arg_required_else_help = true,
)]
pub struct Cli {
    /// Path to a TOML config file. Defaults to `$CALIBRATE_CONFIG`, then
    /// `~/.config/calibrate/config.toml` (Linux) or the platform equivalent.
    #[arg(long, global = true, value_name = "FILE", env = "CALIBRATE_CONFIG")]
    pub config: Option<PathBuf>,

    /// Never make network requests (air-gapped clusters). Uses the built-in GPU
    /// spec database and requires `--prices-file` / `--params-b` for `plan`.
    #[arg(long, global = true, env = "CALIBRATE_OFFLINE",
          action = clap::ArgAction::SetTrue,
          value_parser = clap::builder::BoolishValueParser::new())]
    pub offline: bool,

    /// Diagnostic log format written to stderr.
    #[arg(long, global = true, value_enum, default_value = "text")]
    pub log_format: LogFormat,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Check that all calibrate dependencies are installed and working correctly.
    ///
    /// Inspects the NVIDIA driver stack (nvidia-smi, NVML library, /dev/nvidiactl,
    /// GPM/DCGM profiling support), or the Apple IOKit GPU service on macOS, and
    /// prints a pass/fail report with remediation steps for every failed check.
    Diagnose(DiagnoseArgs),
    /// Attach to a training process (or a set of GPUs) and analyze efficiency live.
    Watch(WatchArgs),
    /// Verify the collector pipeline: print raw samples as JSON to stdout.
    Probe(ProbeArgs),
    /// Benchmark a model file across multiple inference runtimes.
    ///
    /// Measures p50/p95/p99 latency, throughput, and memory for each
    /// (runtime, batch size) combination and recommends the best configuration.
    Bench(BenchArgs),
    /// Estimate VRAM requirements and compare GPU pricing across cloud providers.
    ///
    /// Resolves the model's parameter count, estimates per-GPU VRAM for the given
    /// fine-tuning method, fetches pricing from RunPod, Lambda and Vast.ai (or a
    /// local prices file), and recommends the cheapest GPU that fits.
    Plan(PlanArgs),
}

/// Arguments accepted by `calibrate diagnose`.
#[derive(Debug, clap::Args)]
pub struct DiagnoseArgs {}

/// Parse a sampling interval, rejecting values that would spin or overflow.
fn parse_interval(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a number of seconds"))?;
    if !v.is_finite() || !(0.1..=3600.0).contains(&v) {
        return Err(format!(
            "interval must be between 0.1 and 3600 seconds (got {s})"
        ));
    }
    Ok(v)
}

fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!("value must be a positive number (got {s})"));
    }
    Ok(v)
}

fn parse_non_negative_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !v.is_finite() || v < 0.0 {
        return Err(format!("value must be zero or positive (got {s})"));
    }
    Ok(v)
}

fn parse_ratio(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !v.is_finite() || v <= 0.0 || v > 1.0 {
        return Err(format!(
            "value must be a fraction in (0, 1], e.g. 0.38 for 38% (got {s})"
        ));
    }
    Ok(v)
}

fn parse_target_pct(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !v.is_finite() || v <= 0.0 || v > 100.0 {
        return Err(format!("target must be a percentage in (0, 100] (got {s})"));
    }
    Ok(v)
}

/// `--gpus all` or `--gpus 0,1,3`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSelection {
    All,
    Indices(Vec<u32>),
}

fn parse_gpu_selection(s: &str) -> Result<GpuSelection, String> {
    if s.eq_ignore_ascii_case("all") {
        return Ok(GpuSelection::All);
    }
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        out.push(
            part.parse::<u32>()
                .map_err(|_| format!("'{part}' is not a GPU index (use e.g. 0,1 or 'all')"))?,
        );
    }
    if out.is_empty() {
        return Err("expected 'all' or a comma-separated list of GPU indices".into());
    }
    out.sort_unstable();
    out.dedup();
    Ok(GpuSelection::Indices(out))
}

fn parse_header(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("header must be KEY=VALUE (got '{s}')"))?;
    let k = k.trim();
    if k.is_empty() {
        return Err("header name must not be empty".into());
    }
    Ok((k.to_string(), v.trim().to_string()))
}

/// Arguments accepted by `calibrate probe`.
#[derive(Debug, clap::Args)]
#[command(group(clap::ArgGroup::new("probe_target").required(true).args(["pid", "gpus"])))]
pub struct ProbeArgs {
    /// Process ID of the running GPU job to probe.
    #[arg(short, long, value_name = "PID")]
    pub pid: Option<u32>,

    /// Probe GPUs directly instead of a process (`all` or e.g. `0,1`).
    #[arg(long, value_name = "all|INDICES", value_parser = parse_gpu_selection)]
    pub gpus: Option<GpuSelection>,

    /// Number of sample ticks to collect before exiting.
    #[arg(short = 'n', long, value_name = "COUNT", default_value = "5")]
    pub count: u32,

    /// Sampling interval in seconds (0.1 – 3600).
    #[arg(short, long, value_name = "SECS", default_value = "2", value_parser = parse_interval)]
    pub interval: f64,

    /// Which GPU profiling source to use (Linux/NVIDIA).
    #[arg(long, value_enum, default_value = "auto")]
    pub metrics_source: MetricsSource,
}

/// Arguments accepted by `calibrate watch`.
///
/// Exactly one target is required: `--pid`, `--process-name`, `--auto`,
/// `--gpus`, or `--replay`. All other options combine freely with any target.
#[derive(Debug, clap::Args)]
#[command(group(
    clap::ArgGroup::new("target")
        .required(true)
        .multiple(false)
        .args(["pid", "process_name", "auto", "gpus", "replay"])
))]
pub struct WatchArgs {
    /// Process ID of the running training job (its child processes, e.g.
    /// DataLoader workers and torchrun ranks, are included automatically).
    #[arg(short, long, value_name = "PID")]
    pub pid: Option<u32>,

    /// Attach by process name (regex match against the full command line).
    ///
    /// If several processes match and one is the parent of the others (e.g. a
    /// torchrun launcher and its ranks), the parent is chosen. Otherwise the
    /// matches are listed and calibrate exits.
    #[arg(short = 'n', long, value_name = "PATTERN")]
    pub process_name: Option<String>,

    /// Auto-discover training processes; shows a numbered menu if more than one.
    #[arg(short = 'a', long)]
    pub auto: bool,

    /// Monitor GPUs directly with no target process (`all` or e.g. `0,1`).
    ///
    /// Device-level mode for node agents (Kubernetes DaemonSet) and for
    /// containers where NVML cannot map process IDs.
    #[arg(long, value_name = "all|INDICES", value_parser = parse_gpu_selection)]
    pub gpus: Option<GpuSelection>,

    /// Replay a trace recorded with `--record-raw` instead of live collection.
    #[arg(long, value_name = "FILE")]
    pub replay: Option<PathBuf>,

    /// Replay speed multiplier (0 = as fast as possible, 1 = original timing).
    #[arg(long, value_name = "X", value_parser = parse_non_negative_f64,
          conflicts_with_all = ["pid", "process_name", "auto", "gpus"])]
    pub replay_speed: Option<f64>,

    /// Only watch the target PID itself, not its descendants.
    #[arg(long)]
    pub no_children: bool,

    /// Hourly GPU cost in USD (e.g. 2.49). Enables dollar-denominated waste.
    #[arg(short = 'c', long, value_name = "USD/HR", value_parser = parse_non_negative_f64)]
    pub cost_per_hour: Option<f64>,

    /// Sampling interval in seconds (0.1 – 3600). Default 2, or the config file value.
    #[arg(short, long, value_name = "SECS", value_parser = parse_interval)]
    pub interval: Option<f64>,

    /// Target MFU percentage used for "below target" and waste calculations.
    #[arg(long, value_name = "PCT", value_parser = parse_target_pct)]
    pub target_mfu: Option<f32>,

    /// Stop automatically after this many seconds.
    #[arg(long, value_name = "SECS", value_parser = parse_positive_f64)]
    pub duration: Option<f64>,

    /// Output format on stdout.
    #[arg(short, long, value_enum, default_value = "terminal")]
    pub output: OutputFormat,

    /// Which GPU profiling source to use for utilization (Linux/NVIDIA).
    ///
    /// `auto` prefers NVML GPM (Hopper and newer), then DCGM (`dcgmi`), then
    /// plain NVML kernel-busy utilization, which is only an upper bound on MFU.
    #[arg(long, value_enum)]
    pub metrics_source: Option<MetricsSource>,

    /// Directory where the calibrate Python hook publishes its sockets.
    #[arg(long, value_name = "DIR", env = "CALIBRATE_HOOK_DIR")]
    pub hook_dir: Option<PathBuf>,

    /// Append every snapshot as JSON Lines to this file.
    #[arg(long, value_name = "FILE")]
    pub record: Option<PathBuf>,

    /// Append every raw sample tick as JSON Lines (replayable with `--replay`).
    #[arg(long, value_name = "FILE")]
    pub record_raw: Option<PathBuf>,

    /// Serve Prometheus metrics on this address (e.g. `0.0.0.0:9400`).
    #[arg(long, value_name = "ADDR")]
    pub prometheus_listen: Option<String>,

    /// Push metrics to an OpenTelemetry collector over OTLP/HTTP
    /// (e.g. `http://otel-collector:4318`).
    #[arg(long, value_name = "URL")]
    pub otlp_endpoint: Option<String>,

    /// Extra header for OTLP requests, e.g. `Authorization=Bearer abc` (repeatable).
    #[arg(long = "otlp-header", value_name = "KEY=VALUE", value_parser = parse_header)]
    pub otlp_headers: Vec<(String, String)>,
}

/// Where per-GPU compute-utilization comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricsSource {
    /// Best available: GPM, then DCGM, then NVML.
    Auto,
    /// NVML GPU Performance Monitoring (Hopper+): SM / tensor-core activity.
    Gpm,
    /// DCGM profiling fields via `dcgmi dmon` (A100, H100, L4, ...).
    Dcgm,
    /// Plain NVML utilization (kernel busy time). Works everywhere; MFU upper bound only.
    Nvml,
}

/// Available output renderers for `watch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Live TUI dashboard (default). Press q, Esc or Ctrl+C to quit.
    Terminal,
    /// Newline-delimited JSON snapshots (versioned schema) on stdout.
    Json,
    /// Nothing on stdout — for headless use with exporters or `--record`.
    None,
}

//  calibrate bench

/// Arguments accepted by `calibrate bench`.
#[derive(Debug, clap::Args)]
pub struct BenchArgs {
    /// Path to the model file (.onnx, .gguf, .pt, .safetensors, .engine).
    #[arg(short, long, value_name = "FILE")]
    pub model: PathBuf,

    /// Comma-separated list of runtimes to test (e.g. onnxruntime,llamacpp).
    /// If omitted, all installed runtimes compatible with the model format are tested.
    #[arg(short, long, value_name = "RUNTIMES", value_delimiter = ',')]
    pub runtimes: Option<Vec<String>>,

    /// Batch sizes to benchmark, comma-separated.
    #[arg(short, long, value_name = "SIZES", value_delimiter = ',', default_values_t = vec![1u32, 8, 32],
          value_parser = clap::value_parser!(u32).range(1..))]
    pub batch_sizes: Vec<u32>,

    /// Primary metric to optimize for when selecting the recommended configuration.
    #[arg(long, value_enum, default_value = "latency")]
    pub optimize_for: OptimizeFor,

    /// Number of timed measurement iterations per (runtime, batch) pair.
    #[arg(long, value_name = "N", default_value = "100",
          value_parser = clap::value_parser!(u32).range(1..))]
    pub iterations: u32,

    /// Number of warm-up iterations (discarded before timing starts).
    #[arg(long, value_name = "N", default_value = "20")]
    pub warmup: u32,

    /// Length of the sustained-throughput window in seconds.
    #[arg(long, value_name = "SECS", default_value = "5", value_parser = parse_positive_f64)]
    pub throughput_window: f64,

    /// Input dimensions after the batch dimension, e.g. `3,224,224` or `512`.
    /// Inferred from the model when omitted.
    #[arg(long, value_name = "DIMS", value_delimiter = ',',
          value_parser = clap::value_parser!(u32).range(1..))]
    pub input_shape: Option<Vec<u32>>,

    /// Sequence length for token-input models (transformers, ONNX dynamic axes, llama.cpp).
    #[arg(long, value_name = "TOKENS", default_value = "512",
          value_parser = clap::value_parser!(u32).range(1..))]
    pub seq_len: u32,

    /// Kill a runtime's benchmark subprocess if it runs longer than this.
    #[arg(long, value_name = "SECS", default_value = "600", value_parser = parse_positive_f64)]
    pub timeout: f64,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "terminal")]
    pub output: BenchOutputFormat,

    /// Compare results against a previously saved JSON baseline.
    #[arg(long, value_name = "BASELINE_JSON")]
    pub compare: Option<PathBuf>,

    /// Save the full benchmark report to a JSON file for future comparison.
    #[arg(long, value_name = "OUTPUT_JSON")]
    pub save: Option<PathBuf>,
}

/// Which metric `calibrate bench` should optimise when making its recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OptimizeFor {
    /// Minimise p99 latency — best for interactive / real-time serving.
    Latency,
    /// Maximise sustained requests per second — best for batch inference pipelines.
    Throughput,
    /// Minimise peak memory consumption — best for memory-constrained hardware.
    Memory,
}

/// Output format for `calibrate bench` results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BenchOutputFormat {
    /// Formatted table printed to stdout (default).
    Terminal,
    /// Full JSON report including per-iteration timing arrays.
    Json,
    /// Markdown report suitable for committing to a repository.
    Markdown,
}

//  calibrate plan

/// Arguments accepted by `calibrate plan`.
#[derive(Debug, clap::Args)]
pub struct PlanArgs {
    /// Hugging Face model ID (e.g. `meta-llama/Llama-3.1-8B`) or local path to a
    /// model directory containing config.json.
    #[arg(short, long, value_name = "MODEL_ID_OR_PATH")]
    pub model: String,

    /// Override the parameter count (billions), e.g. `7.6`.
    #[arg(long, value_name = "BILLIONS", value_parser = parse_positive_f64)]
    pub params_b: Option<f64>,

    /// Fine-tuning method.
    #[arg(long, value_enum, default_value = "lora")]
    pub method: FinetuneMethod,

    /// Memory-efficiency library applied during training.
    #[arg(long, value_enum, default_value = "none")]
    pub optimizer: OptimizerLib,

    /// Weight quantization level (QLoRA always uses 4-bit base weights).
    #[arg(long, value_enum, default_value = "none")]
    pub quantization: QuantLevel,

    /// Number of training examples in the dataset. Required for cost estimation.
    #[arg(long, value_name = "ROWS", value_parser = clap::value_parser!(u64).range(1..))]
    pub dataset_rows: Option<u64>,

    /// Tokens per training example after padding/packing. Default 2048.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub seq_len: Option<u32>,

    /// Per-device micro-batch size (sequences per GPU per forward pass).
    #[arg(long, value_name = "N", default_value = "1",
          value_parser = clap::value_parser!(u32).range(1..))]
    pub batch_size: u32,

    /// Number of training epochs.
    #[arg(long, value_name = "N", default_value = "1",
          value_parser = clap::value_parser!(u32).range(1..))]
    pub epochs: u32,

    /// Total spend limit in USD. Listings above it are flagged and never recommended.
    #[arg(long, value_name = "USD", value_parser = parse_positive_f64)]
    pub budget: Option<f64>,

    /// GPU availability requirement.
    #[arg(long, value_enum, default_value = "flexible")]
    pub availability: Availability,

    /// Restrict results to specific providers (comma-separated: runpod, lambda, vastai).
    #[arg(long, value_name = "PROVIDERS", value_delimiter = ',')]
    pub providers: Option<Vec<String>>,

    /// Read GPU listings from a JSON file instead of live provider APIs.
    #[arg(long, value_name = "FILE")]
    pub prices_file: Option<PathBuf>,

    /// Ignore cached provider prices and fetch fresh ones.
    #[arg(long)]
    pub refresh_prices: bool,

    /// Hardware utilization to assume (fraction, e.g. 0.38) — use the value
    /// `calibrate watch` reports for a comparable job. Default 0.30.
    #[arg(long, value_name = "0.0–1.0", value_parser = parse_ratio)]
    pub mfu: Option<f64>,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "terminal")]
    pub output: PlanOutputFormat,
}

/// Fine-tuning method, which determines which model parameters require
/// gradients and optimizer states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FinetuneMethod {
    /// Update all model parameters — highest VRAM, best accuracy ceiling.
    Full,
    /// Low-Rank Adaptation — only small adapter matrices are trained.
    Lora,
    /// Quantized LoRA — base model in 4-bit, adapters in full precision.
    Qlora,
}

/// Memory-efficiency library applied on top of the base framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerLib {
    /// No additional memory optimizations beyond the base framework.
    None,
    /// Unsloth — fused kernels, activation offloading and chunked cross-entropy.
    Unsloth,
    /// DeepSpeed ZeRO-3 — shards weights, gradients and optimizer states across
    /// the GPUs of an instance (optimizer offload to CPU on single-GPU instances).
    DeepSpeed,
}

/// Weight quantization level applied to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum QuantLevel {
    /// Full precision (bfloat16 / float16).
    None,
    /// 8-bit quantization via bitsandbytes.
    #[value(name = "8bit")]
    #[serde(rename = "8bit")]
    EightBit,
    /// 4-bit NormalFloat quantization via bitsandbytes.
    #[value(name = "4bit")]
    #[serde(rename = "4bit")]
    FourBit,
}

/// GPU availability requirement for the plan query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Availability {
    /// Only show GPU instances that are available to launch immediately.
    Now,
    /// Show all options, including those with a queue or currently unavailable.
    Flexible,
}

/// Output format for `calibrate plan` results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PlanOutputFormat {
    /// Workload analysis + ranked pricing table printed to stdout (default).
    Terminal,
    /// Full JSON report including the complete VRAM breakdown and provider listings.
    Json,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut full = vec!["calibrate"];
        full.extend_from_slice(args);
        Cli::try_parse_from(full)
    }

    fn watch(args: &[&str]) -> WatchArgs {
        let mut full = vec!["watch"];
        full.extend_from_slice(args);
        match parse(&full).expect("should parse").command {
            Commands::Watch(w) => w,
            other => panic!("expected watch, got {other:?}"),
        }
    }

    #[test]
    fn clap_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn readme_watch_example_parses() {
        let w = watch(&["--pid", "38291", "--cost-per-hour", "0.34"]);
        assert_eq!(w.pid, Some(38291));
        assert_eq!(w.cost_per_hour, Some(0.34));
    }

    #[test]
    fn pid_combines_with_every_option() {
        let w = watch(&[
            "--pid",
            "1",
            "--interval",
            "0.5",
            "--output",
            "json",
            "--cost-per-hour",
            "2.5",
            "--target-mfu",
            "50",
            "--duration",
            "10",
            "--metrics-source",
            "nvml",
            "--prometheus-listen",
            "127.0.0.1:9400",
            "--record",
            "out.jsonl",
        ]);
        assert_eq!(w.interval, Some(0.5));
        assert_eq!(w.output, OutputFormat::Json);
        assert_eq!(w.metrics_source, Some(MetricsSource::Nvml));
    }

    #[test]
    fn process_name_and_auto_combine_with_options() {
        let w = watch(&["--process-name", "train\\.py", "-i", "1", "-o", "json"]);
        assert_eq!(w.process_name.as_deref(), Some("train\\.py"));
        let w = watch(&["--auto", "--cost-per-hour", "1"]);
        assert!(w.auto);
    }

    #[test]
    fn watch_requires_a_target() {
        let err = parse(&["watch", "--interval", "1"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn watch_rejects_two_targets() {
        let err = parse(&["watch", "--pid", "1", "--auto"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn interval_is_validated() {
        for bad in ["0", "-1", "0.01", "nan", "inf", "99999", "abc"] {
            assert!(
                parse(&["watch", "--pid", "1", "--interval", bad]).is_err(),
                "interval {bad} should be rejected"
            );
        }
        assert!(parse(&["watch", "--pid", "1", "--interval", "0.1"]).is_ok());
    }

    #[test]
    fn gpus_selection_parses() {
        assert_eq!(watch(&["--gpus", "all"]).gpus, Some(GpuSelection::All));
        assert_eq!(
            watch(&["--gpus", "3,1,1"]).gpus,
            Some(GpuSelection::Indices(vec![1, 3]))
        );
        assert!(parse(&["watch", "--gpus", "x"]).is_err());
    }

    #[test]
    fn replay_speed_requires_replay() {
        assert!(parse(&["watch", "--pid", "1", "--replay-speed", "2"]).is_err());
        let w = watch(&["--replay", "t.jsonl", "--replay-speed", "2"]);
        assert_eq!(w.replay_speed, Some(2.0));
    }

    #[test]
    fn otlp_headers_parse() {
        let w = watch(&[
            "--gpus",
            "all",
            "--otlp-endpoint",
            "http://c:4318",
            "--otlp-header",
            "Authorization=Bearer x",
        ]);
        assert_eq!(
            w.otlp_headers,
            vec![("Authorization".to_string(), "Bearer x".to_string())]
        );
    }

    #[test]
    fn global_flags_work_after_subcommand() {
        let cli = parse(&["plan", "--model", "x", "--params-b", "7", "--offline"]).unwrap();
        assert!(cli.offline);
    }

    #[test]
    fn offline_env_accepts_common_spellings() {
        for (v, expected) in [
            ("1", true),
            ("yes", true),
            ("true", true),
            ("0", false),
            ("no", false),
        ] {
            std::env::set_var("CALIBRATE_OFFLINE", v);
            let cli = parse(&["diagnose"]).unwrap();
            std::env::remove_var("CALIBRATE_OFFLINE");
            assert_eq!(cli.offline, expected, "CALIBRATE_OFFLINE={v}");
        }
    }

    #[test]
    fn plan_validates_numbers() {
        assert!(parse(&["plan", "-m", "x", "--mfu", "1.5"]).is_err());
        assert!(parse(&["plan", "-m", "x", "--mfu", "0"]).is_err());
        assert!(parse(&["plan", "-m", "x", "--batch-size", "0"]).is_err());
        assert!(parse(&["plan", "-m", "x", "--seq-len", "0"]).is_err());
        assert!(parse(&["plan", "-m", "x", "--mfu", "0.38", "--seq-len", "4096"]).is_ok());
    }

    #[test]
    fn bench_rejects_zero_batch_and_iterations() {
        assert!(parse(&["bench", "-m", "a.onnx", "-b", "0"]).is_err());
        assert!(parse(&["bench", "-m", "a.onnx", "--iterations", "0"]).is_err());
        assert!(parse(&["bench", "-m", "a.onnx", "--timeout", "30"]).is_ok());
    }

    #[test]
    fn probe_accepts_pid_or_gpus() {
        assert!(parse(&["probe", "--pid", "1"]).is_ok());
        assert!(parse(&["probe", "--gpus", "0"]).is_ok());
        assert!(parse(&["probe"]).is_err());
    }
}
