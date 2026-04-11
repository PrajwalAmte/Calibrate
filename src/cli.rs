use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// GPU training efficiency analyzer.
///
/// Attach to a running training job and instantly see your MFU,
/// where compute time is being lost, and the single change that
/// would make the biggest difference.
#[derive(Debug, Parser)]
#[command(
    name = "calibrate",
    version,
    about,
    long_about = None,
    arg_required_else_help = true,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Attach to a training process and analyze GPU efficiency in real time.
    Watch(WatchArgs),
    /// Verify the collector pipeline: print raw RawSamples as JSON to stdout.
    ///
    /// Use this to confirm that NVML + /proc collectors are working and that
    /// all fields (including cpu_utilization) arrive populated before starting
    /// a full monitoring session.
    Probe(ProbeArgs),
    /// Benchmark a model file across multiple inference runtimes.
    ///
    /// Measures p50/p95/p99 latency, throughput, and memory for each
    /// (runtime, batch size) combination and recommends the best configuration.
    Bench(BenchArgs),
    /// Estimate VRAM requirements and compare GPU pricing across cloud providers.
    ///
    /// Resolves the model's parameter count, calculates accurate VRAM usage for
    /// the given fine-tuning method and optimization stack, fetches live pricing
    /// from RunPod, Lambda Labs, and Vast.ai, and recommends the cheapest GPU
    /// that can reliably run the workload.
    Plan(PlanArgs),
}

/// Arguments accepted by `calibrate probe`.
#[derive(Debug, clap::Args)]
pub struct ProbeArgs {
    /// Process ID of the running GPU job to probe.
    #[arg(short, long, value_name = "PID")]
    pub pid: u32,

    /// Number of RawSamples to collect before exiting.
    #[arg(short = 'n', long, value_name = "COUNT", default_value = "5")]
    pub count: u32,

    /// Sampling interval in seconds.
    #[arg(short, long, value_name = "SECS", default_value = "2")]
    pub interval: f64,
}

/// Arguments accepted by `calibrate watch`.
#[derive(Debug, clap::Args)]
pub struct WatchArgs {
    /// Process ID of the running training job.
    #[arg(short, long, value_name = "PID")]
    pub pid: u32,

    /// Hourly GPU cost in USD (e.g. 0.34). Used to show dollar-denominated waste.
    #[arg(short = 'c', long, value_name = "USD/HR")]
    pub cost_per_hour: Option<f64>,

    /// Sampling interval in seconds.
    #[arg(short, long, value_name = "SECS", default_value = "2")]
    pub interval: f64,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "terminal")]
    pub output: OutputFormat,
}

/// Available output renderers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Live TUI dashboard (default).
    Terminal,
    /// Newline-delimited JSON — suitable for piping into other tools.
    Json,
}

// ── calibrate bench───

/// Arguments accepted by `calibrate bench`.
#[derive(Debug, clap::Args)]
pub struct BenchArgs {
    /// Path to the model file (.onnx, .gguf, .pt, .safetensors).
    #[arg(short, long, value_name = "FILE")]
    pub model: PathBuf,

    /// Comma-separated list of runtimes to test (e.g. onnxruntime,llamacpp).
    /// If omitted, all installed runtimes compatible with the model format are tested.
    #[arg(short, long, value_name = "RUNTIMES", value_delimiter = ',')]
    pub runtimes: Option<Vec<String>>,

    /// Batch sizes to benchmark, comma-separated.
    #[arg(short, long, value_name = "SIZES", value_delimiter = ',', default_values_t = vec![1u32, 8, 32])]
    pub batch_sizes: Vec<u32>,

    /// Primary metric to optimize for when selecting the recommended configuration.
    #[arg(long, value_enum, default_value = "latency")]
    pub optimize_for: OptimizeFor,

    /// Number of timed measurement iterations per (runtime, batch) pair.
    #[arg(long, value_name = "N", default_value = "100")]
    pub iterations: u32,

    /// Number of warm-up iterations (discarded before timing starts).
    #[arg(long, value_name = "N", default_value = "20")]
    pub warmup: u32,

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

// ── calibrate plan────

/// Arguments accepted by `calibrate plan`.
#[derive(Debug, clap::Args)]
pub struct PlanArgs {
    /// Hugging Face model ID (e.g. `meta-llama/Llama-3-8B`) or local path to a
    /// model directory containing config.json.
    #[arg(short, long, value_name = "MODEL_ID_OR_PATH")]
    pub model: String,

    /// Override the parameter count when the model cannot be resolved automatically.
    /// Specify in billions (e.g. `7.0` for a 7B model).
    #[arg(long, value_name = "BILLIONS")]
    pub params_b: Option<f64>,

    /// Fine-tuning method.
    #[arg(long, value_enum, default_value = "lora")]
    pub method: FinetuneMethod,

    /// Memory-efficiency library applied during training.
    #[arg(long, value_enum, default_value = "none")]
    pub optimizer: OptimizerLib,

    /// Weight quantization level.
    #[arg(long, value_enum, default_value = "none")]
    pub quantization: QuantLevel,

    /// Number of training examples in the dataset. Required for cost estimation.
    #[arg(long, value_name = "ROWS")]
    pub dataset_rows: Option<u64>,

    /// Effective batch size (including gradient accumulation steps).
    #[arg(long, value_name = "N", default_value = "1")]
    pub batch_size: u32,

    /// Number of training epochs.
    #[arg(long, value_name = "N", default_value = "1")]
    pub epochs: u32,

    /// Total spend limit in USD. Providers whose estimated cost exceeds this
    /// value are excluded from the recommendation (but still shown).
    #[arg(long, value_name = "USD")]
    pub budget: Option<f64>,

    /// GPU availability requirement.
    #[arg(long, value_enum, default_value = "flexible")]
    pub availability: Availability,

    /// Restrict results to specific providers (comma-separated).
    /// Valid values: runpod, lambda, vastai.
    #[arg(long, value_name = "PROVIDERS", value_delimiter = ',')]
    pub providers: Option<Vec<String>>,

    /// Actual MFU observed from a previous `calibrate watch` run.
    /// Replaces the conservative 0.30 default for duration estimation.
    #[arg(long, value_name = "0.0–1.0")]
    pub mfu: Option<f64>,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "terminal")]
    pub output: PlanOutputFormat,
}

/// Fine-tuning method, which determines which model parameters require
/// gradients and optimizer states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FinetuneMethod {
    /// Update all model parameters — highest VRAM, best accuracy ceiling.
    Full,
    /// Low-Rank Adaptation — only small adapter matrices are trained.
    Lora,
    /// Quantized LoRA — base model in 4-bit, adapters in full precision.
    Qlora,
}

/// Memory-efficiency library applied on top of the base framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OptimizerLib {
    /// No additional memory optimizations beyond the base framework.
    None,
    /// Unsloth — memory-efficient attention kernels + optimized quantization.
    /// Reduces total VRAM by approximately 40-60 % compared to standard HF training.
    Unsloth,
    /// DeepSpeed ZeRO — shards optimizer states and gradients across GPUs.
    DeepSpeed,
}

/// Weight quantization level applied to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum QuantLevel {
    /// Full precision (bfloat16 / float16).
    None,
    /// 8-bit quantization via bitsandbytes.
    #[value(name = "8bit")]
    EightBit,
    /// 4-bit NormalFloat quantization via bitsandbytes or GGUF.
    #[value(name = "4bit")]
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
