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
