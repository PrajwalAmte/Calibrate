use anyhow::Result;
use clap::Parser;

mod analysis;
mod bench;
mod cli;
mod collectors;
mod commands;
mod config;
mod error;
mod export;
mod gpu_specs;
mod metrics;
mod net;
mod output;
mod plan;
mod process;
mod session;
mod util;

use cli::{Cli, Commands, LogFormat};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Diagnostics go to stderr so stdout stays machine-readable.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    use std::io::IsTerminal;
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(std::io::stderr().is_terminal())
        .with_writer(std::io::stderr);
    match cli.log_format {
        LogFormat::Json => builder.json().init(),
        LogFormat::Text => builder.init(),
    }

    let file = config::ConfigFile::load(cli.config.as_deref())?;
    config::init(config::Settings::resolve(file, cli.offline));

    match cli.command {
        Commands::Diagnose(args) => commands::diagnose::run(args).await,
        Commands::Watch(args) => commands::watch::run(args).await,
        Commands::Probe(args) => commands::probe::run(args).await,
        Commands::Bench(args) => commands::bench::run(args).await,
        Commands::Plan(args) => commands::plan::run(args).await,
    }
}
