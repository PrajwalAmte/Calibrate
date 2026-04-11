use anyhow::Result;
use clap::Parser;

mod analysis;
mod bench;
mod cli;
mod collectors;
mod commands;
mod error;
mod gpu_specs;
mod metrics;
mod output;
mod plan;
mod process;
mod session;

use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging — respects RUST_LOG env var.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Watch(args) => {
            commands::watch::run(args).await?;
        }
        Commands::Probe(args) => {
            commands::probe::run(args).await?;
        }
        Commands::Bench(args) => {
            commands::bench::run(args).await?;
        }
        Commands::Plan(args) => {
            commands::plan::run(args).await?;
        }
    }

    Ok(())
}
