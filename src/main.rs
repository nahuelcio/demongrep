// Allow dead code - public APIs for future use
#![allow(dead_code)]

mod bench;
mod cache;
mod chunker;
mod cli;
mod config;
mod database;
mod embed;
mod error;
mod file;
mod fts;
mod index;
mod mcp;
mod output;
mod rerank;
mod search;
mod server;
mod vectordb;
mod watch;

use anyhow::Result;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Check for quiet mode early (before tracing init)
    let args: Vec<String> = std::env::args().collect();
    let is_quiet = args.iter().any(|a| a == "-q" || a == "--quiet");
    let is_json = args.iter().any(|a| a == "--json");
    let is_agent = args.iter().any(|a| a == "--agent");

    // Skip tracing in quiet mode, JSON output, or agent mode
    if !is_quiet && !is_json && !is_agent {
        // Initialize tracing
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "demongrep=info".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();

        info!("Starting demongrep v{}", env!("CARGO_PKG_VERSION"));
    }

    // Parse CLI and execute command
    cli::run().await
}
