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

#[cfg(target_os = "macos")]
fn auto_configure_ort_dylib_path() -> Option<String> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return None;
    }

    let candidates = [
        "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.1.24.1.dylib",
        "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
        "/usr/local/opt/onnxruntime/lib/libonnxruntime.1.24.1.dylib",
        "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib",
    ];

    for path in candidates {
        if std::path::Path::new(path).is_file() {
            std::env::set_var("ORT_DYLIB_PATH", path);
            return Some(path.to_string());
        }
    }

    None
}

#[cfg(not(target_os = "macos"))]
fn auto_configure_ort_dylib_path() -> Option<String> {
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    let auto_ort_path = auto_configure_ort_dylib_path();

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
        if let Some(path) = auto_ort_path {
            info!("Auto-configured ORT_DYLIB_PATH={}", path);
        }
    }

    // Parse CLI and execute command
    cli::run().await
}
