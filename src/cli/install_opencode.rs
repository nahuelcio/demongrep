use anyhow::{anyhow, Result};
use std::path::PathBuf;

use crate::cli::install_common;

fn get_config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    Ok(home.join(".config").join("opencode").join("config.json"))
}

pub fn run(project_path: Option<PathBuf>, dry_run: bool) -> Result<()> {
    let config_path = get_config_path()?;
    install_common::install_agent_mcp("OpenCode", &config_path, project_path, dry_run)
}
