use anyhow::{anyhow, Context, Result};
use std::env;
use std::path::PathBuf;

use crate::cli::install_common;

fn get_config_path() -> Result<PathBuf> {
    if cfg!(target_os = "windows") {
        let appdata =
            env::var("APPDATA").context("APPDATA environment variable not set on Windows")?;
        Ok(PathBuf::from(appdata).join("Claude").join("settings.json"))
    } else {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
        Ok(home.join(".config/claude-code/config.json"))
    }
}

pub fn run(_global: bool, project_path: Option<PathBuf>, dry_run: bool) -> Result<()> {
    let config_path = get_config_path()?;
    install_common::install_agent_mcp("Claude Code", &config_path, project_path, dry_run)
}
